"""Real HF APIs, tied embeddings, and fp32-master weight export."""

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("GDTENSOR_TORCH_TESTS"), reason="set GDTENSOR_TORCH_TESTS=1"
)


def _fixture(key):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    root = Path(__file__).resolve().parents[1] / "fixtures/gdcp-v2/configs"
    config = AutoConfig.from_pretrained(root / key, local_files_only=True)
    config.hidden_size, config.intermediate_size = 32, 64
    config.num_hidden_layers, config.num_attention_heads = 2, 4
    config.num_key_value_heads, config.head_dim, config.vocab_size = 2, 8, 67
    for name in ("layer_types", "mlp_layer_types"):
        value = getattr(config, name, None)
        if isinstance(value, list):
            setattr(config, name, value[:2])
    model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)
    model.tie_weights()
    optimizer = torch.optim.AdamW(model.parameters())
    for index, param in enumerate(model.parameters()):
        _parameter_state(param, index, optimizer)
    return model, optimizer


def _parameter_state(param, index, optimizer):
    import torch

    master = (
        torch.arange(param.numel(), dtype=torch.float32).reshape(param.shape) / 3000
        + index / 100
    )
    optimizer.state[param] = {
        "step": torch.tensor(3.0),
        "master_param": master,
        "exp_avg": master.clone(),
        "exp_avg_sq": master.square(),
    }
    with torch.no_grad():
        param.copy_(master.to(torch.bfloat16))


@pytest.mark.parametrize("key", ["4b", "14b", "24b"])
@pytest.mark.parametrize("dtype", ["fp32", "bf16"])
@pytest.mark.parametrize("limit", [None, 20000])
def test_weights_export_real_hf_readers_and_native_writers(tmp_path, key, dtype, limit):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM
    from gdcp import load_hf, save_hf
    from gdcp.hf import read_headers

    model, optimizer = _fixture(key)
    source, native = tmp_path / "source", tmp_path / "native"
    state = {"model": model, "optimizer": optimizer}
    save_hf(source, state, dtype=dtype, max_shard_size=limit)
    headers = read_headers(source)
    parameters = dict(model.named_parameters())
    loaded = {}
    for name, entry in headers.items():
        with safe_open(str(source / entry["file"]), framework="pt") as file:
            tensor = file.get_tensor(name)
        expected = (
            optimizer.state[parameters[name]]["master_param"]
            if dtype == "fp32"
            else parameters[name]
        )
        assert tensor.dtype == expected.dtype
        assert torch.equal(tensor, expected), name
        loaded[name] = tensor
    if key == "4b":
        assert "lm_head.weight" not in headers
    restored = AutoModelForCausalLM.from_pretrained(
        source, local_files_only=True, dtype="auto"
    )
    assert restored.get_input_embeddings().weight.dtype == (
        torch.float32 if dtype == "fp32" else torch.bfloat16
    )
    native.mkdir()
    save_file(loaded, str(native / "model.safetensors"), metadata={"format": "pt"})
    with torch.no_grad():
        for param in parameters.values():
            target = (
                optimizer.state[param]["master_param"] if dtype == "fp32" else param
            )
            target.zero_()
    load_hf(native, state if dtype == "fp32" else model, dtype=dtype)
    for name, param in parameters.items():
        target = optimizer.state[param]["master_param"] if dtype == "fp32" else param
        with safe_open(str(native / "model.safetensors"), framework="pt") as file:
            assert torch.equal(target, file.get_tensor(name))

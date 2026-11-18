"""Verify exact official config bytes and count unique meta parameters."""

import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "tests/fixtures/gdcp-v2/configs"


def describe(key):
    directory = CONFIGS / key
    provenance = json.loads((directory / "provenance.json").read_text())
    digest = hashlib.sha256((directory / "config.json").read_bytes()).hexdigest()
    assert digest == provenance["sha256"], (key, digest)
    config = AutoConfig.from_pretrained(directory, local_files_only=True)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)
    model.tie_weights()
    parameters = sum(p.numel() for p in model.parameters())
    names = dict(model.named_parameters())
    aliases = {}
    canonical = {id(p): n for n, p in names.items()}
    for name, param in model.named_parameters(remove_duplicate=False):
        if name != canonical[id(param)]:
            aliases[name] = canonical[id(param)]
    return dict(
        provenance,
        parameters=parameters,
        parameter_tensors=len(names),
        nominal_state_bytes=14 * parameters,
        full_state_payload_bytes=14 * parameters + 4 * len(names),
        aliases=aliases,
        model_class=type(model).__name__,
    )


if __name__ == "__main__":
    import sys

    result = {key: describe(key) for key in ("4b", "14b", "24b")}
    data = json.dumps(result, indent=2, sort_keys=True) + "\n"
    Path(sys.argv[1]).write_text(data)
    print(data)

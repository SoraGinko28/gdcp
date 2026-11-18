"""Actual HF file layouts must match the acceptance cell being exercised."""

import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
pytest.importorskip("transformers")
pytest.importorskip("huggingface_hub")

from safetensors import safe_open
from safetensors.torch import save_file

from scripts import gdcp_v2_hf_interop as interop


def write_fixture(root, *, sharded):
    root.mkdir()
    tensors = {"first": torch.arange(8, dtype=torch.float32), "second": torch.ones(8)}
    if not sharded:
        save_file(tensors, str(root / "model.safetensors"))
        return tensors
    weight_map = {}
    for number, (name, tensor) in enumerate(tensors.items(), 1):
        filename = f"model-{number:05d}-of-00002.safetensors"
        save_file({name: tensor}, str(root / filename))
        weight_map[name] = filename
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 64}, "weight_map": weight_map})
    )
    return tensors


@pytest.mark.parametrize("sharded", [False, True])
def test_actual_single_and_sharded_layout_facts(tmp_path, sharded):
    root = tmp_path / "weights"
    write_fixture(root, sharded=sharded)
    facts = interop.validate_layout(root, sharded=sharded)
    assert facts["status"] == "pass"
    assert facts["requested"] == facts["actual"] == ("sharded" if sharded else "single")
    assert facts["file_count"] == (2 if sharded else 1)
    assert facts["tensor_count"] == 2
    assert facts["index_file"] == ("model.safetensors.index.json" if sharded else None)
    assert facts["indexed_file_coverage"] == ("pass" if sharded else "not-applicable")


def test_requested_sharded_rejects_single_file_even_with_valid_tensors(tmp_path):
    root = tmp_path / "weights"
    write_fixture(root, sharded=False)
    with pytest.raises(
        ValueError, match="requires multiple safetensors files and an index"
    ):
        interop.validate_layout(root, sharded=True)


def test_requested_single_rejects_sharded_files(tmp_path):
    root = tmp_path / "weights"
    write_fixture(root, sharded=True)
    with pytest.raises(ValueError, match="requested single layout"):
        interop.validate_layout(root, sharded=False)


def test_index_must_cover_every_on_disk_safetensors_file(tmp_path):
    root = tmp_path / "weights"
    write_fixture(root, sharded=True)
    save_file({"unindexed": torch.ones(1)}, str(root / "extra.safetensors"))
    with pytest.raises(ValueError, match="index file coverage"):
        interop.validate_layout(root, sharded=True)


def test_index_tensor_names_must_match_actual_shard_headers(tmp_path):
    root = tmp_path / "weights"
    write_fixture(root, sharded=True)
    path = root / "model.safetensors.index.json"
    index = json.loads(path.read_text())
    index["weight_map"]["wrong_name"] = index["weight_map"].pop("second")
    path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="index does not match header"):
        interop.validate_layout(root, sharded=True)


def test_shard_names_must_have_complete_canonical_numbering(tmp_path):
    root = tmp_path / "weights"
    write_fixture(root, sharded=True)
    path = root / "model.safetensors.index.json"
    index = json.loads(path.read_text())
    old_name = index["weight_map"]["second"]
    new_name = "model-00003-of-00003.safetensors"
    (root / old_name).rename(root / new_name)
    index["weight_map"]["second"] = new_name
    path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="complete canonical HF shard filenames"):
        interop.validate_layout(root, sharded=True)


@pytest.mark.parametrize("sharded", [False, True])
def test_native_writer_validates_layout_and_preserves_actual_tensor_values(
    monkeypatch, tmp_path, sharded
):
    root = tmp_path / "source"
    tensors = write_fixture(root, sharded=sharded)
    monkeypatch.setattr(
        interop, "check_disk", lambda additional=0: {"additional_bytes": additional}
    )
    args = SimpleNamespace(
        path=root,
        native_output=tmp_path / "native",
        max_shard_size=32 if sharded else None,
    )
    result = interop.write_native(args, interop.read_headers(root))
    assert result["status"] == "pass"
    layout = result["layout"]
    assert isinstance(layout, dict)
    assert layout["file_count"] == (2 if sharded else 1)
    restored = {}
    for filename in result["files"]:
        with safe_open(str(args.native_output / filename), framework="pt") as reader:
            restored.update({name: reader.get_tensor(name) for name in reader.keys()})
    assert restored.keys() == tensors.keys()
    assert all(torch.equal(tensors[name], restored[name]) for name in tensors)


def test_native_writer_rejects_requested_sharding_that_emits_one_file(
    monkeypatch, tmp_path
):
    root = tmp_path / "source"
    write_fixture(root, sharded=False)
    monkeypatch.setattr(interop, "check_disk", lambda additional=0: {})
    args = SimpleNamespace(
        path=root, native_output=tmp_path / "native", max_shard_size=65536
    )
    with pytest.raises(
        ValueError, match="requires multiple safetensors files and an index"
    ):
        interop.write_native(args, interop.read_headers(root))

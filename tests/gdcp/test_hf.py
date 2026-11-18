"""HF headers, deterministic shared-file output, and native-file resharding."""

import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from safetensors.numpy import save_file

from gdcp.hf import load_state, read_headers, save_state, shard_size_bytes
from gdcp.state import RankContext
from gdtensor_conn.coord import ThreadCoordinator
from tests.support.gdcp import assert_ok, assert_values, numpy_checkpoint


def _collective(call, path, fixture, **kwargs):
    world = len(fixture.states)
    coordinator = ThreadCoordinator(world, timeout=30)

    def rank_call(rank):
        return call(
            path, fixture.states[rank], RankContext(rank, world, coordinator), **kwargs
        )

    with ThreadPoolExecutor(max_workers=world) as executor:
        return list(executor.map(rank_call, range(world)))


@pytest.mark.parametrize("world", [1, 2, 3, 4])
@pytest.mark.parametrize("limit", [None, 180])
def test_hf_full_state_shared_files_reshard_and_byte_identity(tmp_path, world, limit):
    source = numpy_checkpoint(world, initialized=True, names=("model", "optimizer"))
    first, second = tmp_path / "first", tmp_path / "second"
    assert_ok(_collective(save_state, first, source, max_shard_size=limit))
    assert_ok(_collective(save_state, second, source, max_shard_size=limit))
    files = sorted(
        path.relative_to(first) for path in first.rglob("*") if path.is_file()
    )
    assert files == sorted(
        path.relative_to(second) for path in second.rglob("*") if path.is_file()
    )
    for filename in files:
        assert (first / filename).read_bytes() == (second / filename).read_bytes(), (
            filename
        )
    headers = read_headers(first)
    assert headers["step"]["shape"] == []
    assert headers["empty"]["shape"] == [0, 3]
    assert headers["optimizer.step"]["shape"] == []
    # A native reader needs neither a GDCP manifest nor a connector marker.
    assert not (first / "gdcp_manifest.json").exists()
    (first / ".gdtensor_committed").unlink()
    target = numpy_checkpoint(
        5 - world, initialized=False, names=("optimizer", "model")
    )
    assert_ok(_collective(load_state, first, target))
    assert_values(target, source)
    if limit is not None:
        index = json.loads((first / "model.safetensors.index.json").read_text())
        assert set(index["weight_map"]) == set(headers)
        assert all("/slot" not in name for name in headers)


@pytest.mark.parametrize("world", [1, 2, 3, 4])
def test_native_safetensors_output_loads_without_private_metadata(tmp_path, world):
    fixture = numpy_checkpoint(world, initialized=False)
    tensors = {
        name: value.reshape(()) if name == "step" else value
        for name, value in fixture.values["model"].items()
    }
    # BF16 is represented as raw U16 in the NumPy fixture. Emit its expected
    # BF16 code by changing only the equally-sized on-disk header dtype.
    save_file(tensors, str(tmp_path / "model.safetensors"))
    filename = tmp_path / "model.safetensors"
    data = filename.read_bytes().replace(b'"dtype":"U16"', b'"dtype":"BF16"')
    # U16 and BF16 differ by one character, so reserialize a valid padded header.
    import struct

    length = struct.unpack("<Q", filename.read_bytes()[:8])[0]
    header = json.loads(filename.read_bytes()[8 : 8 + length])
    header["bf16"]["dtype"] = "BF16"
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    filename.write_bytes(
        struct.pack("<Q", len(raw)) + raw + filename.read_bytes()[8 + length :]
    )
    assert data  # the native payload itself was not converted or gathered
    assert_ok(_collective(load_state, tmp_path, fixture))
    fixture.fabrics["model"].assert_values(fixture.values["model"])


@pytest.mark.parametrize(
    "value, expected",
    [(None, None), (20, 20), ("5GB", 5_000_000_000), ("1.5MiB", 1572864)],
)
def test_hf_shard_size_units(value, expected):
    assert shard_size_bytes(value) == expected


@pytest.mark.parametrize("value", [0, -1, "wrong", "0GB", 2.5])
def test_hf_shard_size_rejects_invalid_limits(value):
    with pytest.raises(ValueError, match="max_shard_size"):
        shard_size_bytes(value)

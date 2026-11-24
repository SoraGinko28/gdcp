"""Operation-local GDCP inventory indexes and metadata lifetimes."""

import copy
import json
import weakref
from types import SimpleNamespace

import pytest

import gdtensor.builder as gb
from gdtensor.tag import WorldSize
from gdcp import core, desc
from gdcp.policy import PerRankPolicy
from gdcp.state import RankContext
from tests.support.gdcp import assert_ok, collective, numpy_checkpoint


def _layout(rows=32, world=8):
    table = gb.DTensorTable(
        [
            gb.NamedDTensor(
                f"weight.{row}",
                gb.NaiveDTensor(
                    gb.GlobalTensorMeta([128, 4], "F32", 4),
                    [gb.Shard(0)],
                    gb.ArrangedMesh(WorldSize(world)),
                ),
            )
            for row in range(rows)
        ]
    )
    return PerRankPolicy().plan(SimpleNamespace(table=table), world)


@pytest.mark.parametrize("world", [4, 8, 16])
def test_inventory_classifies_each_group_once(monkeypatch, world):
    table, layout = _layout(world=world)
    counts = {"equalities": 0, "rectangles": 0, "hashes": 0}
    equality = gb.DisjointStorageMeshList.__eq__
    group_hash = gb.DisjointStorageMeshList.__hash__
    rectangles = desc._layout_rectangles

    def equal(left, right):
        counts["equalities"] += 1
        return equality(left, right)

    def hashed(group):
        counts["hashes"] += 1
        return group_hash(group)

    def convert(group):
        counts["rectangles"] += 1
        return rectangles(group)

    monkeypatch.setattr(gb.DisjointStorageMeshList, "__eq__", equal)
    monkeypatch.setattr(gb.DisjointStorageMeshList, "__hash__", hashed)
    monkeypatch.setattr(desc, "_layout_rectangles", convert)
    value = desc.describe_component(table, layout, world)
    assert len(value["files"]) == world
    assert counts["equalities"] <= 2 * world
    assert counts["hashes"] <= 3 * world
    assert counts["rectangles"] == world


def test_inventory_preserves_equal_distinct_and_first_path(monkeypatch):
    table, layout = _layout()
    before = desc.describe_component(table, layout, 8)
    pos = list(layout.tensor_entries)[1]
    layout.tensor_entries[pos] = copy.deepcopy(layout.tensor_entries[pos])
    assert desc.describe_component(table, layout, 8) == before
    # Each call observes mutable file paths; no identity cache crosses calls.
    leader = next(iter(layout.file_paths))
    old = layout.file_paths[leader]
    layout.file_paths[leader] = "renamed.safetensors"
    value = desc.describe_component(table, layout, 8)
    assert old not in value["files"]
    assert "renamed.safetensors" in value["files"]


def test_inventory_rejects_group_without_path():
    table, layout = _layout()
    del layout.file_paths[next(iter(layout.file_paths))]
    with pytest.raises(ValueError, match="no file path"):
        desc.describe_component(table, layout, 8)


def test_load_rebuilds_each_component_once_including_unrequested(tmp_path, monkeypatch):
    source = numpy_checkpoint(2, initialized=True, names=("model", "optimizer"))
    assert_ok(collective(core.save, tmp_path, source))
    target = numpy_checkpoint(2, initialized=False)
    calls = []
    original = desc.rebuild_component

    def rebuild(component, world):
        calls.append(component)
        return original(component, world)

    monkeypatch.setattr(desc, "rebuild_component", rebuild)
    assert_ok(collective(core.load, tmp_path, target))
    assert len(calls) == 2 * 2


def test_reused_layout_still_rejects_invalid_unrequested_component(tmp_path):
    source = numpy_checkpoint(2, initialized=True, names=("model", "optimizer"))
    assert_ok(collective(core.save, tmp_path, source))
    manifest = tmp_path / desc.MANIFEST_NAME
    document = json.loads(manifest.read_text())
    files = document["components"]["optimizer"]["files"]
    next(iter(files.values()))["slots"][0][0] = "missing.tensor"
    manifest.write_bytes(desc.json_bytes(document))
    target = numpy_checkpoint(2, initialized=False)
    results = collective(core.load, tmp_path, target)
    assert all(type(result) is gb.Err for result in results)
    assert all("missing tensor" in result.value.msg for result in results)


def test_load_releases_manifest_before_payload(tmp_path, monkeypatch):
    source = numpy_checkpoint(2, initialized=True)
    assert_ok(collective(core.save, tmp_path, source))
    target = numpy_checkpoint(2, initialized=False)
    references = []
    original_read = desc.read_manifest
    original_copy = core._copy_load_payloads

    def read(*args, **kwargs):
        description = original_read(*args, **kwargs)
        references.append(weakref.ref(description))
        return description

    def payload(*args, **kwargs):
        assert references and all(ref() is None for ref in references)
        return original_copy(*args, **kwargs)

    monkeypatch.setattr(desc, "read_manifest", read)
    monkeypatch.setattr(core, "_copy_load_payloads", payload)
    assert_ok(collective(core.load, tmp_path, target))


def test_agreement_releases_schema_before_votes():
    references = []

    class Schema(dict):
        pass

    class Coordinator:
        def wait(self):
            pass

        def vote(self, ok):
            assert references and references[0]() is None
            return ok

    def prepare():
        schema = Schema({"ordered tensor schema": {"tensor": [1, 2, 3]}})
        references.append(weakref.ref(schema))
        return 7, schema

    result = core._agree(RankContext(0, 1, Coordinator()), prepare)
    assert type(result) is gb.Ok, result
    assert result.value == 7


def test_inventory_preserves_first_equal_file_leader():
    table, layout = _layout()
    before = desc.describe_component(table, layout, 8)
    pos = list(layout.tensor_entries)[1]
    layout.tensor_entries[pos] = copy.deepcopy(layout.tensor_entries[pos])
    layout.file_paths[pos] = "later.safetensors"
    assert desc.describe_component(table, layout, 8) == before


def test_fragmented_rectangles_are_recorded_once_and_in_order(monkeypatch):
    table, layout = _layout()
    leader = next(iter(layout.file_paths))
    group = layout.tensor_entries[leader]
    original = group.disjoint_storage_mesh_list[0]
    group.disjoint_storage_mesh_list = [
        gb.SlicedMesh(original.underlying_mesh, [gb.Slice(0, 16), original.slices[1]]),
        gb.SlicedMesh(original.underlying_mesh, [gb.Slice(16, 32), original.slices[1]]),
    ]
    assert type(layout.check_constraints()) is gb.Ok
    calls = []
    convert = desc._layout_rectangles

    def rectangles(value):
        calls.append(value)
        return convert(value)

    monkeypatch.setattr(desc, "_layout_rectangles", rectangles)
    inventory = desc.describe_component(table, layout, 8)
    assert len(calls) == 8
    path = layout.file_paths[leader]
    assert inventory["files"][path]["rectangles"] == [
        [[0, 16, None], [0, 1, None]],
        [[16, 32, None], [0, 1, None]],
    ]

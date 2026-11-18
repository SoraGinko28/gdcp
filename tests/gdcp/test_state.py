"""Torch-free policy and deterministic manifest round trips."""

from gdcp import desc
from gdcp.policy import PerRankPolicy
from tests.support.gdcp import numpy_checkpoint


def test_manifest_roundtrip_rebuilds_equal_storage_semantics(tmp_path):
    fixture = numpy_checkpoint(2, initialized=True)
    component = fixture.states[0].components[0]
    table, layout = PerRankPolicy().plan(component, 2)
    document = desc.describe_component(
        table, layout, 2, objects=True, logical_shapes=component.logical_shapes
    )
    rebuilt_table, rebuilt_layout = desc.rebuild_component(document, 2)
    assert rebuilt_table == table
    assert rebuilt_layout == layout
    checkpoint = desc.CheckpointDesc(
        {
            "gdcp_version": 1,
            "framework": "fsdp2",
            "world_size": 2,
            "global_steps": None,
            "components": {"model": document},
            "rank_state": {},
        }
    )
    desc.write_manifest(tmp_path, checkpoint)
    assert desc.read_manifest(tmp_path).to_dict() == checkpoint.to_dict()
    original = (tmp_path / desc.MANIFEST_NAME).read_bytes()
    desc.write_manifest(tmp_path, checkpoint)
    assert (tmp_path / desc.MANIFEST_NAME).read_bytes() == original


def test_manifest_preserves_logical_scalar_and_empty_shapes():
    fixture = numpy_checkpoint(3, initialized=True)
    component = fixture.states[0].components[0]
    table, layout = PerRankPolicy().plan(component, 3)
    document = desc.describe_component(
        table, layout, 3, logical_shapes=component.logical_shapes
    )
    tensors = {entry["id"]: entry for entry in document["tensors"]}
    assert tensors["step"]["shape"] == [1]
    assert tensors["step"]["logical_shape"] == []
    assert tensors["empty"]["shape"] == [0, 3]
    assert tensors["bf16"]["dtype"] == "BF16"


def test_policy_deduplicates_replicated_tensors_across_writer_ranks():
    fixture = numpy_checkpoint(3, initialized=True)
    component = fixture.states[0].components[0]
    table, layout = PerRankPolicy().plan(component, 3)
    document = desc.describe_component(table, layout, 3)
    for key in ("buffer", "step"):
        occurrences = sum(
            slot[0] == key
            for file in document["files"].values()
            for slot in file["slots"]
        )
        assert occurrences == 1, (key, document)

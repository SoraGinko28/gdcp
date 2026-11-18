"""python -m tests.gdcp.spikes.b_scalars_empty (single-rank connector probe)."""

import os
from pathlib import Path

import torch

import gdtensor.builder as gb
import gdtensor.tag as gt
from gdtensor.io import rank_load, rank_save
from tests.support.gdcp_spikes import BytesAccessor, storage


def table_for(scalar_shape):
    mesh = gb.ArrangedMesh(gt.WorldSize(1))
    entries = [("step", scalar_shape, "F32", 4), ("empty", [0, 3], "BF16", 2)]
    return gb.DTensorTable(
        [
            gb.NamedDTensor(
                key,
                gb.NaiveDTensor(
                    gb.GlobalTensorMeta(shape, dtype, size), [gb.Replicate()], mesh
                ),
            )
            for key, shape, dtype, size in entries
        ]
    )


def roundtrip(table, root):
    constraints = table.check_constraints()
    if type(constraints) is gb.Err:
        return constraints
    values = {
        "step": torch.tensor(17.0),
        "empty": torch.empty((0, 3), dtype=torch.bfloat16),
    }
    accessor = BytesAccessor(values, {key: 0 for key in values})
    view = storage(root, table, 1)
    compute = gb.ComputeDTensorsView(list(values), table, accessor)
    result = rank_save(0, compute, view)
    if type(result) is gb.Err:
        return result
    values["step"].zero_()
    result = rank_load(0, compute, view)
    assert type(result) is gb.Ok, result
    assert values["step"].item() == 17.0
    assert values["empty"].shape == (0, 3)
    return result


def main():
    root = Path(os.environ["GDCP_SPIKE_ROOT"]) / "b"
    direct = roundtrip(table_for([]), root / "direct")
    print(f"SPIKE B native scalar: {direct}", flush=True)
    adapted = roundtrip(table_for([1]), root / "adapted")
    assert type(adapted) is gb.Ok, adapted
    print(
        "SPIKE B PASS scalar encoded as [1], real scalar payload restored in place; zero-size BF16 retained with no payload",
        flush=True,
    )


if __name__ == "__main__":
    main()

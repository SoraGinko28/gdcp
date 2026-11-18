"""torchrun --standalone --nproc-per-node=2 --module tests.gdcp.spikes.c_restore."""

import torch
from torch.distributed import destroy_process_group, get_rank, init_process_group
from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict
from torch.distributed.checkpoint.state_dict import set_optimizer_state_dict

from tests.support.gdcp_spikes import model


def locals_of(state):
    return {key: value.to_local().detach().clone() for key, value in state.items()}


def check_optimizer(expected, actual):
    assert expected["param_groups"] == actual["param_groups"]
    for key, fields in expected["state"].items():
        for name, tensor in fields.items():
            got = actual["state"][key][name]
            left = tensor.to_local() if hasattr(tensor, "to_local") else tensor
            right = got.to_local() if hasattr(got, "to_local") else got
            assert torch.equal(left, right), (key, name)


def main():
    init_process_group("gloo")
    source = model()
    optimizer = torch.optim.AdamW(source.parameters(), lr=0.03)
    for parameter in source.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    saved_model = {
        key: value.detach().clone() for key, value in source.state_dict().items()
    }
    saved_optimizer = get_optimizer_state_dict(source, optimizer)
    target = model()
    target_optimizer = torch.optim.AdamW(target.parameters(), lr=0.1)
    before = locals_of(target.state_dict())
    initialized = get_optimizer_state_dict(target, target_optimizer)
    assert initialized["state"], (
        "get_optimizer_state_dict must initialize empty AdamW state"
    )
    assert all(
        torch.equal(before[key], value)
        for key, value in locals_of(target.state_dict()).items()
    )
    identities = [id(parameter) for parameter in target.parameters()]
    set_optimizer_state_dict(target, target_optimizer, saved_optimizer)
    target.load_state_dict(saved_model)
    assert identities == [id(parameter) for parameter in target.parameters()]
    assert all(
        torch.equal(value, before_value)
        for value, before_value in zip(
            locals_of(target.state_dict()).values(), locals_of(saved_model).values()
        )
    )
    check_optimizer(saved_optimizer, get_optimizer_state_dict(target, target_optimizer))
    print(
        f"SPIKE C PASS rank={get_rank()} optimizer initialization preserved parameters; optimizer then model restored in place",
        flush=True,
    )
    destroy_process_group()


if __name__ == "__main__":
    main()

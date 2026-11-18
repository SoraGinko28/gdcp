"""torchrun --standalone --nproc-per-node=2 --module tests.gdcp.spikes.d_gloo."""

from torch.distributed import destroy_process_group, get_rank, get_world_size
from torch.distributed import init_process_group

from gdtensor_conn.coord import GlooCoordinator
from tests.support.gdcp_spikes import model


def main():
    init_process_group("gloo")
    sharded = model()
    assert all(hasattr(parameter, "to_local") for parameter in sharded.parameters())
    coord = GlooCoordinator()
    assert coord.vote(True)
    for failing_rank in range(get_world_size()):
        assert not coord.vote(get_rank() != failing_rank)
    coord.wait()
    assert coord.vote(True)
    print(
        f"SPIKE D PASS rank={get_rank()} CPU FSDP2 default Gloo vote/barrier and recovery",
        flush=True,
    )
    destroy_process_group()


if __name__ == "__main__":
    main()

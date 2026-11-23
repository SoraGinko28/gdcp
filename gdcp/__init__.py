"""Framework-neutral checkpoints over GDTensor, with an FSDP2 facade."""

from .access import DataAccessorRequest
from .checkpointer import FSDP2Checkpointer
from .core import inspect, load, save


def save_hf(path, model_or_state, dtype="fp32", max_shard_size=None):
    """Export HF weights; fp32 exports master weights when optimizer is supplied."""
    from .adapters.hf import save_hf as implementation

    return implementation(
        path, model_or_state, dtype=dtype, max_shard_size=max_shard_size
    )


def load_hf(path, model_or_state, **kwargs):
    """Load single-file or indexed HF weights directly into local tensor shards."""
    from .adapters.hf import load_hf as implementation

    return implementation(path, model_or_state, **kwargs)


__all__ = [
    "DataAccessorRequest",
    "FSDP2Checkpointer",
    "inspect",
    "load",
    "save",
    "save_hf",
    "load_hf",
]

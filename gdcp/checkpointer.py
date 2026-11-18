"""Reference checkpointer parameter surface with explicit option validation."""

from pathlib import Path


def _reject(options, defaults, kwargs):
    for name, value in options.items():
        if value != defaults.get(name):
            raise NotImplementedError(f"{name} is not supported in GDCP v1")
    if kwargs:
        raise NotImplementedError(f"{next(iter(kwargs))} is not supported in GDCP v1")


class FSDP2Checkpointer:
    """Synchronous FSDP2/AdamW checkpoints over Gloo or CUDA/CPU groups."""

    @classmethod
    def save(
        cls,
        path,
        checkpoint_state,
        async_fast_checkpoint=False,
        enable_shm_upload_ckpt_tmp=False,
        global_steps=None,
        io_metrics_on=False,
        loader_in_split_mode=False,
        callback=None,
        modifying_ckpt=False,
        role=None,
        barrier_rpc_tag_suffix=None,
        ignore_append_global_steps_to_folder=False,
        process_group=None,
        **kwargs,
    ) -> None:
        options = {
            "async_fast_checkpoint": async_fast_checkpoint,
            "enable_shm_upload_ckpt_tmp": enable_shm_upload_ckpt_tmp,
            "loader_in_split_mode": loader_in_split_mode,
            "callback": callback,
            "modifying_ckpt": modifying_ckpt,
            "role": role,
            "process_group": process_group,
        }
        defaults = {
            name: False
            for name in (
                "async_fast_checkpoint",
                "enable_shm_upload_ckpt_tmp",
                "loader_in_split_mode",
                "modifying_ckpt",
            )
        }
        checkpoint_format = kwargs.pop("checkpoint_format", "safetensors")
        hf_dtype = kwargs.pop("hf_dtype", None)
        max_shard_size = kwargs.pop("max_shard_size", None)
        _reject(options, defaults, kwargs)
        if checkpoint_format not in ("safetensors", "dcp", "hf"):
            raise ValueError(f"unsupported checkpoint_format {checkpoint_format!r}")
        if checkpoint_format != "hf" and (
            hf_dtype is not None or max_shard_size is not None
        ):
            raise ValueError(
                "hf_dtype and max_shard_size require checkpoint_format='hf'"
            )
        from .adapters.fsdp2 import save_checkpoint

        root = Path(path)
        if global_steps is not None and not ignore_append_global_steps_to_folder:
            root = root / f"global_step_{global_steps}"
        save_checkpoint(
            root,
            checkpoint_state,
            global_steps,
            checkpoint_format=checkpoint_format,
            hf_dtype=hf_dtype,
            max_shard_size=max_shard_size,
        )

    @classmethod
    def load(
        cls,
        path,
        checkpoint_state,
        enable_shm_download_ckpt_tmp=False,
        broadcast_model_tensors=False,
        read_comm_overlap_model_tensors=False,
        loader_in_split_mode=False,
        io_metrics_on=False,
        role=None,
        ignore_model_keys=None,
        process_group=None,
        rank=None,
        global_batch_size=None,
        seqlen=None,
        num_workers=None,
        dsp_size=1,
        cp_size=1,
        load_common_state_only=False,
        **kwargs,
    ) -> None:
        options = {
            "enable_shm_download_ckpt_tmp": enable_shm_download_ckpt_tmp,
            "broadcast_model_tensors": broadcast_model_tensors,
            "read_comm_overlap_model_tensors": read_comm_overlap_model_tensors,
            "loader_in_split_mode": loader_in_split_mode,
            "role": role,
            "ignore_model_keys": ignore_model_keys,
            "process_group": process_group,
            "rank": rank,
            "global_batch_size": global_batch_size,
            "seqlen": seqlen,
            "num_workers": num_workers,
            "dsp_size": dsp_size,
            "cp_size": cp_size,
        }
        defaults = {
            "enable_shm_download_ckpt_tmp": False,
            "broadcast_model_tensors": False,
            "read_comm_overlap_model_tensors": False,
            "loader_in_split_mode": False,
            "dsp_size": 1,
            "cp_size": 1,
        }
        _reject(options, defaults, kwargs)
        from .adapters.fsdp2 import load_checkpoint

        load_checkpoint(path, checkpoint_state, load_common_state_only)

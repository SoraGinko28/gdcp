"""Reference checkpointer parameter surface with explicit option validation."""

from .desc import checkpoint_path

_SAVE_FORMATS = ("safetensors", "dcp", "hf")


def _save_route(checkpoint_format, hf_dtype, max_shard_size):
    route = next((fmt for fmt in _SAVE_FORMATS if fmt == checkpoint_format), None)
    if route is None:
        raise ValueError(f"unsupported checkpoint_format {checkpoint_format!r}")
    if route != "hf" and (hf_dtype is not None or max_shard_size is not None):
        raise ValueError("hf_dtype and max_shard_size require checkpoint_format='hf'")
    if route == "hf" and hf_dtype is not None:
        return "HF weights"
    return route


def _save_root(path, global_steps, ignore_append):
    root = checkpoint_path(path)
    if global_steps is not None and not ignore_append:
        return root / f"global_step_{global_steps}"
    return root


def _reject(options, defaults, kwargs):
    for name, value in options.items():
        if value != defaults.get(name):
            raise NotImplementedError(f"{name} is not supported in GDCP v1")
    if kwargs:
        raise NotImplementedError(f"{next(iter(kwargs))} is not supported in GDCP v1")


def _fsdp2_adapter(preflight):
    try:
        from .adapters import fsdp2
    except ModuleNotFoundError as exc:
        # Diagnose unsupported arguments even without the optional Torch dependency.
        if exc.name == "torch":
            preflight()
        raise
    return fsdp2


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

        def preflight():
            _reject(options, defaults, kwargs)
            route = _save_route(checkpoint_format, hf_dtype, max_shard_size)
            root = _save_root(path, global_steps, ignore_append_global_steps_to_folder)
            return root, route

        _fsdp2_adapter(preflight).save_checkpoint(
            path,
            checkpoint_state,
            global_steps,
            checkpoint_format=checkpoint_format,
            hf_dtype=hf_dtype,
            max_shard_size=max_shard_size,
            _preflight=preflight,
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

        def preflight():
            _reject(options, defaults, kwargs)

        _fsdp2_adapter(preflight).load_checkpoint(
            path, checkpoint_state, load_common_state_only, _preflight=preflight
        )

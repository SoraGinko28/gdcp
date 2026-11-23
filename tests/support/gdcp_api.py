"""Signatures of the reference FSDP2 checkpointer that GDCP keeps parity with.

The reference's io_metrics_on default reads its environment-backed config.
This fixture captures that config with I/O metrics off, matching the GDCP v1 spec.
Annotations and decorators are deliberately excluded from API parity.
"""

REQUIRED = "<required>"
SAVE_PARAMETERS = [
    ("path", REQUIRED),
    ("checkpoint_state", REQUIRED),
    ("async_fast_checkpoint", False),
    ("enable_shm_upload_ckpt_tmp", False),
    ("global_steps", None),
    ("io_metrics_on", False),
    ("loader_in_split_mode", False),
    ("callback", None),
    ("modifying_ckpt", False),
    ("role", None),
    ("barrier_rpc_tag_suffix", None),
    ("ignore_append_global_steps_to_folder", False),
    ("process_group", None),
    ("data_accessor", None),
    ("kwargs", REQUIRED),
]
LOAD_PARAMETERS = [
    ("path", REQUIRED),
    ("checkpoint_state", REQUIRED),
    ("enable_shm_download_ckpt_tmp", False),
    ("broadcast_model_tensors", False),
    ("read_comm_overlap_model_tensors", False),
    ("loader_in_split_mode", False),
    ("io_metrics_on", False),
    ("role", None),
    ("ignore_model_keys", None),
    ("process_group", None),
    ("rank", None),
    ("global_batch_size", None),
    ("seqlen", None),
    ("num_workers", None),
    ("dsp_size", 1),
    ("cp_size", 1),
    ("load_common_state_only", False),
    ("data_accessor", None),
    ("kwargs", REQUIRED),
]
SAVE_REJECTED = {
    "async_fast_checkpoint": True,
    "enable_shm_upload_ckpt_tmp": True,
    "loader_in_split_mode": True,
    "callback": "unsupported_callback",
    "modifying_ckpt": True,
    "role": "actor",
}
LOAD_REJECTED = {
    "enable_shm_download_ckpt_tmp": True,
    "broadcast_model_tensors": True,
    "read_comm_overlap_model_tensors": True,
    "loader_in_split_mode": True,
    "role": "actor",
    "ignore_model_keys": set(),
    "rank": 0,
    "global_batch_size": 1,
    "seqlen": 1,
    "num_workers": 0,
    "dsp_size": 2,
    "cp_size": 2,
}

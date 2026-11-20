# GDCP v2 formats and real models

The synchronous `FSDP2Checkpointer` retains every positional parameter and default of the reference checkpointer. New save options are passed as keyword arguments:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `checkpoint_format` | `"safetensors"` | `"safetensors"`: one file per rank; `"dcp"`: native PyTorch DCP in `model/` and `optimizer/`; `"hf"`: shared HF-layout safetensors containing model and optimizer state. |
| `hf_dtype` | `None` | `None` preserves full state. `"fp32"` or `"bf16"` selects weights-only HF export, equivalent to `gdcp.save_hf` below. |
| `max_shard_size` | `None` | HF file size limit in bytes or a string such as `"5GB"`; `None` writes one file. Tensors never cross files; an oversized tensor occupies its own file. Applies only to HF saves. |

`load(path, checkpoint_state, ...)` detects GDCP checkpoints from regular
completion files at the root. `gdcp_manifest.json` identifies per-rank
safetensors or DCP and records the component formats. A root
`.gdtensor_committed` identifies a committed HF checkpoint, including older
saves with no tensor files. If both regular root completion files are present,
load reports an ambiguous format. Saves reserve both root completion names and
reject foreign entries at those names before publication. For native checkpoints
without GDCP completion files, load recognizes regular HF tensor files or an HF
index at the root, or regular DCP `.metadata` in a component directory. If both
native layouts are present, load reports an ambiguous format.

```python
FSDP2Checkpointer.save(
    path, {"model": model, "optimizer": optimizer}, checkpoint_format="dcp"
)
FSDP2Checkpointer.load(path, {"model": model, "optimizer": optimizer})
gdcp.save_hf(
    path, {"model": model, "optimizer": optimizer}, dtype="fp32", max_shard_size="5GB"
)
gdcp.load_hf(path, model)
```

`gdcp.save_hf(path, model_or_state, dtype="fp32", max_shard_size=None)` exports weights only. Inputs can be a module, tensor mapping, or a checkpoint dictionary containing `model` and optionally `optimizer`. The dtype strings `"fp32"` and `"bf16"` apply only to floating-point tensors. With an optimizer, fp32 exports use `master_param` for floating-point parameters. Integer and boolean tensors are stored with their original dtypes and values.

`gdcp.load_hf(path, model_or_state, dtype=None)` loads weights into existing local tensor shards. When omitted, `dtype` is inferred from the requested floating-point model tensors and their aliases. In full-state HF checkpoints with a recorded optimizer inventory, inference excludes every key owned by the optimizer. Floating-point targets must have the selected dtype. With an optimizer checkpoint dictionary, `dtype="fp32"` targets its fp32 master weights. Integer and boolean tensors are restored bit for bit into targets with matching dtypes. A target containing only integer and boolean tensors can load with `dtype=None`. This function does not infer or update missing Adam moments from a weights-only checkpoint.

Per-rank safetensors, DCP, and HF all preserve `uint16`, `uint32`, and `uint64` buffers and `int16` buffers with `Shard(0)` placement. HF uses the native safetensors dtype codes `U16`, `U32`, `U64`, and `I16` in both export modes.

Apply FSDP2 `fully_shard` to every decoder layer and then the root. Tied embedding keys are represented by a canonical parameter and aliases: payload and optimizer state are stored once. The adapter restores the complete state dictionary while preserving parameter identity. AdamW subclasses may keep any number of per-parameter tensor fields matching parameter geometry, plus replicated scalar fields. Optimizer fields must already exist in the target, including custom fields. The mixed-precision validation fixture uses a real AdamW implementation that updates fp32 masters and casts them to bf16 parameters.

In all three full-state formats, AdamW group membership is matched by fully
qualified model parameter name. Group and parameter order in the target may
differ from the checkpoint. Load restores the saved order and group options,
populates `param_names` to match `params` in every group, and restores moments
and steps to their corresponding parameters. Incompatible or ambiguous groups
fail before requested tensor-key checks and before tensor or rank-state payload
reads. For Torch-written native DCP, GDCP uses a restricted decoder for group
membership. All ranks validate membership and tensor inventory, then check the
backing tensor records with a restricted decoder, before GDCP reads scalar
optimizer state or separately stored group options.

CPU jobs use Gloo. GPU jobs initialize the default group with `backend="cuda:nccl,cpu:gloo"`, so FSDP2 can use NCCL while votes and CPU state use Gloo. The GPU accessor stages one local tensor at a time, copies each save tensor to the host once, and writes back a load tensor once after all its ranges.

HF full state uses model HF keys, `optimizer.state.<fqn>.<field>` tensor keys,
and a typed JSON `optimizer.json` sidecar preserving param groups and aliases.
The optimizer tensor inventory is authoritative when present. Each requested
component validates its keys and aliases against the tensors it owns before any
tensor is copied. Model keys beginning with `optimizer.` are still model keys
unless the inventory lists them. The framework-neutral HF loader supports
model-only, optimizer-only, full, and rank-state-only restores from full-state
checkpoints. Some older sidecars limit which restores work.
With an older optimizer sidecar that lacks
the inventory, GDCP matches requested keys and aliases first, then uses the
saved parameter groups to recognize remaining slots in
`optimizer.state.<fqn>.<field>`. A model-only restore skips these slots and
rejects unexpected model keys. An optimizer-only restore skips identifiable
model keys. Generic legacy sidecars without parameter groups fall back to the
`optimizer.` prefix for unmatched keys.
An HF save with zero tensors writes a valid empty `model.safetensors`. Native
HF loading builds storage from headers and an optional standard index. The
header reader recognizes safetensors dtypes such as `U16`, `U32`, `U64`, and
packed floats. If a requested tensor uses a dtype that the load path cannot
represent, the error names the dtype and key.

Native PyTorch DCP loads accept uneven `Shard(0)` chunks and ignore empty
shards at any row offset. GDCP reads each chunk in logical element order using
its serialized strides, including for transposed local tensors. GDCP applies
serialized tensor-view flags such as lazy negation with PyTorch's copy semantics.
An unsupported flag or dtype, such as a conjugate view of a complex tensor,
produces an error naming the tensor. The nonempty chunks must cover the tensor
along dimension 0. GDCP reports a chunk geometry error for overlaps, gaps, and
sharding along another dimension.

All three formats use the coordinator to compare a digest of the live
world size, rank-state keys, the ordered component list, and format options,
including FSDP2 dispatch, before payload I/O. Saves also check the resolved
target and the ordered per-tensor schema. FSDP2 checks the schema after
extracting state and before staging. GDCP rejects ambiguous flattened AdamW
fields and unsupported `hdfs://` paths. Loads write no files and can use read-only checkpoints. Independent
loads and saves to distinct targets can overlap in one parent. See the
[known limitations](GDCP.md#known-limitations).
Names that resolve to the same directory on the
checkpoint filesystem fail preflight on all ranks before any payload is written.
FSDP2 validates save options collectively and requires every rank to select the
same format and HF export mode. A descriptor close failure during a save aborts
publication on every rank.

GDCP records the saved rank files in its manifest for per-rank safetensors and
DCP, and in the completion marker for HF. A missing declared file fails
validation on all ranks. When the loading world is larger,
ranks outside the saved world warn and use rank 0's file. Older HF checkpoints
without a rank-state inventory still use the rank filenames to look up state.
Requesting rank state from native DCP without a GDCP manifest fails collectively.

All three formats require every rank to request the same components in the same
order. With `load_common_state_only=True`, all three GDCP formats read only rank
state, even when the checkpoint contains model or optimizer components. An
absent `extra_state` leaves the caller's checkpoint dict unchanged. The
low-level `gdcp.hf.load_state` also supports loading an optimizer component
without loading the model component. Tensor loads validate all requested
tensor keys, shapes, dtypes, storage metadata, available
tied-parameter alias information, optimizer objects and parameter groups, and
requested rank state before writing into live targets. If any rank fails
validation, the model, optimizer, and caller's checkpoint dict stay unchanged
on every rank. Native formats without an explicit alias map are checked using
the alias information their metadata provides. Tensor payloads are streamed into
the target, so a read error during
the transfer can leave tensors partially updated.

All three formats and weights-only HF exports use the directory transaction and
full recovery rules in [GDCP.md](GDCP.md). Rank 0 checks mount points, filesystem
identity, leftovers and foreign-name collisions collectively before any payload
write. Use a subdirectory of a mounted volume. Directory symlinks resolve to the
real target, leaving the link untouched. Top-level layout names are unchanged.

A successful overwrite replaces only files identified as old checkpoint files at
preflight from recorded metadata and fixed format names. Foreign files, nested
directories and symlinks keep their contents and inodes. A foreign name collision
detected at preflight fails before payload writes. A collision found in the final
publication scan fails before the new checkpoint is published.
Caught failures and pre-accept interrupts restore the old checkpoint and foreign
entries. If an original name has been recreated, rollback keeps the carried entry
in the stage, leaves the replacement in the target and logs the retained stage.
The original interrupt propagates. Foreign entries found by the final scan move by
rename just before publication, so a hard kill can temporarily leave some in the
stage. The durable transaction journal lets retry return them to the verified old
target. Entries created between that scan and the exchange or old-to-backup
rename stay in the obsolete sibling until cleanup. Cleanup moves each such entry
if its original relative name is free and unreserved. A single-file HF save
reserves the index name because HF loaders check for it first. Cleanup moves old
files and the published journal to private temporary directories before checking
their identities and deleting them. If cleanup cannot move or safely delete an
entry, the save succeeds and logs where it was retained. Linux
exchange retains checkpoint old-or-new visibility; the two-rename fallback has
an absent-target window with the old checkpoint in backup. Unpublished pending
stages are recovered automatically; complete stages and backups are retained for
inspection (a complete backup may be restored when the target is absent).
Automatic recovery assumes no other save to the same path is running. An overwrite
still needs space for both generations; accepted obsolete cleanup may leave a
retained sibling. New HF completion markers also record exact rank-state file
ownership; unrecorded legacy rank-state names remain foreign.

Save preflight and recovery read native DCP `.metadata` as data. Unreadable,
unsupported, or malformed metadata, including a truncated pickle frame,
establishes no native DCP ownership. Recovery leaves a backup in place when that
metadata is needed to verify it. `inspect` reads only the JSON manifest. Native DCP
loads also decode `.metadata` as data and can use pickle for non-tensor objects
after collective validation of metadata and tensor records. Manifest-backed DCP
loads use pickle for component metadata. Only load DCP checkpoints you trust.

HF save groups rows with compatible storage columns before building its shared
file layout. Rectangle count is bounded by the number of column groups rather
than tensor count. This changes file bytes and can change the designated source
writer of a replicated tensor; every replica must hold the same value. Stored
replicated tensors still have one slot. Loading uses native headers and indexes
and does not invoke the current save policy.

# GDCP v1: FSDP2 checkpoints

`gdcp.FSDP2Checkpointer` saves one real FSDP2 module, its AdamW optimizer, and
arbitrary picklable per-rank extra state. All ranks call save/load with the same
path and compatible state. The default process group must be Gloo; the v1
acceptance suite uses CPU tensors and world sizes 1–4.

```python
from gdcp import FSDP2Checkpointer, inspect

# model has already been fully_shard-ed on a one-dimensional world mesh.
checkpoint_state = {
    "model": model,
    "optimizer": optimizer,
    "extra_state": {"step": 100, "rng": rng_state},
}
FSDP2Checkpointer.save("/checkpoints/run", checkpoint_state, global_steps=100)
FSDP2Checkpointer.load("/checkpoints/run/global_step_100", checkpoint_state)
# model and optimizer are restored in place; extra_state is replaced.
description = inspect("/checkpoints/run/global_step_100")  # GDTensor Result
```

The loading world size may differ from the saving world size. GDTensor moves
the intersections of saved storage shards and live compute shards directly;
GDCP never assembles full tensors during save or load. Supported placements
are `Shard(0)` and `Replicate` on a one-dimensional mesh containing all ranks.

Model-only save and load, model-only load from a full checkpoint, extra-only
save, and `load_common_state_only=True` are supported. Optimizer state needs
the accompanying model to establish parameter names, including when loading.
The normal load requires a model. Common-only load restores extra state.

`gdcp_manifest.json` is the checkpoint commit marker. GDCP writes it last,
after every rank commits every requested component and extra-state file.
Nonempty component directories also contain the connector's
`.gdtensor_committed` completion files, which its loader checks. These are
component markers; GDCP's manifest determines whole-checkpoint visibility.
Every save is built in `.NAME.gdcp-stage` beside the resolved real target.
A symlink to a directory keeps its original link text and inode; dangling links
and links to files are errors. The filesystem root and mount-point targets
(including Linux bind mounts) are rejected before payload writes. Pass a
subdirectory of the mount. Rank 0 checks that target and staging parent share a
device, performs recovery and checks collisions inside the collective initialize
step; every rank must pass that step before writing payload.

Ownership comes from the old GDCP manifest (component files, objects and exact
rank-state files), native HF index/recorded shard names, or native DCP metadata,
plus the format's fixed metadata/configuration names. Arbitrary filename patterns
never establish ownership. With no recognizable checkpoint, every existing entry
is foreign. Foreign files, directories (including nested checkpoints) and symlinks
are retained at the same relative names with their original inodes and contents.
Foreign descendants of checkpoint component directories are retained too. A
foreign entry conflicting with a planned checkpoint file or directory causes an
early error naming the entry; no payload is written. Old owned files that the new
checkpoint omits disappear on success. Legacy HF exports that did not record rank
state cannot establish ownership of those files: they are preserved as foreign,
and an attempt to rewrite their names is rejected. New HF completion markers
record the exact rank-state mapping.

The old checkpoint receives no payload writes. After every rank completes the
new generation, rank 0 syncs its directories, moves foreign entries by rename
into the stage, and publishes. Foreign directory trees are neither copied nor
walked. A temporary `.gdcp_transaction` journal records directory/entry device and
inode identities and is synced before any foreign rename. Linux filesystems
supporting `renameat2(RENAME_EXCHANGE)` atomically exchange complete checkpoint
directories; otherwise GDCP uses old-to-backup, then stage-to-target renames.
Publication is followed by parent-directory fsync and a collective success vote.
Before accepted success, a caught failure or `BaseException` (including
`KeyboardInterrupt` and `SystemExit`) rolls back and restores the foreign entries;
interrupts propagate after cleanup. A failed first save leaves the target absent,
or leaves its original foreign entries intact when the directory already existed.
An overwrite needs space for old and new generations together.

Hard termination has these exact namespace windows:

| Step | Checkpoint and foreign entries |
| --- | --- |
| Stage initialization and payload writes | Target retains old checkpoint and all foreign entries; stage is pending/incomplete. |
| New generation complete, before foreign moves | Target retains old checkpoint and all foreign entries; stage is complete. |
| During foreign moves, before exchange/first portable rename | Old checkpoint remains complete at target; each foreign entry is either at its original target name or the same relative name in stage. The durable journal identifies both. |
| After Linux exchange | Target contains complete new checkpoint and all foreign entries; stage contains old checkpoint without those entries. |
| Portable old-to-backup, before stage-to-target | Target is absent; backup contains complete old checkpoint without foreign entries; stage contains complete new checkpoint and all foreign entries. |
| After stage-to-target, before/after acceptance | Target contains complete new checkpoint and all foreign entries; backup contains old checkpoint without those entries. |
| Accepted obsolete-generation cleanup | New checkpoint and foreign entries remain at target; journal is removed, then obsolete sibling is removed. An interrupted/failed cleanup can retain a partially removed sibling. |

A first save has no old-generation rename: before stage-to-target the target is
absent; afterwards it contains the new checkpoint. Recovery does not infer a
complete generation from payload filename patterns, and reads metadata/stat sizes
rather than scanning tensor bytes.

At the next save, recovery follows these rules, assuming **no other save to this
path is running**. Concurrent saves and loads during overwrite remain unsupported.

| Target / stage / backup | Next save |
| --- | --- |
| No reserved siblings | Proceed; remove a leftover published journal only after verifying the target is complete, its new-generation identity matches and all recorded foreign entries are there. |
| Target present, backup present | Refuse and describe all three paths. Never automatically delete a backup. |
| Target absent, complete backup present | Rename backup back to target and log a warning, then apply the stage rules. This restores the old checkpoint; a valid stage journal restores its foreign entries too. |
| Target absent, incomplete/unrecognized backup | Refuse; retain everything for inspection. |
| Empty stage, or only the initializing pending-marker temporary file | With no published journal at target, remove the provably unpublished initialization with a warning and proceed; otherwise retain/refuse the old stage. |
| Pending stage without a journal (legacy attempt) | Remove with a warning, then proceed; legacy pending stages never carried foreign entries. |
| Valid journal, target still has its recorded pre-save identity, stage pending or provably incomplete | Return each moved foreign entry to target, remove stage with a warning, then proceed. A journal created before the pending marker is also recoverable. |
| Stage looks complete without pending marker | Retain and refuse. A valid journal first returns any carried foreign entries to the verified old target; checkpoint payload files are left untouched. |
| Stage without pending marker/provenance, mismatched identity, symlink sibling, or invalid journal | Refuse; inspect the states named in the error. |

A complete stage is retained even if it might be the desired new generation.
Inspect target/stage/backup, restore the desired complete generation and its
foreign entries, then remove only verified obsolete siblings before retrying.
The portable absent-target window restores a complete backup automatically but
still refuses a complete stage after returning foreign entries. After exchange,
a retained old stage is refused; after portable publication, a retained backup is
refused. An accepted cleanup failure logs a warning and returns success with the
new checkpoint complete. An interrupt after the accepted-success boundary during
obsolete cleanup likewise leaves the accepted new generation; old files may
already have been removed. Process death, failed coordination, power/device loss,
or a filesystem that refuses rollback cannot promise a collective error return or
restoration at the original pathname. Data and directory fsyncs retain their
existing durability assumptions.

Each component has one `__{rank}_0.safetensors` file per nonempty storage
rank. Replicated buffers and optimizer steps are stored once. Zero-size
tensors are described in the manifest and need no payload slot. Scalars use
physical shape `[1]` because GDTensor v1 requires nonempty shape metadata;
the manifest's `logical_shape: []` preserves scalar semantics. File entries
include explicit ordered slots and layout rectangles so loading does not
depend on rerunning the current storage policy.

Optimizer non-tensor values live in `optimizer/objects.json`. JSON container
tags preserve tuple/list distinctions. Extra state is stored with `torch.save`
in `extra_state/extra_state_rank_{rank}.pt` and loaded with pickle enabled;
only load trusted checkpoints. If a rank's file is absent, GDCP warns and
loads rank 0's file. Values inside that fallback object are not resharded.

Load validates every requested component's tensor keys, logical and physical
shapes, and dtypes before reading tensor payloads. Missing and unexpected
keys are errors. Unrequested components in a full checkpoint are allowed.
Optimizer restoration precedes model restoration.

The facade preserves the reference checkpointer's parameter names, order, and
defaults. `io_metrics_on` and `barrier_rpc_tag_suffix` are ignored. Nondefault
unsupported options and unknown keywords raise `NotImplementedError` naming
the option. The reference's metrics default depends on its environment; GDCP
uses the plan's `False`.

The neutral `gdcp.state`, `policy`, `desc`, `backend`, and `core` modules import
no torch. Applications can provide their own components/accessors, storage
policy, backend, and connector coordinator. The default backend delegates
formats and byte I/O to `gdtensor_conn`. V1's checkpoint metadata and commit
operations use POSIX paths; remote checkpoint metadata is future work.

The test runner is `scripts/gdcp_h20_pytest.sh`. Use it inside the CPU-only
`iobench-gdcp-v1` container after synchronizing the local source. The H20
environment and exact verification commands are recorded in the task report.

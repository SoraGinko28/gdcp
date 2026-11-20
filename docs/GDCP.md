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
Native PyTorch DCP loads support uneven `Shard(0)` chunks and empty shards. GDCP
honors the saved local strides, including transposed shards, when restoring at
the same or a different world size. GDCP applies serialized tensor-view flags
such as lazy negation with PyTorch's copy semantics. An unsupported flag or
dtype, such as a conjugate view of a complex tensor, produces an error naming
the tensor.

All three formats support empty checkpoints, extra-state-only checkpoints,
and model checkpoints with or without optimizer or extra state. Pass the keys
you want to restore to `load`. Optimizer state needs the accompanying model to
establish parameter names. With `load_common_state_only=True`, `load` reads
only rank state and adds `extra_state` to the caller's dict if the checkpoint
contains it. A declared rank-state file that is missing still causes an error.
All ranks must use the same FSDP2 save format and agree on whether an HF save
exports weights only or full state. All three formats require the same ordered
component list on every rank during load and before staging a save.

Before payload I/O, coordinated saves and loads each use one digest to compare
the live world size, ordered component names, rank-state keys, and built-in
format options, including FSDP2 save and load dispatch. Saves also compare the
resolved target. Loads compare the requested state and can read rank-local
copies at different paths. Each rank hashes its canonical description with
SHA-256, and the coordinator broadcasts
the first 64 bits of rank 0's digest. Preparation, broadcast, and comparison
take 66 votes regardless of name lengths, key counts, or world size. On a
mismatch, all ranks report the fields compared and log their own description
at debug level. FSDP2 compares the save request before extracting model or
optimizer state. A disagreement leaves a fresh AdamW optimizer at its initial step.

Agreement does not write any files. Loads can read from a read-only checkpoint
and parent. Independent jobs can load the same checkpoint or checkpoints in
the same parent concurrently. Independent saves to different targets in one
parent do not share agreement state; each still needs a writable location for
its own checkpoint transaction.

Weights-only HF saves apply `fp32` or `bf16` conversion to floating-point tensors.
Integer and boolean buffers round-trip bit for bit into targets of the same dtype.
This includes `uint16`, `uint32`, and `uint64` buffers and `int16` buffers sharded
with `Shard(0)`.
See [GDCP v2 formats](GDCP-V2.md) for `save_hf` and `load_hf` options.

GDCP writes `gdcp_manifest.json` as the checkpoint commit marker for per-rank
safetensors and DCP after every rank commits its components and extra-state
files. Per-rank safetensors component directories also have their own
`.gdtensor_committed` files. HF uses a root `.gdtensor_committed` file as its
checkpoint commit marker. A load with both regular root marker files fails
format detection. Saves reserve both root marker names and reject a foreign
entry at either name before publication.

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
are carried to the same relative names with their original inodes and contents.
This includes foreign descendants of checkpoint component directories. Preflight
rejects an existing foreign name that conflicts with a planned checkpoint file or
directory before any payload write. Old owned files that the new
checkpoint omits disappear on success. Legacy HF exports that did not record rank
state cannot establish ownership of those files: they are preserved as foreign,
and an attempt to rewrite their names is rejected. New HF completion markers
record the exact rank-state mapping.

Save preflight and recovery read native DCP `.metadata` as data without invoking
pickle constructors or importing modules named by the file. Unreadable,
unsupported, or malformed native metadata, including a truncated pickle frame,
establishes no ownership, so every entry in that candidate checkpoint is foreign.
Recovery leaves a backup in place when that metadata is needed to verify it.
`inspect` reads only the JSON manifest. Native DCP loads also decode `.metadata`
as data. They can use pickle
for non-tensor objects after collective validation of metadata and tensor
records. Manifest-backed DCP loads use pickle for component metadata. Only load
DCP checkpoints you trust.

The old checkpoint receives no payload writes. After every rank completes the
new generation, rank 0 syncs its directories and rescans the old target against
the entries it identified as owned at preflight. A foreign name that now
conflicts with the new checkpoint aborts publication and leaves the old target
intact. Rank 0 updates and syncs `.gdcp_transaction` with the directory and entry
device and inode identities, then moves the foreign entries into the stage by
rename and publishes. Foreign directories move as whole trees. Linux filesystems
supporting `renameat2(RENAME_EXCHANGE)` atomically exchange complete checkpoint
directories; otherwise GDCP uses old-to-backup, then stage-to-target renames.
Publication is followed by parent-directory fsync and a collective success vote.
Before accepted success, a caught failure or `BaseException` (including
`KeyboardInterrupt` and `SystemExit`) rolls back and restores the foreign entries;
interrupts propagate after cleanup. If a carried entry's original name has been
recreated, rollback leaves the replacement in the old target and the carried entry
in the stage. It logs the retained stage and the blocked name, and the original
interrupt still propagates. A failed first save leaves the target absent, or leaves
its original foreign entries intact when the directory already existed. An
overwrite needs space for old and new generations together.

An entry created in the old directory between the final publication scan and
the exchange (or the old-to-backup rename on the portable path) stays with the
old directory through publication. Entries created while the scan is in progress
can also be missed if their directory was already enumerated. After acceptance,
cleanup scans the old sibling and moves the entry to the new target when an
exclusive rename is available and the name is free and unreserved. Reservations
include format-specific names such as `model.safetensors.index.json` for an HF
single-file save, where an index would take precedence during load. A collision,
reserved name, or failed move leaves the entry in the sibling and logs a warning
naming the sibling and entry. The save returns success with the new checkpoint.

Cleanup renames each deletion candidate into a private workspace in the stage
sibling and verifies its identity there before deleting it. Cleanup recreates
the stage sibling for this purpose after the stage has been published. A captured
replacement goes back to its original name if available. If that name has been
recreated, both entries are retained and the warning identifies the temporary
path. Directory removal
requires an empty directory, so arrivals during deletion can also leave the
obsolete sibling in place with a warning. The published journal stays until the
sibling is removed. Its cleanup uses the identity recorded when the journal was
written, preserving a foreign replacement at the journal name.

Hard termination has these exact namespace windows:

| Step | Checkpoint and foreign entries |
| --- | --- |
| Stage initialization and payload writes | Target retains old checkpoint and all foreign entries; stage is pending/incomplete. |
| New generation complete, before foreign moves | Target retains old checkpoint and all foreign entries; stage is complete. |
| During foreign moves, before exchange/first portable rename | Old checkpoint remains complete at target. Each entry recorded by the final scan is at its original target name or the same relative name in stage, as recorded in the durable journal. Later arrivals can still be in the target. |
| After Linux exchange | Target contains the complete new checkpoint and the carried entries. Stage contains the old checkpoint and any arrivals missed by the final scan. |
| Portable old-to-backup, before stage-to-target | Target is absent. Backup contains the complete old checkpoint and any arrivals missed by the final scan. Stage contains the complete new checkpoint and the carried entries. |
| After stage-to-target, before acceptance | Target contains the complete new checkpoint and the carried entries. Backup contains the old checkpoint and any arrivals missed by the final scan. |
| Accepted cleanup | Uncontested late entries move to the new target. An interrupted or blocked cleanup can retain the obsolete sibling and a claim in the stage sibling. After successful obsolete-generation cleanup, the journal is claimed in the stage sibling and removed. |

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
| Empty stage while a journal exists at target | Validate the published generation and its journal first. Keep the stage for inspection if it identifies the old directory left by exchange. Otherwise remove the empty cleanup stage and the verified journal. |
| Nonempty stage without its own journal while a journal exists at target | Retain the stage and refuse recovery because it may be the old generation. |
| Empty stage, or only the initializing pending-marker temporary file | Remove the provably unpublished initialization with a warning and proceed. |
| Pending stage without a journal (legacy attempt) | Remove with a warning and proceed. Legacy pending stages could only contain unpublished checkpoint output. |
| Valid journal, target still has its recorded pre-save identity, stage pending or provably incomplete | Return each moved foreign entry to target, remove stage with a warning, then proceed. A journal created before the pending marker is also recoverable. |
| Stage looks complete without pending marker | Retain and refuse. A valid journal first returns any carried foreign entries to the verified old target; checkpoint payload files are left untouched. |
| Stage without pending marker/provenance, mismatched identity, symlink sibling, or invalid journal | Refuse; inspect the states named in the error. |

A complete stage is retained even if it might be the desired new generation.
Inspect target/stage/backup, restore the desired complete generation and its
foreign entries, then remove only verified obsolete siblings before retrying.
The portable absent-target window restores a complete backup automatically but
still refuses a complete stage after returning foreign entries. After exchange,
a retained old stage is refused; after portable publication, a retained backup is
refused. Recoverable cleanup failures log a warning and return success with the
new checkpoint complete. If an error escapes transaction cleanup after acceptance,
every rank reports a cleanup failure and the new checkpoint is already committed.
An interrupt during cleanup also leaves the accepted new generation. If staging
holds a claim that could not be restored or a hard termination leaves it in
place, inspect the stage before removing it and retrying.
Process death, failed coordination, power/device loss,
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
only load trusted checkpoints. GDCP writes each rank's snapshot of its rank
state and checks the rank configuration, component sequence, and key sets
between save phases using coordinator votes alone.
A detected change returns the same error on every rank, even if a callback
restores the original value. GDCP also checks that every declared file exists
before publication. A failed check preserves any previous checkpoint.
If different names resolve to the same directory on the checkpoint filesystem,
the save fails on every rank before any tensor or rank-state payload is written.
The GDCP manifest or HF completion marker lists the files saved for each key.
A missing declared file fails the load on every rank. When loading with a
larger world size, only ranks outside the saved world warn and load rank 0's
file. Values inside that fallback object are not resharded.

Before writing into the target model, optimizer, or caller's checkpoint dict,
all ranks must pass validation of the requested tensor keys, shapes, dtypes,
recorded tied-parameter aliases, optimizer objects and group membership, storage
metadata, and requested rank state. A failed validation leaves those targets
unchanged. Missing and unexpected keys in requested components are errors.
Unrequested components in a full checkpoint are allowed. HF loads use the
optimizer tensor inventory in `optimizer.json` to distinguish optimizer slots
from model weights, even when a model key starts with `optimizer.`. For older
full-state HF checkpoints without this inventory, GDCP matches requested keys
and aliases first. It classifies the remaining tensors using the saved parameter
groups and the `optimizer.state.<fqn>.<field>` layout. A model-only load skips
these optimizer slots and still rejects unexpected model keys such as
`optimizer.bias`. An optimizer-only load skips tensors identified as model
weights. Generic legacy optimizer sidecars without parameter groups use the
`optimizer.` prefix for unmatched keys.
Optimizer restoration precedes model restoration.

Optimizer groups are matched by their model parameter names. The target may
reorder groups or parameters within a group. Load restores the saved group
options and order, and fills each group's `param_names` with the corresponding
fully qualified model names in `params` order. This also applies to unnamed
optimizers, and a subsequent save records the restored names. AdamW moments
and steps follow their parameters. Per-rank safetensors, DCP, and HF report
incompatible or ambiguous groups before checking requested tensor keys or
reading tensor and extra-state payloads. For Torch-written native DCP, GDCP
decodes membership with a restricted loader. Every rank must pass the group and
tensor inventory checks, then validate the backing tensor records with a
restricted loader, before GDCP reads scalar optimizer state or separately
stored group options.

Tensor payloads are streamed into the target. An I/O error during those reads
can leave tensors partially updated, and copied bytes are retained if a
framework load hook fails during installation.

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

Core selects a default save policy or a default save or load backend only when
the corresponding option is `None`. An exception during option preparation
returns an error on every rank.

The test runner is `scripts/gdcp_h20_pytest.sh`. Use it inside the CPU-only
`iobench-gdcp-v1` container after synchronizing the local source. The H20
environment and exact verification commands are recorded in the task report.

## Known limitations

- The process group or coordinator must supply unique rank IDs, including
  rank 0. GDCP agrees on world size but does not census rank IDs.
- Concurrent writers to the checkpoint target during a save are unsupported. If
  a second process creates, replaces, or swaps an entry in the target or its
  obsolete sibling during publication or cleanup, that entry can be lost or
  the checkpoint can become unreadable.
- GDCP compares reserved checkpoint file names case-sensitively. On a
  case-insensitive filesystem, a foreign file whose name differs only in case
  from a GDCP marker can make the checkpoint unloadable.
- If another process truncates a payload file while it is memory-mapped for a
  native DCP read, the reading process can terminate.
- A legacy HF-layout checkpoint without an optimizer tensor inventory can fail
  a model-only restore if its optimizer sidecar lists parameter IDs as integers
  and the checkpoint contains optimizer tensors. The neutral HF loader can
  restore model and optimizer together when both tensor inventories match.
  FSDP2 requires named optimizer groups. For that path, use a reader that can
  restore the original checkpoint and re-save it with the current version.
- Without an optimizer tensor inventory, the HF loader can classify an
  unmatched model tensor named `optimizer.state.<parameter name>.<field>` as
  optimizer state if the legacy sidecar lists that parameter by name. The
  loader matches requested model keys and aliases before applying this rule.
- The native DCP reader does not support lazily negated tensors whose serialized
  byte order differs from the host's. Such a load can terminate the process.
  PyTorch's own DCP reader handles the byte-order conversion.
- Save agreement is checked at phase boundaries and at selected accessor hooks.
  A rank's rank-state keys or world size can change and change back between two
  tensor reads without detection if neither check observes the change.

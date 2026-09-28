# Syncing `scratch/` to S3

Notes for [`s3_sync.py`](s3_sync.py). Running `python s3_sync.py` works, but for a
large transfer prefer driving the AWS CLI directly — see why below.

## Running the AWS CLI directly (recommended for bulk transfers)

`aws` is not installed in a fresh capsule. Install v2 (not the pip `awscli` v1
package, which is slower on large syncs):

```bash
cd /tmp && curl -sSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o awscliv2.zip
unzip -q awscliv2.zip -d aws-cli-install && ./aws-cli-install/aws/install
aws --version   # -> /usr/local/bin/aws, expect aws-cli/2.x
```

One-time tuning. The defaults are wrong for this tree: `max_concurrent_requests`
is 10, so ~109k sub-MiB files are uploaded 10-at-a-time, and the default 8 MB
chunk splits the 20.5 GiB master table into ~2600 parts.

```bash
aws configure set default.s3.max_concurrent_requests 32
aws configure set default.s3.max_queue_size 10000
aws configure set default.s3.multipart_threshold 64MB
aws configure set default.s3.multipart_chunksize 64MB
```

The full run. Detach it (this takes tens of minutes) and keep the log **outside**
the tree being synced, or the log file mutates mid-sync:

```bash
AWS_REGION=us-west-2 nohup aws s3 sync \
  /root/capsule/scratch/ \
  s3://aind-scratch-data/jason_lee/DRN_beh_physiology/ \
  --no-progress > /tmp/drn_sync.log 2>&1 &
```

`AWS_REGION` is set explicitly because the bucket is in us-west-2 while the
capsule env ships `AWS_REGION=us-west-1` (disagreeing with its own
`AWS_DEFAULT_REGION=us-west-2`). Requests still succeed via redirect, but don't
rely on it across 136k objects.

## Verifying afterwards

Do **not** compare against hardcoded totals — `scratch/` changes under you. (The
first full run "lost" 30 GiB against a 30-minute-old snapshot; two large pickles
had simply been deleted in between. See the note at the end.) Compute the source
totals fresh, then diff on key *and* size:

```bash
# local: paths relative to /scratch, tab-separated from size
find /scratch -type f -printf '%P\t%s\n' | sort > /tmp/local_now.tsv

# S3: list-objects-v2 (not `s3 ls`) so keys with spaces stay tab-delimited
AWS_REGION=us-west-2 aws s3api list-objects-v2 \
  --bucket aind-scratch-data --prefix jason_lee/DRN_beh_physiology/ \
  --query 'Contents[].[Key,Size]' --output text \
  | sed 's|^jason_lee/DRN_beh_physiology/||' | sort > /tmp/s3_now.tsv

comm -23 /tmp/local_now.tsv /tmp/s3_now.tsv   # local only -> not uploaded
comm -13 /tmp/local_now.tsv /tmp/s3_now.tsv   # S3 only    -> stale/deleted locally
```

Both should be empty. Empty directories never appear (S3 has no directories) and
there are no zero-byte files, so object count should equal file count exactly.

Two traps worth knowing. `aws s3 ls --recursive --summarize | tail -2` gives
totals but no per-key detail, so two offsetting errors look like success. And
grepping the log for `error|fail` yields false positives — real filenames include
`failed_sessions_*.csv` and `scan_errors.csv`. Match `^upload:` at line start and
trust the exit code instead.

## Flags deliberately NOT used

- `--delete` — a no-op against an empty destination, and a foot-gun later: with
  `combine_only = True` it would delete every non-`combined/` key.
- `--storage-class` — the bucket lifecycle already transitions everything to
  INTELLIGENT_TIERING at day 0. Incomplete multipart uploads are aborted after
  7 days, and current object versions are never expired.
- `cp --recursive` (the `if_copy=True` path) — not restartable. A 177 GiB
  transfer will get interrupted; `sync` skips what already landed, `cp`
  re-uploads everything. `--delete` is also silently ignored by `cp`.
- `--quiet` — hides errors as well as progress. `--no-progress` keeps both the
  per-file lines and any failures.

## Gotchas in the script

- `sync_directory` routes the real `aws` output to `logger.info`, so per-file
  detail only appears if logging is configured — it now is, as the first
  statement in `__main__`. `basicConfig` has to run *before* the first log call
  or it silently does nothing.
- Status strings now distinguish the three outcomes: `dry run, N file(s) would
  upload` / `successfully uploaded` / `already exists, skip`, plus
  `error during sync (aws exit N)` on a non-zero exit. Previously a dry run
  reported `successfully uploaded` (the `(dryrun) upload: ...` lines matched a
  bare `"upload:" in output` test) and a failed sync reported
  `already exists, skip`.
- `CAPSULE_ROOT + "/scratch/"` resolves through a symlink
  (`/root/capsule/scratch -> /scratch`, an NFS mount). The trailing slash is what
  makes `aws` follow it; don't drop it.

## Measured characteristics (2026-09-28)

- Source: 136,003 files, 146.6 GiB (157,456,815,061 bytes) as of the completed
  run. No symlinks inside, no `__pycache__`/checkpoint/temp junk, nothing worth
  excluding. 99 filenames contain a space and one an `&` — legal S3 keys that
  `sync` handles, but quote them in any hand-written command.
- Shape: ~109k files under 1 MiB, with a long tail of large pickles in
  `combined/master_unit_tables/`.
- **Completed full run: 134,976 files / 146 GiB in 1086 s (18 min), exit 0, no
  errors.** Verified byte-exact by the key+size diff above — 136,003 objects,
  zero differences in either direction.
- Throughput **~144 MB/s and ~124 files/s** with the settings above.
- An 852 MiB pilot beforehand measured only ~85 MB/s, close to the 89.5 MB/s
  single-stream NFS read, which led to a wrong inference that NFS was the ceiling
  and concurrency past 32 was pointless. The full run sustained 144 MB/s, well
  above single-stream, so concurrency *does* help. Don't extrapolate a rate from
  a sub-minute transfer; ramp-up dominates the sample.
- Multipart is exercised: the 15.24 GiB `master_all_units_opto_ccf_raw_260817.pkl`
  uploaded cleanly at `multipart_chunksize = 64MB`.
- Re-syncs are idempotent and cheap — re-running over an already-synced prefix
  uploaded 0 files in 2 s, so an interrupted transfer can just be restarted.

## `scratch/` mutates during long runs

Between the pre-run snapshot and the sync, two master tables were deleted locally
(`master_all_units_opto_ccf_raw_fix_260708.pkl`, 9.74 GiB, and
`master_all_units_opto_ccf_raw_fix_260710_rebuilt_corr_from_metrics.pkl`,
20.49 GiB). `sync` correctly transferred what existed at run time and exited 0 —
only the stale expected-total made it look like a 30 GiB failure. Snapshot the
file list immediately before launching if you need a point-in-time check.

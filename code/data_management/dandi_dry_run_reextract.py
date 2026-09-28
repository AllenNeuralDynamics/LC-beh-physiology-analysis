"""
Dry-run preview for `dandi service-scripts reextract-metadata`.

The CLI's own --diff flag only *displays* a diff -- it still writes the new
metadata immediately after. This script reproduces the same extraction logic
(dandi.metadata.nwb.nwb2asset, the function reextract-metadata itself calls)
but separates "compute + diff" from "write", so you can review every asset's
diff before anything is persisted to DANDI.

Requires: pip install -U dandi "fsspec[http]"
Requires: DANDI_API_KEY env var with write access to the dandiset.

Usage:
    python dry_run_reextract.py                 # check only, prints diffs, writes nothing
    python dry_run_reextract.py --apply         # actually persist the changes shown above
    python dry_run_reextract.py --apply --path sub-123  # only touch assets whose path contains this substring
    python dry_run_reextract.py --limit 3        # only look at the first 3 matching assets (quick smoke test)

Each remote NWB file is streamed over HTTP and its full HDF5 structure is
walked to extract metadata (dandi.metadata.nwb.nwb2asset) -- this can take
tens of seconds to a few minutes per file depending on size/network latency,
and 385 assets are processed one at a time. Progress is printed per-asset
below so you can tell it's working rather than stuck; use --limit or --path
to test on a handful of files before running the full dandiset.
"""

import argparse
import os
import time

import yaml
from dandi.dandiapi import DandiAPIClient
from dandi.exceptions import NotFoundError
from difflib import unified_diff
from pathlib import PurePosixPath

DANDISET_ID = "001950"
VERSION = "draft"


def yaml_dump(d: dict) -> str:
    return yaml.safe_dump(d, default_flow_style=False, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Persist changes (default: dry run only)")
    parser.add_argument("--path", default=None, help="Only process assets whose path contains this substring")
    parser.add_argument("--limit", type=int, default=None, help="Stop after this many matching assets (for a quick test)")
    args = parser.parse_args()

    from dandi.metadata.nwb import nwb2asset  # heavy import, deferred like upstream

    api_key = os.environ["DANDI_API_KEY"]

    processed = 0

    with DandiAPIClient() as client:
        client.authenticate(token=api_key)
        dandiset = client.get_dandiset(DANDISET_ID, VERSION)

        for asset in dandiset.get_assets():
            if PurePosixPath(asset.path).suffix.lower() != ".nwb":
                continue
            if args.path and args.path not in asset.path:
                continue
            if args.limit is not None and processed >= args.limit:
                print(f"--limit {args.limit} reached, stopping")
                break
            processed += 1

            print(f"[{processed}] {asset.path}: fetching digest/metadata...", flush=True)
            t0 = time.monotonic()

            try:
                digest = asset.get_digest()
            except NotFoundError:
                digest = None

            oldmd = asset.get_raw_metadata()

            print(f"[{processed}] {asset.path}: streaming NWB file for reextraction...", flush=True)
            metadata = nwb2asset(asset.as_readable(), digest=digest)
            metadata.path = asset.path
            mddict = metadata.model_dump(mode="json", exclude_none=True)
            print(f"[{processed}] {asset.path}: done in {time.monotonic() - t0:.1f}s", flush=True)

            diff_text = "".join(
                unified_diff(
                    yaml_dump(oldmd).splitlines(True),
                    yaml_dump(mddict).splitlines(True),
                    fromfile=f"{asset.path}:old",
                    tofile=f"{asset.path}:new",
                )
            )

            if not diff_text:
                print(f"{asset.path}: no change")
                continue

            print(f"=== {asset.path} ===")
            print(diff_text)

            if args.apply:
                print(f"  -> writing new metadata for {asset.path}")
                asset.set_raw_metadata(mddict)
            else:
                print("  -> dry run, nothing written (rerun with --apply to persist)")


if __name__ == "__main__":
    main()

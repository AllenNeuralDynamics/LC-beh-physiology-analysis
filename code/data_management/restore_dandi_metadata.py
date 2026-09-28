"""
Restore DANDI asset metadata from a backup produced by
print_dandi_asset_metadata.py --all --out <file>.yaml.

Use this if patch_dandi_modality.py --apply (or anything else) needs undoing.
It writes back each asset's *entire* raw metadata exactly as it was captured
in the backup file -- not just approach/measurementTechnique -- so it's a
full revert, not a partial one.

Requires: pip install -U dandi
Requires: DANDI_API_KEY env var with write access to the dandiset -- but only
for --apply; a dry run reads the current live state and needs no key.

Usage:
    python restore_dandi_metadata.py dandi_001950_metadata_backup_20260928.yaml
        # dry run: prints which assets differ from the backup, writes nothing
    python restore_dandi_metadata.py dandi_001950_metadata_backup_20260928.yaml --apply
        # actually restores every differing asset to the backed-up state
    python restore_dandi_metadata.py dandi_001950_metadata_backup_20260928.yaml --apply --path sub-669489
        # restore only assets whose path contains this substring
"""

from __future__ import annotations

import argparse
import os

import yaml
from dandi.dandiapi import DandiAPIClient

DANDISET_ID = "001950"
VERSION = "draft"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("backup_file", help="yaml file produced by print_dandi_asset_metadata.py --all --out ...")
    parser.add_argument("--apply", action="store_true", help="Persist the restore (default: dry run only)")
    parser.add_argument("--path", default=None, help="Only restore assets whose path contains this substring")
    args = parser.parse_args()

    with open(args.backup_file, encoding="utf-8") as f:
        backup = yaml.safe_load(f)

    backup_by_path = {entry["path"]: entry["metadata"] for entry in backup}
    print(f"Loaded backup for {len(backup_by_path)} assets from {args.backup_file}")

    with DandiAPIClient() as client:
        if args.apply:
            client.authenticate(token=os.environ["DANDI_API_KEY"])
        dandiset = client.get_dandiset(DANDISET_ID, VERSION)

        seen = set()
        for asset in dandiset.get_assets():
            if asset.path not in backup_by_path:
                continue
            if args.path and args.path not in asset.path:
                continue
            seen.add(asset.path)

            backed_up = backup_by_path[asset.path]
            current = asset.get_raw_metadata()

            if current == backed_up:
                print(f"{asset.path}: already matches backup, no change")
                continue

            old_approach = [a["name"] for a in (current.get("approach") or [])]
            new_approach = [a["name"] for a in (backed_up.get("approach") or [])]
            old_technique = [t["name"] for t in (current.get("measurementTechnique") or [])]
            new_technique = [t["name"] for t in (backed_up.get("measurementTechnique") or [])]
            print(
                f"{asset.path}: differs from backup\n"
                f"  approach  : {old_approach} -> {new_approach}\n"
                f"  technique : {old_technique} -> {new_technique}"
            )

            if args.apply:
                asset.set_raw_metadata(backed_up)
                print("  -> restored")
            else:
                print("  -> dry run, nothing written (rerun with --apply to persist)")

        missing = set(backup_by_path) - seen
        if args.path:
            missing = {p for p in missing if args.path in p}
        if missing:
            print(f"\nNote: {len(missing)} backed-up asset(s) were not found on the server (renamed/deleted?):")
            for p in sorted(missing):
                print(f"  {p}")


if __name__ == "__main__":
    main()

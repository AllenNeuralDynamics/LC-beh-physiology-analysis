"""
Print (and optionally save) the raw metadata for one or more DANDI assets --
read-only, no API key or write access needed (as long as the dandiset/draft
is publicly readable).

Useful for sanity-checking the current approach/measurementTechnique values
before/after running patch_dandi_modality.py or dandi_dry_run_reextract.py.

Usage:
    python print_dandi_asset_metadata.py                          # first matching .nwb asset, printed only
    python print_dandi_asset_metadata.py --path sub-669489         # first asset whose path contains this
    python print_dandi_asset_metadata.py --path sub-669489 --all   # every matching asset, not just the first

    python print_dandi_asset_metadata.py --out sample.yaml                     # save the one asset to this file
    python print_dandi_asset_metadata.py --all --out all_assets.yaml
        # saves EVERY asset into one combined yaml file (a list of {path, metadata} entries)
    python print_dandi_asset_metadata.py --path sub-669489 --all --out-dir dandi_metadata_samples
        # saves one <asset-path>.yaml per matching asset into that directory (created if needed)
"""

import argparse
from pathlib import Path, PurePosixPath

import yaml
from dandi.dandiapi import DandiAPIClient

DANDISET_ID = "001950"
VERSION = "draft"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=None, help="Only consider assets whose path contains this substring")
    parser.add_argument("--all", action="store_true", help="Process every matching asset instead of just the first")
    parser.add_argument("--out", default=None, help="Save the (single) asset's metadata to this yaml file")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Save each matching asset's metadata as its own yaml file in this directory (created if needed)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    combined = []  # only used when --all is combined with --out (one file for everything)

    with DandiAPIClient() as client:
        dandiset = client.get_dandiset(DANDISET_ID, VERSION)

        for asset in dandiset.get_assets():
            if PurePosixPath(asset.path).suffix.lower() != ".nwb":
                continue
            if args.path and args.path not in asset.path:
                continue

            raw = asset.get_raw_metadata()
            text = yaml.safe_dump(raw, default_flow_style=False, sort_keys=True)

            print(f"=== {asset.path} ===")
            print(text)

            if out_dir:
                dest = out_dir / (PurePosixPath(asset.path).name.replace(".nwb", "") + ".yaml")
                dest.write_text(text, encoding="utf-8")
                print(f"  -> saved to {dest}")
            elif args.out and args.all:
                combined.append({"path": asset.path, "metadata": raw})
            elif args.out:
                Path(args.out).write_text(text, encoding="utf-8")
                print(f"  -> saved to {args.out}")

            if not args.all:
                break

    if args.out and args.all and not out_dir:
        text = yaml.safe_dump(combined, default_flow_style=False, sort_keys=True)
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"-> saved {len(combined)} assets to {args.out}")


if __name__ == "__main__":
    main()

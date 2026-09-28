"""
Set approach/measurementTechnique on DANDI assets purely from the modality
tags already encoded in this dandiset's filenames -- no NWB file streaming,
and no reliance on dandi-cli's own (unreliable / incomplete) auto-detection.

Asset paths in dandiset 001950 look like:
    sub-669489/sub-669489_ses-..._behavior+fib.nwb
    sub-ZS061/sub-ZS061_ses-..._behavior+ecephys.nwb
    sub-ZS061/sub-ZS061_ses-..._behavior.nwb

The "+"-joined tag(s) right before ".nwb" (behavior / ecephys / fib) say
exactly what each file contains. This script trusts that convention as the
source of truth and REPLACES each asset's approach/measurementTechnique
with exactly what the filename says -- it does not merge with, or defer to,
whatever dandi-cli previously computed from the file content. (Reason:
dandi-cli derives these fields from neurodata types via a fixed lookup table
-- neurodata_typemap in dandi/metadata/util.py's process_ndtypes() -- which
has no entry for fiber photometry at all, and may also just be wrong/stale
for ecephys/behavior on this dandiset already.)

No NWB file is streamed -- only small get/set-metadata API calls per asset,
so this runs in seconds across the whole dandiset.

Requires: pip install -U dandi
Requires: DANDI_API_KEY env var with write access to the dandiset -- but only
for --apply; a dry run reads the public draft and needs no key.

No real dandiset on DANDI currently has "fiber photometry" as an approach or
technique (checked 001528, 001767, 001084, 000351 -- all genuine, published
fiber photometry datasets, all missing it for the same reason: dandi-cli's
own extraction has no mapping for this modality at all). So "fiber
photometry approach"/"fiber photometry" are not an established convention,
just free text dandischema happens to allow. If DANDI's write API ever
rejects those specific values for the "fib" tag, this script falls back to
matching what 000351 does instead: no approach entry, and
measurementTechnique = "analytical technique".

Usage:
    python patch_dandi_modality.py                        # dry run, prints planned changes, no key needed
    python patch_dandi_modality.py --apply                # persist changes (needs DANDI_API_KEY)
    python patch_dandi_modality.py --apply --path sub-669489  # limit to matching assets
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import PurePosixPath

from dandi.dandiapi import DandiAPIClient

DANDISET_ID = "001950"
VERSION = "draft"

SUFFIX_RE = re.compile(r"_([a-zA-Z0-9+]+)\.nwb$")

TAG_APPROACH = {
    "ecephys": "electrophysiological approach",
    "behavior": "behavioral approach",
    "fib": "fiber photometry approach",
}
TAG_TECHNIQUE = {
    # "spike sorting technique" is dandi-cli's own confirmed string for files
    # containing a SpikeEventSeries object (dandi/metadata/util.py's
    # neurodata_typemap) -- matches this dandiset's ecephys files.
    "ecephys": "spike sorting technique",
    "behavior": "behavioral technique",
    "fib": "fiber photometry",
}

# Used only if the server rejects "fib"'s entries above -- mirrors dandiset
# 000351's (a real, published fiber photometry dandiset) actual tagging.
FALLBACK_TAG_APPROACH = {"fib": None}
FALLBACK_TAG_TECHNIQUE = {"fib": "analytical technique"}


def tags_for_path(path: str) -> list[str]:
    m = SUFFIX_RE.search(PurePosixPath(path).name)
    if not m:
        return []
    return m.group(1).split("+")


def build_metadata(tags: list[str], fallback_tags: frozenset[str] = frozenset()) -> tuple[list[dict], list[dict]]:
    approach = []
    technique = []
    for tag in tags:
        approach_map = FALLBACK_TAG_APPROACH if tag in fallback_tags else TAG_APPROACH
        technique_map = FALLBACK_TAG_TECHNIQUE if tag in fallback_tags else TAG_TECHNIQUE

        aname = approach_map.get(tag, TAG_APPROACH.get(tag))
        if aname and aname not in {a["name"] for a in approach}:
            approach.append({"schemaKey": "ApproachType", "name": aname})
        tname = technique_map.get(tag, TAG_TECHNIQUE.get(tag))
        if tname and tname not in {t["name"] for t in technique}:
            technique.append({"schemaKey": "MeasurementTechniqueType", "name": tname})
    return approach, technique


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Persist changes (default: dry run only)")
    parser.add_argument("--path", default=None, help="Only process assets whose path contains this substring")
    args = parser.parse_args()

    with DandiAPIClient() as client:
        if args.apply:
            # Only writing requires a token; reads work against the public draft without one.
            client.authenticate(token=os.environ["DANDI_API_KEY"])
        dandiset = client.get_dandiset(DANDISET_ID, VERSION)

        for asset in dandiset.get_assets():
            if PurePosixPath(asset.path).suffix.lower() != ".nwb":
                continue
            if args.path and args.path not in asset.path:
                continue

            tags = tags_for_path(asset.path)
            if not tags:
                print(f"{asset.path}: no recognized modality suffix, skipping")
                continue

            new_approach, new_technique = build_metadata(tags)

            raw = asset.get_raw_metadata()
            old_approach = raw.get("approach") or []
            old_technique = raw.get("measurementTechnique") or []

            if (
                sorted(a["name"] for a in old_approach) == sorted(a["name"] for a in new_approach)
                and sorted(t["name"] for t in old_technique) == sorted(t["name"] for t in new_technique)
            ):
                print(f"{asset.path}: tags {tags} already match, no change")
                continue

            print(
                f"{asset.path}: tags {tags}\n"
                f"  approach  : {[a['name'] for a in old_approach]} -> {[a['name'] for a in new_approach]}\n"
                f"  technique : {[t['name'] for t in old_technique]} -> {[t['name'] for t in new_technique]}"
            )

            raw["approach"] = new_approach
            raw["measurementTechnique"] = new_technique

            if not args.apply:
                print("  -> dry run, nothing written (rerun with --apply to persist)")
                continue

            try:
                asset.set_raw_metadata(raw)
                print("  -> written")
            except Exception as exc:
                if "fib" not in tags:
                    raise
                print(f"  -> DANDI rejected fiber-photometry values ({exc!r}); retrying with 000351-style fallback")
                fallback_approach, fallback_technique = build_metadata(tags, fallback_tags=frozenset({"fib"}))
                raw["approach"] = fallback_approach
                raw["measurementTechnique"] = fallback_technique
                asset.set_raw_metadata(raw)
                print(
                    f"  -> written with fallback: approach={[a['name'] for a in fallback_approach]}, "
                    f"technique={[t['name'] for t in fallback_technique]}"
                )


if __name__ == "__main__":
    main()

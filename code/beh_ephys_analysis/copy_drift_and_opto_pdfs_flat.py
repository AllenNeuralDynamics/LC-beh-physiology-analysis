#!/usr/bin/env python3
"""
Copy per-unit drift and optotagging PDFs from capsule scratch into flat folders.

Input examples:
  /root/capsule/scratch/826159/behavior_826159_2026-01-22_12-58-47/ephys/opto/raw/drift/145_drift.pdf
  /root/capsule/scratch/826159/behavior_826159_2026-01-22_12-58-47/ephys/opto/raw/figures/unit_145_pulse_width_4_opto_tagging.pdf

Output examples:
  /root/capsule/scratch/unit_drift_pdfs/behavior_826159_2026-01-22_12-58-47_unit_145_drift.pdf
  /root/capsule/scratch/unit_opto_pdfs/behavior_826159_2026-01-22_12-58-47_unit_145_opto.pdf

The script uses only Python standard-library modules.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import sys
import time
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


SESSION_PREFIX = "behavior_"
DRIFT_RE = re.compile(r"^(?P<unit_id>[^_][^/]*)_drift\.pdf$", re.IGNORECASE)
OPTO_RE = re.compile(r"^unit_(?P<unit_id>\d+)(?:_.*)?_opto_tagging\.pdf$", re.IGNORECASE)


@dataclass(frozen=True)
class PdfRecord:
    category: str
    session: str
    unit_id: str
    source_path: Path
    target_name: str
    target_path: Path
    status: str = "pending"
    note: str = ""


def default_scratch_root() -> Path:
    """Pick the most likely capsule scratch path."""
    for candidate in (Path("/root/capsule/scratch"), Path("/scratch")):
        if candidate.exists():
            return candidate
    return Path("/root/capsule/scratch")


def normalize_mouse_id(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def parse_mouse_id_from_session(session: str) -> str:
    # behavior_826159_2026-01-22_12-58-47 -> 826159
    parts = session.split("_")
    if len(parts) >= 2 and parts[0] == "behavior":
        return normalize_mouse_id(parts[1])
    return ""


def find_session_from_path(path: Path) -> Optional[str]:
    """Return the nearest behavior_* parent folder name."""
    for part in reversed(path.parts):
        if part.startswith(SESSION_PREFIX):
            return part
    return None


def is_under_any(path: Path, roots: Sequence[Path]) -> bool:
    try:
        p = path.resolve()
    except OSError:
        p = path.absolute()
    for root in roots:
        try:
            r = root.resolve()
        except OSError:
            r = root.absolute()
        try:
            p.relative_to(r)
            return True
        except ValueError:
            continue
    return False


def iter_relevant_dirs(
    scratch_root: Path,
    data_types: Sequence[str],
    skip_roots: Sequence[Path],
    scan_errors: List[Dict[str, str]],
) -> Iterable[Tuple[str, Path]]:
    """
    Yield (category, directory) for folders matching:
      ephys/opto/<data_type>/drift
      ephys/opto/<data_type>/figures
    """
    data_type_set = {str(d).strip() for d in data_types if str(d).strip()}

    def onerror(exc: OSError) -> None:
        scan_errors.append(
            {
                "path": getattr(exc, "filename", ""),
                "error_type": exc.__class__.__name__,
                "message": str(exc),
            }
        )

    for root, dirnames, _filenames in os.walk(scratch_root, topdown=True, onerror=onerror):
        root_path = Path(root)

        # Do not re-scan output folders or their zip staging folders.
        if is_under_any(root_path, skip_roots):
            dirnames[:] = []
            continue

        parts = root_path.parts
        if len(parts) >= 4 and parts[-4] == "ephys" and parts[-3] == "opto":
            data_type = parts[-2]
            leaf = parts[-1]
            if data_type in data_type_set:
                if leaf == "drift":
                    yield "drift", root_path
                    dirnames[:] = []
                elif leaf == "figures":
                    yield "opto", root_path
                    dirnames[:] = []


def build_drift_records(src_dir: Path, out_dir: Path) -> List[PdfRecord]:
    records: List[PdfRecord] = []
    session = find_session_from_path(src_dir)
    if not session:
        return records

    try:
        files = sorted(src_dir.iterdir())
    except OSError:
        return records

    for path in files:
        if not path.is_file():
            continue
        m = DRIFT_RE.match(path.name)
        if not m:
            continue
        unit_id = m.group("unit_id")
        target_name = f"{session}_unit_{unit_id}_drift.pdf"
        records.append(
            PdfRecord(
                category="drift",
                session=session,
                unit_id=unit_id,
                source_path=path,
                target_name=target_name,
                target_path=out_dir / target_name,
            )
        )
    return records


def collect_opto_candidates(src_dir: Path) -> List[Tuple[str, str, Path]]:
    """Return (session, unit_id, source_path) candidates from one opto figures folder."""
    session = find_session_from_path(src_dir)
    if not session:
        return []

    try:
        files = sorted(src_dir.iterdir())
    except OSError:
        return []

    candidates: List[Tuple[str, str, Path]] = []
    for path in files:
        if not path.is_file():
            continue
        m = OPTO_RE.match(path.name)
        if not m:
            continue
        candidates.append((session, m.group("unit_id"), path))
    return candidates


def choose_opto_records(
    candidates: Sequence[Tuple[str, str, Path]],
    out_dir: Path,
    preferred_substring: str,
) -> List[PdfRecord]:
    """
    Collapse optotagging PDFs to one output PDF per (session, unit_id).

    If multiple source files exist for the same unit, prefer the file whose name
    contains preferred_substring, e.g. pulse_width_4. Otherwise choose the first
    path in sorted order and mark the manifest note.
    """
    grouped: Dict[Tuple[str, str], List[Path]] = defaultdict(list)
    for session, unit_id, path in candidates:
        grouped[(session, unit_id)].append(path)

    records: List[PdfRecord] = []
    preferred = preferred_substring.strip().lower()

    for (session, unit_id), paths in sorted(grouped.items()):
        paths = sorted(paths)
        chosen = paths[0]
        note = ""

        if len(paths) > 1:
            preferred_matches = [p for p in paths if preferred and preferred in p.name.lower()]
            if preferred_matches:
                chosen = sorted(preferred_matches)[0]
                note = (
                    f"multiple_opto_pdfs_for_unit={len(paths)}; "
                    f"selected_preferred_substring={preferred_substring!r}"
                )
            else:
                note = f"multiple_opto_pdfs_for_unit={len(paths)}; selected_first_sorted"

        target_name = f"{session}_unit_{unit_id}_opto.pdf"
        records.append(
            PdfRecord(
                category="opto",
                session=session,
                unit_id=unit_id,
                source_path=chosen,
                target_name=target_name,
                target_path=out_dir / target_name,
                note=note,
            )
        )

    return records


def filter_records(
    records: Sequence[PdfRecord],
    mouse_ids: Optional[Sequence[str]] = None,
    session_ids: Optional[Sequence[str]] = None,
) -> List[PdfRecord]:
    mouse_set = None
    if mouse_ids:
        mouse_set = {normalize_mouse_id(m) for m in mouse_ids if str(m).strip()}
    session_set = None
    if session_ids:
        session_set = {str(s).strip() for s in session_ids if str(s).strip()}

    out: List[PdfRecord] = []
    for rec in records:
        if mouse_set is not None:
            if parse_mouse_id_from_session(rec.session) not in mouse_set:
                continue
        if session_set is not None:
            if rec.session not in session_set:
                continue
        out.append(rec)
    return out


def copy_records(
    records: Sequence[PdfRecord],
    dry_run: bool = False,
    overwrite: bool = False,
) -> List[PdfRecord]:
    copied: List[PdfRecord] = []
    seen_targets: Dict[Path, PdfRecord] = {}

    for rec in records:
        target = rec.target_path
        note = rec.note

        if target in seen_targets:
            prev = seen_targets[target]
            copied.append(
                PdfRecord(
                    **{
                        **rec.__dict__,
                        "status": "skipped_duplicate_target",
                        "note": (
                            note + "; " if note else ""
                        ) + f"duplicate_target_of={prev.source_path}",
                    }
                )
            )
            continue

        seen_targets[target] = rec

        if target.exists() and not overwrite:
            copied.append(
                PdfRecord(
                    **{
                        **rec.__dict__,
                        "status": "skipped_exists",
                        "note": note,
                    }
                )
            )
            continue

        if dry_run:
            copied.append(
                PdfRecord(
                    **{
                        **rec.__dict__,
                        "status": "dry_run",
                        "note": note,
                    }
                )
            )
            continue

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            # copyfile avoids permission/metadata errors that can happen with copy2
            # on mounted capsule filesystems.
            shutil.copyfile(rec.source_path, target)
            copied.append(
                PdfRecord(
                    **{
                        **rec.__dict__,
                        "status": "copied",
                        "note": note,
                    }
                )
            )
        except Exception as exc:  # noqa: BLE001 - manifest should record all copy failures
            copied.append(
                PdfRecord(
                    **{
                        **rec.__dict__,
                        "status": "copy_failed",
                        "note": (
                            note + "; " if note else ""
                        ) + f"{exc.__class__.__name__}: {exc}",
                    }
                )
            )

    return copied


def write_manifest(path: Path, records: Sequence[PdfRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "category",
        "session",
        "mouse_id",
        "unit_id",
        "source_path",
        "target_name",
        "target_path",
        "status",
        "note",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rec in records:
            writer.writerow(
                {
                    "category": rec.category,
                    "session": rec.session,
                    "mouse_id": parse_mouse_id_from_session(rec.session),
                    "unit_id": rec.unit_id,
                    "source_path": str(rec.source_path),
                    "target_name": rec.target_name,
                    "target_path": str(rec.target_path),
                    "status": rec.status,
                    "note": rec.note,
                }
            )


def write_scan_errors(path: Path, errors: Sequence[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["path", "error_type", "message"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for err in errors:
            writer.writerow({field: err.get(field, "") for field in fields})


def zip_folder(folder: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=str(path.relative_to(folder.parent)))


def parse_csv_arg(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Copy drift and optotagging per-unit PDFs from capsule scratch into "
            "flat downloadable folders."
        )
    )
    parser.add_argument(
        "--scratch-root",
        type=Path,
        default=None,
        help="Scratch root to scan. Default: /root/capsule/scratch if present, else /scratch.",
    )
    parser.add_argument(
        "--drift-out-dir",
        type=Path,
        default=None,
        help="Output folder for renamed drift PDFs. Default: <scratch-root>/unit_drift_pdfs.",
    )
    parser.add_argument(
        "--opto-out-dir",
        type=Path,
        default=None,
        help="Output folder for renamed optotagging PDFs. Default: <scratch-root>/unit_opto_pdfs.",
    )
    parser.add_argument(
        "--data-types",
        nargs="+",
        default=["raw"],
        help="Data types under ephys/opto/<data_type> to scan. Default: raw.",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=["drift", "opto"],
        default=["drift", "opto"],
        help="Which categories to copy. Default: drift opto.",
    )
    parser.add_argument(
        "--preferred-opto-substring",
        default="pulse_width_4",
        help=(
            "If multiple optotagging PDFs exist for a unit, prefer filenames containing this substring. "
            "Default: pulse_width_4. Use an empty string to simply choose first sorted."
        ),
    )
    parser.add_argument(
        "--mouse-ids",
        default=None,
        help="Optional comma-separated mouse IDs to include, e.g. 826159,835444.",
    )
    parser.add_argument(
        "--session-ids",
        default=None,
        help="Optional comma-separated exact session IDs to include.",
    )
    parser.add_argument(
        "--clean-out-dirs",
        action="store_true",
        help="Delete existing output folders before copying.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing target PDFs. Default: skip existing targets.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and write manifests without copying files.",
    )
    parser.add_argument(
        "--make-zip",
        action="store_true",
        help="Create <output-folder>.zip for each copied category.",
    )
    args = parser.parse_args(argv)

    scratch_root = args.scratch_root or default_scratch_root()
    scratch_root = scratch_root.expanduser().resolve()

    drift_out_dir = args.drift_out_dir or (scratch_root / "unit_drift_pdfs")
    opto_out_dir = args.opto_out_dir or (scratch_root / "unit_opto_pdfs")
    drift_out_dir = drift_out_dir.expanduser().resolve()
    opto_out_dir = opto_out_dir.expanduser().resolve()

    categories = set(args.categories)
    mouse_ids = parse_csv_arg(args.mouse_ids)
    session_ids = parse_csv_arg(args.session_ids)

    if not scratch_root.exists():
        print(f"ERROR: scratch root does not exist: {scratch_root}", file=sys.stderr)
        return 2

    if args.clean_out_dirs and not args.dry_run:
        for folder in (drift_out_dir, opto_out_dir):
            if folder.exists():
                shutil.rmtree(folder)

    if not args.dry_run:
        if "drift" in categories:
            drift_out_dir.mkdir(parents=True, exist_ok=True)
        if "opto" in categories:
            opto_out_dir.mkdir(parents=True, exist_ok=True)

    skip_roots = [drift_out_dir, opto_out_dir]
    scan_errors: List[Dict[str, str]] = []
    drift_records: List[PdfRecord] = []
    opto_candidates: List[Tuple[str, str, Path]] = []

    t0 = time.time()
    print(f"Scanning scratch root: {scratch_root}")
    print(f"Data types: {tuple(args.data_types)}")
    print(f"Categories: {tuple(sorted(categories))}")
    if mouse_ids:
        print(f"Mouse filter: {tuple(mouse_ids)}")
    if session_ids:
        print(f"Session filter: {tuple(session_ids)}")

    for category, src_dir in iter_relevant_dirs(
        scratch_root=scratch_root,
        data_types=args.data_types,
        skip_roots=skip_roots,
        scan_errors=scan_errors,
    ):
        if category == "drift" and "drift" in categories:
            drift_records.extend(build_drift_records(src_dir, drift_out_dir))
        elif category == "opto" and "opto" in categories:
            opto_candidates.extend(collect_opto_candidates(src_dir))

    opto_records: List[PdfRecord] = []
    if "opto" in categories:
        opto_records = choose_opto_records(
            opto_candidates,
            out_dir=opto_out_dir,
            preferred_substring=args.preferred_opto_substring,
        )

    drift_records = filter_records(drift_records, mouse_ids=mouse_ids, session_ids=session_ids)
    opto_records = filter_records(opto_records, mouse_ids=mouse_ids, session_ids=session_ids)

    print(f"Found drift PDFs to consider: {len(drift_records)}")
    print(f"Found optotagging unit PDFs to consider: {len(opto_records)}")

    drift_results: List[PdfRecord] = []
    opto_results: List[PdfRecord] = []

    if "drift" in categories:
        drift_results = copy_records(drift_records, dry_run=args.dry_run, overwrite=args.overwrite)
        manifest = drift_out_dir / "unit_drift_pdfs_manifest.csv"
        write_manifest(manifest, drift_results)
        print(f"Wrote drift manifest: {manifest}")

    if "opto" in categories:
        opto_results = copy_records(opto_records, dry_run=args.dry_run, overwrite=args.overwrite)
        manifest = opto_out_dir / "unit_opto_pdfs_manifest.csv"
        write_manifest(manifest, opto_results)
        print(f"Wrote opto manifest: {manifest}")

    # Write scan errors next to whichever outputs are requested.
    if "drift" in categories:
        write_scan_errors(drift_out_dir / "scan_errors.csv", scan_errors)
    if "opto" in categories:
        write_scan_errors(opto_out_dir / "scan_errors.csv", scan_errors)

    def count_status(records: Sequence[PdfRecord]) -> Dict[str, int]:
        out: Dict[str, int] = defaultdict(int)
        for r in records:
            out[r.status] += 1
        return dict(sorted(out.items()))

    print("\nSummary:")
    if "drift" in categories:
        print(f"  drift output folder: {drift_out_dir}")
        print(f"  drift status counts: {count_status(drift_results)}")
    if "opto" in categories:
        print(f"  opto output folder:  {opto_out_dir}")
        print(f"  opto status counts:  {count_status(opto_results)}")
    print(f"  scan errors:         {len(scan_errors)}")
    print(f"  elapsed seconds:     {time.time() - t0:.1f}")

    if args.make_zip and not args.dry_run:
        if "drift" in categories:
            zip_path = drift_out_dir.with_suffix(".zip")
            zip_folder(drift_out_dir, zip_path)
            print(f"Created drift zip: {zip_path}")
        if "opto" in categories:
            zip_path = opto_out_dir.with_suffix(".zip")
            zip_folder(opto_out_dir, zip_path)
            print(f"Created opto zip:  {zip_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

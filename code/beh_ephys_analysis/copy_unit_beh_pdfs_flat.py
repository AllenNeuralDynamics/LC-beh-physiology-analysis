#!/usr/bin/env python3
"""
Collect unit_beh_analysis_JL per-unit PDFs from /root/capsule/scratch and copy them
into a single flat folder for use by the clickable Plotly CCF HTML.

Input files are expected to look like:
  .../<session>/ephys/<data_type>/figures/<align>/unit_<unit_id>_<align>.pdf

Examples:
  /root/capsule/scratch/838332/behavior_838332_2026-03-10_13-23-52/ephys/raw/figures/go_cue/unit_11_go_cue.pdf
  /root/capsule/scratch/838332/behavior_838332_2026-03-10_13-23-52/ephys/raw/figures/response/unit_44_response.pdf

Output files are renamed to:
  <session>_unit_<unit_id>_<align>.pdf

Example:
  behavior_838332_2026-03-10_13-23-52_unit_11_go_cue.pdf
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_SCRATCH_ROOT = Path("/root/capsule/scratch")
DEFAULT_OUT_DIR = Path("/root/capsule/scratch/unit_pdfs")
DEFAULT_ALIGN_NAMES = ("go_cue", "response")
DEFAULT_DATA_TYPES = ("raw",)


@dataclass(frozen=True)
class PdfHit:
    source_path: Path
    session: str
    unit_id: str
    align: str
    data_type: str | None
    target_name: str


def _split_csv_or_space(values: Sequence[str] | None, default: Sequence[str]) -> tuple[str, ...]:
    if not values:
        return tuple(default)
    out: list[str] = []
    for value in values:
        for item in str(value).split(","):
            item = item.strip()
            if item:
                out.append(item)
    return tuple(out) if out else tuple(default)


def _normalize_id_text(value: str) -> str:
    """Normalize parsed unit ids while preserving non-integer text if present."""
    value = str(value).strip()
    if re.fullmatch(r"\d+\.0+", value):
        value = value.split(".", 1)[0]
    return value


def _find_session_from_path(path: Path) -> str | None:
    for part in reversed(path.parts):
        if part.startswith("behavior_"):
            return part
    return None


def _find_data_type_from_path(path: Path) -> str | None:
    parts = path.parts
    for i, part in enumerate(parts[:-1]):
        if part == "ephys" and i + 1 < len(parts):
            return parts[i + 1]
    return None


def _is_under(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


def _candidate_pdf_paths(scratch_root: Path,
                         out_dir: Path,
                         align_names: Sequence[str],
                         follow_symlinks: bool = False) -> Iterable[Path]:
    """
    Walk scratch_root and yield candidate unit_*.pdf paths.

    os.walk is used instead of a single broad rglob so we can prune the output
    folder and common junk folders while scanning a large scratch tree.
    """
    scratch_root = Path(scratch_root)
    out_dir = Path(out_dir)
    align_set = set(align_names)

    prune_names = {
        ".git",
        ".ipynb_checkpoints",
        "__pycache__",
        ".Trash",
        ".Trash-0",
        ".cache",
    }

    for root, dirs, files in os.walk(scratch_root, followlinks=follow_symlinks):
        root_path = Path(root)

        # Do not recursively copy/scan the destination folder.
        if _is_under(root_path, out_dir):
            dirs[:] = []
            continue

        dirs[:] = [d for d in dirs if d not in prune_names]

        parent_align = root_path.name
        if parent_align not in align_set:
            continue

        for filename in files:
            if not filename.endswith(".pdf"):
                continue
            if not filename.startswith("unit_"):
                continue
            yield root_path / filename


def _parse_pdf_hit(path: Path,
                   align_names: Sequence[str],
                   data_types: Sequence[str] | None) -> PdfHit | None:
    align_expr = "|".join(re.escape(a) for a in sorted(align_names, key=len, reverse=True))
    pattern = re.compile(rf"^unit_(?P<unit_id>.+)_(?P<align>{align_expr})\.pdf$")

    match = pattern.match(path.name)
    if match is None:
        return None

    unit_id = _normalize_id_text(match.group("unit_id"))
    align = match.group("align")

    # Require the parent folder to match the parsed alignment. This avoids false
    # positives from unrelated files named similarly elsewhere.
    if path.parent.name != align:
        return None

    session = _find_session_from_path(path)
    if session is None:
        return None

    data_type = _find_data_type_from_path(path)
    if data_types is not None and data_type not in set(data_types):
        return None

    target_name = f"{session}_unit_{unit_id}_{align}.pdf"
    return PdfHit(
        source_path=path,
        session=session,
        unit_id=unit_id,
        align=align,
        data_type=data_type,
        target_name=target_name,
    )


def _unique_keep_both_path(target_path: Path, source_path: Path) -> Path:
    """Create a deterministic conflict filename if the flat target already exists."""
    data_type = _find_data_type_from_path(source_path) or "unknown_data_type"
    stem = target_path.stem
    suffix = target_path.suffix

    candidate = target_path.with_name(f"{stem}_{data_type}{suffix}")
    if not candidate.exists():
        return candidate

    parent = target_path.parent
    n = 2
    while True:
        candidate = parent / f"{stem}_{data_type}_dup{n}{suffix}"
        if not candidate.exists():
            return candidate
        n += 1


def copy_unit_pdfs_flat(scratch_root: str | os.PathLike = DEFAULT_SCRATCH_ROOT,
                        out_dir: str | os.PathLike = DEFAULT_OUT_DIR,
                        align_names: Sequence[str] = DEFAULT_ALIGN_NAMES,
                        data_types: Sequence[str] | str | None = DEFAULT_DATA_TYPES,
                        on_conflict: str = "skip",
                        clean_out_dir: bool = False,
                        dry_run: bool = False,
                        make_zip: bool = False,
                        follow_symlinks: bool = False) -> dict[str, int | str]:
    """
    Copy matching unit_beh PDFs into a flat output folder.

    Parameters
    ----------
    scratch_root:
        Root folder to scan.
    out_dir:
        Destination folder for flattened PDFs.
    align_names:
        Alignments to collect, usually ("go_cue", "response").
    data_types:
        Data types to include, e.g. ("raw",), ("raw", "curated"), "all", or None.
    on_conflict:
        What to do if two source PDFs map to the same flattened filename.
        Choices: "skip", "overwrite", "keep_both", "error".
    clean_out_dir:
        Remove existing PDFs/manifests in out_dir before copying.
    dry_run:
        Report what would be copied without writing files.
    make_zip:
        Also create out_dir.zip after copying.
    follow_symlinks:
        Whether os.walk should follow symlinks.
    """
    scratch_root = Path(scratch_root).expanduser()
    out_dir = Path(out_dir).expanduser()
    align_names = tuple(align_names)

    if isinstance(data_types, str):
        if data_types.lower() == "all":
            data_types_tuple: tuple[str, ...] | None = None
        else:
            data_types_tuple = _split_csv_or_space([data_types], DEFAULT_DATA_TYPES)
    elif data_types is None:
        data_types_tuple = None
    else:
        data_types_tuple = tuple(data_types)

    if on_conflict not in {"skip", "overwrite", "keep_both", "error"}:
        raise ValueError("on_conflict must be one of: skip, overwrite, keep_both, error")

    if not scratch_root.exists():
        raise FileNotFoundError(f"Scratch root does not exist: {scratch_root}")

    if clean_out_dir and out_dir.exists() and not dry_run:
        for child in out_dir.iterdir():
            if child.is_file() or child.is_symlink():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)

    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_path = out_dir / f"unit_pdf_copy_manifest_{timestamp}.csv"
    conflict_path = out_dir / f"unit_pdf_copy_conflicts_{timestamp}.csv"

    summary = {
        "candidate_files_seen": 0,
        "matched_files": 0,
        "copied": 0,
        "overwritten": 0,
        "skipped_existing": 0,
        "skipped_conflict": 0,
        "kept_both_conflict": 0,
        "parse_skipped": 0,
        "errors": 0,
        "out_dir": str(out_dir),
    }

    # Track targets created during this run so duplicates are caught even during dry runs.
    target_sources: dict[str, Path] = {}
    rows: list[dict[str, str | int]] = []
    conflicts: list[dict[str, str]] = []

    for pdf_path in _candidate_pdf_paths(
        scratch_root=scratch_root,
        out_dir=out_dir,
        align_names=align_names,
        follow_symlinks=follow_symlinks,
    ):
        summary["candidate_files_seen"] += 1
        hit = _parse_pdf_hit(pdf_path, align_names=align_names, data_types=data_types_tuple)
        if hit is None:
            summary["parse_skipped"] += 1
            continue

        summary["matched_files"] += 1
        target_path = out_dir / hit.target_name
        action = "copy"
        note = ""

        existing_source = target_sources.get(hit.target_name)
        target_already_exists = target_path.exists() or existing_source is not None

        if target_already_exists:
            if on_conflict == "skip":
                action = "skip_existing_or_duplicate"
                summary["skipped_existing"] += 1
                note = f"target already exists or was already planned from {existing_source or target_path}"
                conflicts.append({
                    "target_name": hit.target_name,
                    "kept_or_existing": str(existing_source or target_path),
                    "skipped_source": str(hit.source_path),
                    "reason": "target_exists_or_duplicate",
                })
            elif on_conflict == "overwrite":
                action = "overwrite"
                summary["overwritten"] += 1
            elif on_conflict == "keep_both":
                target_path = _unique_keep_both_path(target_path, hit.source_path)
                action = "copy_keep_both"
                summary["kept_both_conflict"] += 1
                conflicts.append({
                    "target_name": hit.target_name,
                    "kept_or_existing": str(existing_source or out_dir / hit.target_name),
                    "skipped_source": "",
                    "reason": f"kept duplicate as {target_path.name}",
                })
            else:
                summary["errors"] += 1
                raise FileExistsError(
                    f"Conflict for {hit.target_name}: existing/planned target from "
                    f"{existing_source or target_path}; new source {hit.source_path}"
                )

        if action in {"copy", "overwrite", "copy_keep_both"}:
            if not dry_run:
                try:
                    out_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(hit.source_path, target_path)
                except Exception as exc:
                    summary["errors"] += 1
                    action = "error"
                    note = repr(exc)
                else:
                    if action == "copy":
                        summary["copied"] += 1
            else:
                if action == "copy":
                    summary["copied"] += 1
                note = "dry_run"

        target_sources.setdefault(hit.target_name, hit.source_path)

        try:
            stat = hit.source_path.stat()
            size_bytes = stat.st_size
            source_mtime = datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds")
        except Exception:
            size_bytes = ""
            source_mtime = ""

        rows.append({
            "action": action,
            "session": hit.session,
            "unit_id": hit.unit_id,
            "align": hit.align,
            "data_type": hit.data_type or "",
            "source_path": str(hit.source_path),
            "target_name": target_path.name,
            "target_path": str(target_path),
            "source_size_bytes": size_bytes,
            "source_mtime": source_mtime,
            "note": note,
        })

    if not dry_run:
        with manifest_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "action",
                    "session",
                    "unit_id",
                    "align",
                    "data_type",
                    "source_path",
                    "target_name",
                    "target_path",
                    "source_size_bytes",
                    "source_mtime",
                    "note",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        if conflicts:
            with conflict_path.open("w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["target_name", "kept_or_existing", "skipped_source", "reason"],
                )
                writer.writeheader()
                writer.writerows(conflicts)

    summary["manifest_path"] = str(manifest_path)
    summary["conflict_path"] = str(conflict_path) if conflicts else ""

    if make_zip and not dry_run:
        # Creates /path/to/unit_pdfs.zip containing a top-level unit_pdfs/ folder.
        zip_base = str(out_dir)
        zip_path = shutil.make_archive(
            base_name=zip_base,
            format="zip",
            root_dir=str(out_dir.parent),
            base_dir=out_dir.name,
        )
        summary["zip_path"] = zip_path
    elif make_zip:
        summary["zip_path"] = "dry_run"

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Copy unit_beh_analysis_JL unit PDFs into a flat unit_pdfs folder and "
            "rename them to {session}_unit_{unit_id}_{align}.pdf."
        )
    )
    parser.add_argument("--scratch-root", default=str(DEFAULT_SCRATCH_ROOT),
                        help="Scratch folder to recursively scan. Default: /root/capsule/scratch")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                        help="Destination folder. Default: /root/capsule/scratch/unit_pdfs")
    parser.add_argument("--align-names", nargs="*", default=list(DEFAULT_ALIGN_NAMES),
                        help="Alignments to copy. Accepts space or comma separated values. Default: go_cue response")
    parser.add_argument("--data-types", nargs="*", default=list(DEFAULT_DATA_TYPES),
                        help="Data types to include. Use 'all' for all. Default: raw")
    parser.add_argument("--on-conflict", default="skip",
                        choices=["skip", "overwrite", "keep_both", "error"],
                        help="What to do when multiple PDFs flatten to the same target filename. Default: skip")
    parser.add_argument("--clean-out-dir", action="store_true",
                        help="Delete existing contents of out-dir before copying.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Scan and report without copying files.")
    parser.add_argument("--make-zip", action="store_true",
                        help="Also create <out-dir>.zip for easier download.")
    parser.add_argument("--follow-symlinks", action="store_true",
                        help="Follow symlinked directories while scanning.")

    args = parser.parse_args()

    align_names = _split_csv_or_space(args.align_names, DEFAULT_ALIGN_NAMES)
    data_types_raw = _split_csv_or_space(args.data_types, DEFAULT_DATA_TYPES)
    data_types: Sequence[str] | str | None
    if len(data_types_raw) == 1 and data_types_raw[0].lower() == "all":
        data_types = "all"
    else:
        data_types = data_types_raw

    print("Scanning for unit_beh_analysis_JL PDFs")
    print(f"  scratch_root: {args.scratch_root}")
    print(f"  out_dir:      {args.out_dir}")
    print(f"  align_names:  {align_names}")
    print(f"  data_types:   {data_types}")
    print(f"  conflict:     {args.on_conflict}")
    print(f"  dry_run:      {args.dry_run}")

    summary = copy_unit_pdfs_flat(
        scratch_root=args.scratch_root,
        out_dir=args.out_dir,
        align_names=align_names,
        data_types=data_types,
        on_conflict=args.on_conflict,
        clean_out_dir=args.clean_out_dir,
        dry_run=args.dry_run,
        make_zip=args.make_zip,
        follow_symlinks=args.follow_symlinks,
    )

    print("\nDone.")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

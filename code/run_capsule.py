"""Capsule entry point: dispatch to one pipeline based on the app-panel arguments.

The arguments mirror the parameters declared in .codeocean/app-panel.json, and the
defaults here match the panel defaults so a bare `python run_capsule.py` behaves
like clicking "Reproducible Run" with nothing filled in:

    processing   : pack_nwb | data_attachment | analysis   (default pack_nwb)
    mode         : test | production                       (default test,  used by pack_nwb)
    dry_run      : True | False                            (default True,  used by the other two)
    add_metadata : True | False                            (default False, used by pack_nwb)

Values arrive either as flags (--processing pack_nwb --mode test) or, when the app
panel hands them over positionally, in panel order (pack_nwb test True False). Anything
this script does not recognise is forwarded to the pipeline script unchanged.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = CODE_DIR.parent

PACK_NWB_SCRIPT = CODE_DIR / "data_management" / "build_all_nwb.py"
DATA_ATTACHMENT_SCRIPT = CODE_DIR / "run_data_attachment.py"
ANALYSIS_SCRIPT = CODE_DIR / "run_analysis.py"

PROCESSING_CHOICES = ("pack_nwb", "data_attachment", "analysis")
MODE_CHOICES = ("test", "production")

DEFAULT_PROCESSING = "pack_nwb"
DEFAULT_MODE = "test"
DEFAULT_DRY_RUN = "True"
DEFAULT_ADD_METADATA = "False"

TRUE_STRINGS = ("t", "T", "true", "True", "TRUE", "1", "y", "Y", "yes", "Yes")


def parse_args(argv: list[str] | None = None) -> tuple[str, str, bool, bool, list[str]]:
    """Resolve the panel parameters from flags, positional values, then defaults.

    Returns:
        tuple: (processing, mode, dry_run, add_metadata, passthrough_args)
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--processing", "--processing_type", dest="processing", type=str, default=None,
        help=f"which pipeline to run: {', '.join(PROCESSING_CHOICES)} (default {DEFAULT_PROCESSING})",
    )
    parser.add_argument(
        "--mode", type=str, default=None,
        help=f"pack_nwb build scope: {', '.join(MODE_CHOICES)} (default {DEFAULT_MODE})",
    )
    parser.add_argument(
        "--dry_run", "--dry", "--check-only", dest="dry_run", type=str, nargs="?", const="True",
        default=None,
        help=f"skip the real work for data_attachment / analysis (default {DEFAULT_DRY_RUN})",
    )
    parser.add_argument(
        "--add_metadata", "--add-metadata", dest="add_metadata", type=str, nargs="?", const="True",
        default=None,
        help=f"bundle the AIND metadata JSON into each NWB, pack_nwb only (default {DEFAULT_ADD_METADATA})",
    )
    parser.add_argument(
        "panel_args", nargs="*",
        help="panel values given positionally, in app-panel order: processing mode dry_run add_metadata",
    )
    args, passthrough = parser.parse_known_args(argv)

    # Pad so positional values can stand in for any flag that was not supplied.
    positional = (list(args.panel_args) + [None, None, None, None])[:4]
    processing = args.processing or positional[0] or DEFAULT_PROCESSING
    mode = args.mode or positional[1] or DEFAULT_MODE
    dry_run_value = args.dry_run or positional[2] or DEFAULT_DRY_RUN
    add_metadata_value = args.add_metadata or positional[3] or DEFAULT_ADD_METADATA

    if processing not in PROCESSING_CHOICES:
        parser.error(f"unknown processing {processing!r}; expected one of {', '.join(PROCESSING_CHOICES)}")
    if mode not in MODE_CHOICES:
        parser.error(f"unknown mode {mode!r}; expected one of {', '.join(MODE_CHOICES)}")

    return (
        processing,
        mode,
        dry_run_value in TRUE_STRINGS,
        add_metadata_value in TRUE_STRINGS,
        passthrough,
    )


def build_command(
    processing: str, mode: str, dry_run: bool, add_metadata: bool, passthrough: list[str]
) -> list[str]:
    """Build the pipeline command line for the selected processing type."""
    if processing == "pack_nwb":
        script = PACK_NWB_SCRIPT
        options = ["--mode", mode, "--add-metadata" if add_metadata else "--no-add-metadata"]
    elif processing == "data_attachment":
        script, options = DATA_ATTACHMENT_SCRIPT, ["--check-only"] if dry_run else []
    else:
        script, options = ANALYSIS_SCRIPT, ["--check-only"] if dry_run else []

    if not script.is_file():
        raise FileNotFoundError(f"Pipeline script not found: {script}")

    return [sys.executable, "-u", str(script), *options, *passthrough]


def run(argv: list[str] | None = None) -> int:
    """Dispatch to the pipeline selected by the panel arguments."""
    processing, mode, dry_run, add_metadata, passthrough = parse_args(argv)
    command = build_command(processing, mode, dry_run, add_metadata, passthrough)

    print("=" * 80, flush=True)
    print(f"processing   = {processing}", flush=True)
    if processing == "pack_nwb":
        print(f"mode         = {mode}", flush=True)
        print(f"add_metadata = {add_metadata}", flush=True)
    else:
        print(f"dry_run      = {dry_run}", flush=True)
    print(f"command      = {' '.join(command)}", flush=True)
    print("=" * 80, flush=True)

    completed = subprocess.run(command, cwd=str(WORKSPACE_DIR), check=False)
    if completed.returncode != 0:
        print(
            f"ERROR: {processing} failed with exit code {completed.returncode}",
            file=sys.stderr,
            flush=True,
        )
    return completed.returncode


def main() -> int:
    """CLI entry point."""
    try:
        return run()
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


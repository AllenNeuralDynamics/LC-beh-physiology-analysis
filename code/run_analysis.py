"""Top-level runner for figure-preparation scripts and manuscript notebooks."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = CODE_DIR.parent
DATA_ATTACH_SCRIPT = CODE_DIR / "data_management" / "attach_all_data_capsule.py"
FIGURE_CSV_PATH = CODE_DIR / "data_management" / "figure_csv_organization.py"
FIG_PREP_DIR = CODE_DIR / "beh_ephys_analysis" / "session_combine" / "figure_preparation"
FIG_PREP_SEQUENCE_FILE = FIG_PREP_DIR / "sequence.txt"
MANUSCRIPT_FIG_DIR = CODE_DIR / "beh_ephys_analysis" / "session_combine" / "manuscript_figures"
FIG_NOTEBOOK_LIST = MANUSCRIPT_FIG_DIR / "fig_notebook_list.txt"
SUPPRESSED_WARNINGS = "ignore::FutureWarning,ignore::DeprecationWarning,ignore::UserWarning"


def load_sequence(sequence_file: Path) -> list[str]:
    """Load an ordered list of files from a plain-text sequence file."""
    if not sequence_file.exists():
        raise FileNotFoundError(f"Sequence file not found: {sequence_file}")

    entries = []
    for line in sequence_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        entries.append(line)

    if not entries:
        raise ValueError(f"No entries found in: {sequence_file}")

    return entries


def run_script(script_name: str, check_only: bool = False) -> float:
    """Run one figure-preparation Python script, or just validate that it exists.

    Returns:
        float: Duration in seconds (0.0 if check_only mode).
    """
    script_path = FIG_PREP_DIR / script_name
    if not script_path.is_file():
        raise FileNotFoundError(f"Script listed in sequence does not exist: {script_path}")

    if check_only:
        print(f"[CHECK] {script_path}", flush=True)
        return 0.0

    print(f"\n=== Running {script_name} ===", flush=True)
    start_time = time.time()
    completed = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(WORKSPACE_DIR),
        check=False,
    )
    duration = time.time() - start_time

    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, completed.args)

    print(f"✓ Completed in {duration:.2f}s", flush=True)
    return duration


def build_subprocess_env() -> dict[str, str]:
    """Create a subprocess environment that suppresses noisy Python warnings."""
    env = os.environ.copy()
    existing = env.get("PYTHONWARNINGS", "").strip()
    env["PYTHONWARNINGS"] = (
        f"{existing},{SUPPRESSED_WARNINGS}" if existing else SUPPRESSED_WARNINGS
    )
    return env


def run_notebook(notebook_name: str, check_only: bool = False) -> float:
    """Execute one manuscript-figure notebook in place, or just validate it exists.

    Returns:
        float: Duration in seconds (0.0 if check_only mode).
    """
    notebook_path = MANUSCRIPT_FIG_DIR / notebook_name
    if not notebook_path.is_file():
        raise FileNotFoundError(f"Notebook listed in figure order does not exist: {notebook_path}")

    if check_only:
        print(f"[CHECK] {notebook_path}", flush=True)
        return 0.0

    print(f"\n=== Executing {notebook_name} ===", flush=True)
    start_time = time.time()
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--inplace",
            "--ExecutePreprocessor.timeout=-1",
            str(notebook_path),
        ],
        cwd=str(WORKSPACE_DIR),
        env=build_subprocess_env(),
        check=False,
    )
    duration = time.time() - start_time

    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, completed.args)

    print(f"✓ Completed in {duration:.2f}s", flush=True)
    return duration


def run_data_attachment(check_only: bool = False) -> None:
    """Run data attachment helper before executing the main sequence."""
    if not DATA_ATTACH_SCRIPT.is_file():
        raise FileNotFoundError(f"Data-attachment script not found: {DATA_ATTACH_SCRIPT}")

    if check_only:
        print(f"[CHECK] {DATA_ATTACH_SCRIPT}", flush=True)
        return

    print("\n=== Attaching data ===", flush=True)
    completed = subprocess.run(
        [sys.executable, str(DATA_ATTACH_SCRIPT)],
        cwd=str(WORKSPACE_DIR),
        env=build_subprocess_env(),
        check=False,
    )
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, completed.args)


def save_timing_csv(timings: list[tuple[str, float]], output_path: Path, category: str) -> None:
    """Save timing data to a CSV file, updating existing rows or appending new ones.

    Args:
        timings: List of (filename, duration_seconds) tuples
        output_path: Path to the output CSV file
        category: Category label (e.g., "prep_script" or "manuscript_figure")
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Read existing data if file exists
    existing_data = {}
    if output_path.exists():
        with open(output_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['filename'] != 'TOTAL':  # Skip TOTAL row
                    existing_data[row['filename']] = row

    # Update with new timings
    for filename, duration in timings:
        existing_data[filename] = {
            'filename': filename,
            'duration_seconds': f"{duration:.2f}",
            'duration_minutes': f"{duration/60:.2f}",
            'category': category,
            'timestamp': timestamp
        }

    # Write all data back
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'duration_seconds', 'duration_minutes', 'category', 'timestamp'])

        for row in existing_data.values():
            writer.writerow([
                row['filename'],
                row['duration_seconds'],
                row['duration_minutes'],
                row['category'],
                row['timestamp']
            ])

        # Add total row
        total_duration = sum(float(row['duration_seconds']) for row in existing_data.values())
        writer.writerow([
            'TOTAL',
            f"{total_duration:.2f}",
            f"{total_duration/60:.2f}",
            category,
            timestamp
        ])


def run(check_only: bool = False, update_timing: bool = False) -> int:
    """Run figure-preparation scripts, then manuscript notebooks, in their listed order.

    Args:
        check_only: Validate file existence without executing anything.
        update_timing: Write timing CSVs after each step. Disabled by default so that
            reproducible capsule runs do not modify files under code/.
    """
    scripts = load_sequence(FIG_PREP_SEQUENCE_FILE)
    notebooks = load_sequence(FIG_NOTEBOOK_LIST)
    print(f"Loaded {len(scripts)} scripts from {FIG_PREP_SEQUENCE_FILE}", flush=True)
    print(f"Loaded {len(notebooks)} notebooks from {FIG_NOTEBOOK_LIST}", flush=True)

    # Track timing for all scripts and notebooks
    script_timings = []
    notebook_timings = []

    # run_data_attachment(check_only=check_only)

    print("\n" + "="*80, flush=True)
    print("FIGURE PREPARATION SCRIPTS", flush=True)
    print("="*80, flush=True)

    prep_csv_path = FIG_PREP_DIR / "timing_report.csv"

    # for idx, script_name in enumerate(scripts, start=1):
    #     print(f"\n[prep {idx}/{len(scripts)}] {script_name}", flush=True)
    #     duration = run_script(script_name, check_only=check_only)
    #     script_timings.append((script_name, duration))

    #     # Save timing after each script completes
    #     if not check_only:
    #         save_timing_csv(script_timings, prep_csv_path, "prep_script")
    #         print(f"  → Updated timing report: {prep_csv_path}", flush=True)

    notebook_csv_path = MANUSCRIPT_FIG_DIR / "timing_report.csv"

    for idx, notebook_name in enumerate(notebooks, start=1):
        print(f"[figure {idx}/{len(notebooks)}] {notebook_name}", flush=True)
        duration = run_notebook(notebook_name, check_only=check_only)
        notebook_timings.append((notebook_name, duration))
    
        # Save timing after each notebook completes
        if not check_only and update_timing:
            save_timing_csv(notebook_timings, notebook_csv_path, "manuscript_figure")
            print(f"  → Updated timing report: {notebook_csv_path}", flush=True)

    # Print timing summary and save CSV files
    if not check_only:
        print("\n" + "="*80, flush=True)
        print("TIMING SUMMARY", flush=True)
        print("="*80, flush=True)

        if script_timings:
            print("\nFigure Preparation Scripts:", flush=True)
            print("-" * 80, flush=True)
            total_script_time = 0.0  
            for name, duration in script_timings:
                print(f"  {name:60s} {duration:8.2f}s", flush=True)
                total_script_time += duration
            print("-" * 80, flush=True)
            print(f"  {'TOTAL PREP TIME':60s} {total_script_time:8.2f}s", flush=True)

            if update_timing:
                print(f"\n  → Final timing report: {prep_csv_path}", flush=True)

        if notebook_timings:
            print("\nManuscript Figure Notebooks:", flush=True)
            print("-" * 80, flush=True)
            total_notebook_time = 0.0
            for name, duration in notebook_timings:
                print(f"  {name:60s} {duration:8.2f}s", flush=True)
                total_notebook_time += duration
            print("-" * 80, flush=True)
            print(f"  {'TOTAL NOTEBOOK TIME':60s} {total_notebook_time:8.2f}s", flush=True)

            if update_timing:
                print(f"\n  → Final timing report: {notebook_csv_path}", flush=True)

        total_time = sum(d for _, d in script_timings) + sum(d for _, d in notebook_timings)
        print("\n" + "="*80, flush=True)
        print(f"  {'GRAND TOTAL':60s} {total_time:8.2f}s ({total_time/60:.2f} min)", flush=True)
        print("="*80, flush=True)

        # Reorganze figure and csvs in output directory
        if not check_only:
            print("\nReorganizing figure and CSV outputs...", flush=True)
            subprocess.run(
                [sys.executable, str(FIGURE_CSV_PATH)],
                cwd=str(WORKSPACE_DIR),
                env=build_subprocess_env(),
                check=True,
            )
            print("✓ Reorganization complete.", flush=True)

    if check_only:
        print("\nSequence validation completed successfully for scripts and notebooks.", flush=True)
    else:
        print("\nAll scripts and notebooks completed successfully.", flush=True)
    return 0


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run figure-preparation scripts and manuscript notebooks in their listed order."
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate the sequence and file existence without executing the scripts.",
    )
    parser.add_argument(
        "--update-timing",
        action="store_true",
        help="Write timing CSVs after each step. Omit this flag for reproducible capsule runs "
             "so that files under code/ are not modified.",
    )
    args = parser.parse_args()

    try:
        return run(check_only=args.check_only, update_timing=args.update_timing)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 1
    except subprocess.CalledProcessError as exc:
        failed_script = exc.cmd[-1] if isinstance(exc.cmd, (list, tuple)) and exc.cmd else "<unknown>"
        print(
            f"ERROR: script failed with exit code {exc.returncode}: {failed_script}",
            file=sys.stderr,
            flush=True,
        )
        return exc.returncode or 1


if __name__ == "__main__":
    raise SystemExit(main())
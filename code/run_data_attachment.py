"""Top-level runner for data attachment."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = CODE_DIR.parent
DATA_ATTACH_SCRIPT = CODE_DIR / "data_management" / "attach_all_data_capsule.py"
SUPPRESSED_WARNINGS = "ignore::FutureWarning,ignore::DeprecationWarning,ignore::UserWarning"


def build_subprocess_env() -> dict[str, str]:
    """Create a subprocess environment that suppresses noisy Python warnings."""
    env = os.environ.copy()
    existing = env.get("PYTHONWARNINGS", "").strip()
    env["PYTHONWARNINGS"] = (
        f"{existing},{SUPPRESSED_WARNINGS}" if existing else SUPPRESSED_WARNINGS
    )
    return env


def run_data_attachment(check_only: bool = False) -> None:
    """Attach the data assets this capsule needs."""
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


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Attach the data assets this capsule needs.")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate that the data-attachment script exists without executing it.",
    )
    args = parser.parse_args()

    try:
        run_data_attachment(check_only=args.check_only)
        return 0
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

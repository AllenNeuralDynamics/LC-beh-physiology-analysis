"""
Download dandiset 001950 from DANDI Archive.

Usage:
    python download_dandiset_001950.py --data_dir /path/to/save

Requires:
    pip install dandi
"""

import argparse
import subprocess
from pathlib import Path


DANDISET_ID = "001950"
DANDI_URL = f"https://dandiarchive.org/dandiset/{DANDISET_ID}"


def download_dandiset(data_dir: str) -> None:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dandiset {DANDISET_ID} to: {data_dir}")
    subprocess.run(
        ["dandi", "download", "-o", str(data_dir), DANDI_URL],
        check=True,
    )
    print("Download complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Download dandiset {DANDISET_ID}")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory where the dandiset will be saved",
    )
    args = parser.parse_args()
    download_dandiset(args.data_dir)

"""Sync the capsule `scratch/` tree to S3.

For bulk transfers, drive the AWS CLI directly instead -- install steps, tuned
`aws configure` settings, the flags to avoid and measured throughput are in the
sibling `s3_sync.md`.
"""

import os, sys
# Resolve code/beh_ephys_analysis (the folder containing `utils`) relative to this
# file's location, so imports work no matter where the repo is checked out.
_anchor = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.path.abspath(os.getcwd())
while _anchor != os.path.dirname(_anchor):
    _beh_ephys_root = os.path.join(_anchor, "code", "beh_ephys_analysis")
    if os.path.isdir(os.path.join(_beh_ephys_root, "utils")):
        if _beh_ephys_root in sys.path:
            sys.path.remove(_beh_ephys_root)
        sys.path.insert(0, _beh_ephys_root)
        break
    _anchor = os.path.dirname(_anchor)
from utils.capsule_migration import CAPSULE_ROOT
# %%
import concurrent.futures
import logging
import os
import subprocess
from threading import local

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


def sync_directory(local_dir, destination, if_copy=False, if_dry_run=True, if_delete=False):
    """
    Sync the local directory with the given S3 destination using aws s3 sync.
    Returns a status string based on the command output.
    """
    try:
        if if_copy:
            cmd = ["aws", "s3", "cp", local_dir, destination]
            if os.path.isdir(local_dir):
                cmd.append("--recursive")

            if if_dry_run:
                cmd.append("--dryrun")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
            )
        else:
            # Run aws s3 sync command and capture the output
            cmd = ["aws", "s3", "sync", local_dir, destination]
            if if_delete:
                cmd.append("--delete")
            if if_dry_run:
                cmd.append("--dryrun")
            result = subprocess.run(
                cmd, capture_output=True, text=True
            )
        output = result.stdout + result.stderr

        # A non-zero exit has to be reported, or a failed sync falls through to
        # the "nothing needed uploading" branch below and looks like success.
        if result.returncode != 0:
            logger.error(output)
            return f"error during sync (aws exit {result.returncode})"

        # Real transfers print lines starting with "upload:", but --dryrun prefixes
        # them with "(dryrun) ", so a bare `"upload:" in output` substring test
        # claims an upload that never happened. Match at line start instead.
        lines = output.splitlines()
        uploaded = [line for line in lines if line.startswith("upload:")]
        would_upload = [line for line in lines if line.startswith("(dryrun)")]

        if if_dry_run:
            logger.info(output)
            logger.info(f"Dry run: {len(would_upload)} file(s) would upload from {local_dir}.")
            return f"dry run, {len(would_upload)} file(s) would upload"
        elif uploaded:
            logger.info(f"Uploaded {len(uploaded)} file(s) from {local_dir} to {destination}!")
            return "successfully uploaded"
        else:
            logger.info(output)
            logger.info(f"Already exists, skip {local_dir}.")
            return "already exists, skip"
    except Exception as e:
        return f"error during sync: {e}"


# %%
if __name__ == "__main__":
    # Must run before any logger call, or the root logger's WARNING default
    # silently drops every logger.info below and a long run leaves no record.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    s3_bucket_dest = "s3://aind-scratch-data/jason_lee/DRN_beh_physiology/"
    local_dir = CAPSULE_ROOT + "/scratch/"
    combine_only = False
    manuscript = False
    results = False
    if combine_only:
        s3_bucket_dest += "combined/"
        local_dir += "combined/"
    elif manuscript:
        s3_bucket_dest += "manuscript/"
        local_dir += "manuscript/"
    elif results:
        s3_bucket_dest += "results/"
        local_dir += "results/"
    
    out = sync_directory(local_dir, s3_bucket_dest, if_copy=False, if_dry_run=True, if_delete=False)
   # out = sync_directory(local_dir, s3_bucket_dest, if_copy=True, if_dry_run=False, if_delete=True)
    print(out)



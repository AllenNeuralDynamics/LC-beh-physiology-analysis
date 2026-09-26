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

        # Check output: if "upload:" appears, files were sent;
        # otherwise, assume that nothing needed uploading.
        if "upload:" in output:
            logger.info(f"Uploaded {local_dir} to {destination}!")
            return "successfully uploaded"
        else:
            logger.info(output)
            logger.info(f"Already exists, skip {local_dir}.")
            return "already exists, skip"
    except Exception as e:
        return f"error during sync: {e}"


# %%
if __name__ == "__main__":
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



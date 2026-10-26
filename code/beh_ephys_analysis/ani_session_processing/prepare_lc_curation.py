# %%
import os
import sys
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
from utils.capsule_migration import capsule_directories
import pandas as pd
import json
import shutil

from utils.beh_functions import session_dirs, get_unit_tbl


def prepare_lc_curation(session):
    """
    Set up the LC-range curation folder for a session in the current scratch directory.

    Creates ``opto_dir_curated`` (under ``derived_dir_curr``) for the session, copies
    any opto summary PDFs from the legacy scratch location into that folder, and writes
    a placeholder ``{session}_lc_range.json`` if one does not already exist.

    The LC range JSON has the form::

        {
            "lc_range": {
                "a": null,   // µm from first channel to top of LC  (= LC_range_top)
                "b": null    // µm from first channel to bottom of LC (= LC_range_bottom)
            }
        }

    Values are pre-filled from ``LC_range_top`` / ``LC_range_bottom`` columns in the
    unit table when available.  An existing JSON whose values are already non-null is
    assumed to be manually curated and is never overwritten.

    Semantics of ``a`` / ``b``:
        - Both null  → no LC detected on this probe for this session.
        - ``a`` set, ``b`` null → everything below ``a`` µm is in LC.
        - Both set   → LC occupies the range [a, b] µm from the first channel.

    The PDF copy step is non-destructive: existing files in the destination are never
    overwritten, and the source folder is left intact.

    Parameters
    ----------
    session : str
        Session identifier, e.g. ``'behavior_751181_2025-02-26_11-51-19'``.
    """
    caps = capsule_directories()

    # Destination: opto curated dir in current scratch (/root/capsule/scratch/...)
    sd_curr = session_dirs(session, scratch_dir=caps['derived_dir_curr'])
    dest_dir = sd_curr['opto_dir_curated']          # auto-created by session_dirs/makedirs
    os.makedirs(dest_dir, exist_ok=True)

    # Source: opto curated dir in legacy scratch (/root/capsule/data/scratch_data/...)
    sd_legacy = session_dirs(session, scratch_dir=caps['derived_dir'])
    src_dir = sd_legacy['opto_dir_curated']

    # --- copy opto summary PDFs ---
    if os.path.isdir(src_dir):
        for fname in os.listdir(src_dir):
            if fname.endswith('.pdf') and 'opto' in fname.lower() and 'figures' not in fname:
                src_path = os.path.join(src_dir, fname)
                dst_path = os.path.join(dest_dir, fname)
                if not os.path.exists(dst_path):
                    shutil.copy2(src_path, dst_path)
                    print(f'  copied  {fname}')
                else:
                    print(f'  exists  {fname}')
    else:
        print(f'  [warn] legacy opto dir not found: {src_dir}')

    # --- write LC range JSON, pre-filled from unit table when available ---
    json_path = os.path.join(dest_dir, f'{session}_lc_range.json')

    # Load existing values if the file already exists.
    existing_a, existing_b = None, None
    if os.path.exists(json_path):
        with open(json_path) as f:
            existing = json.load(f)
        existing_a = existing.get('lc_range', {}).get('a')
        existing_b = existing.get('lc_range', {}).get('b')
        if existing_a is not None or existing_b is not None:
            # At least one value was set manually — preserve the file as-is.
            print(f'  exists  {os.path.basename(json_path)}  (skipped, manually curated)')
            return

    # Try to pre-fill from the unit table (LC_range_top → a, LC_range_bottom → b).
    a, b = None, None
    try:
        unit_tbl = get_unit_tbl(session, 'curated', summary=True)
        if unit_tbl is not None:
            if 'LC_range_top' in unit_tbl.columns:
                val = unit_tbl['LC_range_top'].dropna()
                if len(val) > 0:
                    a = float(val.iloc[0])
            if 'LC_range_bottom' in unit_tbl.columns:
                val = unit_tbl['LC_range_bottom'].dropna()
                if len(val) > 0:
                    b = float(val.iloc[0])
    except Exception as e:
        print(f'  [warn] could not read unit table: {e}')

    payload = {'lc_range': {'a': a, 'b': b}}
    with open(json_path, 'w') as f:
        json.dump(payload, f, indent=2)
    src = 'unit table' if (a is not None or b is not None) else 'placeholder (no unit table)'
    print(f'  wrote   {os.path.basename(json_path)}  a={a}, b={b}  [{src}]')


if __name__ == '__main__':
    session_assets = pd.read_csv(CAPSULE_ROOT + '/code/data_management/session_assets.csv')
    session_list = [s for s in session_assets['session_id'] if isinstance(s, str)]

    for session in session_list:
        print(f'--- {session}')
        try:
            prepare_lc_curation(session)
        except Exception as e:
            print(f'  [error] {e}')

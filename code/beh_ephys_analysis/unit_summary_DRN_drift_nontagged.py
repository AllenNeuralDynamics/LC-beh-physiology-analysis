# %% Drift summary plots for the matched non-tagged units, for manual drift cuts.
#
# For each session in the selection from unit_summary_DRN_select_nontagged.py:
#   - per-unit drift PDFs (same figure as drift_analysis.plot_session_opto_drift) go to
#     ephys/opto/raw/drift_nontagged/, combined into {session}_drift_nontagged.pdf;
#     the tagged units' drift/ folder, drift PDFs and {session}_opto_drift_tbl.csv are untouched
#   - drift metrics go to {session}_opto_drift_tbl_nontagged.csv with ephys_cut = [nan, nan],
#     to be edited by hand like the tagged table. Rerunning keeps ephys_cut and drift_unit
#     for units already in that file.
#   - all PDFs are also copied flat into REVIEW_DIR as {session}_unit_{id}_drift.pdf
#
#     python unit_summary_DRN_drift_nontagged.py [session ...]   # default: all sessions

import os
import sys
import shutil

curr_path = os.path.dirname(os.path.abspath(__file__))
if curr_path not in sys.path:
    sys.path.insert(0, curr_path)

import matplotlib
matplotlib.use('Agg')
import pandas as pd

from drift_analysis import plot_session_opto_drift
from utils.beh_functions import session_dirs

SELECTION_CSV = '/root/capsule/scratch/combined/unit_summary_DRN/nontagged_matched_selection.csv'
REVIEW_DIR = '/root/capsule/scratch/combined/unit_summary_DRN/drift_nontagged_review'
DATA_TYPE = 'raw'
KEEP_COLS = ['ephys_cut', 'drift_unit']  # hand-edited; never overwritten


def run_session(session, rows):
    if len(rows) == 0:
        print(f'{session}: no selected units')
        return
    opto_dir = session_dirs(session)[f'opto_dir_{DATA_TYPE}']
    out_dir = os.path.join(opto_dir, 'drift_nontagged')
    tbl = plot_session_opto_drift(session, DATA_TYPE, plot=True, update_csv=False, update_cut=False,
                                  unit_ids=rows['unit_id'].tolist(), out_dir=out_dir)
    if tbl is None:  # no motion info
        print(f'{session}: skipped, no motion info')
        return
    tbl = tbl.merge(rows[['unit_id', 'matched_tagged_unit', 'distance_um', 'tier']], on='unit_id', how='left')

    tbl_file = os.path.join(opto_dir, f'{session}_opto_drift_tbl_nontagged.csv')
    if os.path.exists(tbl_file):
        old = pd.read_csv(tbl_file, index_col=0).set_index('unit_id')
        for col in KEEP_COLS:
            keep = tbl['unit_id'].isin(old.index)
            tbl.loc[keep, col] = tbl.loc[keep, 'unit_id'].map(old[col])
        # units dropped from the selection stay in the file with their edits
        tbl = pd.concat([tbl, old[~old.index.isin(tbl['unit_id'])].reset_index()], ignore_index=True)
    tbl.to_csv(tbl_file)

    os.makedirs(REVIEW_DIR, exist_ok=True)
    for unit_id in rows['unit_id']:
        src = os.path.join(out_dir, f'{unit_id}_drift.pdf')
        if os.path.exists(src):
            shutil.copy(src, os.path.join(REVIEW_DIR, f'{session}_unit_{unit_id}_drift.pdf'))
    print(f'{session}: {len(rows)} units -> {out_dir}')


if __name__ == '__main__':
    selection = pd.read_csv(SELECTION_CSV)
    sessions = sys.argv[1:] or sorted(selection['session'].unique())
    for session in sessions:
        try:
            run_session(session, selection[selection['session'] == session])
        except Exception as e:
            print(f'{session}: failed, {type(e).__name__}: {e}')

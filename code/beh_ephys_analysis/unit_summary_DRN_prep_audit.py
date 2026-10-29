# %% Which per-session inputs of unit_summary_DRN.py cover non-tagged units?
#
# Read-only: for every session in the master unit table, counts how many units each input
# file holds, split into tagged (opto_pass & default_qc) and non-tagged non-noise units.
# Writes one row per session to OUT_CSV and prints a per-input summary.
#
#     python unit_summary_DRN_prep_audit.py

import os
import sys
import json
import pickle

curr_path = os.path.dirname(os.path.abspath(__file__))
if curr_path not in sys.path:
    sys.path.insert(0, curr_path)

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from utils.beh_functions import session_dirs

MASTER_CSV = '/root/capsule/scratch/combined/master_unit_tables/master_all_units_opto_ccf_raw_260817.csv'
OUT_CSV = '/root/capsule/scratch/combined/unit_summary_DRN/prep_audit.csv'
DATA_TYPE = 'raw'
TARGET = 'soma'


def _pickle(path):
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def unit_sets(session):
    """Unit ids each unit_summary_DRN input holds for this session (None = file missing)."""
    d = session_dirs(session)
    opto_dir = d[f'opto_dir_{DATA_TYPE}']
    out = {}

    metrics = _pickle(os.path.join(opto_dir, f'{session}_opto_tagging_metrics.pkl'))
    out['opto_tagging_df'] = None if metrics is None else set(metrics['opto_tagging_df']['unit_id'])
    out['opto_tagging_df_metrics'] = (None if metrics is None or 'opto_tagging_df_metrics' not in metrics
                                      else set(metrics['opto_tagging_df_metrics']['unit_id']))
    summary = _pickle(os.path.join(opto_dir, f'{session}_{DATA_TYPE}_{TARGET}_opto_tagging_summary.pkl'))
    out['summary_pkl'] = None if summary is None else set(summary['unit_id'])

    ccf = os.path.join(d[f'ephys_processed_dir_{DATA_TYPE}'], 'ccf_unit_locations.csv')
    out['ccf_unit_locations'] = set(pd.read_csv(ccf)['unit_id']) if os.path.exists(ccf) else None
    drift = os.path.join(opto_dir, f'{session}_opto_drift_tbl.csv')
    out['opto_drift_tbl'] = set(pd.read_csv(drift)['unit_id']) if os.path.exists(drift) else None

    resp = _pickle(os.path.join(opto_dir, f'{session}_opto_responses_{TARGET}.pkl'))
    out['opto_responses'] = None if resp is None else set(resp['resp_p'].keys())
    wfs = _pickle(os.path.join(opto_dir, f'{session}_waveforms_{TARGET}.pkl'))
    out['waveforms_soma'] = None if wfs is None else set(wfs.keys())
    owf = _pickle(os.path.join(opto_dir, f'{session}_opto_waveform_metrics.pkl'))
    out['opto_waveform_metrics'] = (None if owf is None
                                    else set(pd.to_numeric(owf['unit_id'], errors='coerce').dropna().astype(int)))
    cc = _pickle(os.path.join(opto_dir, f'{session}_correlogram_laser_short.pkl'))
    out['correlogram_laser'] = None if cc is None else set(np.asarray(cc['unit_ids_laser']).tolist())
    out['correlogram_nolaser'] = None if cc is None else set(np.asarray(cc['unit_ids_nolaser']).tolist())

    # whole-session assets: present or not
    out['_has_behavior_nwb'] = os.path.exists(os.path.join(d['beh_fig_dir'], f'{session}.nwb'))
    out['_has_postprocessed'] = (d[f'postprocessed_dir_{DATA_TYPE}'] is not None
                                 and os.path.exists(d[f'postprocessed_dir_{DATA_TYPE}']))
    return out


def audit_session(session, units):
    tagged = set(units.loc[units['tagged'], 'unit_id'])
    untagged = set(units.loc[~units['tagged'] & units['is_nonartifact_unit'], 'unit_id'])
    row = {'session': session, 'n_tagged': len(tagged), 'n_untagged_nonnoise': len(untagged)}
    try:
        sets = unit_sets(session)
    except Exception as e:
        row['error'] = f'{type(e).__name__}: {e}'
        return row
    for name, ids in sets.items():
        if name.startswith('_'):
            row[name[1:]] = ids
            continue
        if ids is None:
            row[f'{name}_tagged'] = row[f'{name}_untagged'] = np.nan
        else:
            ids = {int(i) for i in ids}
            row[f'{name}_tagged'] = len(tagged & ids)
            row[f'{name}_untagged'] = len(untagged & ids)
    return row


if __name__ == '__main__':
    master = pd.read_csv(MASTER_CSV, usecols=['session', 'unit_id', 'opto_pass', 'default_qc',
                                              'is_nonartifact_unit'])
    master['tagged'] = master['opto_pass'].astype(bool) & master['default_qc'].astype(bool)
    rows = Parallel(n_jobs=8)(delayed(audit_session)(s, u) for s, u in master.groupby('session'))
    audit = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    audit.to_csv(OUT_CSV, index=False)

    print(f"{len(audit)} sessions; tagged units {audit['n_tagged'].sum()}, "
          f"non-tagged non-noise units {audit['n_untagged_nonnoise'].sum()}")
    if 'error' in audit:
        print('errors:', audit['error'].notna().sum())
    for flag in ['has_behavior_nwb', 'has_postprocessed']:
        print(f'{flag}: {int(audit[flag].sum())} / {len(audit)} sessions')
    names = [c[:-len('_tagged')] for c in audit.columns if c.endswith('_tagged') and c != 'n_tagged']
    print(f"\n{'input':28s} {'sessions w/ file':>16s} {'tagged covered':>16s} {'non-tagged covered':>20s}")
    for name in names:
        has = audit[f'{name}_tagged'].notna()
        print(f"{name:28s} {int(has.sum()):>16d} "
              f"{int(audit[f'{name}_tagged'].sum()):>7d}/{int(audit['n_tagged'].sum()):<8d} "
              f"{int(audit[f'{name}_untagged'].sum()):>9d}/{int(audit['n_untagged_nonnoise'].sum()):<10d}")
    print(f'\nper-session table: {OUT_CSV}')

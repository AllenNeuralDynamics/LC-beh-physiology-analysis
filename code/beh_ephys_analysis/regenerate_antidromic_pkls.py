"""Regenerate the per-session antidromic result pkls with the current antidromic_funcs code.

The across-session combining step (session_combine/figure_preparation/antidromic_generation.py)
only concatenates these cached pkls, so any change to antidromic_funcs (min_spikes gate, FWHM
floor, soma_site handling, tier attachment) only reaches the combined frame after this rerun.

Writes {session}_antidromic_results.pkl into each session's ephys/opto/{data_type}/ and a
status table next to this script's log so partial failures are visible instead of silent.
"""

import os
import sys
import time
import traceback
import warnings

import pandas as pd
from joblib import Parallel, delayed

sys.path.append('/root/capsule/code/beh_ephys_analysis')
from utils.beh_functions import session_dirs
from antidromic_funcs import analyze_antidromic_responses

DATA_TYPE = 'raw'
SOMA_SITE = 'surface_DRN'
N_JOBS = 6
SESSION_ASSETS = '/root/capsule/code/data_management/session_assets.csv'
STATUS_CSV = '/root/capsule/scratch/results/manuscript/prep/antidromic_analysis/regeneration_status.csv'


def process(session):
    """Run the antidromic analysis for one session, never raising."""
    warnings.simplefilter('ignore')
    t0 = time.time()
    try:
        opto_dir = session_dirs(session).get(f'opto_dir_{DATA_TYPE}')
        if opto_dir is None or not os.path.exists(os.path.join(opto_dir, f'{session}_opto_session.csv')):
            return dict(session=session, status='no_opto_csv', n_units=0, sites='', seconds=time.time() - t0)

        out = analyze_antidromic_responses(
            session, data_type=DATA_TYPE, tier_cat=True, soma_site=SOMA_SITE
        )
        if out is None or len(out) == 0:
            # None: no unit table, or no unit x site combination survived the drift cut.
            return dict(session=session, status='no_results', n_units=0, sites='', seconds=time.time() - t0)

        sites = sorted({c[1] for c in out.columns if isinstance(c, tuple) and c[1]})
        return dict(
            session=session,
            status='ok',
            n_units=len(out),
            sites=';'.join(sites),
            seconds=time.time() - t0,
        )
    except Exception:
        print(f'FAILED {session}\n{traceback.format_exc()}', flush=True)
        return dict(session=session, status='error', n_units=0, sites='', seconds=time.time() - t0)


def main():
    sessions = pd.read_csv(SESSION_ASSETS)['session_id'].dropna().unique().tolist()
    print(f'Regenerating antidromic pkls ({DATA_TYPE}) for {len(sessions)} sessions', flush=True)

    results = Parallel(n_jobs=N_JOBS, verbose=10)(delayed(process)(s) for s in sessions)

    status = pd.DataFrame(results).sort_values(['status', 'session'])
    os.makedirs(os.path.dirname(STATUS_CSV), exist_ok=True)
    status.to_csv(STATUS_CSV, index=False)
    print('\n=== status counts ===', flush=True)
    print(status['status'].value_counts().to_string(), flush=True)
    print(f'\nunits written: {int(status["n_units"].sum())}', flush=True)
    print(f'total minutes: {status["seconds"].sum() / 60:.1f}', flush=True)
    print(f'status table: {STATUS_CSV}', flush=True)
    for row in status.query("status != 'ok'").itertuples():
        print(f'  {row.status}: {row.session}', flush=True)


if __name__ == '__main__':
    main()

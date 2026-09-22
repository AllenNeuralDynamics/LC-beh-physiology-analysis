# %%
"""
Backfill the laser-free ISI-violation metric for every session, without rerunning opto
tagging.

The pipeline's isi_violations_ratio is computed over the whole recording, opto blocks
included, so laser artifacts and laser-driven bursts both count as refractory-period
violations -- and the units the laser drives are the ones that suffer for it. This script
recomputes the same metric on the laser-free portion of each session (utils/isi_qc) and
writes it as a sidecar next to the opto tagging outputs, where get_unit_tbl picks it up.

Nothing existing is overwritten: the opto tagging pickles, and the default_qc in them,
are left alone. The sidecar carries default_qc_nonopto beside them, so an analysis opts
in by using that column. opto_tagging.opto_plotting_session(qc_isi_source='nonopto')
makes it the primary flag on the next full rerun.

Usage:
    python add_isi_nonopto.py                       # all sessions, block exclusion
    python add_isi_nonopto.py --mode train          # drop trains only, keep the gaps
    python add_isi_nonopto.py --sessions behavior_814515_2025-10-24_13-22-07
"""

import argparse
import os
import sys
sys.path.append('/root/capsule/code/beh_ephys_analysis')

import numpy as np
import pandas as pd

from utils.beh_functions import get_unit_tbl
from utils.isi_qc import (DEFAULT_QC_ISI_MAX, default_qc, non_opto_isi_paths,
                          non_opto_isi_tbl, save_non_opto_isi)

SESSION_CSV = '/root/capsule/code/data_management/session_assets.csv'
SESSION_EXCLUDE = '/root/capsule/code/data_management/sessions_to_exclude.txt'
SUMMARY_DIR = '/root/capsule/scratch/combined'


def session_list():
    """Sessions from the asset table, minus the exclusion list."""
    tbl = pd.read_csv(SESSION_CSV)
    sessions = [s for s in tbl['session_id'] if isinstance(s, str)]
    if os.path.exists(SESSION_EXCLUDE):
        with open(SESSION_EXCLUDE) as f:
            exclude = {line.strip() for line in f}
        sessions = [s for s in sessions if s not in exclude]
    return sessions


def process_session(session, data_type='raw', mode='block', isi_max=DEFAULT_QC_ISI_MAX,
                    overwrite=True):
    """
    Compute and save the sidecar for one session, and return its rows for the summary.

    The QC flags need decoder_label, which lives in the opto tagging table rather than in
    the spike times, so they are added here rather than in non_opto_isi_tbl.
    """
    csv_file, _ = non_opto_isi_paths(session, data_type)
    if os.path.exists(csv_file) and not overwrite:
        print(f'{session}: sidecar exists, skipping.')
        return None

    isi_tbl = non_opto_isi_tbl(session, data_type=data_type, mode=mode)
    if isi_tbl is None:
        return None

    unit_tbl = get_unit_tbl(session, data_type, summary=False)
    if unit_tbl is None:
        print(f'{session}: no opto tagging table, saving ISI columns without QC flags.')
        save_non_opto_isi(session, data_type=data_type, tbl=isi_tbl)
        return None

    qc_cols = ['unit_id', 'decoder_label', 'isi_violations_ratio', 'opto_pass']
    qc_cols = [col for col in qc_cols if col in unit_tbl.columns]
    merged = isi_tbl.merge(unit_tbl[qc_cols], on='unit_id', how='left')
    # units the sorting has but the opto tagging table does not cannot be QC'd
    merged['decoder_label'] = merged.get('decoder_label', pd.Series(index=merged.index)).fillna('unknown')

    merged['default_qc_all_isi'] = default_qc(merged, isi_col='isi_violations_ratio',
                                              isi_max=isi_max)
    merged['default_qc_nonopto'] = default_qc(merged, isi_col='isi_violations_ratio_nonopto',
                                              fallback_isi_col='isi_violations_ratio',
                                              isi_max=isi_max)

    sidecar_cols = [col for col in merged.columns
                    if col in isi_tbl.columns or col.startswith('default_qc')]
    sidecar = merged[sidecar_cols].copy()
    sidecar.attrs['isi_nonopto_params'] = dict(isi_tbl.attrs.get('isi_nonopto_params', {}),
                                               isi_max=isi_max)
    save_non_opto_isi(session, data_type=data_type, tbl=sidecar)

    merged['session'] = session
    if 'default_qc' in unit_tbl.columns:
        merged = merged.merge(unit_tbl[['unit_id', 'default_qc']], on='unit_id', how='left')
    return merged


def summarize(units, isi_max=DEFAULT_QC_ISI_MAX):
    """Print what switching default_qc onto the laser-free ISI would do."""
    n_sessions = units['session'].nunique()
    print(f'\n{len(units)} units over {n_sessions} sessions; '
          f'laser-free fraction of the session window: '
          f'{units["frac_nonopto"].min():.2f}-{units["frac_nonopto"].max():.2f} '
          f'(median {units["frac_nonopto"].median():.2f})')

    ratio_all = units['isi_violations_ratio']
    ratio_win = units['isi_violations_ratio_window']
    ratio_non = units['isi_violations_ratio_nonopto']
    ratio_opto = units['isi_violations_ratio_opto']
    print('\nmedian ISI violations ratio')
    print(f'  pipeline (whole recording)  {ratio_all.median():.3f}')
    print(f'  session window, recomputed  {ratio_win.median():.3f}')
    print(f'  laser-free                  {ratio_non.median():.3f}')
    print(f'  laser windows only          {ratio_opto.median():.3f}')

    for label, subset in (('all units', units),
                          ('opto_pass units', units[units.get('opto_pass', False) == True]),
                          ('non-artifact units', units[~units['decoder_label'].isin(['noise', 'artifact'])])):
        if len(subset) == 0:
            continue
        gained = int((subset['default_qc_nonopto'] & ~subset['default_qc_all_isi']).sum())
        lost = int((~subset['default_qc_nonopto'] & subset['default_qc_all_isi']).sum())
        print(f'\n{label}: n={len(subset)}, default_qc at isi < {isi_max}')
        print(f'  pass on pipeline ISI   {int(subset["default_qc_all_isi"].sum())}')
        print(f'  pass on laser-free ISI {int(subset["default_qc_nonopto"].sum())} '
              f'(+{gained} gained, -{lost} lost)')
        drop = (subset['isi_violations_ratio_window'] - subset['isi_violations_ratio_nonopto'])
        print(f'  median drop window -> laser-free: {drop.median():.3f}')


# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-type', default='raw', choices=['raw', 'curated'])
    parser.add_argument('--mode', default='block', choices=['block', 'train'],
                        help='drop whole opto blocks, or only the trains themselves')
    parser.add_argument('--sessions', nargs='*', default=None,
                        help='sessions to process; defaults to the whole asset table')
    parser.add_argument('--isi-max', type=float, default=DEFAULT_QC_ISI_MAX,
                        help='default_qc bound on the ISI violations ratio')
    parser.add_argument('--no-overwrite', action='store_true',
                        help='skip sessions that already have a sidecar')
    parser.add_argument('--summary-csv', default=os.path.join(SUMMARY_DIR, 'isi_nonopto_units.csv'),
                        help='where to write the per-unit summary table')
    args = parser.parse_args()

    sessions = args.sessions if args.sessions else session_list()
    all_units = []
    for session in sessions:
        try:
            units = process_session(session, data_type=args.data_type, mode=args.mode,
                                    isi_max=args.isi_max, overwrite=not args.no_overwrite)
        except Exception as error:
            print(f'{session}: failed ({type(error).__name__}: {error})')
            continue
        if units is not None:
            all_units.append(units)
            print(f'{session}: {len(units)} units, laser-free fraction '
                  f'{units["frac_nonopto"].iloc[0]:.2f}')

    if not all_units:
        print('No sessions processed.')
        sys.exit(0)

    units = pd.concat(all_units, ignore_index=True)
    summarize(units, isi_max=args.isi_max)

    if args.summary_csv:
        os.makedirs(os.path.dirname(args.summary_csv), exist_ok=True)
        keep = [col for col in units.columns if col != 'spike_times']
        units[keep].to_csv(args.summary_csv, index=False)
        print(f'\nPer-unit summary written to {args.summary_csv}')

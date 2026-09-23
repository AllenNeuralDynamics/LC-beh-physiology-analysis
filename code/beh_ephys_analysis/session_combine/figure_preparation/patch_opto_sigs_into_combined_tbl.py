# %%
"""Patch the opto-significance columns onto the existing combined unit table.

``make_combined_unit_tbl_DRN.py`` hardcodes ``session_opto_sig = None``, so every
table it has produced carries an all-NaN ``sig_counts`` and an all-NaN
``all_sig_counts``. The script also cannot simply be re-run: it calls
``get_unit_tbl(session, data_type, ccf_suffix=...)`` and ``get_unit_tbl`` takes no
``ccf_suffix`` argument, so ``safe_process`` swallows a TypeError for every
session and the rebuild yields nothing. This patches the columns onto the
pickle in place of a rebuild.

The ``sig_counts`` / ``all_sig_counts`` fill replicates the DRN script's own
alignment (its lines 184-198) so the values are what a working rebuild would
have produced. The ``_exc`` / ``_inh`` / ``_50ms`` columns are new.
"""
import sys
import os
sys.path.append('/root/capsule/code/beh_ephys_analysis')
import pickle
import shutil
import numpy as np
import pandas as pd
from utils.beh_functions import session_dirs
from utils.opto_utils import opto_metrics, load_opto_sig
from utils.capsule_migration import capsule_directories

DATA_TYPE = 'raw'
# count columns in the opto_sigs pkl to lift onto the unit table, mapped to the
# unit-table column they populate
SIG_COLS = {
    'p_sig_count': 'sig_counts',
    'p_sig_count_exc': 'sig_counts_exc',
    'p_sig_count_inh': 'sig_counts_inh',
    'p_sig_count_50ms': 'sig_counts_50ms',
}


def unit_sig_values(unit_opto, unit_opto_sig, is_zs, sig_col):
    """Return (per-condition array aligned to unit_opto, max over all conditions).

    Mirrors make_combined_unit_tbl_DRN.py: match each opto_metrics condition to
    an opto_sigs row on power/site (and pre_post when the session has both), and
    take the session-wide max as the scalar summary.
    """
    per_cond = np.full(len(unit_opto), np.nan)
    if unit_opto_sig is None or len(unit_opto_sig) == 0 or sig_col not in unit_opto_sig.columns:
        return per_cond, np.nan
    if is_zs:
        per_cond[:] = unit_opto_sig[sig_col].values[0]
    else:
        for pos, (_, row) in enumerate(unit_opto.iterrows()):
            filt = ((unit_opto_sig['power'] == row['powers'])
                    & (unit_opto_sig['site'] == row['sites']))
            if len(unit_opto_sig['pre_post'].unique()) > 1:
                filt &= (unit_opto_sig['pre_post'] == row['stim_times'])
            curr = unit_opto_sig[filt]
            if len(curr) >= 1:
                per_cond[pos] = curr[sig_col].values[0]
    return per_cond, unit_opto_sig[sig_col].max()


def main(tbl_path=None, backup=True):
    if tbl_path is None:
        tbl_path = os.path.join(capsule_directories()['manuscript_fig_prep_dir'],
                                'combined_unit_tbl', 'combined_unit_tbl.pkl')
    with open(tbl_path, 'rb') as f:
        tbl = pickle.load(f)
    print(f'loaded {tbl_path}: {tbl.shape}, {tbl["session"].nunique()} sessions')

    # allocate the new columns
    out_scalar = {v: np.full(len(tbl), np.nan) for v in SIG_COLS.values()}
    out_arrays = {f'all_{v}': [None] * len(tbl) for v in SIG_COLS.values()}
    # The drift cut can remove every train of a condition -- for some units the
    # whole post-laser block, which carries 5 of the 7 power conditions. Those
    # units' sig_counts is then a max over the pre block only and is NOT
    # comparable to an uncut unit's, so record what was actually testable.
    diag = {
        'sig_counts_post': np.full(len(tbl), np.nan),
        'n_cond_testable': np.full(len(tbl), np.nan),
        'n_cond_total': np.full(len(tbl), np.nan),
        'post_block_testable': np.full(len(tbl), np.nan),
    }

    n_filled = 0
    no_sig_pkl, no_metrics, unit_missing = [], [], 0
    for session, idx in tbl.groupby('session').groups.items():
        # load_opto_sig is a loader object, not a DataFrame; .opto_sigs is the frame
        sig_loader = load_opto_sig(session, data_type=DATA_TYPE)
        if sig_loader.opto_sigs is None or len(sig_loader.opto_sigs) == 0:
            no_sig_pkl.append(session)
            continue
        try:
            om = opto_metrics(session, data_type=DATA_TYPE)
        except Exception as e:
            no_metrics.append((session, repr(e)[:80]))
            continue
        is_zs = str(session_dirs(session)['aniID']).startswith('ZS')

        for i in idx:
            unit_id = tbl.at[i, 'unit']
            unit_opto = om.load_unit(unit_id)
            unit_sig = sig_loader.load_unit(unit_id)
            if unit_opto is None or len(unit_opto) == 0 or unit_sig is None or len(unit_sig) == 0:
                unit_missing += 1
                continue
            for sig_col, out_col in SIG_COLS.items():
                per_cond, scalar = unit_sig_values(unit_opto, unit_sig, is_zs, sig_col)
                out_scalar[out_col][i] = scalar
                out_arrays[f'all_{out_col}'][i] = per_cond

            testable = unit_sig['n_trains'] > 0 if 'n_trains' in unit_sig.columns \
                else unit_sig['p_sig_count'].notna()
            is_post = unit_sig['pre_post'] == 'post'
            diag['n_cond_total'][i] = len(unit_sig)
            diag['n_cond_testable'][i] = int(testable.sum())
            diag['post_block_testable'][i] = int((testable & is_post).sum() > 0)
            if (testable & is_post).any():
                diag['sig_counts_post'][i] = unit_sig.loc[testable & is_post, 'p_sig_count'].max()
            n_filled += 1

    # keep the original array lengths where a unit could not be filled
    for out_col in SIG_COLS.values():
        src = f'all_{out_col}'
        old = tbl['all_sig_counts'].values
        out_arrays[src] = [
            new if new is not None else np.full(len(np.atleast_1d(o)), np.nan)
            for new, o in zip(out_arrays[src], old)
        ]

    if backup:
        bak = tbl_path.replace('.pkl', '_pre_opto_sigs_backup.pkl')
        if not os.path.exists(bak):
            shutil.copy2(tbl_path, bak)
            print(f'backed up original -> {bak}')

    for out_col in SIG_COLS.values():
        tbl[out_col] = out_scalar[out_col]
        tbl[f'all_{out_col}'] = out_arrays[f'all_{out_col}']
    for k, v in diag.items():
        tbl[k] = v
    tbl['opto_sigs_post_win'] = 0.025

    with open(tbl_path, 'wb') as f:
        pickle.dump(tbl, f)

    print(f'\nfilled {n_filled} / {len(tbl)} unit rows')
    if no_sig_pkl:
        print(f'sessions with no opto_sigs pkl ({len(no_sig_pkl)}): {no_sig_pkl}')
    if no_metrics:
        print(f'sessions whose opto_metrics failed ({len(no_metrics)}): {no_metrics}')
    print(f'unit rows with no matching opto_sigs/opto_metrics entry: {unit_missing}')
    print('\n--- resulting coverage ---')
    for out_col in SIG_COLS.values():
        print(f'  {out_col:18s} non-NaN: {tbl[out_col].notna().sum()} / {len(tbl)}'
              f'   mean(non-NaN)={tbl[out_col].mean():.3f}')
    nfilled = tbl['n_cond_testable'].notna()
    partial = nfilled & (tbl['n_cond_testable'] < tbl['n_cond_total'])
    no_post = nfilled & (tbl['post_block_testable'] == 0)
    print('\n--- drift-cut coverage warnings ---')
    print(f'  units with >=1 untestable condition : {int(partial.sum())}')
    print(f'  units with NO testable post block   : {int(no_post.sum())}'
          f'  <- sig_counts for these is a pre-block-only max')
    print(f'\nwrote {tbl_path}')
    return tbl


if __name__ == '__main__':
    main()

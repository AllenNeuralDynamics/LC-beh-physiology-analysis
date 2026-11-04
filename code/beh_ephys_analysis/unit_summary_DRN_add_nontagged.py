# %% Add another round of matched non-tagged controls, keeping everything already selected.
#
# Same matching as unit_summary_DRN_select_nontagged.py (nearest unused unit on the same
# insertion, tier 1 then tier 2) with the unit criteria of unit_summary_DRN_swap_low_fr.py
# (firing rate > MIN_FR, ISI <= MAX_ISI). No unit that has ever had a drift-table row is
# reused, so units excluded in review stay out.
# Controls are filled by level so they spread over tagged units: first controls for tagged
# units with none, then second controls for those with one, then third, ..., closest pairs
# first within a level, until N_ADD units in total.
#
# For the added units: drift PDFs into drift_nontagged/ (also copied to the review folder and
# REVIEW_SUBDIR), rows appended to {session}_opto_drift_tbl_nontagged.csv with
# ephys_cut = [nan, nan] and needs_review = True, and rows appended to the selection with
# status = added. Selection, tables and tracking sheet are backed up first. Then build the
# next tracking sheet with unit_summary_DRN_drift_cut_sheet.py make.
#
#     python unit_summary_DRN_add_nontagged.py --dry-run
#     python unit_summary_DRN_add_nontagged.py [--n 60]

import os
import sys
import glob
import shutil
import argparse
import datetime

curr_path = os.path.dirname(os.path.abspath(__file__))
if curr_path not in sys.path:
    sys.path.insert(0, curr_path)

import matplotlib
matplotlib.use('Agg')
import pandas as pd

from unit_summary_DRN_select_nontagged import FLAGS_CSV, TIERS, candidate_pool, match_session
from unit_summary_DRN_swap_low_fr import (MASTER_CSV, MIN_FR, MAX_ISI, OUT_ROOT, ever_listed, tbl_path)
from unit_summary_DRN_drift_nontagged import SELECTION_CSV, REVIEW_DIR, DATA_TYPE
from utils.beh_functions import session_dirs

N_ADD = 60
ROUND = f'added_{datetime.date.today():%y%m%d}'
REVIEW_SUBDIR = f'needs_review_{datetime.date.today():%y%m%d}'
ADD_LOG = os.path.join(OUT_ROOT, 'nontagged_add_log.csv')


def match(need, flags, eligible, used, step):
    rows, matched = [], set()
    for tier, margin, isi_max in TIERS:
        pool = candidate_pool(flags, margin, isi_max)
        pool = pool[pool.index.isin(eligible.index)]
        for session, tagged in need[~need.index.isin(matched)].groupby('session'):
            for t_idx, c_idx, dist in match_session(tagged, pool[pool['session'] == session], used):
                used.add(c_idx)
                matched.add(t_idx)
                rows.append({'session': session, 'unit_id': int(flags.at[c_idx, 'unit_id']),
                             'matched_tagged_unit': int(flags.at[t_idx, 'unit_id']),
                             'distance_um': round(1000 * dist, 1), 'tier': tier,
                             'decoder_label': flags.at[c_idx, 'decoder_label'],
                             'isi_violations_ratio': flags.at[c_idx, 'isi_violations_ratio'],
                             'firing_rate': flags.at[c_idx, 'firing_rate'], 'step': step,
                             '_c_idx': c_idx})
    return pd.DataFrame(rows)


def plan(n_add):
    flags = pd.read_csv(FLAGS_CSV).merge(pd.read_csv(MASTER_CSV, usecols=['session', 'unit_id', 'firing_rate']),
                                         on=['session', 'unit_id'], how='left')
    sel = pd.read_csv(SELECTION_CSV)
    listed = ever_listed() | set(zip(sel['session'], sel['unit_id']))
    used = set(flags.index[[k in listed for k in zip(flags['session'], flags['unit_id'])]])
    eligible = flags[(flags['firing_rate'] > MIN_FR) & (flags['isi_violations_ratio'] <= MAX_ISI)]
    tagged = flags[flags['tagged']]
    counts = sel.groupby(['session', 'matched_tagged_unit']).size()
    n_controls = pd.Series([int(counts.get(k, 0)) for k in zip(tagged['session'], tagged['unit_id'])],
                           index=tagged.index)

    picks, level = [], 0
    while sum(map(len, picks)) < n_add and level <= n_controls.max() + 1:
        need = tagged[n_controls == level]
        m = match(need, flags, eligible, set(used), f'control #{level + 1}')
        if len(m):  # closest pairs first; tier 1 before tier 2 at equal distance
            m = m.sort_values(['distance_um', 'tier']).head(n_add - sum(map(len, picks)))
            used |= set(m['_c_idx'])
            picks.append(m)
        # tagged units that got a control here move up a level and can get the next one
        for k in zip(m['session'], m['matched_tagged_unit']) if len(m) else []:
            n_controls[(tagged['session'] == k[0]) & (tagged['unit_id'] == k[1])] += 1
        level += 1
    return pd.concat(picks, ignore_index=True).drop(columns='_c_idx')


def apply_session(session, rows, bdir):
    from drift_analysis import plot_session_opto_drift
    opto_dir = session_dirs(session)[f'opto_dir_{DATA_TYPE}']
    drift_dir = os.path.join(opto_dir, 'drift_nontagged')
    new = plot_session_opto_drift(session, DATA_TYPE, plot=True, update_csv=False, update_cut=False,
                                  unit_ids=rows['unit_id'].tolist(), out_dir=drift_dir)
    new = new.merge(rows[['unit_id', 'matched_tagged_unit', 'distance_um', 'tier']], on='unit_id', how='left')
    new['needs_review'] = True
    new['replaces_unit'] = pd.NA
    f = tbl_path(session)
    if os.path.exists(f):
        shutil.copy(f, bdir)
        new = pd.concat([pd.read_csv(f, index_col=0), new], ignore_index=True)
    new.to_csv(f)
    os.makedirs(os.path.join(REVIEW_DIR, REVIEW_SUBDIR), exist_ok=True)
    for unit_id in rows['unit_id']:
        src = os.path.join(drift_dir, f'{unit_id}_drift.pdf')
        name = f'{session}_unit_{unit_id}_drift.pdf'
        shutil.copy(src, os.path.join(REVIEW_DIR, name))
        shutil.copy(src, os.path.join(REVIEW_DIR, REVIEW_SUBDIR, name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=N_ADD)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    add = plan(args.n)
    pd.set_option('display.width', 200)
    print(add.drop(columns=['decoder_label']).to_string(index=False))
    print(f'\nadd {len(add)} units in {add["session"].nunique()} sessions: '
          f'{add["step"].value_counts().to_dict()}, {add["tier"].value_counts().to_dict()}; '
          f'distance median {add["distance_um"].median():.0f} um, {int((add["distance_um"] > 250).sum())} over 250 um')
    if args.dry_run:
        sys.exit()

    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    bdir = os.path.join(OUT_ROOT, f'backup_before_add_{stamp}')
    os.makedirs(bdir)
    shutil.copy(SELECTION_CSV, bdir)
    sheets = sorted(glob.glob(os.path.join(REVIEW_DIR, 'drift_cut_nontagged_units_*.csv')))
    if sheets:
        shutil.copy(sheets[-1], bdir)
    print(f'backup: {bdir}')

    done = []
    for session, rows in add.groupby('session'):
        try:
            apply_session(session, rows, bdir)
            done.append(rows)
            print(f'{session}: {len(rows)} units added')
        except Exception as e:
            print(f'{session}: failed, {type(e).__name__}: {e}')
    done = pd.concat(done, ignore_index=True) if done else add.iloc[0:0]

    sel = pd.read_csv(SELECTION_CSV)
    new_sel = done.drop(columns=['firing_rate', 'step']).assign(status='added', replaces_unit=pd.NA,
                                                                needs_review=True)
    pd.concat([sel, new_sel], ignore_index=True).sort_values(['session', 'unit_id']).to_csv(SELECTION_CSV, index=False)
    done.assign(round=ROUND).to_csv(ADD_LOG, mode='a', header=not os.path.exists(ADD_LOG), index=False)
    print(f'selection {len(sel)} -> {len(sel) + len(done)}; log {ADD_LOG}')

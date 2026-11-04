# %% Remove matched non-tagged units that fail a unit criterion, keeping everything else.
#
# Criteria: firing rate above MIN_FR and ISI violations ratio at most MAX_ISI (the tagged units'
# bound). Units that pass keep their match and their hand-placed drift cuts. For each unit that
# fails, its tagged partner is re-matched to the nearest unused candidate on the same insertion
# with the tier rules of unit_summary_DRN_select_nontagged.py plus both criteria (unless
# --no-replace). Only the affected tagged units are re-matched, so no other pair changes.
#
# Writes (after copying the current selection and drift tables to BACKUP_DIR):
#   selection CSV       status = kept / replacement, replaces_unit, needs_review
#   per-session table   {session}_opto_drift_tbl_nontagged.csv: removed units' rows moved to
#                       {session}_opto_drift_tbl_nontagged_removed.csv; replacement rows added
#                       with ephys_cut = [nan, nan] and needs_review = True
#   drift PDFs          new units plotted into drift_nontagged/; removed units' PDFs moved to
#                       drift_nontagged/removed_fr/ (also in the review folder)
#   SWAP_LOG            one row per removed unit: rate, its cut, and what replaced it
#
#   tracking sheet      removed units' rows dropped from the latest drift_cut_nontagged_units_*.csv
#
#     python unit_summary_DRN_swap_low_fr.py --dry-run     # print the plan only
#     python unit_summary_DRN_swap_low_fr.py [--no-replace]

import os
import sys
import shutil
import argparse
import datetime
import glob

curr_path = os.path.dirname(os.path.abspath(__file__))
if curr_path not in sys.path:
    sys.path.insert(0, curr_path)

import matplotlib
matplotlib.use('Agg')
import pandas as pd

from unit_summary_DRN_select_nontagged import FLAGS_CSV, TIERS, candidate_pool, match_session
from unit_summary_DRN_drift_nontagged import SELECTION_CSV, REVIEW_DIR, DATA_TYPE
from utils.beh_functions import session_dirs

MIN_FR = 0.5  # Hz; units must fire above this
MAX_ISI = 0.1  # ISI violations ratio; same bound as the tagged units (basic_ephys_all_DRN)
MASTER_CSV = '/root/capsule/scratch/combined/master_unit_tables/master_all_units_opto_ccf_raw_260817.csv'
OUT_ROOT = '/root/capsule/scratch/combined/unit_summary_DRN'
# nontagged_fr_swap_log.csv holds the first (firing-rate only) swap
SWAP_LOG = os.path.join(OUT_ROOT, 'nontagged_swap_log.csv')
REMOVED_SUBDIRS = {'firing_rate': 'removed_fr', 'isi': 'removed_isi'}
NEEDS_REVIEW_SUBDIR = 'needs_review'


def tbl_path(session, suffix=''):
    return os.path.join(session_dirs(session)[f'opto_dir_{DATA_TYPE}'],
                        f'{session}_opto_drift_tbl_nontagged{suffix}.csv')


def ever_listed():
    """(session, unit_id) of every unit with a row in any non-tagged drift table, current or
    removed: kept units, units excluded by hand from the selection, and earlier swaps."""
    keys = set()
    for f in glob.glob('/root/capsule/scratch/*/behavior_*/ephys/opto/raw/*_opto_drift_tbl_nontagged*.csv'):
        session = os.path.basename(f).split('_opto_drift_tbl_nontagged')[0]
        keys |= {(session, int(u)) for u in pd.read_csv(f, index_col=0)['unit_id']}
    return keys


def plan_swaps(sel, flags, ineligible=(), replace=True):
    """Removed rows and their replacements, re-matching only the partners of removed units.
    ineligible: (session, unit_id) never used as a replacement."""
    flags = flags.copy()
    flags['firing_rate'] = pd.to_numeric(flags['firing_rate'], errors='coerce')
    flags['isi_violations_ratio'] = pd.to_numeric(flags['isi_violations_ratio'], errors='coerce')
    vals = sel[['session', 'unit_id']].merge(flags[['session', 'unit_id', 'firing_rate', 'isi_violations_ratio']],
                                             on=['session', 'unit_id'], how='left')
    low_fr = ~(vals['firing_rate'] > MIN_FR).values
    high_isi = ~(vals['isi_violations_ratio'] <= MAX_ISI).values
    removed = sel[low_fr | high_isi].copy()
    removed['firing_rate'] = vals['firing_rate'].values[low_fr | high_isi]
    removed['isi_violations_ratio'] = vals['isi_violations_ratio'].values[low_fr | high_isi]
    removed['removed_reason'] = ['; '.join(r for r, bad in [(f'firing_rate <= {MIN_FR} Hz', f),
                                                             (f'isi_violations_ratio > {MAX_ISI}', i)] if bad)
                                 for f, i in zip(low_fr[low_fr | high_isi], high_isi[low_fr | high_isi])]
    if not replace:
        return removed, pd.DataFrame()

    # never reuse a unit that is kept or was just removed
    used = set(flags.index[flags.set_index(['session', 'unit_id']).index.isin(
        list(zip(sel['session'], sel['unit_id'])) + list(ineligible))])
    tagged_all = flags[flags['tagged']]
    need = tagged_all[tagged_all.set_index(['session', 'unit_id']).index.isin(
        list(zip(removed['session'], removed['matched_tagged_unit'])))]
    eligible = flags[(flags['firing_rate'] > MIN_FR) & (flags['isi_violations_ratio'] <= MAX_ISI)]

    replacements, matched = [], set()
    for tier, margin, isi_max in TIERS:
        pool = candidate_pool(flags, margin, isi_max)
        pool = pool[pool.index.isin(eligible.index)]
        for session, tagged in need[~need.index.isin(matched)].groupby('session'):
            for t_idx, c_idx, dist in match_session(tagged, pool[pool['session'] == session], used):
                used.add(c_idx)
                matched.add(t_idx)
                old = removed[(removed['session'] == session)
                              & (removed['matched_tagged_unit'] == flags.at[t_idx, 'unit_id'])].iloc[0]
                replacements.append({'session': session, 'unit_id': int(flags.at[c_idx, 'unit_id']),
                                     'matched_tagged_unit': int(flags.at[t_idx, 'unit_id']),
                                     'distance_um': round(1000 * dist, 1), 'tier': tier,
                                     'decoder_label': flags.at[c_idx, 'decoder_label'],
                                     'isi_violations_ratio': flags.at[c_idx, 'isi_violations_ratio'],
                                     'status': 'replacement', 'replaces_unit': int(old['unit_id']),
                                     'needs_review': True,
                                     'firing_rate': flags.at[c_idx, 'firing_rate']})
    return removed, pd.DataFrame(replacements)


def backup(sel_sessions):
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    bdir = os.path.join(OUT_ROOT, f'backup_before_fr_swap_{stamp}')
    os.makedirs(bdir)
    shutil.copy(SELECTION_CSV, bdir)
    for session in sel_sessions:
        for suffix in ['', '_removed']:
            if os.path.exists(tbl_path(session, suffix)):
                shutil.copy(tbl_path(session, suffix), bdir)
    return bdir


def move_pdf(src_dir, name, dst_dir):
    src = os.path.join(src_dir, name)
    if os.path.exists(src):
        os.makedirs(dst_dir, exist_ok=True)
        shutil.move(src, os.path.join(dst_dir, name))


def apply_session(session, removed, repl):
    from drift_analysis import plot_session_opto_drift  # heavy import, only when applying
    from utils.plot_utils import combine_pdf_big
    opto_dir = session_dirs(session)[f'opto_dir_{DATA_TYPE}']
    drift_dir = os.path.join(opto_dir, 'drift_nontagged')

    # removed units: table rows to the _removed file, PDFs to removed_fr/
    tbl = pd.read_csv(tbl_path(session), index_col=0)
    out = tbl[tbl['unit_id'].isin(removed['unit_id'])].copy()
    for col in ['removed_reason', 'firing_rate', 'isi_violations_ratio']:
        out[col] = out['unit_id'].map(removed.set_index('unit_id')[col])
    out['replaced_by'] = out['unit_id'].map(repl.set_index('replaces_unit')['unit_id']) if len(repl) else pd.NA
    if os.path.exists(tbl_path(session, '_removed')):
        out = pd.concat([pd.read_csv(tbl_path(session, '_removed'), index_col=0), out], ignore_index=True)
    out.to_csv(tbl_path(session, '_removed'))
    tbl = tbl[~tbl['unit_id'].isin(removed['unit_id'])].copy()
    for unit_id, reason in zip(removed['unit_id'], removed['removed_reason']):
        subdir = REMOVED_SUBDIRS['firing_rate' if 'firing_rate' in reason else 'isi']
        move_pdf(drift_dir, f'{unit_id}_drift.pdf', os.path.join(drift_dir, subdir))
        move_pdf(REVIEW_DIR, f'{session}_unit_{unit_id}_drift.pdf', os.path.join(REVIEW_DIR, subdir))

    if 'needs_review' not in tbl:
        tbl['needs_review'] = False
    if 'replaces_unit' not in tbl:
        tbl['replaces_unit'] = pd.NA

    # replacements: drift PDFs and metrics; ephys_cut stays [nan, nan] until reviewed
    if len(repl):
        new = plot_session_opto_drift(session, DATA_TYPE, plot=True, update_csv=False, update_cut=False,
                                      unit_ids=repl['unit_id'].tolist(), out_dir=drift_dir)
        new = new.merge(repl[['unit_id', 'matched_tagged_unit', 'distance_um', 'tier', 'replaces_unit',
                              'needs_review']], on='unit_id', how='left')
        tbl = pd.concat([tbl, new], ignore_index=True)
        for unit_id in repl['unit_id']:
            src = os.path.join(drift_dir, f'{unit_id}_drift.pdf')
            name = f'{session}_unit_{unit_id}_drift.pdf'
            shutil.copy(src, os.path.join(REVIEW_DIR, name))
            os.makedirs(os.path.join(REVIEW_DIR, NEEDS_REVIEW_SUBDIR), exist_ok=True)
            shutil.copy(src, os.path.join(REVIEW_DIR, NEEDS_REVIEW_SUBDIR, name))
    else:  # nothing new plotted, so rebuild the combined PDF without the removed pages
        combine_pdf_big(drift_dir, os.path.join(opto_dir, f'{session}_drift_nontagged.pdf'))
    tbl.to_csv(tbl_path(session))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--no-replace', action='store_true', help='remove failing units without re-matching')
    args = parser.parse_args()

    sel = pd.read_csv(SELECTION_CSV)
    flags = pd.read_csv(FLAGS_CSV)
    fr = pd.read_csv(MASTER_CSV, usecols=['session', 'unit_id', 'firing_rate'])
    flags = flags.merge(fr, on=['session', 'unit_id'], how='left')
    ineligible = ever_listed()
    print(f'{len(ineligible)} units already in non-tagged drift tables are not eligible as replacements')
    removed, repl = plan_swaps(sel, flags, ineligible, replace=not args.no_replace)

    cuts = {}
    for session in removed['session'].unique():
        t = pd.read_csv(tbl_path(session), index_col=0).set_index('unit_id')['ephys_cut']
        cuts.update({(session, u): t.get(u) for u in removed.loc[removed['session'] == session, 'unit_id']})
    log = removed[['session', 'unit_id', 'removed_reason', 'firing_rate', 'isi_violations_ratio',
                   'matched_tagged_unit', 'tier']].copy()
    log['ephys_cut_entered'] = [cuts.get((s, u)) for s, u in zip(log['session'], log['unit_id'])]
    log['had_cut'] = log['ephys_cut_entered'] != '[nan, nan]'
    rmap = {(r.session, r.replaces_unit): r for r in repl.itertuples()} if len(repl) else {}
    log['replaced_by'] = [getattr(rmap.get((s, u)), 'unit_id', pd.NA) for s, u in zip(log['session'], log['unit_id'])]
    log['replacement_distance_um'] = [getattr(rmap.get((s, u)), 'distance_um', pd.NA) for s, u in zip(log['session'], log['unit_id'])]
    log['replacement_tier'] = [getattr(rmap.get((s, u)), 'tier', pd.NA) for s, u in zip(log['session'], log['unit_id'])]

    pd.set_option('display.width', 200)
    print(log.to_string(index=False))
    print(f'\nremove {len(removed)} units (firing rate <= {MIN_FR} Hz or ISI > {MAX_ISI}; '
          f'{int(log["had_cut"].sum())} had cuts entered), '
          f'{len(repl)} replacements, {int(log["replaced_by"].isna().sum())} without a replacement; '
          f'{removed["session"].nunique()} sessions')
    if args.dry_run:
        sys.exit()

    bdir = backup(sel['session'].unique())
    sheet = sorted(glob.glob(os.path.join(REVIEW_DIR, 'drift_cut_nontagged_units_*.csv')))
    sheet = sheet[-1] if sheet else None
    if sheet:
        shutil.copy(sheet, bdir)
    print(f'backed up selection, drift tables and tracking sheet to {bdir}')
    log['swapped_at'] = datetime.datetime.now().isoformat(timespec='seconds')
    log.to_csv(SWAP_LOG, mode='a', header=not os.path.exists(SWAP_LOG), index=False)

    kept = sel[~sel.set_index(['session', 'unit_id']).index.isin(list(zip(removed['session'], removed['unit_id'])))].copy()
    kept['status'] = kept['status'] if 'status' in kept else 'kept'
    kept['replaces_unit'] = kept['replaces_unit'] if 'replaces_unit' in kept else pd.NA
    kept['needs_review'] = kept['needs_review'] if 'needs_review' in kept else False
    new_sel = pd.concat([kept, repl.drop(columns='firing_rate', errors='ignore')], ignore_index=True)
    new_sel.sort_values(['session', 'unit_id']).to_csv(SELECTION_CSV, index=False)

    if sheet:
        ann = pd.read_csv(sheet, encoding='utf-8-sig', dtype=str, keep_default_na=False)
        gone = set(zip(removed['session'], removed['unit_id'].astype(str)))
        keep = [k not in gone for k in zip(ann['session'], ann['unit_id'])]
        ann[keep].to_csv(sheet, index=False, encoding='utf-8-sig')
        print(f'{sheet}: dropped {len(ann) - sum(keep)} removed units')

    for session in removed['session'].unique():
        try:
            apply_session(session, removed[removed['session'] == session],
                          repl[repl['session'] == session] if len(repl) else repl)
            print(f'{session}: done')
        except Exception as e:
            print(f'{session}: failed, {type(e).__name__}: {e}')

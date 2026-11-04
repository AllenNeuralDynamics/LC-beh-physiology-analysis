# %% Tracking sheet for hand-placed drift cuts of the matched non-tagged units.
#
# Cuts are entered in a CSV (one row per unit) and written back into each session's
# {session}_opto_drift_tbl_nontagged.csv. Same rules as the 2026-10-29 write-back:
#   - ephys_cut_start / ephys_cut_end: blank = no cut on that side
#   - drift_unit: TRUE exactly when a cut is entered (or for excluded units)
#   - exclude: TRUE drops the unit from nontagged_matched_selection.csv (its table row is kept);
#     an excluded unit carries no cut
# Extra columns: status (kept / replacement / excluded), replaces_unit, and needs_review, which
# is TRUE for units nobody has looked at yet; set it to FALSE (or blank) once reviewed.
#
#     python unit_summary_DRN_drift_cut_sheet.py make  [--from SHEET]  # new sheet with current units
#     python unit_summary_DRN_drift_cut_sheet.py write SHEET            # write cuts into the tables

import os
import sys
import glob
import shutil
import argparse
import datetime

curr_path = os.path.dirname(os.path.abspath(__file__))
if curr_path not in sys.path:
    sys.path.insert(0, curr_path)

import pandas as pd

from utils.beh_functions import session_dirs

ROOT = '/root/capsule/scratch/combined/unit_summary_DRN'
REVIEW_DIR = os.path.join(ROOT, 'drift_nontagged_review')
SELECTION_CSV = os.path.join(ROOT, 'nontagged_matched_selection.csv')
SHEET_COLS = ['session', 'unit_id', 'ephys_cut_start', 'ephys_cut_end', 'drift_unit', 'exclude', 'notes',
              'needs_review', 'status', 'replaces_unit']


def tbl_path(session):
    return os.path.join(session_dirs(session)['opto_dir_raw'], f'{session}_opto_drift_tbl_nontagged.csv')


def truthy(col):
    return col.astype(str).str.strip().str.upper().isin(['TRUE', '1', '1.0', 'YES'])


def parse_cut(text):
    """'[123, nan]' -> (123.0, nan)."""
    a, b = [x.strip() for x in str(text).strip('[]').split(',')]
    return tuple(float('nan') if x.lower() in ('nan', 'none') else float(x) for x in (a, b))


def fmt(v):
    return 'nan' if pd.isna(v) else (str(int(v)) if float(v).is_integer() else str(v))


def latest_sheet():
    sheets = sorted(glob.glob(os.path.join(REVIEW_DIR, 'drift_cut_nontagged_units_*.csv')))
    return sheets[-1] if sheets else None


def make_sheet(prev_path, out=None):
    """One row per unit in the active drift tables: current selection plus excluded units.
    Values come from the previous sheet where it has the unit, otherwise from the tables."""
    prev = pd.read_csv(prev_path, encoding='utf-8-sig') if prev_path else pd.DataFrame(columns=SHEET_COLS)
    prev = prev.set_index(['session', 'unit_id'])
    sel = pd.read_csv(SELECTION_CSV)
    selected = set(zip(sel['session'], sel['unit_id']))
    sel_info = sel.set_index(['session', 'unit_id'])

    rows = []
    for session in sorted(sel['session'].unique().tolist()
                          + [s for s in prev.index.get_level_values(0).unique() if s not in set(sel['session'])]):
        if not os.path.exists(tbl_path(session)):
            continue
        tbl = pd.read_csv(tbl_path(session), index_col=0)
        for _, t in tbl.iterrows():
            key = (session, int(t['unit_id']))
            in_sel = key in selected
            p = prev.loc[key] if key in prev.index else None
            if p is None and not in_sel:
                continue  # neither selected nor previously tracked
            start, end = parse_cut(t['ephys_cut'])
            info = sel_info.loc[key] if in_sel else None
            needs_review = (bool(info.get('needs_review')) if info is not None and pd.notna(info.get('needs_review'))
                            else False)
            replaces = info.get('replaces_unit') if info is not None else pd.NA
            row = {'session': session, 'unit_id': key[1],
                   'ephys_cut_start': p['ephys_cut_start'] if p is not None else start,
                   'ephys_cut_end': p['ephys_cut_end'] if p is not None else end,
                   'drift_unit': (p['drift_unit'] if p is not None
                                  else ('TRUE' if pd.notna([start, end]).any() else '')),
                   'exclude': p['exclude'] if p is not None else '',
                   'notes': p['notes'] if p is not None else '',
                   'needs_review': 'TRUE' if needs_review else '',
                   'status': ('excluded' if not in_sel
                              else (info.get('status') if pd.notna(info.get('status')) else 'kept')),
                   'replaces_unit': '' if pd.isna(replaces) else int(replaces)}
            if needs_review and not row['notes']:
                row['notes'] = (f'new: replaces unit {row["replaces_unit"]} (firing rate <= 0.5 Hz)'
                                if row['status'] == 'replacement' else
                                f'new: added control for tagged unit {int(info["matched_tagged_unit"])} '
                                f'({info["distance_um"]:.0f} um)')
            rows.append(row)
    sheet = pd.DataFrame(rows, columns=SHEET_COLS)
    for col in ['drift_unit', 'exclude', 'notes']:
        sheet[col] = sheet[col].fillna('')
    out = out or os.path.join(REVIEW_DIR, f'drift_cut_nontagged_units_{datetime.date.today():%y%m%d}.csv')
    if os.path.exists(out):
        sys.exit(f'{out} exists; not overwriting a sheet that may hold edits')
    sheet.to_csv(out, index=False, encoding='utf-8-sig')
    print(f'wrote {out}: {len(sheet)} units, {int(truthy(sheet["needs_review"]).sum())} need review, '
          f'{int(truthy(sheet["exclude"]).sum())} excluded, '
          f'{int(sheet[["ephys_cut_start", "ephys_cut_end"]].notna().any(axis=1).sum())} with cuts')


def write_sheet(path):
    ann = pd.read_csv(path, encoding='utf-8-sig')
    ann['exclude'] = truthy(ann['exclude'])
    drift = truthy(ann['drift_unit'])
    has_cut = ann[['ephys_cut_start', 'ephys_cut_end']].notna().any(axis=1)
    bad = ann[ann['exclude'] & has_cut]
    assert bad.empty, f'excluded units with cuts:\n{bad[["session", "unit_id"]]}'
    bad = ann[(drift != has_cut) & ~(drift & ann['exclude'])]
    assert bad.empty, f'drift_unit does not match whether a cut is entered:\n{bad[["session", "unit_id"]]}'

    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    bdir = os.path.join(ROOT, f'backup_before_cut_writeback_{stamp}')
    os.makedirs(bdir)
    shutil.copy(SELECTION_CSV, bdir)

    # selection: drop excluded units, carry review state from the sheet
    sel = pd.read_csv(SELECTION_CSV)
    excl = set(zip(ann.loc[ann['exclude'], 'session'], ann.loc[ann['exclude'], 'unit_id']))
    review = dict(zip(zip(ann['session'], ann['unit_id']), truthy(ann['needs_review'])))
    n_before = len(sel)
    sel = sel[[k not in excl for k in zip(sel['session'], sel['unit_id'])]].copy()
    if 'needs_review' in sel:
        sel['needs_review'] = [review.get(k, v) for k, v in zip(zip(sel['session'], sel['unit_id']), sel['needs_review'])]
    sel.to_csv(SELECTION_CSV, index=False)

    n_cut = 0
    for session, g in ann.groupby('session'):
        f = tbl_path(session)
        t = pd.read_csv(f, index_col=0)
        missing = set(g['unit_id']) - set(t['unit_id'])
        assert not missing, f'{session}: units {missing} not in {f}'
        shutil.copy(f, bdir)
        if 'needs_review' not in t:
            t['needs_review'] = False
        for _, r in g.iterrows():
            m = t['unit_id'] == r['unit_id']
            t.loc[m, 'needs_review'] = bool(review[(session, r['unit_id'])])
            if r['exclude']:
                continue
            c = [r['ephys_cut_start'], r['ephys_cut_end']]
            t.loc[m, 'ephys_cut'] = f'[{fmt(c[0])}, {fmt(c[1])}]'
            t.loc[m, 'drift_unit'] = bool(pd.notna(c).any())
            n_cut += bool(pd.notna(c).any())
        t.to_csv(f)
    pending = ann[truthy(ann['needs_review']) & ~ann['exclude']]
    print(f'backup: {bdir}')
    print(f'sessions {ann["session"].nunique()}, units with cuts {n_cut}, excluded {len(excl)}; '
          f'selection {n_before} -> {len(sel)}')
    if len(pending):
        print(f'{len(pending)} units still marked needs_review:')
        print(pending[['session', 'unit_id', 'notes']].to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('make')
    p.add_argument('--from', dest='prev', default=None, help='previous sheet; default: latest in REVIEW_DIR')
    p.add_argument('--out', default=None, help='sheet to write; default: dated name in REVIEW_DIR')
    p = sub.add_parser('write')
    p.add_argument('sheet')
    args = parser.parse_args()
    if args.cmd == 'make':
        make_sheet(args.prev or latest_sheet(), args.out)
    else:
        write_sheet(args.sheet)

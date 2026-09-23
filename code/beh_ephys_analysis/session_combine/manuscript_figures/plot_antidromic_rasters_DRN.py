"""Raster pages for the DRN antidromic candidate units, for eyeballing real spike vs artifact.

Plots every unit that passed any sharp/collision criterion at either stimulation site
(evidence_rank <= 3 in the per-site lookup tables written by F_antidromic_combined_DRN.ipynb),
one page per unit, all emission sites side by side so the somatic DRN response can be compared
against the putative antidromic PrL/S1 responses in the same unit.

Two PDFs are produced: a zoomed page (-10..40 ms), which is the one that shows whether a
light-locked spike has plausible jitter, and the full window (-100..70 ms) that the original
plot_opto_responses default uses.

Usage:  python plot_antidromic_rasters_DRN.py [--all-units] [--max-units N]
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append('/root/capsule/code/beh_ephys_analysis')

import antidromic_funcs as anf
from utils.beh_functions import get_unit_tbl, session_dirs
from utils.ephys_functions import load_drift
from utils.plot_utils import combine_pdf_big

TABLE_DIR = '/root/capsule/scratch/results/manuscript/figures/F_antidromic_DRN/tables'
OUT_DIR = '/root/capsule/scratch/results/manuscript/figures/F_antidromic_DRN/rasters'
DATA_TYPE = 'raw'
FOCUSES = ['PrL', 'S1']
SITE_ORDER = ['surface_DRN', 'surface_PrL', 'surface_S1']
VIEWS = {
    'zoom': (-0.010, 0.040),
    'full': (-0.100, 0.070),
}


def load_lookup():
    """Per-(unit, site) criteria rows, both focuses stacked."""
    frames = []
    for focus in FOCUSES:
        d = pd.read_csv(os.path.join(TABLE_DIR, f'unit_criteria_lookup_{focus}.csv'))
        d['focus'] = focus
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def page_label(rows):
    """One-line summary of what this unit passed at each site."""
    r0 = rows.iloc[0]
    bits = [f"{r0.session}  unit {int(r0.unit)}", f"5-HT tagged: {bool(r0.opto_tagged)}"]
    for _, r in rows.sort_values('focus').iterrows():
        passed = r.criteria_passed if isinstance(r.criteria_passed, str) and r.criteria_passed else 'none'
        lat = 'nan' if not np.isfinite(r.lat_ms) else f'{r.lat_ms:.1f}'
        jit = 'nan' if not np.isfinite(r.jit_ms) else f'{r.jit_ms:.1f}'
        deg = '  [DEGENERATE FIT]' if bool(r.degenerate_fit) else ''
        bits.append(f"{r.focus}: lat {lat} ms, jitter {jit} ms, tier {r.tier} | {passed}{deg}")
    return '\n'.join(bits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--all-units', action='store_true',
                    help='plot every unit in the lookup table, not just sharp/collision passers')
    ap.add_argument('--max-units', type=int, default=None)
    args = ap.parse_args()

    lookup = load_lookup()
    if args.all_units:
        keep = lookup[['session', 'unit']].drop_duplicates()
    else:
        keep = lookup.loc[lookup.evidence_rank <= 3, ['session', 'unit']].drop_duplicates()
    keep = keep.sort_values(['session', 'unit'])
    if args.max_units:
        keep = keep.head(args.max_units)
    print(f'{len(keep)} units across {keep.session.nunique()} sessions')

    for view in VIEWS:
        os.makedirs(os.path.join(OUT_DIR, view), exist_ok=True)

    failed = []
    for session, units in keep.groupby('session'):
        sdir = session_dirs(session)
        csv = os.path.join(sdir[f'opto_dir_{DATA_TYPE}'], f'{session}_opto_session.csv')
        if not os.path.exists(csv):
            failed.append((session, None, 'no opto_session.csv'))
            continue
        event_ids_all = pd.read_csv(csv)
        try:
            unit_tbl = get_unit_tbl(session, data_type=DATA_TYPE)
        except Exception as exc:  # noqa: BLE001 - one bad session should not stop the batch
            failed.append((session, None, f'get_unit_tbl: {exc}'))
            continue

        for unit in units.unit.tolist():
            row = unit_tbl[unit_tbl.unit_id == unit]
            if row.empty:
                failed.append((session, unit, 'unit not in unit_tbl'))
                continue

            # Same drift handling as plot_opto_responses_session: drop stim events outside the
            # hand-curated window for this unit, so the rasters match the analysed pulses.
            event_ids = event_ids_all.copy()
            try:
                drift = load_drift(session, unit, data_type=DATA_TYPE)
            except Exception:  # noqa: BLE001
                drift = None
            cut = None if drift is None else drift.get('ephys_cut')
            if isinstance(cut, (list, tuple, np.ndarray)) and len(cut) == 2:
                if cut[0] is not None and np.isfinite(cut[0]):
                    event_ids = event_ids[event_ids['time'] >= cut[0]]
                if cut[1] is not None and np.isfinite(cut[1]):
                    event_ids = event_ids[event_ids['time'] <= cut[1]]
            if event_ids.empty:
                failed.append((session, unit, 'no events left after drift cut'))
                continue

            label = page_label(lookup[(lookup.session == session) & (lookup.unit == unit)])
            for view, trange in VIEWS.items():
                try:
                    fig = anf.plot_opto_responses(row, event_ids, time_range_raster=trange,
                                                  sites=SITE_ORDER)
                    fig.suptitle(label, fontsize=9, ha='left', x=0.02, y=0.998)
                    out = os.path.join(OUT_DIR, view, f'{session}_unit{unit}.pdf')
                    fig.savefig(out, bbox_inches='tight')
                    plt.close(fig)
                except Exception as exc:  # noqa: BLE001
                    plt.close('all')
                    failed.append((session, unit, f'{view}: {exc}'))
            print(f'  {session} unit {unit} done', flush=True)

    for view in VIEWS:
        combine_pdf_big(os.path.join(OUT_DIR, view),
                        os.path.join(OUT_DIR, f'antidromic_rasters_{view}.pdf'))
        print(f'wrote {OUT_DIR}/antidromic_rasters_{view}.pdf')

    if failed:
        print(f'\n{len(failed)} failures:')
        for f in failed:
            print('  ', f)


if __name__ == '__main__':
    main()

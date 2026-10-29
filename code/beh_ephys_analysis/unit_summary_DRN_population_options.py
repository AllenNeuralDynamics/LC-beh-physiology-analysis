# %% How many non-tagged DRN units would each population definition select?
#
# Read-only. Tagged units are the basic_ephys_all_DRN selection, built exactly as in
# session_combine/figure_preparation/basic_ephys_generation.py (version 'DRN'). Non-tagged
# candidates come from the master unit table. Prints a table of counts per option and saves
# the per-unit flags so a chosen option can be used to pick units for unit_summary_DRN.py.
#
#     python unit_summary_DRN_population_options.py

import os
import sys
import json
import pickle

curr_path = os.path.dirname(os.path.abspath(__file__))
if curr_path not in sys.path:
    sys.path.insert(0, curr_path)

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from trimesh import load_mesh

from utils.capsule_migration import capsule_directories
from utils.ccf_utils import pir_to_lps
from utils.combine_tools import apply_qc, to_str_intlike

MASTER_CSV = '/root/capsule/scratch/combined/master_unit_tables/master_all_units_opto_ccf_raw_260817.csv'
CRITERIA = '/root/capsule/code/beh_ephys_analysis/session_combine/metrics/basic_ephys_all_DRN.json'
OUT_DIR = '/root/capsule/scratch/combined/unit_summary_DRN'
BREGMA_LPS_MM = np.array([-5.74, 5.4, -0.45])  # as in unit_summary_DRN.py
CCF_COLS = ['x_ccf', 'y_ccf', 'z_ccf']
DRN_MESHES = ['/root/capsule/data/5HT_meshes/B6_7_DR_core_67pct.obj',
              '/root/capsule/data/5HT_meshes/B7l_wing_core_67pct.obj']
VOXEL_MM = 0.01
ISI_MATCHED = 0.1  # the tagged filter's isi bound (non-tier_1 units)
DV, ML = 2, 0
MARGINS_UM = [0, 100, 200]


def tagged_units():
    """basic_ephys_all_DRN selection, as basic_ephys_generation.py builds it for 'DRN'."""
    prep = capsule_directories()['manuscript_fig_prep_dir']
    with open(os.path.join(prep, 'combined_unit_tbl', 'combined_unit_tbl.pkl'), 'rb') as f:
        tbl = pickle.load(f)
    with open(CRITERIA) as f:
        constraints = json.load(f)
    constraints['sd'] = [0, 1]  # basic_ephys_generation's override for DRN
    with open(os.path.join(prep, 'antidromic_analysis', 'PrL_S1', 'combined_antidromic_results.pkl'), 'rb') as f:
        anti = pickle.load(f)[['session', 'unit', 'tier_1']].copy()
    anti['unit_key'] = anti['unit'].apply(to_str_intlike)
    tbl['unit_key'] = tbl['unit'].apply(to_str_intlike)
    tbl = tbl.merge(anti[['session', 'unit_key', 'tier_1']], on=['session', 'unit_key'],
                    how='left').drop(columns='unit_key')
    tbl['tier_1'] = tbl['tier_1'].fillna(False)
    filtered, tbl, _, _ = apply_qc(tbl, constraints)
    candidates = set(zip(tbl['session'], tbl['unit'].astype(int)))  # any opto candidate
    return set(zip(filtered['session'], filtered['unit'].astype(int))), candidates


def drn_distance_mm(X):
    """Distance (mm) from each point to the DR core + wing volume; 0 inside, NaN without CCF."""
    grids = []
    for path in DRN_MESHES:
        mesh = load_mesh(path)
        mesh.vertices = pir_to_lps(np.asarray(mesh.vertices) / 1000) - BREGMA_LPS_MM
        grids.append(mesh)
    lo = np.min([m.bounds[0] for m in grids], axis=0) - 0.5
    hi = np.max([m.bounds[1] for m in grids], axis=0) + 0.5
    shape = np.ceil((hi - lo) / VOXEL_MM).astype(int) + 1
    inside = np.zeros(shape, bool)
    for mesh in grids:
        vg = mesh.voxelized(VOXEL_MM).fill()
        idx = np.argwhere(vg.matrix)
        pts = vg.indices_to_points(idx)
        ijk = np.round((pts - lo) / VOXEL_MM).astype(int)
        inside[tuple(ijk.T)] = True
    dist = distance_transform_edt(~inside) * VOXEL_MM
    out = np.full(len(X), np.nan)
    ok = np.isfinite(X).all(axis=1)
    ijk = np.round((X[ok] - lo) / VOXEL_MM).astype(int)
    in_grid = ((ijk >= 0) & (ijk < shape)).all(axis=1)
    d = np.full(ok.sum(), np.inf)  # beyond the grid: > 0.5 mm away
    d[in_grid] = dist[tuple(ijk[in_grid].T)]
    out[ok] = d
    return out


def within_tagged_range(df, col, tagged_mask, margin):
    """Per session, is `col` within [min, max] of that session's tagged units, +/- margin?"""
    rng = df[tagged_mask].groupby('session')[col].agg(['min', 'max'])
    lo = df['session'].map(rng['min']) - margin
    hi = df['session'].map(rng['max']) + margin
    return (df[col] >= lo) & (df[col] <= hi)


if __name__ == '__main__':
    tagged, candidates = tagged_units()
    m = pd.read_csv(MASTER_CSV, usecols=['session', 'unit_id', 'decoder_label', 'default_qc', 'opto_pass',
                                         'isi_violations_ratio', 'depth'] + CCF_COLS)
    key = list(zip(m['session'], m['unit_id'].astype(int)))
    m['tagged'] = [k in tagged for k in key]
    m['opto_candidate'] = [k in candidates for k in key]
    print(f"tagged (basic_ephys_all_DRN): {len(tagged)}, found in master: {int(m['tagged'].sum())}")

    X = m[CCF_COLS].to_numpy(float) - BREGMA_LPS_MM
    m['ML'], m['AP'], m['DV'] = X[:, 0], X[:, 1], X[:, 2]
    m['drn_dist_mm'] = drn_distance_mm(X)
    m['has_ccf'] = np.isfinite(X).all(axis=1)

    nonnoise = ~m['decoder_label'].isin(['noise', 'artifact'])
    base = {
        'default_qc': nonnoise & m['default_qc'].astype(bool),
        f'default_qc + ISI<={ISI_MATCHED}': nonnoise & m['default_qc'].astype(bool) & (m['isi_violations_ratio'] <= ISI_MATCHED),
    }
    # "non-tagged" two ways: everything outside the 194, or also excluding every opto candidate
    # (units that responded to light but failed the tagged filter)
    exclusions = {'not tagged': ~m['tagged'], 'no opto response': ~m['tagged'] & ~m['opto_candidate'] & ~m['opto_pass'].astype(bool)}
    has_tagged = m['session'].isin(m.loc[m['tagged'], 'session'])

    spatial = {'anywhere (all 58 sessions)': pd.Series(True, index=m.index),
               'sessions with tagged units': has_tagged}
    for mg in MARGINS_UM:
        spatial[f'DR core+wing mesh{"" if mg == 0 else f" + {mg} um"}'] = m['drn_dist_mm'] <= mg / 1000
    for mg in MARGINS_UM:
        spatial[f'CCF DV within session tagged range{"" if mg == 0 else f" +/- {mg} um"}'] = \
            within_tagged_range(m, 'DV', m['tagged'], mg / 1000)
    for mg in MARGINS_UM:
        spatial[f'probe depth within session tagged range{"" if mg == 0 else f" +/- {mg} um"}'] = \
            within_tagged_range(m, 'depth', m['tagged'], mg)
    q = m.loc[m['tagged'], 'DV'].quantile([0.05, 0.95])
    spatial[f'CCF DV within pooled tagged 5-95% ({q.iloc[0]:.2f} to {q.iloc[1]:.2f} mm)'] = m['DV'].between(q.iloc[0], q.iloc[1])
    spatial['CCF DV within session tagged range +/- 100 um AND mesh + 100 um'] = (
        within_tagged_range(m, 'DV', m['tagged'], 0.1) & (m['drn_dist_mm'] <= 0.1))

    rows = []
    for sname, smask in spatial.items():
        row = {'spatial option': sname}
        for bname, bmask in base.items():
            for ename, emask in exclusions.items():
                row[f'{bname} | {ename}'] = int((bmask & emask & smask.fillna(False)).sum())
        row['tagged in option'] = int((m['tagged'] & smask.fillna(False)).sum())
        rows.append(row)
    counts = pd.DataFrame(rows)

    pd.set_option('display.width', 250)
    pd.set_option('display.max_colwidth', 70)
    print(counts.to_string(index=False))
    os.makedirs(OUT_DIR, exist_ok=True)
    counts.to_csv(os.path.join(OUT_DIR, 'population_option_counts.csv'), index=False)
    m.drop(columns=CCF_COLS).to_csv(os.path.join(OUT_DIR, 'population_unit_flags.csv'), index=False)
    print(f'\nsessions with tagged units: {m.loc[m["tagged"], "session"].nunique()} of {m["session"].nunique()}')
    print(f'tagged units without CCF: {int((m["tagged"] & ~m["has_ccf"]).sum())}; '
          f'non-noise units without CCF: {int((nonnoise & ~m["has_ccf"]).sum())}')

# %% How many tagged / untagged units sit in or near the B5 and B8 CS 5-HT core meshes?
#
# Read-only. Tagged = basic_ephys_all_DRN (as in unit_summary_DRN_population_options.py).
# Units must pass default_qc (and not be decoder noise/artifact) and have more than MIN_SPIKES
# spikes over the whole session (uncut spike_times from the per-session unit table).
# Prints counts per mesh and distance margin and saves the per-unit table.
#
#     python cs_mesh_population_counts.py

import os
import sys

curr_path = os.path.dirname(os.path.abspath(__file__))
if curr_path not in sys.path:
    sys.path.insert(0, curr_path)

import contextlib
import io

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from trimesh import load_mesh

from utils.beh_functions import get_unit_tbl
from utils.ccf_utils import pir_to_lps
from unit_summary_DRN_population_options import (MASTER_CSV, BREGMA_LPS_MM, CCF_COLS, VOXEL_MM,
                                                 tagged_units)

OUT_DIR = '/root/capsule/scratch/combined/unit_summary_DRN'
CS_MESHES = {'B5': '/root/capsule/data/5HT_meshes/B5_CS_core_67pct.obj',
             'B8': '/root/capsule/data/5HT_meshes/B8_CS_core_67pct.obj'}
MIN_SPIKES = 200
MARGINS_UM = [0, 100, 200, 300]


def mesh_distance_mm(X, path, pad=0.5):
    """Distance (mm) from each point to the filled mesh volume; 0 inside, inf beyond pad, NaN without CCF."""
    mesh = load_mesh(path)
    mesh.vertices = pir_to_lps(np.asarray(mesh.vertices) / 1000) - BREGMA_LPS_MM
    lo, hi = mesh.bounds[0] - pad, mesh.bounds[1] + pad
    shape = np.ceil((hi - lo) / VOXEL_MM).astype(int) + 1
    inside = np.zeros(shape, bool)
    vg = mesh.voxelized(VOXEL_MM).fill()
    ijk = np.round((vg.indices_to_points(np.argwhere(vg.matrix)) - lo) / VOXEL_MM).astype(int)
    inside[tuple(ijk.T)] = True
    dist = distance_transform_edt(~inside) * VOXEL_MM
    out = np.full(len(X), np.nan)
    ok = np.isfinite(X).all(axis=1)
    ijk = np.round((X[ok] - lo) / VOXEL_MM).astype(int)
    in_grid = ((ijk >= 0) & (ijk < shape)).all(axis=1)
    d = np.full(ok.sum(), np.inf)
    d[in_grid] = dist[tuple(ijk[in_grid].T)]
    out[ok] = d
    return out


def session_spike_counts(sessions):
    rows = []
    for s in sessions:
        with contextlib.redirect_stdout(io.StringIO()):
            tbl = get_unit_tbl(s, 'raw', summary=False)
        if tbl is None:
            print(f'no unit table: {s}')
            continue
        rows.append(pd.DataFrame({'session': s, 'unit_id': tbl['unit_id'].astype(int),
                                  'n_spikes': [len(t) for t in tbl['spike_times']]}))
    return pd.concat(rows, ignore_index=True)


if __name__ == '__main__':
    tagged, _ = tagged_units()
    m = pd.read_csv(MASTER_CSV, usecols=['session', 'unit_id', 'decoder_label', 'default_qc'] + CCF_COLS)
    m['unit_id'] = m['unit_id'].astype(int)
    m['tagged'] = [k in tagged for k in zip(m['session'], m['unit_id'])]
    X = m[CCF_COLS].to_numpy(float) - BREGMA_LPS_MM
    for name, path in CS_MESHES.items():
        m[f'{name}_dist_mm'] = mesh_distance_mm(X, path)
    m['cs_dist_mm'] = m[[f'{n}_dist_mm' for n in CS_MESHES]].min(axis=1)

    # spike counts only needed for sessions with a unit near either mesh
    near = m['cs_dist_mm'] <= max(MARGINS_UM) / 1000
    m = m.merge(session_spike_counts(m.loc[near, 'session'].unique()), on=['session', 'unit_id'], how='left')

    qc = ~m['decoder_label'].isin(['noise', 'artifact']) & m['default_qc'].astype(bool)
    spikes = m['n_spikes'] > MIN_SPIKES
    rows = []
    for col, label in [('B5_dist_mm', 'B5 (CS)'), ('B8_dist_mm', 'B8 (CS)'), ('cs_dist_mm', 'B5 or B8')]:
        for mg in MARGINS_UM:
            loc = m[col] <= mg / 1000
            sel = loc & qc & spikes
            rows.append({'mesh': label, 'margin_um': mg,
                         'tagged': int((sel & m['tagged']).sum()), 'untagged': int((sel & ~m['tagged']).sum()),
                         'sessions': m.loc[sel, 'session'].nunique(),
                         'all units (no QC/spike filter)': int(loc.sum())})
    counts = pd.DataFrame(rows)
    print(counts.to_string(index=False))
    os.makedirs(OUT_DIR, exist_ok=True)
    counts.to_csv(os.path.join(OUT_DIR, 'cs_mesh_population_counts.csv'), index=False)
    out = m[near & qc & spikes].drop(columns=CCF_COLS).sort_values(['cs_dist_mm', 'session', 'unit_id'])
    out.to_csv(os.path.join(OUT_DIR, 'cs_mesh_population_units.csv'), index=False)
    print(f'\nunits with CCF: {int(np.isfinite(X).all(axis=1).sum())} of {len(m)}; '
          f'missing spike counts among near units: {int(m.loc[near, "n_spikes"].isna().sum())}')

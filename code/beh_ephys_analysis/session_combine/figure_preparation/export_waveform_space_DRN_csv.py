"""Export data behind section 9 ("Main DRN feature-in-space figure") of
manuscript_figures/F_waveform_space_DRN_v2.ipynb as CSVs for collaborators.

Outputs (in <manuscript_fig_dir>/F_waveform_space_DRN_v2/csv_export/):
  1. units_coords_features_<criteria>.csv
       All units passing non-opto QC, with boolean 'opto tagged' column,
       with raw CCF and plotting (bregma-relative, |ML|) coordinates.
       Waveform feature columns are filled for opto-tagged units only.
  2. quartile_waveforms_<criteria>.csv
       Long format: mean +/- SEM normalized waveform for each feature quartile
       of opto-tagged units (rightmost column of the figure).
"""
import sys
import os
import json
import pickle

sys.path.append('/root/capsule/code/beh_ephys_analysis')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.combine_tools import apply_qc
from utils.capsule_migration import capsule_directories

capsule_dirs = capsule_directories()

# Same configuration as F_waveform_space_DRN_v2.ipynb
criteria_name = 'waveform_low_DRN'
waveform_version = '_raw'
ISI_MAX = 0.5
UNIT_KEY = ['session', 'unit']
CCF_COLS = ['x_ccf', 'y_ccf', 'z_ccf']
BREGMA_LPS_MM = np.array([-5.74, 5.4, -0.45])
MIRROR_ML = True
BIN_NUM = 4
focus_features = [
    'post_w',
    'trough_post_ratio_1D',
    'post_trough_slope',
    'pre_slope',
    'symmetry_slope_div_log',
    'symmetry_trough_dis',
    'symmetry_inte_div_log',
]

out_dir = os.path.join(capsule_dirs['manuscript_fig_dir'], 'F_waveform_space_DRN_v2', 'csv_export')
os.makedirs(out_dir, exist_ok=True)


def safe_divide(num, den, eps=1e-10):
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    out = np.full(np.broadcast_shapes(num.shape, den.shape), np.nan, dtype=float)
    valid = np.isfinite(num) & np.isfinite(den) & (np.abs(den) > eps)
    np.divide(num, den, out=out, where=valid)
    return out


def ccf_to_bregma_relative(df):
    ccf = df[CCF_COLS].to_numpy(dtype=float).copy()
    ccf -= BREGMA_LPS_MM
    if MIRROR_ML:
        ccf[:, 0] = np.abs(ccf[:, 0])
    return ccf


# ---------------- Load + QC ----------------
with open(os.path.join(capsule_dirs['manuscript_fig_prep_dir'], 'combined_unit_tbl', 'combined_unit_tbl.pkl'), 'rb') as f:
    combined_tagged_units = pickle.load(f)
with open(os.path.join('/root/capsule/code/beh_ephys_analysis/session_combine/metrics', f'{criteria_name}.json')) as f:
    constraints = json.load(f)

opto_units, combined_tagged_units, fig, _ = apply_qc(combined_tagged_units, constraints)
plt.close(fig)

wf_features = pd.read_csv(os.path.join(capsule_dirs['manuscript_fig_prep_dir'], 'waveforms_np', 'combined_features.csv'))
wf_features = wf_features[[c for c in wf_features.columns if not c.startswith('wf_pc')]]
feature_cols = [c for c in wf_features.columns
                if c not in UNIT_KEY + ['probe'] and np.issubdtype(wf_features[c].dtype, np.number)]

# ---------------- Normalized waveforms (opto units, as in notebook section 4) ----------------
all_wfs = np.stack(opto_units[f'wf{waveform_version}'].to_numpy())
baseline = np.nanmean(all_wfs[:, :3], axis=1)
denom = all_wfs[:, 30] - baseline
valid_norm = np.isfinite(denom) & (np.abs(denom) > 1e-10)
all_wfs_nl = np.full_like(all_wfs, np.nan, dtype=float)
all_wfs_nl[valid_norm] = (all_wfs[valid_norm] - baseline[valid_norm, None]) / denom[valid_norm, None]
all_wfs_nl_raw = safe_divide(all_wfs, all_wfs[:, 30][:, None])
bad_shape = np.isfinite(all_wfs_nl_raw[:, 0]) & (all_wfs_nl_raw[:, 0] > 1)
all_wfs_nl[bad_shape] = np.nan

waveform_lookup = opto_units[UNIT_KEY].copy()
waveform_lookup['wf_nl'] = list(all_wfs_nl)
waveform_lookup['wf_norm_valid'] = np.isfinite(all_wfs_nl).all(axis=1)

# ---------------- CSV 1: all QC units, coords, label, opto features ----------------
qc_units = combined_tagged_units.loc[combined_tagged_units['selected_no_opto'],
                                     UNIT_KEY + CCF_COLS + ['isi_violations', 'selected']].copy()
qc_units['opto tagged'] = qc_units['selected'].astype(bool)
plot_xyz = ccf_to_bregma_relative(qc_units)
qc_units['ml_mm'] = plot_xyz[:, 0]
qc_units['ap_mm'] = plot_xyz[:, 1]
qc_units['dv_mm'] = plot_xyz[:, 2]

# Opto features: inner merge as in the notebook (opto units lacking features keep NaN)
opto_feat = wf_features[UNIT_KEY + feature_cols].merge(opto_units[UNIT_KEY], on=UNIT_KEY, how='inner',
                                                       validate='one_to_one')
opto_feat = opto_feat.merge(waveform_lookup[UNIT_KEY + ['wf_norm_valid']], on=UNIT_KEY, how='left',
                            validate='one_to_one')
opto_feat['has_wf_features'] = True
units_out = qc_units.merge(opto_feat, on=UNIT_KEY, how='left', validate='one_to_one')
units_out['has_wf_features'] = units_out['has_wf_features'].fillna(False).astype(bool)
units_out['wf_norm_valid'] = units_out['wf_norm_valid'].astype('boolean')
num_cols = units_out.select_dtypes(include=np.number).columns
units_out[num_cols] = units_out[num_cols].replace([np.inf, -np.inf], np.nan)

# Units actually plotted in section 9 (base_df): opto + features + finite CCF + ISI < ISI_MAX
units_out['in_section9_figure'] = (
    units_out['has_wf_features']
    & units_out['isi_violations'].lt(ISI_MAX)
    & np.isfinite(units_out[CCF_COLS].to_numpy(dtype=float)).all(axis=1)
)

units_out = units_out[UNIT_KEY + ['opto tagged', 'in_section9_figure', 'has_wf_features', 'wf_norm_valid',
                                  'ml_mm', 'ap_mm', 'dv_mm'] + CCF_COLS + ['isi_violations'] + feature_cols]
units_out = units_out.rename(columns={'x_ccf': 'x_ccf_raw', 'y_ccf': 'y_ccf_raw', 'z_ccf': 'z_ccf_raw'})
units_file = os.path.join(out_dir, f'units_coords_features_{criteria_name}.csv')
units_out.to_csv(units_file, index=False)

# ---------------- CSV 2: quartile waveforms (as in notebook section 9) ----------------
base_df = units_out.loc[units_out['in_section9_figure'], UNIT_KEY + feature_cols].merge(
    waveform_lookup, on=UNIT_KEY, how='left', validate='one_to_one')
t_ms = (np.arange(all_wfs_nl.shape[1]) - 30) / 30000 * 1000

rows = []
for feature in feature_cols:
    df_f = base_df.loc[np.isfinite(base_df[feature].to_numpy(dtype=float))]
    wf_mask = df_f['wf_norm_valid'].to_numpy(dtype=bool)
    wf_values = df_f[feature].to_numpy(dtype=float)[wf_mask]
    wf_arrays = df_f.loc[wf_mask, 'wf_nl'].to_numpy()
    if len(wf_values) < BIN_NUM or np.unique(wf_values).size < BIN_NUM:
        continue
    edges = np.nanquantile(wf_values, np.linspace(0, 1, BIN_NUM + 1))
    for b in range(BIN_NUM):
        bm = ((wf_values >= edges[b]) &
              (wf_values <= edges[b + 1] if b == BIN_NUM - 1 else wf_values < edges[b + 1]))
        if not bm.any():
            continue
        wfs = np.stack(wf_arrays[bm])
        mean_wf = np.nanmean(wfs, axis=0)
        sem = safe_divide(np.nanstd(wfs, axis=0), np.sqrt(np.sum(np.isfinite(wfs), axis=0)))
        rows.append(pd.DataFrame({
            'feature': feature,
            'is_focus_feature': feature in focus_features,
            'quartile': b + 1,
            'bin_low': edges[b],
            'bin_high': edges[b + 1],
            'n_units': int(bm.sum()),
            'sample': np.arange(len(t_ms)),
            'time_ms': t_ms,
            'mean': mean_wf,
            'sem': sem,
        }))
quartile_out = pd.concat(rows, ignore_index=True)
quartile_file = os.path.join(out_dir, f'quartile_waveforms_{criteria_name}.csv')
quartile_out.to_csv(quartile_file, index=False)

print(units_out['opto tagged'].value_counts())
print('Units in section 9 figure:', units_out['in_section9_figure'].sum())
print('Quartile features exported:', quartile_out['feature'].nunique())
print('Wrote', units_file)
print('Wrote', quartile_file)

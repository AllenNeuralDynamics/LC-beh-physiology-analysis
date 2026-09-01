# %%

from scipy.optimize import curve_fit
import sys
import os
from matplotlib.colors import LinearSegmentedColormap
sys.path.append('/root/capsule/code/beh_ephys_analysis')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from pynwb import NWBFile, TimeSeries, NWBHDF5IO
from scipy.io import loadmat
from scipy.stats import zscore
from pathlib import Path
import glob
import json
import seaborn as sns
from PyPDF2 import PdfMerger
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import re
from utils.beh_functions import session_dirs, parseSessionID, load_model_dv, makeSessionDF, get_session_tbl, get_unit_tbl, get_history_from_nwb
from utils.ephys_functions import*
from utils.ccf_utils import ccf_pts_convert_to_mm, pir_to_lps, project_to_plane
from utils.combine_tools import apply_qc, to_str_intlike, spatial_dependence_summary, binary_shift_P_vs_U, welch_shift_P_vs_U
from utils.plot_utils import combine_pdf_big
import pickle
import scipy.stats as stats
import spikeinterface as si
import shutil
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import r2_score
import warnings
from scipy.stats import gaussian_kde
import trimesh
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from utils.ccf_utils import ccf_pts_convert_to_mm
from trimesh import load_mesh
from scipy.stats import pearsonr
import statsmodels.api as sm
from aind_ephys_utils import align
import k3d
from scipy.stats import rankdata
from scipy.ndimage import binary_dilation
from skimage.measure import find_contours
from joblib import Parallel, delayed
import seaborn as sns
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')



# %% [markdown]
# ## Load data

# %%
# load basic ephys
target_folder = '/root/capsule/scratch/manuscript/F_basicephys'
target_folder = os.path.join(target_folder, 'acg', 'generation')
if not os.path.exists(target_folder):
    os.makedirs(target_folder, exist_ok=True)
be_folder = os.path.join('/root/capsule/scratch/combined/beh_plots', 'basic_ephys_low')
be_file = os.path.join(be_folder, f'basic_ephys.pkl')
with open(be_file, 'rb') as f:
    basic_ephys_df = pickle.load(f)
filter = basic_ephys_df['be_filter'].values
filter = np.array(filter, dtype=bool)
basic_ephys_df['be_filter'] = filter
basic_ephys_df.rename(columns={'unit': 'unit_id'}, inplace=True)
basic_ephys_df['unit_id'] = basic_ephys_df['unit_id'].apply(to_str_intlike)

# %%
# load and add model variables

beh_folder = os.path.join('/root/capsule/scratch/combined/beh_plots', 'beh_all')

model_combined = pd.read_csv(os.path.join(beh_folder, 'figures_in_generation', 'model_combined_beh_all.csv'), index_col=0)
model_combined['theta'] = model_combined['theta'] - 0.5
model_combined['unit_id'] = model_combined['unit_id'].apply(to_str_intlike)

versions = ['e', 'l', 'com']
for version in versions:
    all_vec = np.column_stack((
        model_combined[f'coef_outcome_{version}_mc'],
        model_combined[f'coef_Qchosen_{version}_ori']
    ))
    theta, rho = np.arctan2(all_vec[:, 1], all_vec[:, 0]), np.hypot(all_vec[:, 1], all_vec[:, 0])
    bound_1, bound_2, bound_3 = -(1 / 4) * np.pi, np.pi, -np.pi
    theta_scaled_dis = np.zeros_like(theta)
    for ind, angle_curr in enumerate(theta):
        if bound_1 < angle_curr <= bound_2:
            theta_scaled_dis[ind] = (angle_curr - bound_1) / (bound_2 - bound_1)
        else:
            theta_scaled_dis[ind] = (bound_1 - angle_curr) / (bound_1 - bound_3)
    theta_scaled_dis_all = 1 - theta_scaled_dis - 0.5
    model_combined[f'theta_{version}'] = theta_scaled_dis_all

# derived features
model_combined['coef_outcome|(|coef_outcome| + |coef_Q|)'] = (
    model_combined['coef_outcome_com_mc'] /
    (np.abs(model_combined['coef_outcome_com_mc']) + np.abs(model_combined['coef_Qchosen_com_mc']))
)
model_combined['outcome_ipsi'] = (
    model_combined['coef_outcome_com_mc'] + model_combined['coef_outcome:ipsi_com_mc']
)
model_combined['outcome_contra'] = (
    model_combined['coef_outcome_com_mc'] - model_combined['coef_outcome:ipsi_com_mc']
)
# combined-beh
# model_combined = model_combined.merge(combined_labeled_beh[['session', 'selected', 'diff_1']], on=['session'], how='left')
# model_combined = model_combined[model_combined['selected']]

# load response
response_tbl = pd.read_csv(f'/root/capsule/scratch/combined/beh_plots/beh_all/response_ratio_beh_all_go_cue.csv')
response_tbl['unit_id'] = response_tbl['unit'].apply(to_str_intlike)

# %%
with open(os.path.join('/root/capsule/scratch/combined/combine_unit_tbl', 'combined_unit_tbl.pkl'), 'rb') as f:
    combined_tagged_units = pickle.load(f)
combined_tagged_units.rename(columns={'unit': 'unit_id'}, inplace=True)
combined_tagged_units['unit_id'] = combined_tagged_units['unit_id'].apply(to_str_intlike)
# antidromic data
version = 'PrL_S1'
antidromic_file = f'/root/capsule/scratch/combined/beh_plots/basic_ephys/{version}/combined_antidromic_results.pkl'
with open(antidromic_file, 'rb') as f:
    antidromic_df = pickle.load(f)

antidromic_df.rename(columns={'unit': 'unit_id'}, inplace=True)
antidromic_df['unit_id'] = antidromic_df['unit_id'].apply(to_str_intlike)
antidromic_df = antidromic_df[['unit_id', 'session', 'p_auto_inhi', 't_auto_inhi',
       'p_collision', 't_collision', 'p_antidromic', 't_antidromic', 'tier_1',
       'tier_2', 'tier_1_long', 'tier_2_long']].copy()
combined_tagged_units = combined_tagged_units.merge(antidromic_df, on=['session', 'unit_id'], how='left')
combined_tagged_units['tier_1'].fillna(False, inplace=True)
combined_tagged_units['tier_2'].fillna(False, inplace=True)
combined_tagged_units['tier_1_long'].fillna(False, inplace=True)
combined_tagged_units['tier_2_long'].fillna(False, inplace=True)


# %%
density = False
criteria_name = 'basic_ephys_all'
with open(os.path.join('/root/capsule/code/beh_ephys_analysis/session_combine/metrics', f'{criteria_name}.json'), 'r') as f:
    constraints = json.load(f)
    
combined_tagged_units_filtered, combined_tagged_units, fig, axes = apply_qc(combined_tagged_units, constraints, density=density, plot_all=False)
combined_tagged_units.rename(columns={'unit': 'unit_id'}, inplace=True)
combined_tagged_units['unit_id'] = combined_tagged_units['unit_id'].apply(to_str_intlike)
combined_tagged_units_filtered.rename(columns={'unit': 'unit_id'}, inplace=True)
combined_tagged_units_filtered['unit_id'] = combined_tagged_units_filtered['unit_id'].apply(to_str_intlike)

fig.savefig(fname=os.path.join(target_folder, f'{criteria_name}_qc.pdf'), bbox_inches='tight', dpi=300)


# %%
features_combined = basic_ephys_df.merge(combined_tagged_units_filtered, on=['session', 'unit_id'], how='inner')
# merge with model variables


# %%
features_combined = features_combined.merge(model_combined, on=['session', 'unit_id'], how='left')
# merge with response
features_combined = features_combined.merge(response_tbl, on=['session', 'unit_id'], how='left')

# %%
# plot all autocorrelograms
filter = features_combined['be_filter'].values
auto_inhi_bin = 0.03
window_length = 3
auto_corr_mat_bl = np.array(features_combined['acg_bl'].values.tolist())[filter][:, 1:]
auto_corr_mat = np.array(features_combined['acg'].values.tolist())[filter][:, 1:]
units = features_combined['unit_id'].values[filter]
sessions = features_combined['session'].values[filter]

x_vals = np.array(range(int(window_length/auto_inhi_bin))) * auto_inhi_bin

# %%
fig = plt.figure(figsize=(5, 3))
plt.plot(x_vals, auto_corr_mat.T, color='k', alpha=0.5, linewidth=0.2);
plt.xlabel('Lag (s)')
fig.savefig(os.path.join(target_folder, f'all_acg_{criteria_name}.pdf'), dpi=300, bbox_inches='tight')

# %%
# All acg denoise acf with PCA


# ---- inputs ----
X = np.asarray(auto_corr_mat, float)   # shape (n_units, n_lags)
x = np.asarray(x_vals, float)          # shape (n_lags,)

n_components = 15
pca = PCA(n_components=n_components, svd_solver="full", random_state=0)

# Fit PCA on ACF matrix
scores = pca.fit_transform(X)          # (n_units, n_components)
explained_variance = pca.explained_variance_ratio_

# ---- Plot cumulative explained variance ----
fig = plt.figure(figsize=(4, 3))
plt.plot(np.arange(1, n_components + 1), np.cumsum(explained_variance), marker="o")
plt.xlabel("Number of PCA components")
plt.ylabel("Cumulative Explained Variance")
plt.tight_layout()
fig.savefig(fname=os.path.join(target_folder, f'pca_explained_variance_all_session.pdf'), dpi=300, bbox_inches='tight')

# ---- Plot each PCA component (loading over lag) ----
n_cols = 3
n_rows = int(np.ceil(n_components / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 2.3 * n_rows), sharex=True)
axes = np.atleast_2d(axes)

for i in range(n_components):
    ax = axes[i // n_cols, i % n_cols]
    ax.plot(x, pca.components_[i], color="k")
    ax.set_title(f"PC{i+1} (EV={explained_variance[i]:.2%})")

# turn off unused axes
for j in range(n_components, n_rows * n_cols):
    axes[j // n_cols, j % n_cols].axis("off")

plt.tight_layout()

fig.savefig(fname=os.path.join(target_folder, f'pca_components_all_session.pdf'), dpi=300, bbox_inches='tight')

# ---- Reconstruct using first k PCs ----
n_components_use = 10  # choose e.g. 9 (change as desired)
k = min(n_components_use, n_components)

# Reconstruction MUST add the mean back
auto_corr_mat_denoised = scores[:, :k] @ pca.components_[:k, :] + pca.mean_
auto_corr_mat_denoised = auto_corr_mat_denoised - np.mean(auto_corr_mat_denoised[:, -3:], axis=1, keepdims=True)  # re-zero at lag 0

# ---- Plot denoised ACF for 9 random examples ----
fig, axes = plt.subplots(5, 3, figsize=(20, 12), sharex=True, sharey=True)
samples = np.random.default_rng(0).choice(X.shape[0], size=15, replace=False)

for i, idx in enumerate(samples):
    ax = axes[i // 3, i % 3]
    ax.plot(x, X[idx], color="k", alpha=0.5, label="Original")
    ax.plot(x, auto_corr_mat_denoised[idx], color="r", lw=1.5, label=f"Denoised (k={k})")
    ax.set_title(f"Unit {idx}")

# show legend once
axes[0, 0].legend()
plt.tight_layout()
fig.savefig(fname=os.path.join(target_folder, f'pca_denoised_acg_examples_all_session.pdf'), dpi=300, bbox_inches='tight')


# %%
# baseline acg denoise acf with PCA
# ---- inputs ----
X = np.asarray(auto_corr_mat_bl, float)   # shape (n_units, n_lags)
x = np.asarray(x_vals, float)          # shape (n_lags,)

n_components = 15
pca = PCA(n_components=n_components, svd_solver="full", random_state=0)

# Fit PCA on ACF matrix
scores = pca.fit_transform(X)          # (n_units, n_components)
explained_variance = pca.explained_variance_ratio_

# ---- Plot cumulative explained variance ----
fig = plt.figure(figsize=(4, 3))
plt.plot(np.arange(1, n_components + 1), np.cumsum(explained_variance), marker="o")
plt.xlabel("Number of PCA components")
plt.ylabel("Cumulative Explained Variance")
plt.tight_layout()
fig.savefig(fname=os.path.join(target_folder, f'pca_explained_variance_bl.pdf'), dpi=300, bbox_inches='tight')

# ---- Plot each PCA component (loading over lag) ----
n_cols = 3
n_rows = int(np.ceil(n_components / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 2.3 * n_rows), sharex=True)
axes = np.atleast_2d(axes)

for i in range(n_components):
    ax = axes[i // n_cols, i % n_cols]
    ax.plot(x, pca.components_[i], color="k")
    ax.set_title(f"PC{i+1} (EV={explained_variance[i]:.2%})")

# turn off unused axes
for j in range(n_components, n_rows * n_cols):
    axes[j // n_cols, j % n_cols].axis("off")

plt.tight_layout()
fig.savefig(fname=os.path.join(target_folder, f'pca_components_bl.pdf'), dpi=300, bbox_inches='tight')

# ---- Reconstruct using first k PCs ----
n_components_use = 10  # choose e.g. 9 (change as desired)
k = min(n_components_use, n_components)

# Reconstruction MUST add the mean back
auto_corr_mat_bl_denoised = scores[:, :k] @ pca.components_[:k, :] + pca.mean_
auto_corr_mat_bl_denoised = auto_corr_mat_bl_denoised - np.mean(auto_corr_mat_bl_denoised[:, -3:], axis=1, keepdims=True)  # re-zero at lag 0
# ---- Plot denoised ACF for 9 random examples ----
fig, axes = plt.subplots(5, 3, figsize=(20, 12), sharex=True, sharey=True)
samples = np.random.default_rng(0).choice(X.shape[0], size=15, replace=False)

for i, idx in enumerate(samples):
    ax = axes[i // 3, i % 3]
    ax.plot(x, X[idx], color="k", alpha=0.5, label="Original")
    ax.plot(x, auto_corr_mat_bl_denoised[idx], color="r", lw=1.5, label=f"Denoised (k={k})")
    ax.set_title(f"Unit {idx}")

# show legend once
axes[0, 0].legend()
plt.tight_layout()
fig.savefig(fname=os.path.join(target_folder, f'pca_denoised_acg_examples_bl.pdf'), dpi=300, bbox_inches='tight')


# %%
# Compare same neuron's baseline and overall acg after cleaning
fig, axes = plt.subplots(5, 3, figsize=(20, 12), sharex=True, sharey=True)
samples = np.random.default_rng(0).choice(X.shape[0], size=15, replace=False) + 5
for i, idx in enumerate(samples):
    ax = axes[i // 3, i % 3]
    ax.plot(x, auto_corr_mat_bl_denoised[idx], color="b", alpha=0.5, label="Baseline ACG (denoised)")
    ax.plot(x, auto_corr_mat_denoised[idx], color="r", alpha=0.5, label="Overall ACG (denoised)")
    ax.set_title(f"Unit {idx}")
# show legend once
axes[0, 0].legend()
plt.tight_layout()
fig.savefig(fname=os.path.join(target_folder, f'pca_denoised_acg_comparison_bl_overall_examples.pdf'), dpi=300, bbox_inches='tight')


# %%
from scipy.optimize import least_squares

def gamma_peak(t, A, peak, k):
    k = np.maximum(k, 1.01)
    tau = peak / (k - 1.0)
    tau = np.maximum(tau, 1e-9)
    x = np.maximum(t, 0.0) / tau
    return A * (x ** (k - 1.0)) * np.exp(-x)

def model(t, theta, K3):
    A1, tau1, A2, peak2, k2, A3, peak3 = theta
    C1 = -A1 * np.exp(-t / tau1)
    C2 =  gamma_peak(t, A2, peak2, k2)
    C3 = -gamma_peak(t, A3, peak3, K3)
    return C1 + C2 + C3

def fit_acf(acf, dt, ftol=1e-12, xtol=1e-12, gtol=1e-12, verbose=0, p0=None):
    """
    Additive model:
      y(t) = C1(t) + C2(t) + C3(t)

      C1: short inhibition (negative exponential)
          C1(t) = -A1 * exp(-t/tau1)

      C2: mid excitation (positive gamma bump) with FREE k2 in [3, 7]
          C2(t) = +A2 * ( (t/tau2)^(k2-1) * exp(-t/tau2) )
          parameterized by peak2 so tau2 = peak2/(k2-1)

      C3: long inhibition (negative gamma bump) with fixed k3
          C3(t) = -A3 * ( (t/tau3)^(k3-1) * exp(-t/tau3) )
          parameterized by peak3 so tau3 = peak3/(k3-1)
    """
    acf = np.asarray(acf, float)
    t = np.arange(len(acf)) * dt

    # ---------- Model ----------
    K3 = 4.0



    # ---------- Residual ----------
    def residual(theta, k3 = K3):
        return model(t, theta, k3) - acf

    # ---------- Initial guesses ----------
    if p0 is None:
        
        p0 = np.array([
            acf[0],                 # A1
            0.015,                                  # tau1
            float(max(np.max(acf[t <= 0.25] - acf[-1]), 0.0)),  # A2
            0.10,                                  # peak2
            6.0,                                   # k2
            float(max(-np.min(acf[t >= 0.2]), 0.0)),# A3
            0.70                                   # peak3
        ], dtype=float)

    print("Initial parameter guesses:", p0)

    # ---------- Bounds ----------
    # [A1, tau1, A2, peak2,  k2,  A3, peak3]
    lower = [-1.0, 0.01,   -1.0, dt,  2.0, 0.0, 0.4]
    upper = [1.0, 0.20, 1.0, 0.4, 8, 0.2, 1.20]

    # ---------- Least Squares ----------
    res = least_squares(
        residual,
        p0,
        bounds=(lower, upper),
        method="trf",
        x_scale="jac",   # automatic scaling
        ftol=ftol,
        xtol=xtol,
        gtol=gtol,
        max_nfev=10000,
        verbose=verbose
    )

    theta = res.x
    y_fit = model(t, theta, K3)

    # ---------- GOF ----------
    y = acf
    resid = y - y_fit
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - np.mean(y))**2))

    r2 = np.nan if ss_tot == 0 else (1.0 - ss_res / ss_tot)
    n = int(len(y))
    p = int(len(theta))
    adj_r2 = np.nan if (ss_tot == 0 or n <= p + 1) else (1.0 - (1.0 - r2) * (n - 1) / (n - p - 1))
    rmse = float(np.sqrt(ss_res / n)) if n > 0 else np.nan

    # Gaussian-residual AIC/BIC (up to an additive constant)
    eps = 1e-300
    aic = n * np.log(ss_res / max(n, 1) + eps) + 2 * p
    bic = n * np.log(ss_res / max(n, 1) + eps) + p * np.log(max(n, 1))

    # ---------- Plot ----------
    fig, axes = plt.subplots(5, 1, figsize=(12, 12), sharex=True)

    axes[0].plot(t, acf, 'o', label="ACF")
    axes[0].plot(t, y_fit, label="Fit")
    axes[0].axhline(0, color='gray', lw=1, ls='--')
    axes[0].legend(loc="lower right")
    axes[0].set_title(f"Fit (R²={r2:.3f}, RMSE={rmse:.4g})")

    axes[1].plot(t, y_fit - acf)
    axes[1].axhline(0, color='gray', lw=1, ls='--')
    axes[1].set_title("Residual (fit - data)")

    A1, tau1, A2, peak2, k2, A3, peak3 = theta
    tau2 = peak2 / (max(k2 - 1.0, 1e-6))
    tau3 = peak3 / (K3 - 1.0)

    C1 = -A1 * np.exp(-t / tau1)
    C2 =  gamma_peak(t, A2, peak2, k2)
    C3 = -gamma_peak(t, A3, peak3, K3)

    axes[2].plot(t, C1)
    axes[2].axhline(0, color='gray', lw=1, ls='--')
    axes[2].set_title(f"Short inhibition C1: tau1={tau1:.3f}s, A1={A1:.3f}")

    axes[3].plot(t, C2)
    axes[3].axhline(0, color='gray', lw=1, ls='--')
    axes[3].set_title(f"Mid excitation C2: peak2={peak2:.3f}s, tau2≈{tau2:.3f}s, k2={k2:.2f}, A2={A2:.3f}")

    axes[4].plot(t, C3)
    axes[4].axhline(0, color='gray', lw=1, ls='--')
    axes[4].set_title(f"Long inhibition C3: peak3={peak3:.3f}s, tau3≈{tau3:.3f}s, k3={K3:g}, A3={A3:.3f}")

    plt.tight_layout()

    outcome = {
        # parameters
        "A1": float(A1), "tau1": float(tau1),
        "A2": float(A2), "peak2": float(peak2), "k2": float(k2), "tau2": float(tau2),
        "A3": float(A3), "peak3": float(peak3), "k3": float(K3), "tau3": float(tau3),
        "P1": C1[0], 
        "P2": np.max(C2) if A2 > 0 else np.min(C2),
        "P3": np.min(C3) if A3 > 0 else np.max(C3),

        # fit diagnostics
        "success": bool(res.success),
        "status": int(res.status),
        "message": str(res.message),
        "nfev": int(res.nfev),
        "cost": float(res.cost),

        # GOF
        "ss_res": ss_res,
        "ss_tot": ss_tot,
        "r2": float(r2) if np.isfinite(r2) else np.nan,
        "adj_r2": float(adj_r2) if np.isfinite(adj_r2) else np.nan,
        "rmse": rmse,
        "aic": float(aic),
        "bic": float(bic),
    }

    return outcome, fig, axes


# %%
# In parallel loop through all neurons
fit_folder = os.path.join(target_folder, 'acf_fits')
if not os.path.exists(fit_folder):
    os.makedirs(fit_folder)
dt = 0.03
fit_sample = (0, int(np.round(1.5/dt)))

# fit in parallel

def _fit_one_unit(unit_ind, unit_id, session_id,
                  auto_corr_mat_denoised, auto_corr_mat,
                  fit_sample, dt, x, fit_folder):
    print(f"Fitting unit {unit_id} from session {session_id} (index {unit_ind})...")

    params, fig, axes = fit_acf(
        auto_corr_mat_denoised[unit_ind][fit_sample[0]:fit_sample[1]],
        dt,
        ftol=10**-17,
        xtol=10**-50,
        verbose=1
    )

    # overlay raw ACF on the first axis returned by fit_acf
    axes[0].plot(
        x[fit_sample[0]:fit_sample[1]],
        auto_corr_mat[unit_ind][fit_sample[0]:fit_sample[1]],
        'o', alpha=0.6, label="Raw ACF"
    )

    fig.suptitle(f"Unit {unit_id} from session {session_id}")
    outpath = os.path.join(fit_folder, f"session_{session_id} unit_{unit_id}_acf_fit.pdf")
    fig.savefig(outpath, dpi=300, bbox_inches="tight")

    plt.close(fig)          # close only this figure
    params['unit_id'] = unit_id
    params['session_id'] = session_id
    return params

# Run in parallel (processes)
all_params = Parallel(n_jobs=-1, prefer="processes", verbose=10)(
    delayed(_fit_one_unit)(
        unit_ind, unit_id, session_id,
        auto_corr_mat_denoised, auto_corr_mat,
        fit_sample, dt, x, fit_folder
    )
    for unit_ind, (unit_id, session_id) in enumerate(zip(units, sessions))
)


# %%
# # combine single pdfs into a big one
# combine_pdf_big(fit_folder, os.path.join(target_folder, 'combined_fitting.pdf'))

# %%
unit_session_included = features_combined[features_combined['be_filter']][['unit_id','session']]
all_params_df = pd.DataFrame(all_params)
# reorder rows so that unit and session for each row matches features_combined
all_params_df = all_params_df.merge(unit_session_included[['session', 'unit_id']], left_on=['session_id', 'unit_id'], right_on=['session', 'unit_id'], how='right')
# %%
# for each set of parameters, generate C1 C2 C3 and put in one matrix
K3 = 4.0
dt = 0.03
fit_sample = (0, int(np.round(1.5/dt)))
t = np.arange(fit_sample[1] - fit_sample[0]) * dt
t = np.arange(1, np.shape(auto_corr_mat_denoised)[1]+1) * dt
C1_mat = []
C2_mat = []
C3_mat = []
for idx, row in all_params_df.iterrows():
    A1 = row['A1']
    tau1 = row['tau1']
    A2 = row['A2']
    peak2 = row['peak2']
    k2 = row['k2']
    A3 = row['A3']
    peak3 = row['peak3']

    tau2 = peak2 / max(k2 - 1.0, 1e-6)
    tau3 = peak3 / (K3 - 1.0)

    C1 = -A1 * np.exp(-t / tau1)
    C2 =  gamma_peak(t, A2, peak2, k2)
    C3 = -gamma_peak(t, A3, peak3, K3)

    C1_mat.append(C1)
    C2_mat.append(C2)
    C3_mat.append(C3)

C1_mat = np.array(C1_mat)
C2_mat = np.array(C2_mat)
C3_mat = np.array(C3_mat)


# %%
g = sns.pairplot(all_params_df[['A1', 'tau1', 'A2', 'peak2', 'k2', 'A3', 'peak3', 'r2', 'adj_r2', 'P1', 'P2', 'P3']])
g.fig.savefig(os.path.join(target_folder, 'acf_fit_parameters_pairplot.pdf'), dpi=300, bbox_inches='tight')

# %%
# mean acg, C1, C2 and C3 by each parameter quartiles
features_to_plot = ['A1', 'tau1', 'A2', 'peak2', 'k2', 'A3', 'peak3', 'P1', 'P2', 'P3']
fig, axes = plt.subplots(4, len(features_to_plot), figsize=(5 * len(features_to_plot), 4*4), sharey=True)
bin_num = 4
cmap = cm.get_cmap("coolwarm", bin_num)
x_vals = np.arange(1, auto_corr_mat_denoised.shape[1]+1) * dt
for i, feature in enumerate(features_to_plot):
    curr_quantile = np.quantile(all_params_df[feature], q=np.linspace(0, 1, bin_num + 1))

    for q in range(4):
        filter = (all_params_df[feature] >= curr_quantile[q]) & (all_params_df[feature] < curr_quantile[q+1])
        # whole trace
        mean_acf = np.mean(auto_corr_mat_denoised[filter], axis=0)
        sem_acf = np.std(auto_corr_mat_denoised[filter], axis=0) / np.sqrt(np.sum(filter))
        ax = axes[0, i]
        ax.plot(x_vals, mean_acf, label=f"Q{q+1}", color=cmap(q / (bin_num - 1)))
        ax.fill_between(x_vals, mean_acf - sem_acf, mean_acf + sem_acf, alpha=0.3, facecolor = cmap(q / (bin_num - 1)), edgecolor=None)
        ax.set_xlim(-0.1, 1.2)
        if i == 0:
            ax.legend(loc='upper right')
            ax.set_ylabel('Denoised ACF')
        # C1
        mean_C1 = np.mean(C1_mat[filter], axis=0)
        sem_C1 = np.std(C1_mat[filter], axis=0) / np.sqrt(np.sum(filter))
        ax = axes[1, i]
        ax.plot(x_vals, mean_C1, label=f"Q{q+1}", color=cmap(q / (bin_num - 1)))
        ax.fill_between(x_vals, mean_C1 - sem_C1, mean_C1 + sem_C1, alpha=0.3, facecolor = cmap(q / (bin_num - 1)), edgecolor=None)
        ax.set_xlim(-0.1, 1.2)
        if i == 0:
            ax.set_ylabel('C1 (short inhibition)')
        # C2
        mean_C2 = np.mean(C2_mat[filter], axis=0)
        sem_C2 = np.std(C2_mat[filter], axis=0) / np.sqrt(np.sum(filter))
        ax = axes[2, i]
        ax.plot(x_vals, mean_C2, label=f"Q{q+1}", color=cmap(q / (bin_num - 1)))
        ax.fill_between(x_vals, mean_C2 - sem_C2, mean_C2 + sem_C2, alpha=0.3, facecolor = cmap(q / (bin_num - 1)), edgecolor=None)
        ax.set_xlim(-0.1, 1.2)
        if i == 0:
            ax.set_ylabel('C2 (mid excitation)')
        # C3
        mean_C3 = np.mean(C3_mat[filter], axis=0)
        sem_C3 = np.std(C3_mat[filter], axis=0) / np.sqrt(np.sum(filter))
        ax = axes[3, i]
        ax.plot(x_vals, mean_C3, label=f"Q{q+1}", color=cmap(q / (bin_num - 1)))
        ax.fill_between(x_vals, mean_C3 - sem_C3, mean_C3 + sem_C3, alpha=0.3, facecolor = cmap(q / (bin_num - 1)), edgecolor=None)
        ax.set_xlim(-0.1, 1.2)
        if i == 0:
            ax.set_ylabel('C3 (long inhibition)')


    ax.set_title(feature)

    ax.set_xlabel('Lag (s)')

# %%
# pca on parameters
param_cols = ['P1', 'tau1', 'P2', 'peak2', 'k2', 'P3', 'peak3']
pca = PCA(n_components=5)
mask = all_params_df['A1'] > 0
param_data = all_params_df[param_cols].values[mask]
# standardize parameters before PCA
param_data = (param_data - np.mean(param_data, axis=0)) / np.std(param_data, axis=0)
pca_scores = pca.fit_transform(param_data)
explained_variance = pca.explained_variance_ratio_
# Plot PCA results
fig = plt.figure(figsize=(6, 4))
plt.plot(np.arange(1, len(explained_variance) + 1), np.cumsum(explained_variance), marker='o')
plt.xlabel('Number of PCA components')
plt.ylabel('Cumulative Explained Variance')
plt.ylim(0, 1.05)
plt.title('PCA on ACF Fit Parameters')
plt.tight_layout()
fig.savefig(fname=os.path.join(target_folder, 'acf_fit_parameters_pca_explained_variance.pdf'), dpi=300, bbox_inches='tight')
fig = plt.figure(figsize=(12, 10))
# colorcode by r2
color_code = all_params_df['adj_r2'].values[mask]
sort_ind = np.argsort(color_code)[::-1]
for i in range(5):
    plt.subplot(3, 2, i + 1)
    sg = plt.scatter(pca_scores[:, i][sort_ind], pca_scores[:, (i + 1) % 5][sort_ind], alpha=0.7, c = range(len(color_code)), cmap='viridis')
    plt.xlabel(f'PC{i+1} ({explained_variance[i]:.2%} EV)')
    plt.ylabel(f'PC{((i + 1) % 5) + 1} ({explained_variance[(i + 1) % 5]:.2%} EV)')
    plt.title(f'PC{i+1} vs PC{((i + 1) % 5) + 1}')
    if i == 0:
        plt.colorbar(sg, label='R² of fit')

plt.tight_layout()
fig.savefig(os.path.join(target_folder, 'acf_fit_parameters_pca_colored_r2.pdf'), dpi=300, bbox_inches='tight')




# %%
# append pcs to parameters dataframe
for i in range(pca_scores.shape[1]):
    all_params_df[f'PC{i+1}'] = np.nan
all_params_df.loc[mask, [f'PC{i+1}' for i in range(pca_scores.shape[1])]] = pca_scores

# %%
# mean acg, C1, C2 and C3 by each parameter quartiles
fig, axes = plt.subplots(4, pca_scores.shape[1], figsize=(5 * pca_scores.shape[1], 4*4), sharey=True)
bin_num = 4
cmap = cm.get_cmap("coolwarm", bin_num)
x_vals = np.arange(1, auto_corr_mat_denoised.shape[1]+1) * dt
for i in range(pca_scores.shape[1]):
    curr_quantile = np.quantile(pca_scores[:, i], q=np.linspace(0, 1, bin_num + 1))
    for q in range(4):
        filter = (pca_scores[:, i] >= curr_quantile[q]) & (pca_scores[:, i] < curr_quantile[q+1])
        # whole trace
        mean_acf = np.mean(auto_corr_mat_denoised[mask][filter], axis=0)
        sem_acf = np.std(auto_corr_mat_denoised[mask][filter], axis=0) / np.sqrt(np.sum(filter))
        ax = axes[0, i]
        ax.plot(x_vals, mean_acf, label=f"Q{q+1}", color=cmap(q / (bin_num - 1)))
        ax.fill_between(x_vals, mean_acf - sem_acf, mean_acf + sem_acf, alpha=0.3, facecolor = cmap(q / (bin_num - 1)), edgecolor=None)
        ax.set_xlim(-0.1, 1.2)
        if i == 0:
            ax.legend(loc='upper right')
            ax.set_ylabel('Denoised ACF')
        ax.set_title(f'PC{i+1} (EV={explained_variance[i]:.2%})')
        # C1
        mean_C1 = np.mean(C1_mat[mask][filter], axis=0)
        sem_C1 = np.std(C1_mat[mask][filter], axis=0) / np.sqrt(np.sum(filter))
        ax = axes[1, i]
        ax.plot(x_vals, mean_C1, label=f"Q{q+1}", color=cmap(q / (bin_num - 1)))
        ax.fill_between(x_vals, mean_C1 - sem_C1, mean_C1 + sem_C1, alpha=0.3, facecolor = cmap(q / (bin_num - 1)), edgecolor=None)
        ax.set_xlim(-0.1, 1.2)
        if i == 0:
            ax.set_ylabel('C1 (short inhibition)')
        # C2
        mean_C2 = np.mean(C2_mat[mask][filter], axis=0)
        sem_C2 = np.std(C2_mat[mask][filter], axis=0) / np.sqrt(np.sum(filter))
        ax = axes[2, i]
        ax.plot(x_vals, mean_C2, label=f"Q{q+1}", color=cmap(q / (bin_num - 1)))
        ax.fill_between(x_vals, mean_C2 - sem_C2, mean_C2 + sem_C2, alpha=0.3, facecolor = cmap(q / (bin_num - 1)), edgecolor=None)
        ax.set_xlim(-0.1, 1.2)
        if i == 0:
            ax.set_ylabel('C2 (mid excitation)')
        # C3
        mean_C3 = np.mean(C3_mat[mask][filter], axis=0)
        sem_C3 = np.std(C3_mat[mask][filter], axis=0) / np.sqrt(np.sum(filter))
        ax = axes[3, i]
        ax.plot(x_vals, mean_C3, label=f"Q{q+1}", color=cmap(q / (bin_num - 1)))
        ax.fill_between(x_vals, mean_C3 - sem_C3, mean_C3 + sem_C3, alpha=0.3, facecolor = cmap(q / (bin_num - 1)), edgecolor=None)
        ax.set_xlim(-0.1, 1.2)
        if i == 0:
            ax.set_ylabel('C3 (long inhibition)')

    ax.set_xlabel('Lag (s)')

fig.savefig(os.path.join(target_folder, 'acf_components_fit_parameters_pca_quartiles.pdf'), dpi=300, bbox_inches='tight')

# %%
# save data:
# %%
save_file = os.path.join(target_folder, 'acf_fit_parameters_and_components.pkl')
with open(save_file, 'wb') as f:
    pickle.dump({
        'params_df': all_params_df,
        'C1_mat': C1_mat,
        'C2_mat': C2_mat,
        'C3_mat': C3_mat,
        'acg_mat': auto_corr_mat_denoised,
    }, f)




# %%
# PSTH and outcome AUC aligned to go cue vs choice/reward, same DRN units in the same order.
# Shows that the all-trial PSTH (dominated by the go-cue transient, shared by every trial) and the
# outcome AUC (rewarded vs unrewarded difference, emerging after the choice) are different views
# of the same neurons.
# run: python psth_auc_cue_vs_outcome_DRN.py
import sys
import os
sys.path.append('/root/capsule/code/beh_ephys_analysis')
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import mannwhitneyu
from joblib import Parallel, delayed
from utils.beh_functions import makeSessionDF, get_unit_tbl
from utils.ephys_functions import load_drift
from utils.combine_tools import to_str_intlike
from utils.capsule_migration import capsule_directories

capsule_dirs = capsule_directories()
criteria_name = 'beh_all_DRN'
model_name = 'stan_qLearning_5params'
data_type = 'raw'
bin_size = 0.02      # s
smooth_sd = 0.04     # s, gaussian smoothing of the PSTH
pre, post = -1.0, 5.0
bl_win = (-1.0, 0.0)  # pre-cue baseline used to z-score every unit, for both alignments
sort_feature = 'T_outcome_com_mc'
target_folder = os.path.join(capsule_dirs['manuscript_fig_dir'], f'F_ephys_beh_action&outcome_DRN_beh_none')
os.makedirs(target_folder, exist_ok=True)

model_combined = pd.read_csv(os.path.join(capsule_dirs['manuscript_fig_prep_dir'], 'outcome_regressions',
                                          f'model_combined_{criteria_name}.csv'), index_col=0)
model_combined['unit_id'] = model_combined['unit_id'].apply(to_str_intlike)
edges = np.arange(pre, post + bin_size / 2, bin_size)
time = edges[:-1] + bin_size / 2


def trial_rates(spike_times, align_times):
    """trials x time firing rate (Hz), gaussian smoothed"""
    counts = np.array([np.histogram(spike_times - t, bins=edges)[0] for t in align_times], dtype=float)
    return gaussian_filter1d(counts / bin_size, smooth_sd / bin_size, axis=1)


def process_session(session, unit_ids):
    unit_tbl = get_unit_tbl(session, data_type)
    session_df = makeSessionDF(session, model_name=model_name)   # responded trials only
    out = []
    for unit_id in unit_ids:
        key = float(unit_id) if unit_id.isdigit() else unit_id
        spike_times = unit_tbl.query('unit_id == @key')['spike_times'].values[0]
        df = session_df.copy()
        drift = load_drift(session, key, data_type=data_type)
        if drift is not None:
            t0, t1 = drift['ephys_cut']
            if t0 is not None:
                spike_times = spike_times[spike_times >= t0]
                df = df[df['go_cue_time'] >= t0]
            if t1 is not None:
                spike_times = spike_times[spike_times <= t1]
                df = df[df['go_cue_time'] <= t1]
        rewarded = df['outcome'].values.astype(bool)
        res = {'session': session, 'unit_id': unit_id}
        cue = trial_rates(spike_times, df['go_cue_time'].values)
        bl_mask = (time >= bl_win[0]) & (time < bl_win[1])
        bl = cue[:, bl_mask].mean(axis=1)
        bl_mean, bl_sd = bl.mean(), bl.std() + 1e-9
        for name, col in [('cue', 'go_cue_time'), ('choice', 'choice_time')]:
            rates = cue if name == 'cue' else trial_rates(spike_times, df[col].values)
            z = (rates - bl_mean) / bl_sd
            res[f'{name}_all'] = z.mean(axis=0)
            res[f'{name}_rew'] = z[rewarded].mean(axis=0)
            res[f'{name}_unrew'] = z[~rewarded].mean(axis=0)
            u, p = mannwhitneyu(rates[rewarded], rates[~rewarded], axis=0)
            res[f'{name}_auc'] = u / (rewarded.sum() * (~rewarded).sum())
            res[f'{name}_p'] = p
        out.append(res)
    return out


groups = list(model_combined.groupby('session')['unit_id'])
results = Parallel(n_jobs=8)(delayed(process_session)(s, list(u)) for s, u in groups)
res = pd.DataFrame([r for g in results for r in g])
res = res.merge(model_combined[['session', 'unit_id', sort_feature]], on=['session', 'unit_id'])
res = res.sort_values(sort_feature).reset_index(drop=True)   # most negative outcome t at the bottom
n_units = len(res)


# %%
def stack(col):
    return np.vstack(res[col].values)


fig, axes = plt.subplots(2, 4, figsize=(20, 9), gridspec_kw={'width_ratios': [1, 1, 1, 1.1]})
for row, (name, xlabel) in enumerate([('cue', 'Time from go cue (s)'), ('choice', 'Time from choice = reward (s)')]):
    all_z = stack(f'{name}_all')
    diff_z = stack(f'{name}_rew') - stack(f'{name}_unrew')
    auc = stack(f'{name}_auc')
    extent = [time[0], time[-1], 0, n_units]
    lim = np.nanquantile(np.abs(all_z), 0.98)
    panels = [
        (all_z, 'RdBu_r', -lim, lim, 'All trials: z-scored PSTH', 'z (vs pre-cue baseline)'),
        (diff_z, 'RdBu_r', -lim, lim, 'Rewarded − unrewarded PSTH', 'Δz'),
        (auc, 'PiYG', 0, 1, 'Outcome AUC (rewarded vs unrewarded)', 'AUC'),
    ]
    for col, (mat, cmap, vmin, vmax, title, cbar_label) in enumerate(panels):
        ax = axes[row, col]
        im = ax.imshow(mat, aspect='auto', origin='lower', extent=extent, cmap=cmap, vmin=vmin, vmax=vmax,
                       interpolation='none')
        fig.colorbar(im, ax=ax, label=cbar_label)
        ax.axvline(0, color='k', linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f'Units (sorted by {sort_feature})' if col == 0 else '')

    ax = axes[row, 3]
    for col_name, color, label in [(f'{name}_rew', 'tab:blue', 'rewarded'), (f'{name}_unrew', 'tab:red', 'unrewarded')]:
        mat = stack(col_name)
        m, se = mat.mean(axis=0), mat.std(axis=0) / np.sqrt(n_units)
        ax.plot(time, m, color=color, label=f'{label} trials')
        ax.fill_between(time, m - se, m + se, color=color, alpha=0.25, linewidth=0)
    ax.axvline(0, color='k', linewidth=0.8)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Population mean z (± SEM)')
    ax.legend(frameon=False, fontsize=9, loc='upper right')
    ax.set_title('Population average')

fig.suptitle(f'DRN units (n={n_units}, {criteria_name}), responded trials. Same units and order in every heatmap. '
             f'Choice is a median 0.15 s after the go cue; reward is delivered at choice.', fontsize=12)
plt.tight_layout()
for ext in ['pdf', 'png']:
    fig.savefig(os.path.join(target_folder, f'psth_auc_cue_vs_choice_aligned_{criteria_name}.{ext}'),
                bbox_inches='tight', dpi=150)
print('saved to', target_folder)

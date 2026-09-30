# Parallel outcome-window generation over (session, unit) pairs
# Based on `outcome_window_generation.py`

import argparse
import json
import os
import pickle
import sys
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_auc_score

sys.path.append('/root/capsule/code/beh_ephys_analysis')

from utils.beh_functions import makeSessionDF, get_session_tbl, get_unit_tbl, session_dirs
from utils.capsule_migration import capsule_directories
from utils.combine_tools import apply_qc
from utils.ephys_functions import get_spike_matrix, load_drift, load_trial_drift

warnings.filterwarnings('ignore')


def _expected_slide_times(pre_event, post_event, step_size):
    return np.arange(pre_event, post_event, step_size) - 0.5 * step_size


def _safe_roc_auc(focus, spike_counts):
    focus = np.asarray(focus)
    spike_counts = np.asarray(spike_counts)
    valid = np.isfinite(focus) & np.isfinite(spike_counts)
    focus = focus[valid]
    spike_counts = spike_counts[valid]

    if focus.size < 5 or np.unique(focus).size < 2:
        return np.nan

    try:
        return roc_auc_score(focus.astype(int), spike_counts)
    except Exception:
        return np.nan


def _nan_result(session, unit_id, slide_times_auc, labels, error=None):
    return {
        'session': session,
        'unit': unit_id,
        'auc': np.full((len(slide_times_auc), len(labels)), np.nan, dtype=float),
        'auc_max': np.full(len(labels), np.nan, dtype=float),
        'auc_max_ind': np.full(len(labels), np.nan, dtype=float),
        'error': error,
    }


def process_session_unit_pair(
    row,
    pre_event,
    post_event,
    bin_size=1.5,
    step_size=0.1,
    labels=('outcome', 'hit', 'svs'),
    align='go_cue_time',
    data_type='curated',
    model_name='stan_qLearning_5params',
):
    session = row['session']
    unit_id = row['unit']
    rec_side = row.get('rec_side', np.nan)
    slide_times_auc = _expected_slide_times(pre_event, post_event, step_size)

    try:
        unit_tbl = get_unit_tbl(session, data_type)
        whole_session_tbl = get_session_tbl(session)
        whole_session_tbl['hit'] = whole_session_tbl['animal_response'].values == 1

        session_df = makeSessionDF(session, model_name=model_name)
        if 'choice' in session_df.columns and pd.notna(rec_side):
            session_df['ipsi'] = 2 * (session_df['choice'].values - 0.5) * rec_side

        drift_data = load_trial_drift(session, data_type)
        if drift_data is not None:
            try:
                _ = drift_data.load_unit(unit_id)
            except Exception:
                pass

        unit_rows = unit_tbl.query('unit_id == @unit_id')
        if unit_rows.empty:
            raise ValueError(f'Unit not found in unit table: {unit_id}')

        spike_times = unit_rows['spike_times'].values[0]
        session_df_curr = session_df.copy()
        whole_session_df_curr = whole_session_tbl.copy()
        spike_times_curr = spike_times.copy()

        unit_drift = load_drift(session, unit_id, data_type=data_type)
        if unit_drift is not None:
            if unit_drift['ephys_cut'][0] is not None:
                spike_times_curr = spike_times_curr[spike_times_curr >= unit_drift['ephys_cut'][0]]
                session_df_curr = session_df_curr[session_df_curr['go_cue_time'] >= unit_drift['ephys_cut'][0]]
                whole_session_df_curr = whole_session_df_curr[whole_session_df_curr['goCue_start_time'] >= unit_drift['ephys_cut'][0]]
            if unit_drift['ephys_cut'][1] is not None:
                spike_times_curr = spike_times_curr[spike_times_curr <= unit_drift['ephys_cut'][1]]
                session_df_curr = session_df_curr[session_df_curr['go_cue_time'] <= unit_drift['ephys_cut'][1]]
                whole_session_df_curr = whole_session_df_curr[whole_session_df_curr['goCue_start_time'] <= unit_drift['ephys_cut'][1]]

        if len(session_df_curr) < 5:
            return _nan_result(session, unit_id, slide_times_auc, labels)

        if align not in session_df_curr.columns:
            raise KeyError(f'Missing align column in session_df: {align}')

        align_time = session_df_curr[align].values
        spike_matrix_auc, slide_times_auc = get_spike_matrix(
            spike_times_curr,
            align_time,
            pre_event=pre_event,
            post_event=post_event,
            binSize=bin_size,
            stepSize=step_size,
            kernel=False,
            tau_rise=0.001,
            tau_decay=0.08,
        )

        if align == 'go_cue_time':
            all_align_time = whole_session_df_curr['goCue_start_time'].values
        elif align == 'outcome_time':
            all_align_time = whole_session_df_curr['reward_outcome_time'].values
        else:
            all_align_time = whole_session_df_curr[align].values

        spike_matrix_auc_all, _ = get_spike_matrix(
            spike_times_curr,
            all_align_time,
            pre_event=pre_event,
            post_event=post_event,
            binSize=bin_size,
            stepSize=step_size,
            kernel=False,
            tau_rise=0.001,
            tau_decay=0.08,
        )

        curr_auc = np.full((len(slide_times_auc), len(labels)), np.nan, dtype=float)
        for time_ind in range(len(slide_times_auc)):
            spike_counts = spike_matrix_auc[:, time_ind]
            spike_counts_all = spike_matrix_auc_all[:, time_ind]

            for label_ind, label in enumerate(labels):
                if label == 'hit':
                    if label in whole_session_df_curr.columns:
                        curr_auc[time_ind, label_ind] = _safe_roc_auc(
                            whole_session_df_curr[label].values,
                            spike_counts_all,
                        )
                else:
                    if label in session_df_curr.columns:
                        curr_auc[time_ind, label_ind] = _safe_roc_auc(
                            session_df_curr[label].values,
                            spike_counts,
                        )

        curr_max = np.full(len(labels), np.nan, dtype=float)
        curr_max_ind = np.full(len(labels), np.nan, dtype=float)
        for label_ind in range(len(labels)):
            curr_vals = np.abs(curr_auc[:, label_ind] - 0.5)
            if np.all(np.isnan(curr_vals)):
                continue
            best_idx = int(np.nanargmax(curr_vals))
            curr_max_ind[label_ind] = best_idx
            curr_max[label_ind] = curr_auc[best_idx, label_ind]

        return {
            'session': session,
            'unit': unit_id,
            'auc': curr_auc,
            'auc_max': curr_max,
            'auc_max_ind': curr_max_ind,
            'error': None,
        }

    except Exception as exc:
        print(f'[Error] session {session}, unit {unit_id}: {exc}')
        return _nan_result(session, unit_id, slide_times_auc, labels, error=str(exc))


def _estimate_mode(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return np.nan, None, None
    if values.size == 1 or np.allclose(values, values[0]):
        return float(values[0]), np.array([values[0]]), np.array([1.0])

    kde = gaussian_kde(values)
    x_grid = np.linspace(values.min(), values.max(), 1000)
    pdf_values = kde(x_grid)
    mode = x_grid[np.argmax(pdf_values)]
    return float(mode), x_grid, pdf_values


def compute_outcome_window_parallel(criteria_name, pre_event, post_event, n_jobs=-4):
    version = 'PrL_S1'
    capsule_dirs = capsule_directories()
    metrics_folder = os.path.join(capsule_dirs['manuscript_fig_prep_dir'], 'outcome_window', criteria_name)
    os.makedirs(metrics_folder, exist_ok=True)

    with open(os.path.join(capsule_dirs['manuscript_fig_prep_dir'], 'combined_unit_tbl', 'combined_unit_tbl.pkl'), 'rb') as f:
        combined_tagged_units = pickle.load(f)
    with open(os.path.join(capsule_dirs['manuscript_fig_prep_dir'], 'combined_session_tbl', 'combined_beh_sessions.pkl'), 'rb') as f:
        combined_session_qc = pickle.load(f)

    antidromic_file = os.path.join(
        capsule_dirs['manuscript_fig_prep_dir'],
        'antidromic_analysis',
        version,
        'combined_antidromic_results.pkl',
    )
    with open(antidromic_file, 'rb') as f:
        antidromic_df = pickle.load(f)

    antidromic_df = antidromic_df[[
        'unit', 'session', 'p_auto_inhi', 't_auto_inhi',
        'p_collision', 't_collision', 'p_antidromic', 't_antidromic',
        'tier_1', 'tier_2', 'tier_1_long', 'tier_2_long'
    ]].copy()

    combined_tagged_units = combined_tagged_units.merge(antidromic_df, on=['session', 'unit'], how='left')
    combined_tagged_units['tier_1'].fillna(False, inplace=True)
    combined_tagged_units['tier_2'].fillna(False, inplace=True)
    combined_tagged_units['tier_1_long'].fillna(False, inplace=True)
    combined_tagged_units['tier_2_long'].fillna(False, inplace=True)
    combined_tagged_units.drop(columns=['probe'], inplace=True, errors='ignore')
    combined_tagged_units = combined_tagged_units.merge(combined_session_qc, on='session', how='left')

    constraint_file = os.path.join('/root/capsule/code/beh_ephys_analysis/session_combine/metrics', f'{criteria_name}.json')
    with open(constraint_file, 'r') as f:
        constraints = json.load(f)

    combined_tagged_units_filtered, _, _, _ = apply_qc(combined_tagged_units, constraints)
    if combined_tagged_units_filtered.empty:
        raise ValueError(f'No units passed QC for {criteria_name}')

    bin_size = 1.5
    step_size = 0.1
    labels = ['outcome', 'hit', 'svs']
    align = 'go_cue_time'
    data_type = 'raw'
    model_name = 'stan_qLearning_5params'
    slide_times_auc = _expected_slide_times(pre_event, post_event, step_size)

    pair_records = combined_tagged_units_filtered[['session', 'unit', 'rec_side']].to_dict('records')
    print(f'Processing {len(pair_records)} session-unit pairs with n_jobs={n_jobs}...')

    results = Parallel(n_jobs=n_jobs, backend='loky', verbose=10)(
        delayed(process_session_unit_pair)(
            row,
            pre_event=pre_event,
            post_event=post_event,
            bin_size=bin_size,
            step_size=step_size,
            labels=labels,
            align=align,
            data_type=data_type,
            model_name=model_name,
        )
        for row in pair_records
    )

    error_rows = [
        {'session': r['session'], 'unit': r['unit'], 'error': r['error']}
        for r in results if r['error'] is not None
    ]
    if error_rows:
        pd.DataFrame(error_rows).to_csv(
            os.path.join(metrics_folder, f'auc_parallel_errors_{criteria_name}.csv'),
            index=False,
        )

    auc_mat = np.stack([r['auc'] for r in results], axis=0)
    auc_max = np.vstack([r['auc_max'] for r in results])
    auc_max_ind = np.vstack([r['auc_max_ind'] for r in results])

    reward_colors = LinearSegmentedColormap.from_list('outcome', [(0.0, 'red'), (0.5, 'white'), (1.0, 'blue')])
    hit_colors = LinearSegmentedColormap.from_list('hit', [(0.0, 'blue'), (0.5, 'white'), (1.0, 'orange')])
    switch_colors = LinearSegmentedColormap.from_list('switch', [(0.0, 'green'), (0.5, 'white'), (1.0, 'purple')])
    feature_map = {'outcome': reward_colors, 'hit': hit_colors, 'svs': switch_colors}

    for label_ind, label in enumerate(labels):
        fig = plt.figure(figsize=(10, 6))
        sort_ind = np.argsort(np.nan_to_num(auc_max[:, label_ind], nan=0.5), axis=0)
        plt.imshow(
            auc_mat[sort_ind, :, label_ind],
            aspect='auto',
            origin='lower',
            extent=[slide_times_auc[0], slide_times_auc[-1], 0, len(combined_tagged_units_filtered)],
            cmap=feature_map[label],
            vmin=0,
            vmax=1,
            interpolation='none',
        )
        plt.colorbar(label='AUC')
        plt.title(f'AUC for {label} over time')
        plt.xlabel(f'Time from {align} (s)')
        plt.ylabel('Unit index')
        plt.savefig(os.path.join(metrics_folder, f'AUC_{label}_{criteria_name}_{align}.pdf'), bbox_inches='tight')
        plt.close(fig)

    fig, axes = plt.subplots(len(labels), 1, figsize=(10, 8), sharex=True)
    bins = np.linspace(0, 1, 40)
    for label_ind, label in enumerate(labels):
        axes[label_ind].hist(auc_max[:, label_ind], bins=bins, color='gray', alpha=0.7, edgecolor='none')
        axes[label_ind].set_title(label)
        axes[label_ind].set_xlim(0, 1)
    plt.xlabel('AUC')
    plt.suptitle('AUC for each label')
    plt.savefig(os.path.join(metrics_folder, f'AUC_hist_{criteria_name}.pdf'), bbox_inches='tight')
    plt.close(fig)

    max_lag_time = np.full(len(combined_tagged_units_filtered), np.nan)
    auc_max_ind_outcome = auc_max_ind[:, labels.index('outcome')]
    valid = ~np.isnan(auc_max_ind_outcome)
    idx = auc_max_ind_outcome[valid].astype(int)
    max_lag_time[valid] = slide_times_auc[idx]
    max_lag_time[max_lag_time < 0.5 * bin_size] = 0.5 * bin_size

    auc_df = combined_tagged_units_filtered[['session', 'unit']].copy()
    auc_df['max_auc_lag'] = max_lag_time
    auc_df.to_csv(os.path.join(metrics_folder, 'auc_max_lag_indi.csv'), index=False)

    label = 'outcome'
    label_ind = labels.index(label)
    valid_rows = ~np.isnan(auc_max[:, label_ind]) & ~np.isnan(auc_max_ind[:, label_ind])
    lag_values = slide_times_auc[auc_max_ind[valid_rows, label_ind].astype(int)] if np.any(valid_rows) else np.array([])

    fig, axes = plt.subplots(1, 3, figsize=(10, 4), sharex=False, sharey=False)
    if lag_values.size > 0:
        auc_values = auc_max[valid_rows, label_ind]
        axes[0].scatter(auc_values, lag_values, s=10)
        axes[0].set_xlim(0, 1)
        axes[0].set_xlabel('AUC')
        axes[0].set_ylabel('Time lag')

        positive_log = lag_values[auc_values >= 0.5] + 1
        negative_log = lag_values[auc_values < 0.5] + 1
        axes[1].hist(np.log(positive_log[positive_log > 0]), bins=15, color='k', alpha=0.7, edgecolor='none', density=True)
        if negative_log.size:
            axes[1].hist(np.log(negative_log[negative_log > 0]), bins=15, color='lightgray', alpha=0.5, edgecolor='none', density=True)

        mode_p_log, x_grid_p_log, pdf_p_log = _estimate_mode(np.log(positive_log[positive_log > 0]))
        mode_n_log, x_grid_n_log, pdf_n_log = _estimate_mode(np.log(negative_log[negative_log > 0]))
        if x_grid_p_log is not None:
            axes[1].axvline(mode_p_log, color='k', linestyle='--')
            axes[1].plot(x_grid_p_log, pdf_p_log, color='blue')
        if x_grid_n_log is not None:
            axes[1].axvline(mode_n_log, color='gray', linestyle='--')
            axes[1].plot(x_grid_n_log, pdf_n_log, color='red')
        axes[1].set_xlim(-1, 5)
        axes[1].set_xlabel('log(Time lag)')
        axes[1].set_title(f"p: {np.exp(mode_p_log) - 1 if np.isfinite(mode_p_log) else np.nan:.2f}, n: {np.exp(mode_n_log) - 1 if np.isfinite(mode_n_log) else np.nan:.2f}")

        positive = lag_values[auc_values >= 0.5]
        negative = lag_values[auc_values < 0.5]
        axes[2].hist(positive, bins=20, color='k', alpha=0.7, edgecolor='none', density=True)
        if negative.size:
            axes[2].hist(negative, bins=20, color='lightgray', alpha=0.5, edgecolor='none', density=True)

        mode_p, x_grid_p, pdf_p = _estimate_mode(positive)
        mode_n, x_grid_n, pdf_n = _estimate_mode(negative)
        if x_grid_p is not None:
            axes[2].axvline(mode_p, color='k', linestyle='--')
            axes[2].plot(x_grid_p, pdf_p, color='blue')
        if x_grid_n is not None:
            axes[2].axvline(mode_n, color='gray', linestyle='--')
            axes[2].plot(x_grid_n, pdf_n, color='red')
        axes[2].set_xlim(-1, 5)
        axes[2].set_xlabel('Time lag')
        axes[2].set_title(f'p: {mode_p:.2f}, n: {mode_n:.2f}')
    else:
        mode_p = np.nan
        mode_n = np.nan

    plt.suptitle(f'Peak-AUC lag distribution ({label})')
    plt.savefig(os.path.join(metrics_folder, f'AUC_lag_modes_{criteria_name}.pdf'), bbox_inches='tight')
    plt.close(fig)

    window_dict = {'late': mode_n, 'early': mode_p}
    with open(os.path.join(metrics_folder, 'auc_windows.json'), 'w') as f:
        json.dump(window_dict, f)

    print(f'Finished {criteria_name}: {len(results)} pairs, {len(error_rows)} failures.')


def main():
    parser = argparse.ArgumentParser(description='Parallel outcome-window generation over session-unit pairs.')
    parser.add_argument('--criteria-name', dest='criteria_names', action='append', help='Criteria name(s) to run. Can be passed multiple times.')
    parser.add_argument('--pre-event', type=float, default=0.0, help='Seconds before the event.')
    parser.add_argument('--post-event', type=float, default=None, help='Seconds after the event.')
    parser.add_argument('--n-jobs', type=int, default=-4, help='joblib n_jobs value.')
    args = parser.parse_args()

    if args.criteria_names:
        if args.post_event is None:
            parser.error('--post-event is required when using --criteria-name')
        for criteria_name in args.criteria_names:
            compute_outcome_window_parallel(criteria_name, args.pre_event, args.post_event, n_jobs=args.n_jobs)
    else:
        default_runs = [
            ('beh_all_TT', 0, 2),
            ('beh_all_NP', 0, 2.5),
        ]
        for criteria_name, pre_event, post_event in default_runs:
            compute_outcome_window_parallel(criteria_name, pre_event, post_event, n_jobs=args.n_jobs)


if __name__ == '__main__':
    main()

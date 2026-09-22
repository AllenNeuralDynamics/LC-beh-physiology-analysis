# %%
"""
Diagnostic pass for rhythmicity of spontaneous firing in opto-tagged DRN units.

Run this before building anything session-wide. For a handful of tagged units you already
believe are rhythmic, it recomputes the autocorrelogram at 5 ms bins over 4 s of lag (the
pipeline's bin_long = 50 ms has a Nyquist of 10 Hz and cannot measure the 2-12 Hz band)
and plots it against an ISI-shuffle surrogate.

What the figure tells you: peaks that stay above the shuffled band for a run of cycles are
a sustained rhythm. Peaks that fall inside the band are fully explained by ISI regularity,
which is what a classic slow, regular 5-HT pacemaker produces on its own -- in that case
the result to report is ISI-based (cv_isi, isi_mode), not an oscillation.
"""

import sys
import os
sys.path.append('/root/capsule/code/beh_ephys_analysis')
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.beh_functions import session_dirs, get_unit_tbl, get_session_tbl
from utils.ephys_functions import load_drift
from utils.rhythmicity import rhythmicity_metrics, plot_unit_rhythmicity

# 5 ms gives 16 samples/cycle at 12 Hz; 4 s spans 8 cycles at 2 Hz, enough to tell a
# 3-cycle renewal echo from a sustained rhythm.
BIN_SIZE = 0.005
WINDOW_LENGTH = 4.0
BAND = (2.0, 12.0)
GO_CUE_PERIOD = 2.0

# Units judged rhythmic by eye from the existing long-lag autocorrelogram figures, with the
# band they were placed in. This is a labelled validation set, not just a convenience list:
# peak_freq should land in the labelled band. Units where it does not are the ones to look
# at, because the disagreement is either a bad eyeball call or a bad metric, and the figure
# says which. 'slow' = 2-5 Hz, 'fast' = 6-12 Hz.
UNITS_OF_INTEREST = {
    'behavior_808650_2025-09-24_12-38-00': {76: 'fast'},
    'behavior_826159_2026-01-22_12-58-47': {145: 'fast'},
    'behavior_826159_2026-01-23_13-34-11': {92: 'slow'},
    'behavior_826159_2026-01-26_15-15-12': {20: 'slow'},
    'behavior_826159_2026-01-27_08-55-48': {142: 'slow'},
    'behavior_826164_2026-01-27_11-14-13': {22: 'slow', 25: 'fast'},
    'behavior_826164_2026-01-29_11-09-03': {178: 'slow', 180: 'fast'},
    'behavior_826164_2026-01-30_11-12-30': {95: 'slow'},
    'behavior_835444_2026-02-17_13-56-45': {158: 'slow', 141: 'fast', 207: 'fast'},
    'behavior_835444_2026-02-18_13-01-55': {47: 'slow'},
    'behavior_835444_2026-02-19_13-08-36': {66: 'slow'},
    'behavior_835451_2026-02-25_13-19-37': {188: 'fast'},
    'behavior_835451_2026-02-27_14-19-03': {234: 'slow', 115: 'fast', 132: 'fast'},
    'behavior_838332_2026-03-11_13-01-00': {105: 'slow', 164: 'slow', 168: 'fast'},
    'behavior_838332_2026-03-13_12-29-22': {165: 'slow'},
    'behavior_841596_2026-03-25_14-10-54': {16: 'fast', 188: 'fast'},
    'behavior_841596_2026-03-27_13-57-43': {168: 'slow', 259: 'slow'},
    'behavior_841861_2026-04-03_13-18-22': {6: 'fast'},
    'behavior_843661_2026-04-28_09-49-53': {231: 'slow'},
    'behavior_843661_2026-04-29_08-19-05': {180: 'slow'},
    'behavior_848869_2026-05-12_08-51-54': {186: 'slow', 191: 'fast', 193: 'fast'},
    'behavior_848869_2026-05-15_08-59-55': {47: 'fast'},
    'behavior_844917_2026-06-02_11-08-15': {60: 'fast'},
    'behavior_844917_2026-06-04_12-30-01': {206: 'slow', 106: 'fast', 118: 'fast'},
}

EXPECTED_BANDS = {'slow': (2.0, 5.0), 'fast': (6.0, 12.0)}


def spontaneous_window(session, data_type):
    """
    Opto-free analysis window for a session, following cross_auto_corr_DRN exactly:
    the curated ephys_cut, narrowed to the gap between the pre- and post-opto blocks.

    Returns:
    (rec_start, rec_end) or None if the session lacks the required files.
    """
    session_dir = session_dirs(session)
    qm_file = os.path.join(session_dir['processed_dir'], f'{session}_qm.json')
    if not os.path.exists(qm_file):
        return None
    with open(qm_file, 'r') as f:
        session_qm = json.load(f)
    rec_start, rec_end = session_qm['ephys_cut'][0], session_qm['ephys_cut'][1]

    # cross_auto_corr_DRN hardcodes opto_dir_curated, but every opto_session.csv in the
    # current scratch tree sits under ephys/opto/raw. Try the requested data_type first,
    # then the other, and refuse to continue if neither is present: falling back to the
    # full ephys_cut would silently pull the opto-tagging blocks into the "spontaneous"
    # window, and stimulus-locked spikes would dominate the autocorrelogram.
    opto_file = None
    for key in (f'opto_dir_{data_type}', 'opto_dir_curated', 'opto_dir_raw'):
        candidate = os.path.join(str(session_dir[key]), f'{session}_opto_session.csv')
        if os.path.exists(candidate):
            opto_file = candidate
            break
    if opto_file is None:
        print(f'No opto_session.csv for {session}; cannot isolate the opto-free window.')
        return None
    opto_tbl = pd.read_csv(opto_file)
    pre_post = opto_tbl['pre_post'].unique()
    if len(pre_post) > 1:
        rec_start = opto_tbl[opto_tbl['pre_post'] == 'pre']['time'].max()
        rec_end = opto_tbl[opto_tbl['pre_post'] == 'post']['time'].min()
    elif len(pre_post) == 1:
        rec_end = opto_tbl['time'].min()
    return rec_start, rec_end


def unit_window(session, unit_id, data_type, rec_start, rec_end):
    """Intersect the session window with the unit's curated drift cut."""
    drift = load_drift(session, unit_id, data_type)
    start, end = rec_start, rec_end
    if drift is not None:
        if drift['ephys_cut'][0] is not None:
            start = max(start, drift['ephys_cut'][0])
        if drift['ephys_cut'][1] is not None:
            end = min(end, drift['ephys_cut'][1])
    return start, end


def go_cue_intervals(session, start, end):
    """
    Go-cue periods to blank, as (starts, ends).

    Blanking creates gaps, so the valid-pair count per lag falls with lag AND falls
    periodically with trial structure -- which injects trial-rate structure into the very
    band being measured. rhythmicity_metrics applies identical blanking to the surrogates
    so the artifact appears in the null too, and masks lags whose valid-pair count drops
    below min_valid_frac. Run the diagnostic both ways and compare.
    """
    session_tbl = get_session_tbl(session)
    if session_tbl is None:
        return None
    go_cues = session_tbl['goCue_start_time'].values
    go_cues = go_cues[(go_cues >= start - GO_CUE_PERIOD) & (go_cues <= end)]
    if len(go_cues) == 0:
        return None
    return go_cues, go_cues + GO_CUE_PERIOD


def diagnose_session(session, data_type, unit_ids=None, unit_group='tagged',
                     summary=None, blank_go_cue=False, n_surrogates=100, max_units=12,
                     plot=True, save=True, expected=None):
    """
    Compute and plot rhythmicity for the units of one session.

    Parameters:
    session : str
        Session id.
    data_type : str
        'raw' or 'curated'.
    unit_ids : list, optional
        Explicit unit list, overriding unit_group.
    unit_group : {'tagged', 'all'}
        'tagged' keeps only opto_pass units -- right for the first diagnostic pass, where
        the point is to eyeball a dozen figures. 'all' is required for the quantitative
        pass: the untagged population is the only null that carries the real drift, burst,
        sorting and trial-structure artifacts that synthetic surrogates do not, so it is
        how you find out whether a "significant" rate is biology or a pipeline artifact.
        Note untagged is a contaminated null (some untagged units are 5-HT cells that
        failed tagging), which biases against a tagged-vs-untagged difference.
    summary : bool, optional
        Which unit table to load. get_unit_tbl(summary=True) returns the soma opto tagging
        summary, which is already restricted to passing units -- opto_pass is uniformly
        True there, so it cannot provide an untagged group. summary=False loads the full
        opto_tagging_metrics table. Defaults to True for unit_group 'tagged' and False for
        'all'.
    blank_go_cue : bool
        Exclude go-cue periods. See go_cue_intervals for why this needs checking.
    n_surrogates : int
        ISI shuffles per unit. 100 gives p-value resolution of 0.01; use osc_score_z
        rather than osc_score_p when correcting across a population.
    max_units : int, optional
        Cap on units run. None for no cap; the figure is only drawn when plot is True.
    plot, save : bool
        Whether to draw the diagnostic figure and whether to write outputs to disk.
    expected : dict, optional
        unit_id -> band label ('slow'/'fast') from visual inspection. Recorded in the
        output and shown in the panel title so the figure can be checked against the call.

    Returns:
    DataFrame, one row per unit, of the scalar metrics.
    """
    window = spontaneous_window(session, data_type)
    if window is None:
        print(f'No qm file for {session}. Skipping.')
        return None
    rec_start, rec_end = window

    if summary is None:
        # For an explicit unit list, use the full metrics table: it is a superset of the
        # soma summary, so no requested unit can go missing and silently drop out.
        summary = unit_ids is None and unit_group == 'tagged'
    unit_tbl = get_unit_tbl(session, data_type, summary=summary)
    if unit_tbl is None:
        print(f'No unit table for {session}. Skipping.')
        return None
    tagged = unit_tbl['opto_pass'].fillna(False).astype(bool)
    if unit_group == 'all' and tagged.all():
        print(f'{session}: opto_pass is uniformly True, so there is no untagged group. '
              f'The table is pre-filtered -- pass summary=False.')
    if unit_ids is None:
        keep = (
            (unit_tbl['decoder_label'] != 'artifact') &
            (unit_tbl['decoder_label'] != 'noise') &
            (unit_tbl['isi_violations_ratio'] < 0.1)
        )
        if unit_group == 'tagged':
            keep &= tagged
        unit_ids = unit_tbl[keep]['unit_id'].to_list()
    if len(unit_ids) == 0:
        print(f'No {unit_group} units in {session}.')
        return None
    if max_units is not None:
        unit_ids = unit_ids[:max_units]
    is_tagged = dict(zip(unit_tbl['unit_id'], tagged))

    rows = []
    results = []
    for unit_id in unit_ids:
        row = unit_tbl[unit_tbl['unit_id'] == unit_id]
        if len(row) == 0:
            print(f'{session} {unit_id}: not in the {"summary" if summary else "full"} '
                  f'unit table. Skipping.')
            continue
        start, end = unit_window(session, unit_id, data_type, rec_start, rec_end)
        if start >= end - WINDOW_LENGTH:
            print(f'{session} {unit_id}: drift cut leaves too short a window. Skipping.')
            continue
        spike_times = row['spike_times'].values[0]
        blank = go_cue_intervals(session, start, end) if blank_go_cue else None
        print(f'{session} {unit_id}: {end - start:.0f} s window')
        res = rhythmicity_metrics(spike_times, bin_size=BIN_SIZE,
                                  window_length=WINDOW_LENGTH, start=start, end=end,
                                  blank_intervals=blank, band=BAND,
                                  n_surrogates=n_surrogates)
        res['session'] = session
        res['unit_id'] = unit_id
        res['tagged'] = bool(is_tagged.get(unit_id, False))
        res['blank_go_cue'] = blank_go_cue
        res['expected'] = (expected or {}).get(unit_id, '')
        results.append(res)
        rows.append({k: v for k, v in res.items() if np.ndim(v) == 0})

    if len(results) == 0:
        return None

    metrics = pd.DataFrame(rows)
    tag = '_nogo' if blank_go_cue else ''
    stem = f'{session}_{data_type}_rhythmicity{tag}'
    out_dir = str(session_dirs(session)[f'ephys_processed_dir_{data_type}'])
    if save:
        os.makedirs(out_dir, exist_ok=True)
        metrics.to_pickle(os.path.join(out_dir, f'{stem}.pkl'))
        metrics.to_csv(os.path.join(out_dir, f'{stem}.csv'), index=False)

    if plot:
        fig, axes = plt.subplots(len(results), 3, figsize=(15, 3.2 * len(results)),
                                 squeeze=False)
        for res, ax_row in zip(results, axes):
            label = f"{res['unit_id']}{' *tagged' if res['tagged'] else ''}"
            if res['expected']:
                label += f" (eye: {res['expected']})"
            plot_unit_rhythmicity(res, unit_label=label, axes=ax_row, band=BAND)
        fig.tight_layout()
        if save:
            fig.savefig(os.path.join(out_dir, f'{stem}.png'), dpi=150)
        plt.close(fig)

    return metrics


SUMMARY_COLS = ['session', 'unit_id', 'tagged', 'rate', 'n_spikes', 'duration', 'cv_isi',
                'isi_mode', 'peak_freq', 'first_peak_lag', 'osc_score', 'osc_score_p',
                'osc_score_z', 'max_run_sig', 'persist_time']


def run_units_of_interest(units=None, data_type='raw', blank_go_cue=False,
                          n_surrogates=200, plot=True, save=True):
    """
    Run the diagnostic on the hand-picked units, keeping the visual band label alongside.

    Returns:
    DataFrame with an `expected` column and an `agrees` column, True when peak_freq falls
    in the band the unit was visually assigned to.
    """
    units = UNITS_OF_INTEREST if units is None else units
    all_metrics = []
    for session, unit_bands in units.items():
        try:
            metrics = diagnose_session(session, data_type,
                                       unit_ids=list(unit_bands),
                                       expected=unit_bands,
                                       blank_go_cue=blank_go_cue,
                                       n_surrogates=n_surrogates,
                                       max_units=None, plot=plot, save=save)
            if metrics is not None:
                all_metrics.append(metrics)
        except Exception as e:
            print(f'Error processing {session}: {type(e).__name__}: {e}')
            plt.close('all')
    if not all_metrics:
        return None
    metrics = pd.concat(all_metrics, ignore_index=True)
    lo = metrics['expected'].map(lambda b: EXPECTED_BANDS.get(b, (np.nan, np.nan))[0])
    hi = metrics['expected'].map(lambda b: EXPECTED_BANDS.get(b, (np.nan, np.nan))[1])
    metrics['agrees'] = (metrics['peak_freq'] >= lo) & (metrics['peak_freq'] <= hi)
    return metrics


def add_fdr(metrics, alpha=0.05):
    """
    Benjamini-Hochberg q-values across units, from the normal tail of osc_score_z.

    Uses osc_score_z rather than osc_score_p because the permutation p is floored at
    1/(n_surrogates+1) and so cannot survive correction across a large population.
    """
    from scipy.stats import norm, false_discovery_control

    metrics = metrics.copy()
    metrics['osc_score_p_z'] = norm.sf(metrics['osc_score_z'])
    metrics['osc_score_q'] = np.nan
    ok = metrics['osc_score_p_z'].notna()
    if ok.any():
        metrics.loc[ok, 'osc_score_q'] = false_discovery_control(
            metrics.loc[ok, 'osc_score_p_z'].values, method='bh')
    metrics['rhythmic'] = (metrics['osc_score_q'] < alpha) & (metrics['max_run_sig'] >= 3)
    return metrics


# %%
if __name__ == '__main__':
    # Step 1, diagnostic: the hand-picked units, which carry a visual band label so the
    # metric can be checked against the eyeball call before it is trusted anywhere else.
    # Step 2, quantitative: unit_group='all', max_units=None, over whole sessions, so the
    # untagged population can serve as a real-data null for the tagged claim.
    metrics = run_units_of_interest(n_surrogates=200)

    if metrics is not None:
        metrics = add_fdr(metrics)
        cols = SUMMARY_COLS + ['expected', 'agrees', 'osc_score_q', 'rhythmic']
        print(metrics[cols].to_string(index=False))
        print('\npeak_freq lands in the visually assigned band:')
        print(metrics.groupby('expected')['agrees'].agg(['mean', 'sum', 'count']))
        print('\npassing the shuffle test:')
        print(metrics.groupby('expected')['rhythmic'].agg(['mean', 'sum', 'count']))

"""
ISI-violation quality metric restricted to the laser-free portion of a recording.

The pipeline's isi_violations_ratio (nwb.units, and hence unit_tbl) is computed by
spikeinterface over the *whole* recording, opto-tagging blocks included. Two things
inside those blocks inflate it without saying anything about sorting quality:

  * laser artifacts, which the sorter can pick up as extra "spikes" a fraction of a
    millisecond apart, and
  * genuinely laser-driven spikes, which arrive in short high-rate bursts.

Either one puts sub-refractory intervals into a unit's spike train, so a well-isolated
unit can be failed by default_qc on violations it only commits while the laser is on --
and precisely the units of interest (the ones the laser drives) are the ones most
affected. This module recomputes the same metric over the complement of the opto blocks
so that the QC decision can be made on laser-free data.

Three windows are computed per unit, all with the same estimator, so the comparison
between them is meaningful:

  window   the full curated session (qm['ephys_cut']); reproduces the pipeline value up
           to the difference between ephys_cut and the raw recording duration
  nonopto  window minus the laser intervals -- the QC number to use
  opto     the laser intervals alone -- the diagnostic. A unit whose violations sit
           almost entirely in the opto windows is the artifact case this module exists
           for; one that violates just as much outside is genuinely contaminated.

The ratio is duration-normalized (it is a rate ratio, not a fraction of spikes), so it
does not automatically fall when time is removed: with the same contamination rate
throughout, nonopto and window agree. See isi_violations_windowed.

Estimator: mirrors spikeinterface 0.103.0
qualitymetrics.misc_metrics.isi_violations exactly, at the parameters the AIND sorting
pipeline used (isi_threshold_ms=1.5, min_isi_ms=0, read off the quality_metrics
extension params of the postprocessed analyzer). Kept windows are passed as separate
segments, which is how spikeinterface handles multi-segment recordings: intervals that
straddle a window boundary are not counted, because they were not observed.
"""

import json
import os

import numpy as np
import pandas as pd
import pickle

from utils.beh_functions import session_dirs

# pipeline values; see module docstring
ISI_THRESHOLD_MS = 1.5
MIN_ISI_MS = 0.0

# padding around laser intervals, in seconds, as (before, after). Artifacts are confined
# to the pulse itself but laser-driven spiking outlasts it, so the tail is the longer one.
TRAIN_PAD_S = (0.05, 0.2)
BLOCK_PAD_S = (1.0, 1.0)

# trains closer together than this belong to the same opto block
BLOCK_GAP_S = 60.0

# default_qc, as it was defined inline in opto_tagging.opto_plotting_session. The
# criteria that were commented out there (firing_rate > 0.1, presence_ratio > 0.95,
# amplitude_cutoff < 0.05) are deliberately not applied here.
DEFAULT_QC_ISI_MAX = 0.4
DEFAULT_QC_EXCLUDE_LABELS = ('noise', 'artifact')


# ------------------------------------------------------------------ intervals


def merge_intervals(starts, ends):
    """
    Merge overlapping [start, end] intervals.

    Returns:
    starts, ends : ndarray
        Sorted, non-overlapping intervals.
    """
    starts = np.asarray(starts, dtype=float)
    ends = np.asarray(ends, dtype=float)
    if len(starts) == 0:
        return starts, ends
    order = np.argsort(starts)
    starts, ends = starts[order], ends[order]
    keep_starts, keep_ends = [starts[0]], [ends[0]]
    for s, e in zip(starts[1:], ends[1:]):
        if s <= keep_ends[-1]:
            keep_ends[-1] = max(keep_ends[-1], e)
        else:
            keep_starts.append(s)
            keep_ends.append(e)
    return np.array(keep_starts), np.array(keep_ends)


def subtract_intervals(starts, ends, cut_starts, cut_ends):
    """
    Remove [cut_start, cut_end] intervals from [start, end] intervals.

    Returns:
    starts, ends : ndarray
        The remaining intervals, sorted and non-overlapping. Zero-length remainders are
        dropped.
    """
    starts, ends = merge_intervals(starts, ends)
    cut_starts, cut_ends = merge_intervals(cut_starts, cut_ends)
    keep_starts, keep_ends = [], []
    for s, e in zip(starts, ends):
        pieces = [(s, e)]
        for cs, ce in zip(cut_starts, cut_ends):
            remaining = []
            for ps, pe in pieces:
                if ce <= ps or cs >= pe:
                    remaining.append((ps, pe))
                    continue
                if cs > ps:
                    remaining.append((ps, cs))
                if ce < pe:
                    remaining.append((ce, pe))
            pieces = remaining
        for ps, pe in pieces:
            if pe > ps:
                keep_starts.append(ps)
                keep_ends.append(pe)
    return np.array(keep_starts, dtype=float), np.array(keep_ends, dtype=float)


def train_intervals(opto_tbl):
    """
    Span of each laser train in an opto session table, unpadded.

    A row of the table is one train: onset at 'time', num_pulses pulses at freq Hz, each
    'duration' ms long, so the train ends (num_pulses - 1) / freq + duration / 1000 s
    after onset. Missing or degenerate values collapse to a single pulse.

    Returns:
    starts, ends : ndarray
    """
    starts = opto_tbl['time'].to_numpy(dtype=float)

    def column(name, default):
        if name not in opto_tbl.columns:
            return np.full(len(opto_tbl), default, dtype=float)
        values = pd.to_numeric(opto_tbl[name], errors='coerce').to_numpy(dtype=float)
        return np.where(np.isnan(values), default, values)

    num_pulses = column('num_pulses', 1.0)
    freq = column('freq', 0.0)
    duration_s = column('duration', 0.0) / 1000.0
    train_len = np.where((num_pulses > 1) & (freq > 0), (num_pulses - 1) / np.where(freq > 0, freq, 1), 0.0)
    return starts, starts + train_len + duration_s


def block_intervals(opto_tbl, block_gap_s=BLOCK_GAP_S):
    """
    Span of each opto block: a run of trains with no laser-free gap longer than
    block_gap_s.

    Blocks are found from the train times rather than the table's 'pre_post' label so
    that a session with laser interleaved into behavior is handled the same way as the
    usual pre-block / behavior / post-block design.

    Returns:
    starts, ends : ndarray
    """
    starts, ends = merge_intervals(*train_intervals(opto_tbl))
    if len(starts) == 0:
        return starts, ends
    block_starts, block_ends = [starts[0]], [ends[0]]
    for s, e in zip(starts[1:], ends[1:]):
        if s - block_ends[-1] <= block_gap_s:
            block_ends[-1] = max(block_ends[-1], e)
        else:
            block_starts.append(s)
            block_ends.append(e)
    return np.array(block_starts), np.array(block_ends)


def laser_intervals(opto_tbl, mode='block', pad=None, block_gap_s=BLOCK_GAP_S):
    """
    Intervals to treat as laser-contaminated.

    Parameters:
    opto_tbl : DataFrame
        Opto session table for *all* emission sites (i.e. {session}_opto_session.csv,
        not the per-target {session}_opto_session_{target}.csv). Antidromic sites fire
        the same laser and leave the same artifacts, so they have to be excluded too.
    mode : {'block', 'train'}
        'block' drops whole opto blocks, which is what the rest of the repo means by an
        opto-free window (see rhythmicity_DRN.spontaneous_window) and what "the non-opto
        portion of the recording" reads as. 'train' drops only the trains themselves,
        keeping the inter-train gaps; it costs much less time but leaves any
        contamination that is locked to the block rather than to the pulse.
    pad : tuple, optional
        (before, after) padding in seconds. Defaults to BLOCK_PAD_S or TRAIN_PAD_S
        depending on mode.
    block_gap_s : float
        Laser-free gap that separates two blocks, in seconds. Unused in 'train' mode.

    Returns:
    starts, ends : ndarray
    """
    if mode == 'block':
        starts, ends = block_intervals(opto_tbl, block_gap_s=block_gap_s)
        pad = BLOCK_PAD_S if pad is None else pad
    elif mode == 'train':
        starts, ends = train_intervals(opto_tbl)
        pad = TRAIN_PAD_S if pad is None else pad
    else:
        raise ValueError(f"mode must be 'block' or 'train', got {mode}")
    return merge_intervals(starts - pad[0], ends + pad[1])


# ------------------------------------------------------------------ metric


def isi_violations_windowed(spike_times, starts, ends,
                            isi_threshold_ms=ISI_THRESHOLD_MS, min_isi_ms=MIN_ISI_MS):
    """
    ISI violations of one unit over a set of time windows.

    Mirrors spikeinterface.qualitymetrics.misc_metrics.isi_violations, with each window
    passed as a separate segment: intervals spanning a window boundary are not counted,
    and the denominator is the summed window duration rather than the recording
    duration.

    The returned ratio is
        (violations / (2 * n_spikes * isi_threshold)) / (n_spikes / duration),
    the firing rate of the hypothetical contaminating unit as a fraction of the unit's
    own rate. Both terms are rates, so removing clean time from the windows leaves it
    unchanged -- it moves only if the contamination is unevenly distributed.

    Parameters:
    spike_times : array-like
        Spike times in seconds, on the same clock as the windows.
    starts, ends : array-like
        Window bounds in seconds. Overlapping windows are merged.
    isi_threshold_ms, min_isi_ms : float
        Refractory period and enforced dead time, in ms.

    Returns:
    dict with keys 'isi_violations_ratio', 'isi_violations_count', 'num_spikes',
    'duration'. The ratio is NaN when no spikes fall in the windows.
    """
    spike_times = np.sort(np.asarray(spike_times, dtype=float))
    starts, ends = merge_intervals(starts, ends)
    duration = float(np.sum(ends - starts)) if len(starts) else 0.0

    isi_threshold_s = isi_threshold_ms / 1000.0
    min_isi_s = min_isi_ms / 1000.0

    num_spikes = 0
    num_violations = 0
    left = np.searchsorted(spike_times, starts, side='left')
    right = np.searchsorted(spike_times, ends, side='right')
    for i0, i1 in zip(left, right):
        segment = spike_times[i0:i1]
        num_spikes += len(segment)
        if len(segment) > 1:
            num_violations += int(np.sum(np.diff(segment) < isi_threshold_s))

    ratio = np.nan
    if num_spikes > 0 and duration > 0:
        violation_time = 2 * num_spikes * (isi_threshold_s - min_isi_s)
        violation_rate = num_violations / violation_time
        total_rate = num_spikes / duration
        ratio = violation_rate / total_rate
    return {'isi_violations_ratio': ratio,
            'isi_violations_count': num_violations,
            'num_spikes': num_spikes,
            'duration': duration}


# ------------------------------------------------------------------ per session


def session_window(session, spiketimes=None):
    """
    Curated session bounds, from qm['ephys_cut'].

    Falls back to the span of the spike times when the qm file is missing, so that the
    metric can still be computed; the caller can tell which happened from the returned
    source.

    Returns:
    (start, end), source : tuple, str
        source is 'ephys_cut' or 'spike_span'. (None, None), 'none' when neither is
        available.
    """
    qm_file = os.path.join(session_dirs(session)['processed_dir'], f'{session}_qm.json')
    if os.path.exists(qm_file):
        with open(qm_file) as f:
            qm = json.load(f)
        cut = qm.get('ephys_cut')
        if cut is not None and cut[0] is not None and cut[1] is not None:
            return (float(cut[0]), float(cut[1])), 'ephys_cut'
    if spiketimes:
        all_times = [t for t in spiketimes.values() if len(t)]
        if all_times:
            return (float(min(t[0] for t in all_times)),
                    float(max(t[-1] for t in all_times))), 'spike_span'
    return (None, None), 'none'


def load_opto_events(session, data_type='raw'):
    """
    Opto session table covering every emission site, or None if absent.

    The file lives under ephys/opto/{data_type} but historically has been written to
    whichever of raw/curated was processed, so both are tried (as in
    rhythmicity_DRN.spontaneous_window).
    """
    session_dir = session_dirs(session)
    for key in (f'opto_dir_{data_type}', 'opto_dir_raw', 'opto_dir_curated'):
        opto_dir = session_dir.get(key)
        if opto_dir is None:
            continue
        candidate = os.path.join(str(opto_dir), f'{session}_opto_session.csv')
        if os.path.exists(candidate):
            return pd.read_csv(candidate)
    return None


def load_spiketimes(session, data_type='raw'):
    """Spike times per unit id, in session (harp) seconds, or None if absent."""
    spike_file = os.path.join(session_dirs(session)[f'ephys_processed_dir_{data_type}'],
                              'spiketimes.pkl')
    if not os.path.exists(spike_file):
        return None
    with open(spike_file, 'rb') as f:
        return pickle.load(f)


def non_opto_isi_tbl(session, data_type='raw', mode='block', pad=None,
                     block_gap_s=BLOCK_GAP_S, spiketimes=None, unit_ids=None,
                     isi_threshold_ms=ISI_THRESHOLD_MS, min_isi_ms=MIN_ISI_MS):
    """
    ISI violations per unit over the laser-free, laser-only and full session windows.

    Parameters:
    session : str
    data_type : str
        'raw' or 'curated'.
    mode, pad, block_gap_s :
        Passed to laser_intervals.
    spiketimes : dict, optional
        unit_id -> spike times. Loaded from spiketimes.pkl when omitted.
    unit_ids : list, optional
        Subset of units. Defaults to every unit in spiketimes.
    isi_threshold_ms, min_isi_ms : float
        Passed to isi_violations_windowed.

    Returns:
    DataFrame with one row per unit and columns unit_id, and
    {isi_violations_ratio,isi_violations_count,num_spikes,duration}_{nonopto,opto,window}
    plus frac_nonopto (fraction of the session window that is laser-free). The
    parameters used, and the windows themselves, are in df.attrs['isi_nonopto_params'].
    Returns None when the session lacks spike times or an opto table.
    """
    if spiketimes is None:
        spiketimes = load_spiketimes(session, data_type)
    if not spiketimes:
        print(f'{session}: no spiketimes.pkl for {data_type}, skipping non-opto ISI.')
        return None

    opto_tbl = load_opto_events(session, data_type)
    if opto_tbl is None or len(opto_tbl) == 0:
        print(f'{session}: no opto_session.csv, cannot define the laser-free window.')
        return None

    (win_start, win_end), win_source = session_window(session, spiketimes)
    if win_start is None:
        print(f'{session}: no ephys_cut and no spikes, skipping non-opto ISI.')
        return None

    laser_starts, laser_ends = laser_intervals(opto_tbl, mode=mode, pad=pad,
                                               block_gap_s=block_gap_s)
    # clip the laser intervals to the session window before subtracting, so that the
    # opto-window numbers are computed on the same time base as the other two
    laser_starts, laser_ends = subtract_intervals(laser_starts, laser_ends,
                                                  [-np.inf, win_end], [win_start, np.inf])
    keep_starts, keep_ends = subtract_intervals([win_start], [win_end],
                                                laser_starts, laser_ends)

    if unit_ids is None:
        unit_ids = list(spiketimes.keys())

    windows = {'nonopto': (keep_starts, keep_ends),
               'opto': (laser_starts, laser_ends),
               'window': ([win_start], [win_end])}
    rows = []
    for unit_id in unit_ids:
        spike_times = spiketimes.get(unit_id)
        if spike_times is None:
            continue
        row = {'unit_id': unit_id}
        for name, (starts, ends) in windows.items():
            for key, value in isi_violations_windowed(
                    spike_times, starts, ends,
                    isi_threshold_ms=isi_threshold_ms, min_isi_ms=min_isi_ms).items():
                row[f'{key}_{name}'] = value
        rows.append(row)

    tbl = pd.DataFrame(rows)
    if len(tbl):
        tbl['frac_nonopto'] = tbl['duration_nonopto'] / tbl['duration_window']
    tbl.attrs['isi_nonopto_params'] = {
        'session': session,
        'data_type': data_type,
        'mode': mode,
        'pad': list(pad) if pad is not None else
               list(BLOCK_PAD_S if mode == 'block' else TRAIN_PAD_S),
        'block_gap_s': block_gap_s,
        'isi_threshold_ms': isi_threshold_ms,
        'min_isi_ms': min_isi_ms,
        'window': [win_start, win_end],
        'window_source': win_source,
        'n_laser_intervals': int(len(laser_starts)),
        'laser_intervals': np.column_stack([laser_starts, laser_ends]).tolist(),
        'n_units': int(len(tbl)),
    }
    return tbl


# ------------------------------------------------------------------ sidecar io


def non_opto_isi_paths(session, data_type='raw'):
    """(csv, json) sidecar paths for the non-opto ISI metric."""
    opto_dir = session_dirs(session)[f'opto_dir_{data_type}']
    return (os.path.join(opto_dir, f'{session}_isi_nonopto.csv'),
            os.path.join(opto_dir, f'{session}_isi_nonopto.json'))


def save_non_opto_isi(session, data_type='raw', tbl=None, **kwargs):
    """
    Compute (unless given) and write the non-opto ISI sidecar for one session.

    The sidecar is written next to the opto tagging outputs so that every analysis that
    goes through get_unit_tbl picks the columns up without the opto tagging pickles
    having to be regenerated.

    Returns:
    The table that was written, or None if it could not be computed.
    """
    if tbl is None:
        tbl = non_opto_isi_tbl(session, data_type=data_type, **kwargs)
    if tbl is None or len(tbl) == 0:
        return None
    csv_file, json_file = non_opto_isi_paths(session, data_type)
    tbl.to_csv(csv_file, index=False)
    with open(json_file, 'w') as f:
        json.dump(tbl.attrs.get('isi_nonopto_params', {}), f, indent=1)
    return tbl


def load_non_opto_isi(session, data_type='raw'):
    """Non-opto ISI sidecar for one session, or None if it has not been computed."""
    csv_file, json_file = non_opto_isi_paths(session, data_type)
    if not os.path.exists(csv_file):
        return None
    tbl = pd.read_csv(csv_file)
    if os.path.exists(json_file):
        with open(json_file) as f:
            tbl.attrs['isi_nonopto_params'] = json.load(f)
    return tbl


# ------------------------------------------------------------------ qc


def default_qc(tbl, isi_col='isi_violations_ratio', fallback_isi_col=None,
               isi_max=DEFAULT_QC_ISI_MAX, exclude_labels=DEFAULT_QC_EXCLUDE_LABELS):
    """
    The default_qc criterion: a decoder label that is not noise or artifact, and an ISI
    violations ratio below isi_max.

    Single definition, so that switching QC onto the laser-free ISI is one argument
    rather than an edit in every caller.

    Parameters:
    tbl : DataFrame or dict
        Unit table, or one unit's row. Needs 'decoder_label' and isi_col.
    isi_col : str
        Which ISI column to test. 'isi_violations_ratio' is the pipeline's
        whole-recording value; 'isi_violations_ratio_nonopto' is the laser-free one.
    fallback_isi_col : str, optional
        Used where isi_col is NaN -- e.g. units with no spikes left in the laser-free
        window, which should not be failed for having no data. Set to None to fail them.
    isi_max : float
        Upper bound on the ISI violations ratio.
    exclude_labels : tuple
        decoder_label values that fail.

    Returns:
    Boolean Series (DataFrame in) or bool (dict in).
    """
    scalar = not isinstance(tbl, pd.DataFrame)
    if scalar:
        tbl = pd.DataFrame([dict(tbl)])
    if isi_col not in tbl.columns:
        raise KeyError(f'{isi_col} missing from the unit table')

    isi = pd.to_numeric(tbl[isi_col], errors='coerce')
    if fallback_isi_col is not None and fallback_isi_col in tbl.columns:
        isi = isi.fillna(pd.to_numeric(tbl[fallback_isi_col], errors='coerce'))
    keep = (isi < isi_max) & ~tbl['decoder_label'].isin(exclude_labels)
    return bool(keep.iloc[0]) if scalar else keep

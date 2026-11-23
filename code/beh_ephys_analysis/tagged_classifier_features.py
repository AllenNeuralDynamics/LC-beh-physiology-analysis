# %% Per-unit features for tagged_vs_untagged_clustering.ipynb, cached to CACHE.
#
# Units: the tagged and matched non-tagged units of the unit_summary_DRN PDFs, plus a larger
# "unidentified" population: every unit in the same sessions that passes default QC, ISI and
# firing-rate bounds and is not tagged, opto_pass or an opto candidate (see unidentified_units).
#
# Spontaneous spikes are the unit's spike times in a window minus the opto-tagging blocks
# (first train onset to last train offset, +/- BLOCK_PAD_S). Each unit is computed in up to
# two windows:
#   'cut'    the drift cut of its summary page (load_drift for tagged units, the hand-placed
#            {session}_opto_drift_tbl_nontagged.csv cut for matched controls)
#   'nocut'  the whole recording (qm ephys_cut); every unit, so the larger population, which
#            has no drift cuts, can be compared with everyone treated the same way
# From those spikes: firing rate, LV and positive-lag ACGs (rate / mean rate). Waveforms are
# the center channel of {session}_waveforms_soma.pkl; template_metrics are read straight from
# the sorted asset's analyzer zarr (whole-recording templates, multi-channel metrics).
#
#     python tagged_classifier_features.py [--jobs 8] [--rebuild]

import os
import sys
import json
import pickle
import argparse

curr_path = os.path.dirname(os.path.abspath(__file__))
if curr_path not in sys.path:
    sys.path.insert(0, curr_path)

import numpy as np
import pandas as pd
import zarr
from joblib import Parallel, delayed

from utils.beh_functions import session_dirs, get_unit_tbl
from utils.ephys_functions import load_drift
from unit_summary_DRN import criteria_passed, TAGGED_CRITERION, NONTAGGED_SELECTION, nontagged_drift_cut

CACHE = '/root/capsule/scratch/combined/tagged_classifier/features.pkl'
FLAGS_CSV = '/root/capsule/scratch/combined/unit_summary_DRN/population_unit_flags.csv'
MASTER_CSV = '/root/capsule/scratch/combined/master_unit_tables/master_all_units_opto_ccf_raw_260817.csv'
DATA_TYPE = 'raw'
BLOCK_PAD_S = 2.0
# unidentified population bounds: same as the matched controls (unit_summary_DRN_swap_low_fr)
UNID_MIN_FR = 0.5
UNID_MAX_ISI = 0.1
EDGES_SHORT = np.round(np.arange(0, 50.5, 1), 6) / 1000   # 0-50 ms, 1 ms bins
EDGES_LONG = np.round(np.arange(0, 505, 10), 6) / 1000    # 0-500 ms, 10 ms bins
LAG_SHORT_MS = (EDGES_SHORT[:-1] + EDGES_SHORT[1:]) / 2 * 1000
LAG_LONG_MS = (EDGES_LONG[:-1] + EDGES_LONG[1:]) / 2 * 1000
TEMPLATE_METRICS = ['peak_to_valley', 'half_width', 'peak_trough_ratio', 'repolarization_slope',
                    'recovery_slope', 'num_positive_peaks', 'num_negative_peaks', 'exp_decay',
                    'spread', 'velocity_above', 'velocity_below']


def unidentified_units(sessions):
    """QC-passing units of these sessions that never responded to light; with their DV and
    the session's tagged DV range, so the notebook can restrict by distance."""
    m = pd.read_csv(FLAGS_CSV).merge(pd.read_csv(MASTER_CSV, usecols=['session', 'unit_id', 'firing_rate']),
                                     on=['session', 'unit_id'], how='left')
    ok = (m['session'].isin(sessions) & ~m['decoder_label'].isin(['noise', 'artifact'])
          & m['default_qc'].astype(bool) & (m['isi_violations_ratio'] <= UNID_MAX_ISI)
          & (m['firing_rate'] > UNID_MIN_FR)
          & ~m['tagged'] & ~m['opto_pass'].astype(bool) & ~m['opto_candidate'])
    rng = m[m['tagged']].groupby('session')['DV'].agg(['min', 'max'])
    # distance in DV outside the session's tagged range (0 inside it)
    m['dv_outside_tagged_mm'] = np.maximum(m['session'].map(rng['min']) - m['DV'],
                                           m['DV'] - m['session'].map(rng['max'])).clip(lower=0)
    return m.loc[ok, ['session', 'unit_id', 'DV', 'dv_outside_tagged_mm', 'drn_dist_mm']].rename(columns={'unit_id': 'unit'})


def unit_sets():
    """One row per unit: tagged / matched (control of the PDFs) / unidentified flags."""
    tagged = pd.DataFrame(sorted(k for k, v in criteria_passed().items() if TAGGED_CRITERION in v),
                          columns=['session', 'unit'])
    sel = pd.read_csv(NONTAGGED_SELECTION)
    matched = pd.DataFrame({'session': sel['session'], 'unit': sel['unit_id'].astype(int),
                            'matched_tagged_unit': sel['matched_tagged_unit'].astype(int)})
    unid = unidentified_units(set(tagged['session']))
    units = (tagged.assign(tagged=True)
             .merge(matched.assign(matched=True), on=['session', 'unit'], how='outer')
             .merge(unid.assign(unidentified=True), on=['session', 'unit'], how='outer'))
    for c in ['tagged', 'matched', 'unidentified']:
        units[c] = units[c].eq(True)  # NaN (unit not in that set) -> False
    assert not (units['tagged'] & (units['matched'] | units['unidentified'])).any()
    return units.sort_values(['session', 'unit']).reset_index(drop=True)


def spont_segments(session, cut, qm_cut, opto):
    lo, hi = qm_cut
    if cut[0] is not None:
        lo = max(lo, cut[0])
    if cut[1] is not None:
        hi = min(hi, cut[1])
    blocks = sorted((g['time'].min() - BLOCK_PAD_S, g['time'].max() + g['duration'].max() + BLOCK_PAD_S)
                    for _, g in opto.groupby('pre_post'))
    segs, t = [], lo
    for a, b in blocks:
        if a > t:
            segs.append((t, min(a, hi)))
        t = max(t, b)
    if t < hi:
        segs.append((t, hi))
    return [(a, b) for a, b in segs if b > a]


def acg_counts(st, edges):
    """Spike pairs per positive-lag bin (st sorted)."""
    out = np.zeros(len(edges) - 1)
    for k in range(1, len(st)):
        d = st[k:] - st[:-k]
        if d.min() > edges[-1]:
            break
        out += np.histogram(d, edges)[0]
    return out


def lv_terms(st):
    isi = np.diff(st)
    return ((isi[:-1] - isi[1:]) / (isi[:-1] + isi[1:])) ** 2


def spike_features(st_all, segs):
    cs, cl, lv, n, dur = np.zeros(len(LAG_SHORT_MS)), np.zeros(len(LAG_LONG_MS)), [], 0, 0.0
    for a, b in segs:
        st = st_all[(st_all >= a) & (st_all <= b)]
        n += len(st)
        dur += b - a
        if len(st) > 2:
            cs += acg_counts(st, EDGES_SHORT)
            cl += acg_counts(st, EDGES_LONG)
            lv.append(lv_terms(st))
    fr = n / dur if dur > 0 else np.nan
    with np.errstate(invalid='ignore', divide='ignore'):
        acg_s = cs / (n * fr * np.diff(EDGES_SHORT))
        acg_l = cl / (n * fr * np.diff(EDGES_LONG))
    return {'n_spikes': n, 'spont_s': dur, 'fr': fr,
            'lv': 3 * np.concatenate(lv).mean() if lv and len(np.concatenate(lv)) else np.nan,
            'acg_short': acg_s, 'acg_long': acg_l}


def template_metrics(session):
    """template_metrics of the session's analyzer, indexed by unit id; None if absent."""
    path = session_dirs(session)[f'postprocessed_dir_{DATA_TYPE}']
    if path is None or not os.path.exists(path):
        return None
    m = zarr.open(path, mode='r')['extensions/template_metrics/metrics']
    tm = pd.DataFrame({c: np.asarray(m[c][:], dtype=float) for c in TEMPLATE_METRICS if c in m},
                      index=np.asarray(m['index'][:]))
    return tm.add_prefix('tm_')


def session_features(session, units):
    sd = session_dirs(session)
    opto_dir = sd[f'opto_dir_{DATA_TYPE}']
    with open(os.path.join(sd['processed_dir'], f'{session}_qm.json')) as f:
        qm_cut = json.load(f)['ephys_cut']
    opto = pd.read_csv(os.path.join(opto_dir, f'{session}_opto_session.csv'))
    unit_tbl = get_unit_tbl(session, DATA_TYPE, summary=False).set_index('unit_id')
    with open(os.path.join(opto_dir, f'{session}_waveforms_soma.pkl'), 'rb') as f:
        templates = pickle.load(f)
    with open(os.path.join(opto_dir, f'{session}_waveform_params.json')) as f:
        wp = json.load(f)
    n_samples = wp['samples_to_keep'][1] - wp['samples_to_keep'][0]
    tm = template_metrics(session)

    rows = []
    for r in units.itertuples():
        st_all = np.sort(unit_tbl.loc[r.unit, 'spike_times'])
        wf = (np.asarray(templates[r.unit], float)[wp['y_neighbors_to_keep'], :n_samples]
              if r.unit in templates else np.full(n_samples, np.nan))
        tm_row = tm.loc[r.unit].to_dict() if tm is not None and r.unit in tm.index else {}
        modes = {'nocut': [None, None]}
        if r.tagged or r.matched:
            drift = load_drift(session, r.unit, data_type=DATA_TYPE)
            modes['cut'] = list(drift['ephys_cut']) if drift is not None else nontagged_drift_cut(session, r.unit, DATA_TYPE)
        for mode, cut in modes.items():
            rows.append({'session': session, 'unit': r.unit, 'mode': mode, 'cut': cut,
                         **spike_features(st_all, spont_segments(session, cut, qm_cut, opto)),
                         'wf': wf, **tm_row})
    return rows


def build_features(jobs=8, rebuild=False):
    """(units, features): unit flags and one feature row per (unit, mode); cached in CACHE."""
    units = unit_sets()
    if not rebuild and os.path.exists(CACHE):
        with open(CACHE, 'rb') as f:
            cached_units, feats = pickle.load(f)
        if cached_units[['session', 'unit', 'tagged', 'matched', 'unidentified']].equals(
                units[['session', 'unit', 'tagged', 'matched', 'unidentified']]):
            return cached_units, feats
        print('unit sets changed since the cache was written; rebuilding')
    out = Parallel(n_jobs=jobs)(delayed(session_features)(s, g) for s, g in units.groupby('session'))
    feats = pd.DataFrame([row for rows in out for row in rows])
    feats = feats.merge(units[['session', 'unit', 'tagged', 'matched', 'unidentified']], on=['session', 'unit'])
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, 'wb') as f:
        pickle.dump((units, feats), f)
    return units, feats


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobs', type=int, default=8)
    parser.add_argument('--rebuild', action='store_true')
    args = parser.parse_args()
    units, feats = build_features(args.jobs, args.rebuild)
    print(units[['tagged', 'matched', 'unidentified']].value_counts())
    print(feats.groupby(['mode', 'tagged'])[['n_spikes', 'fr', 'lv']].median())
    print(f'wrote {CACHE}')

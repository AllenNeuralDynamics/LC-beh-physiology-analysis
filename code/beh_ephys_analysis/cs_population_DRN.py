# %% Helpers for cs_population_DRN.ipynb: select units in or near chosen 5-HT core meshes
#    (data/5HT_meshes) and summarize them as a population.
#
# Selection (select_units): master unit table + tagged flag (basic_ephys_all_DRN, as in
# unit_summary_DRN_population_options.py) + distance to each chosen mesh surface (0 inside) +
# QC + spike count over the whole session. Per-session caches go to CACHE_DIR:
#   mesh_dist/{mesh}.csv   distance of every master-table unit to the mesh
#   units/{session}.pkl    per unit: spike count, 60 s rate course, peak-channel template,
#                          waveform shape, LV in the opto-free window (no drift cut)
#   psth/{session}_unit_{id}.pkl   task / spontaneous-lick PSTHs as saved by the unit pages
#                          (unit_summary_DRN.store_psth format), recomputed when the cut changes
#
# Drift cuts (drift_cuts): 'none' (whole recording), 'existing' (tagged _opto_drift_tbl.csv, then
# the hand-placed _opto_drift_tbl_nontagged.csv, as unit_summary_DRN.curate_unit_times), or
# 'sheet' (a CSV in the unit_summary_DRN_drift_cut_sheet.py layout, written by write_cut_sheet,
# overriding 'existing' for the units it lists).

import os
import sys
import glob
import json
import pickle
import contextlib
import io

curr_path = os.path.dirname(os.path.abspath(__file__))
if curr_path not in sys.path:
    sys.path.insert(0, curr_path)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from joblib import Parallel, delayed
from trimesh import load_mesh

from unit_summary_DRN import (BREGMA_LPS_MM, CCF_COLS, MESH_DIR, PLANES, AXIS_LABEL, ML, AP, DV,
                              INK, INK_2, MUTED, MESH_EDGE, TAGGED_UNIT, NONTAGGED_POP, SPONT_WINDOW,
                              TASK_TAU_RISE, TASK_TAU_DECAY, style, ipsi_side, causal_kernel_matrix,
                              store_psth, spontaneous_licks, load_drift, nontagged_drift_cut,
                              project_to_plane, pir_to_lps)
from unit_summary_DRN_cover import (PSTH_ROWS, HEAT_PCT, HEAT_CMAP, MIN_TRIALS, WF_ALIGN, WF_PRE, WF_POST,
                                    WF_FS, unit_trace, unit_order, condition_matrix, heatmap_panel,
                                    normalize_waveform)
from unit_summary_DRN_population_options import MASTER_CSV, tagged_units
from unit_response_profiles_DRN import classify, waveform_features, PROFILES, PROFILE_COLORS
from cs_mesh_population_counts import mesh_distance_mm
from utils.beh_functions import session_dirs, makeSessionDF, get_session_tbl, get_unit_tbl
from utils.lick_utils import load_licks
from rhythmicity_DRN import spontaneous_window
from regularity_spatial import local_variation, MIN_SPIKES as LV_MIN_SPIKES


# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------

CACHE_DIR = '/root/capsule/scratch/combined/cs_population_DRN'
MESH_5HT = '/root/capsule/data/5HT_meshes'
# name: (label, file); BrainGlobe allen_mouse_25um, x = AP, y = DV, z = ML, in um
MESHES = {
    'B5': ('B5 (CS)', 'B5_CS_core_67pct.obj'),
    'B8': ('B8 (CS)', 'B8_CS_core_67pct.obj'),
    'DR': ('B6/7 (DR)', 'B6_7_DR_core_67pct.obj'),
    'wing': ('B7l (DR wing)', 'B7l_wing_core_67pct.obj'),
    'B9': ('B9 (PRNr)', 'B9_PRNr_core_67pct.obj'),
    'B3': ('B3 (RM)', 'B3_RM_core_67pct.obj'),
    'B1': ('B1 (MY)', 'B1_MY_core_67pct.obj'),
    'B1+B2': ('B1+B2 (RO/RPA)', 'B1+B2_RO_RPA_core_67pct.obj'),
}
DIST_PAD_MM = 1.0  # distances are computed up to this far from a mesh; beyond it they are inf
RATE_BIN = 60  # s, session rate course for the stability panels
PRE_EVENT, POST_EVENT, STEP = -1.5, 3, 0.05  # task PSTHs, as the unit pages
PSTH_VERSION = 1  # bump to recompute every cached PSTH
N_JOBS = 8

# categorical slots 1-4 of the dataviz reference palette, fixed order: mesh identity
CATEGORICAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
# one hue light -> dark: ordinal groups (PC1 quartiles, distance bins)
ORDINAL = ['#86b6ef', '#3987e5', '#1c5cab', '#0d366b']
MESH_FILL = {name: c for name, c in zip(MESHES, CATEGORICAL + ['#dfded8'] * 4)}
NONTAGGED_RING = '#5c5b57'
STABILITY_CMAP = LinearSegmentedColormap.from_list('stab', ['#1c5cab', '#f0efec', '#d6402a'])

MASTER_COLS = ['session', 'unit_id', 'decoder_label', 'default_qc', 'opto_pass', 'isi_violations_ratio',
               'firing_rate', 'presence_ratio', 'amplitude_cutoff', 'snr', 'depth'] + CCF_COLS


def _pickle(path, obj=None):
    if obj is None:
        with open(path, 'rb') as f:
            return pickle.load(f)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def _quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


# ---------------------------------------------------------------------
# Units, meshes, caches
# ---------------------------------------------------------------------

def master_units():
    """Master unit table with tagged / opto-responsive flags and bregma-relative ML, AP, DV (mm)."""
    m = pd.read_csv(MASTER_CSV, usecols=MASTER_COLS)
    m['unit_id'] = m['unit_id'].astype(int)
    tagged, candidates = _quiet(tagged_units)
    plt.close('all')  # apply_qc draws its own summary figure
    key = list(zip(m['session'], m['unit_id']))
    m['tagged'] = [k in tagged for k in key]
    # light responders that failed the tagged filter, or the master table's opto_pass
    m['opto_responsive'] = np.array([k in candidates for k in key]) | m['opto_pass'].astype(bool).to_numpy()
    X = m[CCF_COLS].to_numpy(float) - BREGMA_LPS_MM
    m['ML'], m['AP'], m['DV'] = X[:, ML], X[:, AP], X[:, DV]
    m['has_ccf'] = np.isfinite(X).all(axis=1)
    return m


def mesh_vertices(name):
    """Vertices of a 5-HT mesh in bregma-relative LPS mm (the units' frame)."""
    verts = np.asarray(load_mesh(os.path.join(MESH_5HT, MESHES[name][1])).vertices, float)
    return pir_to_lps(verts / 1000) - BREGMA_LPS_MM


def mesh_distances(m, names):
    """{name: distance (mm) of each row of m to that mesh}; cached per mesh for the master table."""
    out = {}
    for name in names:
        f = os.path.join(CACHE_DIR, 'mesh_dist', f'{name}.csv')
        cache = pd.read_csv(f) if os.path.exists(f) else None
        if cache is None or len(cache) != len(m) or \
                not (cache[['session', 'unit_id']].values == m[['session', 'unit_id']].values).all():
            X = m[['ML', 'AP', 'DV']].to_numpy(float)
            cache = m[['session', 'unit_id']].copy()
            cache['dist_mm'] = mesh_distance_mm(X, os.path.join(MESH_5HT, MESHES[name][1]), pad=DIST_PAD_MM)
            os.makedirs(os.path.dirname(f), exist_ok=True)
            cache.to_csv(f, index=False)
        out[name] = cache['dist_mm'].to_numpy(float)
    return out


def _session_unit_cache(session):
    """Per-unit spike count, rate course, template and waveform shape of one session."""
    tbl = _quiet(get_unit_tbl, session, 'raw', summary=False)
    if tbl is None:
        return None
    with open(os.path.join(session_dirs(session)['processed_dir'], f'{session}_qm.json')) as f:
        rec = json.load(f)['ephys_cut']
    edges = np.arange(rec[0], rec[1] + RATE_BIN, RATE_BIN)
    spont = spontaneous_window(session, 'raw')
    rows = []
    for _, r in tbl.iterrows():
        st = np.sort(np.asarray(r['spike_times'], float))
        w = np.asarray(r['waveform_mean'], float)
        ch = int(np.argmax(np.abs(w[WF_ALIGN, :])))  # peak channel as unit_summary_DRN_cover
        feats = waveform_features(w)
        lv = np.nan
        if spont is not None:
            s = st[(st >= spont[0]) & (st <= spont[1])]
            if len(s) >= LV_MIN_SPIKES:
                isi = np.diff(s)
                lv = local_variation(isi[isi > 0])
        rows.append({'unit_id': int(r['unit_id']), 'n_spikes': len(st),
                     'rate_course': (np.histogram(st, edges)[0] / RATE_BIN).astype(np.float32),
                     'wf': w[WF_ALIGN - WF_PRE:WF_ALIGN + WF_POST, ch].astype(np.float32),
                     'lv': lv, **{k: feats[k] for k in ['trough_to_peak_ms', 'half_width_ms',
                                                         'peak_trough_ratio', 'wf_amplitude']}})
    return {'session': session, 'rec_cut': rec, 'rate_t0': edges[0], 'spont_window': spont,
            'units': pd.DataFrame(rows)}


def unit_cache(sessions, n_jobs=N_JOBS):
    """Per-unit cache rows (session, unit_id, n_spikes, lv, waveform shape, wf, rate_course,
    rate_t0, spont_window) for the given sessions; builds missing sessions in parallel."""
    path = lambda s: os.path.join(CACHE_DIR, 'units', f'{s}.pkl')
    missing = [s for s in sessions if not os.path.exists(path(s))]
    if missing:
        print(f'building unit cache for {len(missing)} sessions...')
        for s, c in zip(missing, Parallel(n_jobs=n_jobs)(delayed(_session_unit_cache)(s) for s in missing)):
            if c is None:
                print(f'  no unit table: {s}')
                continue
            _pickle(path(s), c)
    parts = []
    for s in sessions:
        if not os.path.exists(path(s)):
            continue
        c = _pickle(path(s))
        u = c['units'].copy()
        u['session'], u['rate_t0'] = s, c['rate_t0']
        u['spont_window'] = [c['spont_window']] * len(u)
        parts.append(u)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ---------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------

def select_units(m, meshes=('B5', 'B8'), dist_um=(0, 100), groups=('untagged',), min_spikes=200,
                 require_default_qc=True, max_isi=None, min_firing_rate=None, exclude_meshes=(),
                 exclude_within_um=0, exclude_opto_responsive=False, sessions=None, n_jobs=N_JOBS):
    """Units within [dist_um[0], dist_um[1]] um of the nearest chosen mesh (0 = inside).

    groups: any of 'tagged', 'untagged'. max_isi: keep isi_violations_ratio < max_isi (None = no limit). exclude_meshes: drop units within exclude_within_um of
    these meshes (e.g. ('DR', 'wing') to keep DR units out of a B8 population).
    exclude_opto_responsive: drop untagged units that responded to light (opto candidates
    failing the tagged filter, or opto_pass). min_spikes: more than this many spikes over the
    whole session. Returns (selection, funnel): one row per selected unit, and the tagged /
    untagged count after each step.
    """
    meshes = list(meshes)
    dist = mesh_distances(m, meshes + [e for e in exclude_meshes if e not in meshes])
    d = m.copy()
    D = np.column_stack([dist[n] for n in meshes])
    d['dist_um'] = 1000 * np.min(D, axis=1)
    near = np.argmin(np.where(np.isfinite(D), D, np.inf), axis=1)
    d['nearest_mesh'] = np.where(np.isfinite(D).any(axis=1), np.array(meshes)[near], None)
    for n in meshes:
        d[f'dist_um_{n}'] = 1000 * dist[n]

    steps = [('all units in master table', np.ones(len(d), bool))]
    if sessions is not None:
        steps.append((f'in {len(sessions)} chosen sessions', d['session'].isin(sessions).values))
    steps.append(('CCF location', d['has_ccf'].values))
    steps.append((f'{dist_um[0]}-{dist_um[1]} um from {" / ".join(meshes)}',
                  d['dist_um'].between(*dist_um).values))
    if exclude_meshes:
        far = np.all([1000 * dist[e] > exclude_within_um for e in exclude_meshes], axis=0)
        steps.append((f'> {exclude_within_um} um from {" / ".join(exclude_meshes)}', far))
    if require_default_qc:
        steps.append(('default_qc, not noise/artifact',
                      (d['default_qc'].astype(bool) & ~d['decoder_label'].isin(['noise', 'artifact'])).values))
    if max_isi is not None:
        # strict, and units without an ISI value are dropped
        steps.append((f'ISI violations ratio < {max_isi}', (d['isi_violations_ratio'] < max_isi).values))
    if min_firing_rate is not None:
        steps.append((f'firing rate >= {min_firing_rate} Hz', (d['firing_rate'] >= min_firing_rate).values))
    if exclude_opto_responsive:
        steps.append(('untagged: no light response', (d['tagged'] | ~d['opto_responsive']).values))

    keep = np.ones(len(d), bool)
    funnel = []
    for name, mask in steps:
        keep &= mask
        funnel.append({'step': name, 'tagged': int((keep & d['tagged']).sum()),
                       'untagged': int((keep & ~d['tagged']).sum())})
    d = d[keep]
    # spike counts only for units that got this far
    cache = unit_cache(sorted(d['session'].unique()), n_jobs=n_jobs)
    d = d.merge(cache, on=['session', 'unit_id'], how='left')
    d = d[d['n_spikes'] > min_spikes]
    funnel.append({'step': f'> {min_spikes} spikes in session', 'tagged': int(d['tagged'].sum()),
                   'untagged': int((~d['tagged']).sum())})
    want = {'tagged': d['tagged'], 'untagged': ~d['tagged']}
    d = d[np.any([want[g] for g in groups], axis=0)]
    funnel.append({'step': f'groups: {", ".join(groups)}', 'tagged': int(d['tagged'].sum()),
                   'untagged': int((~d['tagged']).sum())})
    d['group'] = np.where(d['tagged'], 'tagged', 'untagged')
    return d.reset_index(drop=True), pd.DataFrame(funnel)


def distance_bins(sel, edges_um):
    """Label of each unit's distance bin, e.g. edges (0, 50, 100) -> 'inside', '0-50 um', '50-100 um'."""
    def lab(x):
        if x == 0 and edges_um[0] == 0:
            return 'inside'
        for lo, hi in zip(edges_um[:-1], edges_um[1:]):
            if lo < x <= hi or (lo == x == edges_um[0]):
                return f'{lo:g}-{hi:g} um'
        return None
    return sel['dist_um'].apply(lab)


def make_groups(sel, by='group', order=None):
    """{group name: (list of (session, unit_id) keys, color)} for plotting, split by a column of sel."""
    vals = sel[by].astype(str)
    names = order or sorted(vals.unique(), key=lambda v: (v != 'inside', v))
    if by == 'group':
        names = [n for n in ['tagged', 'untagged'] if n in set(vals)]
    out = {}
    for i, name in enumerate(names):
        s = sel[vals == name]
        if len(s) == 0:
            continue
        if by == 'group':
            color = TAGGED_UNIT if name == 'tagged' else NONTAGGED_POP
        elif by == 'nearest_mesh':
            color = MESH_FILL.get(name, MUTED)
        else:
            pal = ORDINAL if len(names) <= len(ORDINAL) else CATEGORICAL + ORDINAL
            color = pal[i] if i < len(pal) else MUTED  # never cycle hues
        out[f'{name} (n={len(s)})'] = (list(zip(s['session'], s['unit_id'])), color)
    return out


def counts_table(sel, edges_um=None):
    """Unit and session counts per nearest mesh and group (and distance bin if edges_um)."""
    d = sel.copy()
    cols = ['nearest_mesh']
    if edges_um is not None:
        d['distance'] = distance_bins(d, edges_um)
        cols.append('distance')
    t = d.pivot_table(index=cols, columns='group', values='unit_id', aggfunc='count', fill_value=0)
    t['sessions'] = d.groupby(cols)['session'].nunique()
    return t


# ---------------------------------------------------------------------
# Drift cuts
# ---------------------------------------------------------------------

SHEET_COLS = ['session', 'unit_id', 'ephys_cut_start', 'ephys_cut_end', 'drift_unit', 'exclude', 'notes',
              'needs_review', 'group', 'nearest_mesh', 'dist_um', 'existing_cut_source']


def _nan_none(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else float(x)


def existing_cut(session, unit_id):
    """(start, end, source) of the unit's current drift cut; None = no cut on that side."""
    d = load_drift(session, int(unit_id), data_type='raw')
    if d is not None:
        c = list(d['ephys_cut'])
        return _nan_none(c[0]), _nan_none(c[1]), 'opto_drift_tbl'
    c = nontagged_drift_cut(session, int(unit_id), 'raw')
    src = 'opto_drift_tbl_nontagged' if c != [None, None] else 'none'
    return _nan_none(c[0]), _nan_none(c[1]), src


def drift_cuts(sel, mode='none', sheet=None):
    """sel with cut_start, cut_end (s, None = no cut), cut_source and, for mode 'sheet', units the
    sheet marks exclude=TRUE dropped. Modes: 'none', 'existing', 'sheet' (sheet overrides existing)."""
    d = sel.copy()
    if mode == 'none':
        d['cut_start'], d['cut_end'], d['cut_source'] = None, None, 'none'
        return d
    cuts = [existing_cut(s, u) for s, u in zip(d['session'], d['unit_id'])]
    d['cut_start'] = [c[0] for c in cuts]
    d['cut_end'] = [c[1] for c in cuts]
    d['cut_source'] = [c[2] for c in cuts]
    if mode == 'sheet':
        sh = pd.read_csv(sheet)
        sh['unit_id'] = sh['unit_id'].astype(int)
        sh = sh.set_index(['session', 'unit_id'])
        truthy = lambda v: str(v).strip().upper() in ('TRUE', '1', '1.0', 'YES')
        drop = []
        for i, k in enumerate(zip(d['session'], d['unit_id'])):
            if k not in sh.index:
                continue
            r = sh.loc[k]
            if truthy(r.get('exclude', '')):
                drop.append(i)
                continue
            if truthy(r.get('drift_unit', '')) or pd.notna(r['ephys_cut_start']) or pd.notna(r['ephys_cut_end']):
                d.iat[i, d.columns.get_loc('cut_start')] = _nan_none(pd.to_numeric(r['ephys_cut_start'], errors='coerce'))
                d.iat[i, d.columns.get_loc('cut_end')] = _nan_none(pd.to_numeric(r['ephys_cut_end'], errors='coerce'))
                d.iat[i, d.columns.get_loc('cut_source')] = 'sheet'
        if drop:
            print(f'{len(drop)} units excluded by the sheet')
        d = d.drop(index=d.index[drop]).reset_index(drop=True)
    elif mode != 'existing':
        raise ValueError(f'unknown drift-cut mode {mode!r}')
    return d


def write_cut_sheet(sel, path):
    """Blank cut sheet for sel (unit_summary_DRN_drift_cut_sheet.py columns), pre-filled with existing
    cuts; fill ephys_cut_start / ephys_cut_end / exclude, then use drift_cuts(sel, 'sheet', path).
    Never overwrites an existing file."""
    if os.path.exists(path):
        raise FileExistsError(f'{path} exists; pick a new name so hand-entered cuts are not lost')
    cuts = [existing_cut(s, u) for s, u in zip(sel['session'], sel['unit_id'])]
    sh = sel[['session', 'unit_id', 'group', 'nearest_mesh', 'dist_um']].copy()
    sh['ephys_cut_start'] = [c[0] for c in cuts]
    sh['ephys_cut_end'] = [c[1] for c in cuts]
    sh['existing_cut_source'] = [c[2] for c in cuts]
    sh['drift_unit'] = sh['ephys_cut_start'].notna() | sh['ephys_cut_end'].notna()
    sh['exclude'], sh['notes'], sh['needs_review'] = False, '', True
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sh[SHEET_COLS].to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------
# PSTHs
# ---------------------------------------------------------------------

def spont_matrix(spike_times, times, avoid_win=(0, 1.5), tau_rise=0.01, tau_decay=0.08):
    """get_spike_matrix(..., avoid_overlap=True, kernel=True) for spontaneous lick trains, vectorized:
    causal PSP rates, NaN where a bin's kernel window [t - 10 tau_decay, t) meets another train
    start's avoid_win."""
    matrix, bins = causal_kernel_matrix(spike_times, times, SPONT_WINDOW[0], SPONT_WINDOW[1], STEP,
                                        tau_rise=tau_rise, tau_decay=tau_decay)
    reach = 10 * tau_decay
    times = np.asarray(times, float)
    for i, ev in enumerate(times):
        lo, hi = ev + bins - reach, ev + bins
        near = times[(times > ev + SPONT_WINDOW[0] - reach - avoid_win[1]) & (times < ev + SPONT_WINDOW[1] - avoid_win[0])]
        near = near[near != ev]
        if len(near):
            bad = ((near[None, :] + avoid_win[1] >= lo[:, None]) & (near[None, :] + avoid_win[0] < hi[:, None])).any(axis=1)
            matrix[i, bad] = np.nan
    return matrix, bins


def unit_psths(spike_times, session_df, tblTrials, spont, side):
    """The unit pages' task and spontaneous-lick PSTHs (unit_summary_DRN.behavior_row), no figure."""
    store = {}
    if session_df is None or len(session_df) == 0:
        return store
    info = dict(stepSize=STEP, kernel='psp', tau_rise=TASK_TAU_RISE, tau_decay=TASK_TAU_DECAY)

    def put(key, times):
        times = np.asarray(times, float)
        matrix, bins = causal_kernel_matrix(spike_times, times, PRE_EVENT, POST_EVENT, STEP)
        store_psth(store, key, matrix, bins, times, spike_times=spike_times, **info)

    put('go_cue_all', session_df['go_cue_time'].values)
    put('response_all', session_df['choice_time'].values)
    rewarded = session_df['outcome'].values.astype(bool)
    choice = session_df['choice'].values.astype(int)
    ipsi_choice = 1 if side in (None, 'right') else 0  # choice 1 is a right lick
    ipsi = choice == ipsi_choice
    ignored = tblTrials[tblTrials['animal_response'] == 2]
    for col, ignore_col, align in [('go_cue_time', 'goCue_start_time', 'go_cue'),
                                   ('choice_time', 'reward_outcome_time', 'response')]:
        t = session_df[col].values
        for mask, side_key in [(ipsi, 'ipsi'), (~ipsi, 'contra')]:
            put(f'{align}_{side_key}_rewarded', t[mask & rewarded])
            put(f'{align}_{side_key}_unrewarded', t[mask & ~rewarded])
        put(f'{align}_ignored', ignored[ignore_col].values)
    if spont is not None:
        ik, ck = ('R', 'L') if ipsi_choice == 1 else ('L', 'R')
        times = np.concatenate([spont[ik], spont[ck]])
        is_ipsi = np.r_[np.ones(len(spont[ik])), np.zeros(len(spont[ck]))].astype(bool)
        order = np.argsort(times)
        times, is_ipsi = times[order], is_ipsi[order]
        if len(times):
            matrix, bins = spont_matrix(spike_times, times)
            sinfo = dict(kernel='psp', tau_rise=0.01, tau_decay=0.08, avoid_overlap=True)
            store_psth(store, 'spont_ipsi', matrix[is_ipsi], bins, times[is_ipsi], **sinfo)
            store_psth(store, 'spont_contra', matrix[~is_ipsi], bins, times[~is_ipsi], **sinfo)
    return store


def _psth_path(session, unit_id):
    return os.path.join(CACHE_DIR, 'psth', f'{session}_unit_{int(unit_id)}.pkl')


def _session_psths(session, units):
    """units: list of (unit_id, cut_start, cut_end, x_ccf). Writes one pickle per unit."""
    tbl = _quiet(get_unit_tbl, session, 'raw', summary=False)
    sd = session_dirs(session)
    session_df = tblTrials = licks = None
    if os.path.exists(os.path.join(sd['beh_fig_dir'], f'{session}.nwb')):
        session_df = _quiet(makeSessionDF, session)
        tblTrials = _quiet(get_session_tbl, session)
        try:
            licks = _quiet(load_licks, session)
        except Exception as e:
            print(f'No licks for {session}: {type(e).__name__}: {e}')
    with open(os.path.join(sd['processed_dir'], f'{session}_qm.json')) as f:
        qm = json.load(f)
    data = {'licks': licks, 'qm': qm}
    for unit_id, c0, c1, x_ccf in units:
        st = np.sort(np.asarray(tbl.loc[tbl['unit_id'] == unit_id, 'spike_times'].values[0], float))
        sdf, tt = session_df, tblTrials
        lo = -np.inf if c0 is None else c0
        hi = np.inf if c1 is None else c1
        st = st[(st >= lo) & (st <= hi)]
        if sdf is not None:
            sdf = sdf[(sdf['go_cue_time'] >= lo) & (sdf['go_cue_time'] <= hi)]
            tt = tt[(tt['goCue_start_time'] >= lo) & (tt['goCue_start_time'] <= hi)]
        side, ml = ipsi_side({'x_ccf': x_ccf})
        spont = spontaneous_licks(data, tt, [c0, c1])
        rec = [max(qm['ephys_cut'][0], lo), min(qm['ephys_cut'][1], hi)]
        _pickle(_psth_path(session, unit_id), {
            'session': session, 'unit_id': int(unit_id), 'ephys_cut': [c0, c1], 'version': PSTH_VERSION,
            'ipsi_side': side, 'ml_mm': ml, 'n_spikes_cut': len(st),
            'firing_rate_cut': len(st) / max(rec[1] - rec[0], 1e-9), 'n_trials': 0 if sdf is None else len(sdf),
            'psth': unit_psths(st, sdf, tt, spont, side)})
    return session


def population_psths(sel, n_jobs=N_JOBS):
    """{(session, unit_id): PSTH dict} for sel (with cut_start / cut_end from drift_cuts); units
    whose cached PSTHs used a different cut are recomputed, one job per session."""
    todo = {}
    out = {}
    for r in sel.itertuples():
        f = _psth_path(r.session, r.unit_id)
        cut = [_nan_none(r.cut_start), _nan_none(r.cut_end)]
        if os.path.exists(f):
            d = _pickle(f)
            if d['ephys_cut'] == cut and d.get('version') == PSTH_VERSION:
                out[(r.session, r.unit_id)] = d
                continue
        todo.setdefault(r.session, []).append((r.unit_id, *cut, r.x_ccf))
    if todo:
        n = sum(len(v) for v in todo.values())
        print(f'computing PSTHs for {n} units in {len(todo)} sessions...')
        Parallel(n_jobs=n_jobs)(delayed(_session_psths)(s, u) for s, u in todo.items())
        for s, units in todo.items():
            for u in units:
                out[(s, u[0])] = _pickle(_psth_path(s, u[0]))
    return out


def add_cut_stats(sel, psths):
    """sel with n_spikes_cut, firing_rate_cut and n_trials from the PSTH cache."""
    d = sel.copy()
    for col in ['n_spikes_cut', 'firing_rate_cut', 'n_trials']:
        d[col] = [psths[k][col] if k in psths else np.nan for k in zip(d['session'], d['unit_id'])]
    return d


def response_profiles(sel, psths):
    """sel with the unit_response_profiles_DRN classification (profile_go_cue, profile_response, ...)."""
    rows = [classify(psths[k]) if k in psths else {} for k in zip(sel['session'], sel['unit_id'])]
    return pd.concat([sel.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------

def _plane_background(meshes, context):
    """Vertices of the chosen and context meshes and of the aqueduct (None if missing)."""
    verts = {n: mesh_vertices(n) for n in list(meshes) + [c for c in context if c not in meshes]}
    aq = os.path.join(MESH_DIR, 'AQ_bregma_lps_mm.obj')
    aq = np.asarray(load_mesh(aq).vertices, float) if os.path.exists(aq) else None
    return verts, aq


def _draw_plane(ax, verts, aq, meshes, context, xd, yd):
    """Chosen meshes filled, context meshes dashed, aqueduct outlined, projected onto (xd, yd)."""
    for n in meshes:
        for j, c in enumerate(project_to_plane(verts[n], [xd, yd], pitch=0.02, margin=0.5)):
            ax.fill(c[:, 0], c[:, 1], facecolor=MESH_FILL[n], alpha=0.18, edgecolor=MESH_FILL[n],
                    linewidth=0.8, zorder=0, label=MESHES[n][0] if j == 0 else None)
    for n in context:
        if n in meshes:
            continue
        for j, c in enumerate(project_to_plane(verts[n], [xd, yd], pitch=0.02, margin=0.5)):
            ax.plot(c[:, 0], c[:, 1], color=MESH_EDGE, lw=0.9, ls='--', zorder=0,
                    label=f'{MESHES[n][0]} (context)' if j == 0 else None)
    if aq is not None:
        for c in project_to_plane(aq, [xd, yd], pitch=0.02, margin=0.5):
            ax.plot(c[:, 0], c[:, 1], color=MUTED, lw=0.6, zorder=1)


def _finish_plane(ax, verts, meshes, sel, title, xd, yd, pad):
    cols = ['ML', 'AP', 'DV']
    pts = np.vstack([np.vstack([verts[n] for n in meshes]), sel[cols].to_numpy(float)])
    ax.set_xlim(pts[:, xd].min() - pad, pts[:, xd].max() + pad)
    ax.set_ylim(pts[:, yd].min() - pad, pts[:, yd].max() + pad)
    ax.set_aspect('equal')
    ax.set_xlabel(AXIS_LABEL[xd], color=INK_2)
    ax.set_ylabel(AXIS_LABEL[yd], color=INK_2)
    ax.set_title(title, loc='left', color=INK)
    style(ax)


def plane_maps(sel, meshes, context=('DR', 'wing'), color_by='group', palette=None, order=None,
               figsize=(20, 6.5), pad=0.25):
    """Chosen meshes (filled) and context meshes (outlined) in three planes, with the units.
    color_by: 'group' (tagged filled red, untagged open rings), 'nearest_mesh', a categorical column
    with palette {value: color} (drawn in `order`, first at the back; e.g. 'profile_go_cue' with
    PROFILE_COLORS, the default for profile_* columns) or a numeric column such as 'dist_um'."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    verts, aq = _plane_background(meshes, context)
    cols = ['ML', 'AP', 'DV']
    if palette is None and color_by.startswith('profile_'):
        palette, order = PROFILE_COLORS, order or ['non-responsive'] + [p for p in PROFILES if p != 'non-responsive']
    sc = None
    for ax, (title, xd, yd) in zip(axes, PLANES):
        _draw_plane(ax, verts, aq, meshes, context, xd, yd)
        if color_by == 'group':
            u = sel[~sel['tagged']]
            if len(u):
                ax.scatter(u[cols[xd]], u[cols[yd]], s=14, facecolor='none', edgecolor=NONTAGGED_RING, lw=0.6,
                           zorder=2, label=f'untagged (n={len(u)})')
            u = sel[sel['tagged']]
            if len(u):
                ax.scatter(u[cols[xd]], u[cols[yd]], s=16, color=TAGGED_UNIT, lw=0, zorder=3,
                           label=f'tagged (n={len(u)})')
        elif color_by == 'nearest_mesh':
            for n in meshes:
                u = sel[sel['nearest_mesh'] == n]
                ax.scatter(u[cols[xd]], u[cols[yd]], s=14, color=MESH_FILL[n], edgecolor='white', lw=0.3,
                           zorder=2, label=f'nearest {n} (n={len(u)})')
        elif palette is not None:
            for k, v in enumerate(order or list(palette)):
                u = sel[sel[color_by] == v]
                if len(u):
                    ax.scatter(u[cols[xd]], u[cols[yd]], s=14, color=palette[v], edgecolor='white', lw=0.3,
                               zorder=2 + k, label=f'{v} (n={len(u)})')
            n_none = int(sel[color_by].isna().sum())
            if n_none and ax is axes[0]:
                ax.plot([], [], ' ', label=f'not classified (n={n_none}, not drawn)')
        else:
            sc = ax.scatter(sel[cols[xd]], sel[cols[yd]], s=14, c=sel[color_by], cmap='viridis',
                            edgecolor='white', lw=0.3, zorder=2)
        _finish_plane(ax, verts, meshes, sel, title, xd, yd, pad)
    axes[0].legend(fontsize=8, frameon=False, loc='best')
    if sc is not None:
        fig.colorbar(sc, ax=axes, shrink=0.7, label=color_by)
    return fig


def plane_maps_by_category(sel, meshes, col, palette=None, order=None, context=('DR', 'wing'), pad=0.25,
                           panel_size=5.5):
    """One row of three planes per value of a categorical column (e.g. 'profile_go_cue'): that value's
    units in colour over every other unit of sel in light gray, so each class's spatial spread reads
    without the others hiding it. The row title gives the class's count and share of classified units."""
    if palette is None and col.startswith('profile_'):
        palette = PROFILE_COLORS
    order = [v for v in (order or list(palette)) if (sel[col] == v).any()]
    verts, aq = _plane_background(meshes, context)
    cols = ['ML', 'AP', 'DV']
    n_cls = int(sel[col].notna().sum())
    fig, axes = plt.subplots(len(order), 3, figsize=(3.4 * panel_size, panel_size * len(order) * 0.95), squeeze=False)
    for r, v in enumerate(order):
        u = sel[sel[col] == v]
        rest = sel[sel[col] != v]
        for c, (title, xd, yd) in enumerate(PLANES):
            ax = axes[r, c]
            _draw_plane(ax, verts, aq, meshes, context, xd, yd)
            ax.scatter(rest[cols[xd]], rest[cols[yd]], s=6, color='#b9b7b0', alpha=0.6, lw=0, zorder=2)
            ax.scatter(u[cols[xd]], u[cols[yd]], s=14, color=palette[v], edgecolor='white', lw=0.3, zorder=3)
            head = f'{v}: n={len(u)} ({100 * len(u) / max(n_cls, 1):.0f}% of {n_cls} classified)  |  '
            _finish_plane(ax, verts, meshes, sel, head + title if c == 0 else title, xd, yd, pad)
    fig.tight_layout()
    return fig


METRICS = [('firing_rate', 'Firing rate (Hz)', 'log'), ('n_spikes', 'Spikes in session', 'log'),
           ('isi_violations_ratio', 'ISI violations ratio', 'linear'), ('snr', 'SNR', 'linear'),
           ('lv', 'LV (ISI), opto-free window', 'linear'), ('trough_to_peak_ms', 'Trough to peak (ms)', 'linear'),
           ('half_width_ms', 'Trough half-width (ms)', 'linear'), ('wf_amplitude', 'Template amplitude (uV)', 'linear'),
           ('presence_ratio', 'Presence ratio', 'linear'), ('dist_um', 'Distance to nearest mesh (um)', 'linear'),
           ('DV', 'DV from bregma (mm)', 'linear'), ('depth', 'Probe depth (um)', 'linear')]


def metric_hists(sel, groups_by='group', metrics=METRICS, ncols=4):
    """One normalized histogram per metric, one step outline per group (column groups_by)."""
    names = [n for n in (['tagged', 'untagged'] if groups_by == 'group' else sorted(sel[groups_by].astype(str).unique()))
             if (sel[groups_by].astype(str) == n).any()]
    colors = make_groups(sel, groups_by)
    colors = {k.split(' (n=')[0]: v[1] for k, v in colors.items()}
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.3 * nrows))
    for ax, (col, label, scale) in zip(axes.flat, metrics):
        vals = sel[col].to_numpy(float)
        vals = vals[np.isfinite(vals)]
        if scale == 'log':
            vals = vals[vals > 0]
        if len(vals) == 0:
            ax.axis('off')
            continue
        if scale == 'log':
            bins = np.logspace(np.log10(vals.min()), np.log10(np.percentile(vals, 99.5)), 30)
            ax.set_xscale('log')
        else:
            bins = np.linspace(vals.min(), np.percentile(vals, 99.5) + 1e-9, 30)
        for n in names:
            x = sel.loc[sel[groups_by].astype(str) == n, col].to_numpy(float)
            x = np.clip(x[np.isfinite(x)], bins[0], bins[-1])
            if len(x) == 0:
                continue
            w = np.full(len(x), 1 / len(x))
            ax.hist(x, bins=bins, weights=w, histtype='step', color=colors[n], lw=1.5,
                    label=f'{n} (n={len(x)}, median {np.median(x):.3g})')
        ax.set_xlabel(label)
        ax.set_ylabel('Fraction of units')
        ax.legend(fontsize=7, frameon=False)
        style(ax)
    for ax in list(axes.flat)[len(metrics):]:
        ax.axis('off')
    fig.tight_layout()
    return fig


def _xlabel(key):
    if key.startswith('spont'):
        return 'Time from lick-train start (s)'
    if key.startswith('go_cue'):
        return 'Time from go cue (s)'
    return 'Time from response-window end (s)' if key.endswith('ignored') else 'Time from response (s)'


def _norm_label(normalize):
    return 'Rate - baseline (Hz)' if normalize == 'delta' else 'z-score (baseline SD)'


def group_mean(psths, keys, key, normalize):
    """(t, mean, sem, n) of the normalized traces of keys for condition key, or None."""
    traces = [unit_trace(psths[k], key, normalize) for k in keys if k in psths]
    traces = [tr for tr in traces if tr is not None]
    if not traces:
        return None
    M = np.vstack([tr[1] for tr in traces])
    n = np.sum(np.isfinite(M), axis=0).clip(1)
    return traces[0][0], np.nanmean(M, axis=0), np.nanstd(M, axis=0) / np.sqrt(n), len(traces)


def psth_grid(psths, groups, normalize='delta', rows=PSTH_ROWS, share_y=True):
    """Mean +/- SEM population PSTH per condition (rows of (key, title)), one line per group."""
    ncols = max(len(r) for r in rows)
    fig, axes = plt.subplots(len(rows), ncols, figsize=(3.3 * ncols, 3.0 * len(rows)), squeeze=False)
    for r, row in enumerate(rows):
        for c, (key, title) in enumerate(row):
            ax = axes[r, c]
            for name, (keys, color) in groups.items():
                g = group_mean(psths, keys, key, normalize)
                if g is None:
                    continue
                t, mu, se, n = g
                ax.fill_between(t, mu - se, mu + se, color=color, alpha=0.25, lw=0)
                ax.plot(t, mu, color=color, lw=1.8, label=f'{name.split(" (n=")[0]} ({n})')
            ax.axvline(0, color=MUTED, ls='--', lw=1, zorder=0)
            ax.axhline(0, color=MUTED, lw=0.6, zorder=0)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel(_xlabel(key), fontsize=8)
            if c == 0:
                ax.set_ylabel(_norm_label(normalize))
            ax.legend(fontsize=6, frameon=False, loc='upper left')
            style(ax)
        for ax in axes[r, len(row):]:
            ax.axis('off')
    if share_y:
        for kind in ['task', 'spont']:
            axs = [axes[r, c] for r, row in enumerate(rows) for c, (k, _) in enumerate(row)
                   if k.startswith('spont') == (kind == 'spont')]
            if axs:
                lo, hi = min(a.get_ylim()[0] for a in axs), max(a.get_ylim()[1] for a in axs)
                for a in axs:
                    a.set_ylim(lo, hi)
    fig.tight_layout()
    return fig


def population_heatmaps(psths, groups, normalize='delta', keys=None, sort_key='go_cue_all'):
    """Colormap raster per condition (columns) and group (rows): one row per unit, its normalized mean
    trace, units sorted by their 0-0.5 s response to the go cue (strongest on top), one colour scale."""
    keys = keys or ['go_cue_all', 'response_all', 'spont_ipsi', 'spont_contra']
    orders = {g: unit_order(psths, [k for k in ks if k in psths], normalize) for g, (ks, _) in groups.items()}
    mats = {(c, g): condition_matrix(psths, orders[g], c, normalize) for c in keys for g in groups}
    vals = np.concatenate([M[np.isfinite(M)] for _, M in mats.values()] or [np.zeros(1)])
    lims = np.percentile(vals, HEAT_PCT) if len(vals) else (0, 1)
    heights = [max(len(o), 5) for o in orders.values()]
    fig, axes = plt.subplots(len(groups), len(keys), figsize=(3.4 * len(keys), 1.5 + 0.012 * sum(heights) + 1.5 * len(groups)),
                             squeeze=False, gridspec_kw=dict(height_ratios=heights))
    im = None
    for j, c in enumerate(keys):
        for i, g in enumerate(groups):
            t, M = mats[(c, g)]
            h = heatmap_panel(axes[i, j], t, M, *lims, g.split(' (n=')[0] if j == 0 else '', show_x=(i == len(groups) - 1))
            im = h if h is not None else im
            if i == 0:
                axes[i, j].set_title(c.replace('_', ' '), fontsize=10)
            if i == len(groups) - 1:
                axes[i, j].set_xlabel(_xlabel(c), fontsize=8)
    if im is not None:
        fig.colorbar(im, ax=axes, shrink=0.6, label=f'{_norm_label(normalize)}, {HEAT_PCT[0]}-{HEAT_PCT[1]}th pct')
    return fig


def unit_rasters(psths, keys, condition='go_cue_all', ncols=8, vmax_pct=99, titles=None):
    """Trial x time rate colormap rasters of individual units (session order top to bottom), each
    on its own 0..vmax_pct-percentile scale."""
    keys = [k for k in keys if k in psths and condition in psths[k]['psth']]
    nrows = int(np.ceil(len(keys) / ncols)) or 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.4 * ncols, 2.2 * nrows), squeeze=False)
    for ax, k in zip(axes.flat, keys):
        p = psths[k]['psth'][condition]
        M = np.asarray(p['matrix'], float)
        t = p['bin_times']
        vmax = np.nanpercentile(M, vmax_pct) if M.size else 1
        ax.imshow(M, aspect='auto', cmap=HEAT_CMAP, vmin=0, vmax=max(vmax, 1e-3), interpolation='nearest',
                  extent=[t[0], t[-1], len(M) + 0.5, 0.5])
        ax.axvline(0, color='white', ls='--', lw=0.7)
        ax.set_title(titles[k] if titles else f'{k[0].split("_")[1]} u{k[1]}\n0-{vmax:.0f} Hz', fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_yticks([])
    for ax in list(axes.flat)[len(keys):]:
        ax.axis('off')
    fig.supxlabel(_xlabel(condition), fontsize=9)
    fig.tight_layout()
    return fig


def waveform_matrix(sel):
    """(keys, X): normalized (trough = -1) peak-channel templates of sel, malformed shapes left out."""
    keys, rows = [], []
    for r in sel.itertuples():
        x = normalize_waveform(np.asarray(r.wf, float))
        if x is not None:
            keys.append((r.session, r.unit_id))
            rows.append(-x)
    return keys, np.vstack(rows)


def wf_time():
    return 1000 * (np.arange(WF_PRE + WF_POST) - WF_PRE) / WF_FS


def waveform_outliers(X, outlier_sd=6):
    """Rows of X further than outlier_sd robust SDs (1.4826 x MAD) from the median on PC1 or PC2."""
    from sklearn.decomposition import PCA
    S0 = PCA(n_components=2, random_state=0).fit_transform(X)
    med = np.median(S0, axis=0)
    mad = 1.4826 * np.median(np.abs(S0 - med), axis=0)
    return (np.abs(S0 - med) > outlier_sd * mad).any(axis=1)


def waveform_panels(sel, groups, n_show=150, seed=0, ylim='auto', ylim_pct=(0.5, 99.5), outlier_sd=None):
    """Per group: up to n_show individual normalized templates (thin) with mean +/- SD; last panel:
    the group means together. ylim: (lo, hi), 'auto' (the ylim_pct percentiles of every template
    sample, padded 10%, so a few malformed templates cannot squash the rest; traces beyond are clipped
    by the axes) or None (matplotlib's own). outlier_sd: leave out templates flagged by
    waveform_outliers (as in the PCA) from the traces and the means; None keeps all."""
    keys, X = waveform_matrix(sel)
    if outlier_sd is not None:
        out = waveform_outliers(X, outlier_sd)
        print(f'{out.sum()} of {len(keys)} templates left out as outliers (> {outlier_sd} robust SD)')
        keys, X = [k for k, o in zip(keys, out) if not o], X[~out]
    if isinstance(ylim, str) and ylim == 'auto':
        lo, hi = np.percentile(X, ylim_pct)
        ylim = (lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo))
    idx = {k: i for i, k in enumerate(keys)}
    t = wf_time()
    fig, axes = plt.subplots(1, len(groups) + 1, figsize=(4.5 * (len(groups) + 1), 3.8), squeeze=False,
                             sharey=ylim is not None)
    rng = np.random.default_rng(seed)
    for ax, (name, (ks, color)) in zip(axes[0], groups.items()):
        W = X[[idx[k] for k in ks if k in idx]]
        if len(W) == 0:
            ax.axis('off')
            continue
        show = W[rng.choice(len(W), min(n_show, len(W)), replace=False)]
        ax.plot(t, show.T, color=color, lw=0.3, alpha=0.35)
        mu, sd = W.mean(0), W.std(0)
        ax.fill_between(t, mu - sd, mu + sd, color=INK, alpha=0.12, lw=0)
        ax.plot(t, mu, color=INK, lw=1.8)
        ax.set_title(f'{name.split(" (n=")[0]} (n={len(W)}; {min(n_show, len(W))} shown)', fontsize=10)
        axes[0, -1].plot(t, mu, color=color, lw=1.8, label=name.split(' (n=')[0])
        axes[0, -1].fill_between(t, mu - W.std(0) / np.sqrt(len(W)), mu + W.std(0) / np.sqrt(len(W)),
                                 color=color, alpha=0.3, lw=0)
    axes[0, -1].set_title('Group means +/- SEM', fontsize=10)
    axes[0, -1].legend(fontsize=8, frameon=False)
    for ax in axes[0]:
        ax.set_xlabel('Time from trough (ms)')
        ax.axhline(0, color=MUTED, lw=0.5)
        if ylim is not None:
            ax.set_ylim(ylim)
        style(ax)
    axes[0, 0].set_ylabel('Normalized amplitude (trough = -1)')
    fig.tight_layout()
    return fig


def waveform_pca(sel, n_components=3, outlier_sd=6):
    """sel with wf_pc1..3 and wf_pc1_quartile (Q1 = lowest PC1), plus the fitted PCA. One PCA over
    the units in sel, on normalized templates (as F_waveform_space_DRN_v2 / the cover page).
    Templates further than outlier_sd robust SDs (1.4826 x MAD) from the median on PC1 or PC2 of a
    first fit are left out (wf_outlier = True) and the PCA is refit without them; None keeps all."""
    from sklearn.decomposition import PCA
    keys, X = waveform_matrix(sel)
    out = np.zeros(len(keys), bool)
    if outlier_sd is not None:
        out = waveform_outliers(X, outlier_sd)
        print(f'{out.sum()} of {len(keys)} templates left out of the PCA as outliers (> {outlier_sd} robust SD)')
    pca = PCA(n_components=n_components, random_state=0).fit(X[~out])
    S = pca.transform(X)
    S[out] = np.nan
    d = sel.copy()
    sc = pd.DataFrame(S, columns=[f'wf_pc{i + 1}' for i in range(n_components)])
    sc['session'], sc['unit_id'] = [k[0] for k in keys], [k[1] for k in keys]
    d = d.drop(columns=[c for c in sc.columns if c.startswith('wf_pc')], errors='ignore').merge(
        sc, on=['session', 'unit_id'], how='left')
    d['wf_pc1_quartile'] = pd.qcut(d['wf_pc1'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    bad = set(k for k, o in zip(keys, out) if o)
    d['wf_outlier'] = [k in bad for k in zip(d['session'], d['unit_id'])]
    return d, pca


def pca_quartile_panels(selp, pca, psths=None, normalize='delta',
                        conditions=('go_cue_all', 'response_all', 'spont_contra')):
    """PC1 x PC2 scatter, PC1 loading, mean template per PC1 quartile, the quartiles' firing rate /
    LV / trough-to-peak, and (with psths) their population PSTHs."""
    keys, X = waveform_matrix(selp)
    idx = {k: i for i, k in enumerate(keys)}
    t = wf_time()
    q = selp.dropna(subset=['wf_pc1_quartile'])
    qs = ['Q1', 'Q2', 'Q3', 'Q4']
    ncols = 4
    nrows = 2 + (psths is not None)
    fig = plt.figure(figsize=(5 * ncols, 4 * nrows))
    gs = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=0.45, wspace=0.3)
    ax = fig.add_subplot(gs[0, 0])
    for qi, c in zip(qs, ORDINAL):
        u = q[q['wf_pc1_quartile'] == qi]
        ax.scatter(u['wf_pc1'], u['wf_pc2'], s=10, color=c, lw=0, label=f'{qi} (n={len(u)})')
    if q['tagged'].any():
        u = q[q['tagged']]
        ax.scatter(u['wf_pc1'], u['wf_pc2'], s=26, facecolor='none', edgecolor=TAGGED_UNIT, lw=0.9, label='tagged')
    ev = 100 * pca.explained_variance_ratio_
    ax.set_xlabel(f'Waveform PC1 ({ev[0]:.0f}%)')
    ax.set_ylabel(f'Waveform PC2 ({ev[1]:.0f}%)')
    ax.legend(fontsize=7, frameon=False)
    style(ax)
    ax = fig.add_subplot(gs[0, 1])
    sd1 = np.sqrt(pca.explained_variance_[0])
    for s, c, lab in [(-2, ORDINAL[0], '-2 SD'), (0, INK, 'mean'), (2, ORDINAL[3], '+2 SD')]:
        ax.plot(t, pca.mean_ + s * sd1 * pca.components_[0], color=c, lw=1.8, label=f'PC1 {lab}')
    ax.set_title('What PC1 captures', fontsize=10)
    ax.legend(fontsize=7, frameon=False)
    ax.set_xlabel('Time from trough (ms)')
    style(ax)
    ax = fig.add_subplot(gs[0, 2])
    for qi, c in zip(qs, ORDINAL):
        ks = list(zip(q.loc[q['wf_pc1_quartile'] == qi, 'session'], q.loc[q['wf_pc1_quartile'] == qi, 'unit_id']))
        W = X[[idx[k] for k in ks if k in idx]]
        mu, se = W.mean(0), W.std(0) / np.sqrt(len(W))
        ax.fill_between(t, mu - se, mu + se, color=c, alpha=0.3, lw=0)
        ax.plot(t, mu, color=c, lw=1.6, label=qi)
    ax.set_title('Mean template per PC1 quartile (+/- SEM)', fontsize=10)
    ax.set_xlabel('Time from trough (ms)')
    ax.legend(fontsize=7, frameon=False)
    style(ax)
    ax = fig.add_subplot(gs[0, 3])
    for qi, c in zip(qs, ORDINAL):
        u = q[q['wf_pc1_quartile'] == qi]
        ax.scatter(u['AP'], u['DV'], s=8, color=c, lw=0, label=qi)
    ax.set_xlabel(AXIS_LABEL[AP])
    ax.set_ylabel(AXIS_LABEL[DV])
    ax.set_title('Sagittal location by quartile', fontsize=10)
    style(ax)
    for j, (col, label, log) in enumerate([('firing_rate', 'Firing rate (Hz)', True), ('lv', 'LV (ISI)', False),
                                           ('trough_to_peak_ms', 'Trough to peak (ms)', False),
                                           ('dist_um', 'Distance to mesh (um)', False)]):
        ax = fig.add_subplot(gs[1, j])
        data = [q.loc[q['wf_pc1_quartile'] == qi, col].dropna().to_numpy(float) for qi in qs]
        bp = ax.boxplot(data, tick_labels=qs, showfliers=False, patch_artist=True, widths=0.6)
        for patch, c in zip(bp['boxes'], ORDINAL):
            patch.set_facecolor(c)
            patch.set_alpha(0.5)
        for med in bp['medians']:
            med.set_color(INK)
        if log:
            ax.set_yscale('log')
        ax.set_ylabel(label)
        style(ax)
    if psths is not None:
        groups = {qi: (list(zip(q.loc[q['wf_pc1_quartile'] == qi, 'session'], q.loc[q['wf_pc1_quartile'] == qi, 'unit_id'])), c)
                  for qi, c in zip(qs, ORDINAL)}
        for j, key in enumerate(conditions[:ncols]):
            ax = fig.add_subplot(gs[2, j])
            for name, (ks, c) in groups.items():
                g = group_mean(psths, ks, key, normalize)
                if g is None:
                    continue
                tt, mu, se, n = g
                ax.fill_between(tt, mu - se, mu + se, color=c, alpha=0.25, lw=0)
                ax.plot(tt, mu, color=c, lw=1.6, label=f'{name} ({n})')
            ax.axvline(0, color=MUTED, ls='--', lw=1)
            ax.axhline(0, color=MUTED, lw=0.6)
            ax.set_title(key.replace('_', ' '), fontsize=10)
            ax.set_xlabel(_xlabel(key), fontsize=8)
            ax.set_ylabel(_norm_label(normalize))
            ax.legend(fontsize=7, frameon=False)
            style(ax)
    return fig


def stability_index(sel, smooth_bins=5, floor_hz=0.1):
    """sel with rate_stability: log2(max / min) of each unit's rate course (RATE_BIN bins, smoothed
    over smooth_bins), within its opto-free window; large values point at drift or dropouts."""
    out = []
    for r in sel.itertuples():
        rc = np.asarray(r.rate_course, float)
        t = r.rate_t0 + RATE_BIN * np.arange(len(rc))
        w = r.spont_window
        if w is not None:
            rc = rc[(t >= w[0]) & (t + RATE_BIN <= w[1])]
        if len(rc) < smooth_bins:
            out.append(np.nan)
            continue
        s = np.convolve(rc, np.ones(smooth_bins) / smooth_bins, mode='valid')
        out.append(np.log2((s.max() + floor_hz) / (s.min() + floor_hz)))
    d = sel.copy()
    d['rate_stability'] = out
    return d


def stability_heatmap(sel, sort_by='rate_stability', max_units=400, cuts=True):
    """Units x session time: rate course (RATE_BIN bins) as log2 of each unit's median rate, sorted
    by sort_by (largest first). Time is from the recording start; existing / sheet cuts in black."""
    d = sel.sort_values(sort_by, ascending=False).head(max_units)
    n = max(len(r) for r in d['rate_course'])
    M = np.full((len(d), n), np.nan)
    for i, rc in enumerate(d['rate_course']):
        rc = np.asarray(rc, float)
        med = np.median(rc[rc > 0]) if (rc > 0).any() else 1
        M[i, :len(rc)] = np.log2((rc + 0.05) / med)
    fig, ax = plt.subplots(figsize=(14, 2 + 0.025 * len(d)))
    im = ax.imshow(M, aspect='auto', cmap=STABILITY_CMAP, vmin=-3, vmax=3, interpolation='nearest',
                   extent=[0, n * RATE_BIN / 60, len(d) + 0.5, 0.5])
    if cuts and 'cut_start' in d.columns:
        for i, r in enumerate(d.itertuples(), start=1):
            for c in (r.cut_start, r.cut_end):
                if c is not None and np.isfinite(c):
                    x = (c - r.rate_t0) / 60
                    ax.plot([x, x], [i - 0.5, i + 0.5], color=INK, lw=1.5)
    ax.set_xlabel('Time from recording start (min)')
    ax.set_ylabel(f'units, by {sort_by} (n={len(d)})')
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, label='log2(rate / unit median)', shrink=0.8)
    return fig


def profile_counts(selr, by='group', align='go_cue'):
    """Stacked fraction of response profiles (unit_response_profiles_DRN) per value of column by."""
    col = f'profile_{align}'
    d = selr.dropna(subset=[col])
    t = d.groupby(by)[col].value_counts(normalize=True).unstack(fill_value=0).reindex(columns=PROFILES, fill_value=0)
    n = d.groupby(by).size()
    fig, ax = plt.subplots(figsize=(1.6 + 1.1 * len(t), 3.8))
    bottom = np.zeros(len(t))
    for p in PROFILES:
        ax.bar(np.arange(len(t)), t[p], bottom=bottom, color=PROFILE_COLORS[p], width=0.7,
               edgecolor='white', linewidth=1.5, label=p)
        bottom += t[p].to_numpy()
    ax.set_xticks(np.arange(len(t)))
    ax.set_xticklabels([f'{i}\n(n={n[i]})' for i in t.index], fontsize=8)
    ax.set_ylabel(f'Fraction of units ({align.replace("_", " ")})')
    ax.legend(fontsize=7, frameon=False, bbox_to_anchor=(1, 1), loc='upper left')
    style(ax)
    fig.tight_layout()
    return fig, t

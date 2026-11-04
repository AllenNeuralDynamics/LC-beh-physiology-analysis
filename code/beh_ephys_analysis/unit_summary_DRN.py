# %% One-page summary per DRN unit: behavior-aligned rasters and PSTHs, opto-tagging
#    responses and waveforms, and the unit's CCF location in three planes.
#
# Combines the panels of unit_beh_analysis_JL.plot_unit_beh_session,
# opto_tagging.opto_plotting_unit and the plane projections of
# session_combine/figure_preparation/sampling_map.py.
#
# Example notebook use:
#     from unit_summary_DRN import plot_unit_summary_session
#     plot_unit_summary_session('behavior_814515_2025-10-23_14-41-04', units=[30, 87])
#
# Example terminal use:
#     python unit_summary_DRN.py behavior_814515_2025-10-23_14-41-04 30 87

import os
import sys

curr_path = os.path.dirname(os.path.abspath(__file__))
if curr_path in sys.path:
    sys.path.remove(curr_path)
sys.path.insert(0, curr_path)

import argparse
import contextlib
import io
import json
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from trimesh import load_mesh
import spikeinterface as si

from aind_ephys_utils import align
from aind_dynamic_foraging_basic_analysis.plot.plot_foraging_session import plot_foraging_session
from utils.beh_functions import session_dirs, makeSessionDF, get_session_tbl, get_unit_tbl, get_history_from_nwb
from utils.ephys_functions import load_drift, plot_raster_rate, get_spike_matrix, plot_filled_sem
from utils.ccf_utils import project_to_plane, pir_to_lps
from utils.plot_utils import shiftedColorMap
from utils.lick_utils import load_licks
from utils.capsule_migration import capsule_directories
from utils.combine_tools import apply_qc, to_str_intlike
from unit_beh_analysis_JL import save_fig


# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------

OUT_DIR = '/root/capsule/scratch/combined/unit_summary_DRN'

# same frame as sampling_map.py: x/y/z_ccf are LPS mm, meshes are already bregma-relative
BREGMA_LPS_MM = np.array([-5.74, 5.4, -0.45])
CCF_COLS = ['x_ccf', 'y_ccf', 'z_ccf']
MESH_DIR = '/root/capsule/data/ccf_meshes'
# DRN outlines drawn in the plane views. 'bregma_lps_mm' meshes are used as is; 'ccf_pir_um'
# meshes (BrainGlobe allen_mouse_25um: x = AP, y = DV, z = ML, in um) are converted to LPS mm
# and bregma-subtracted like the units. The aqueduct is always drawn as a landmark.
MESH_SETS = {
    '5HT_core': [('DR core (B6/7)', '/root/capsule/data/5HT_meshes/B6_7_DR_core_67pct.obj', 'ccf_pir_um'),
                 ('DR wing (B7l)', '/root/capsule/data/5HT_meshes/B7l_wing_core_67pct.obj', 'ccf_pir_um')],
    'ccf_DR': [('DR (CCF)', os.path.join(MESH_DIR, 'DR_bregma_lps_mm.obj'), 'bregma_lps_mm')],
}
DRN_MESH_SET = '5HT_core'
ML, AP, DV = 0, 1, 2
PLANES = [('Coronal', ML, DV), ('Horizontal', ML, AP), ('Sagittal', AP, DV)]
AXIS_LABEL = {ML: 'ML from midline (mm)', AP: 'AP from bregma (mm)', DV: 'DV from bregma (mm)'}
PLANE_PAD = 0.4  # mm around the DR mesh shown in each plane
# spontaneous licks, as in session_combine/manuscript_figures/F_spont_choice_lick_neuron_DRN.ipynb
SPONT_WINDOW = (-5, 5)
SPIKE_TICK = 0.5  # raster tick height in opto block rasters, in train rows
AMP_YLIM_SD = 2  # spike-amplitude y-axis spans mean +/- this many SD
# session firing-rate panel: peak rate in PEAK_BIN sub-bins within each 30 s bin; bins whose
# peak exceeds the unit's own BURST_PCT percentile of PEAK_BIN rates over the recording are
# marked as candidate bursts. 1 s, not 50-100 ms: at 100 ms the peak of a 3-10 Hz unit takes
# only 5-10 distinct values (each spike adds 10 Hz), and these units fire more regularly than
# Poisson, so a Poisson burst test is too conservative to flag anything.
PEAK_BIN = 1.0  # s
BURST_PCT = 99
PEAK_COLOR = '#e07b00'
IN_TRIAL_PERIOD = (0, 1.5)  # lick-train starts this soon after a go cue are task licks
# per-session CCF unit locations, newest first; _260817 is the version in the master unit table
CCF_FILES = ['ccf_unit_locations_260817.csv', 'ccf_unit_locations.csv']

# population histograms: tagged units (TAGGED_CRITERION) against the matched non-tagged units
MASTER_CSV = '/root/capsule/scratch/combined/master_unit_tables/master_all_units_opto_ccf_raw_260817.csv'
NONTAGGED_SELECTION = os.path.join(OUT_DIR, 'nontagged_matched_selection.csv')
# every PSTH on a page, saved per unit for population averages (see load_population_psth)
PSTH_DIR = os.path.join(OUT_DIR, 'psth')
POP_CACHE = os.path.join(OUT_DIR, 'population_metrics.csv')
POP_METRICS = [('firing_rate', 'Firing rate (Hz)', 'log'),
               ('isi_violations_ratio', 'ISI violations ratio', 'linear'),
               ('snr', 'SNR', 'linear'),
               ('latency_ms', 'Latency at p_max (ms)', 'linear')]

# tagging-summary columns copied onto the full unit table when the summary exists
SUMMARY_COLS = ['p_max', 'bl_max_p', 'p_mean', 'lat_max_p', 'lat_mean', 'euc_max_p',
                'corr_max_p', 'pass_count', 'tagged_loc', 'tagged']

# opto-tagging criteria (session_combine/metrics) listed in the title when the unit passes them,
# applied to combined_unit_tbl as basic_ephys_generation.py does. Left out: beh_all_beh_DRN and
# basic_ephys_low_DRN_nonopto need columns that table lacks (apply_qc would skip them), and
# beh_all_NP_DRN's probe "2" matches no rows.
CRITERIA_DIR = '/root/capsule/code/beh_ephys_analysis/session_combine/metrics'
CRITERIA = ['basic_ephys_all_DRN', 'basic_ephys_low_DRN', 'beh_all_DRN', 'waveform_low_DRN']
# a unit is "tagged" (title label, red/gray units in the plane views) if it passes this one;
# basic_ephys_low_DRN is a superset of the other three, so "any of CRITERIA" would be that.
# Same population unit_summary_DRN_select_nontagged.py matches non-tagged units to.
TAGGED_CRITERION = 'basic_ephys_all_DRN'
# antidromic criteria per projection site, as defined in F_antidromic_combined_DRN.ipynb
ANTI_FOCUSES = ['PrL', 'S1']
ANTI_REG_ALPHA = 0.005
ANTI_REG_JITTER = 0.010

INK = '#0b0b0b'
INK_2 = '#52514e'
MUTED = '#8a8985'
MESH = '#ecebe7'
# fills for the DRN mesh components, in MESH_SETS order; light neutrals that stay behind the units
MESH_FILLS = ['#dfded8', '#f1e8d8']
MESH_EDGE = '#b9b7b0'
TAGGED_UNIT = '#d6402a'
# plane views: tagged units of every session pale red, this session's tagged units yellow
# (dark edge so they hold up on the light mesh), other units of this session gray
ALL_TAGGED_UNIT = '#f0a99c'
SESSION_TAGGED_UNIT = '#f5c400'
SESSION_TAGGED_EDGE = '#6b5500'
SESSION_UNIT = '#cfcdc7'  # the session's untagged units, lighter so they stay behind the tagged ones
TARGET_UNIT = '#1c5cab'
LASER = 'red'
TAGGED_LABEL = 'red'
# trial outcome in the lick-side panels; blue/orange stay distinguishable under CVD
REWARDED = '#e07b00'
UNREWARDED = '#1c5cab'
IGNORE = INK_2
# lick side in the spontaneous-lick panel, kept neutral so it can't be read as outcome
SPONT_IPSI = INK
SPONT_CONTRA = '#a9a7a1'
NONTAGGED_POP = '#8a8985'  # population histograms: tagged in TAGGED_UNIT, non-tagged in this
# sequential, one hue light to dark, for trial x time rate heatmaps
HEAT_CMAP = LinearSegmentedColormap.from_list('white_to_blue', ['#ffffff', '#86b6ef', '#2a78d6', '#0d366b'])

fs = 12
custom_cmap = LinearSegmentedColormap.from_list("blue_white_red", ["blue", "white", "red"])
b_w_r_cmap = LinearSegmentedColormap.from_list("b_w_r", ["blue", "white", "red"])


# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------

def _load_pickle(path):
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


_criteria_passed = None
_tagged_ccf = None


def criteria_passed():
    """{(session, unit_id): [names of CRITERIA the unit passes]}, computed once per process.
    Also keeps the CCF location of every unit passing TAGGED_CRITERION (see tagged_ccf)."""
    global _criteria_passed, _tagged_ccf
    if _criteria_passed is not None:
        return _criteria_passed
    prep = capsule_directories()['manuscript_fig_prep_dir']
    tbl = _load_pickle(os.path.join(prep, 'combined_unit_tbl', 'combined_unit_tbl.pkl'))
    # isi bound in basic_ephys_all_DRN is conditional on antidromic tier_1
    anti = _load_pickle(os.path.join(prep, 'antidromic_analysis', 'PrL_S1', 'combined_antidromic_results.pkl'))
    anti = anti[['session', 'unit', 'tier_1']].copy()
    anti['unit_key'] = anti['unit'].apply(to_str_intlike)
    tbl['unit_key'] = tbl['unit'].apply(to_str_intlike)
    tbl = tbl.merge(anti[['session', 'unit_key', 'tier_1']], on=['session', 'unit_key'],
                    how='left').drop(columns='unit_key')
    tbl['tier_1'] = tbl['tier_1'].fillna(False)
    _criteria_passed = {}
    for name in CRITERIA:
        with open(os.path.join(CRITERIA_DIR, f'{name}.json')) as f:
            constraints = json.load(f)
        if name == 'basic_ephys_all_DRN':
            constraints['sd'] = [0, 1]  # basic_ephys_generation's override for DRN
        with contextlib.redirect_stdout(io.StringIO()):  # apply_qc prints every step
            filtered, _, fig, _ = apply_qc(tbl.copy(), constraints)
        plt.close(fig)
        for key in zip(filtered['session'], filtered['unit'].astype(int)):
            _criteria_passed.setdefault(key, []).append(name)
        if name == TAGGED_CRITERION:
            _tagged_ccf = filtered[['session', 'unit'] + CCF_COLS].reset_index(drop=True)
            _tagged_ccf['unit'] = _tagged_ccf['unit'].astype(int)
    return _criteria_passed


def is_tagged(session, unit_id):
    return TAGGED_CRITERION in criteria_passed().get((session, int(unit_id)), [])


def tagged_ccf():
    """session, unit and CCF location (combined_unit_tbl) of all tagged units, any session."""
    criteria_passed()
    return _tagged_ccf


_antidromic_passed = None


def antidromic_passed():
    """{(session, unit_id): {focus: [antidromic criteria passed]}} for units in the antidromic
    analysis; units not in it were not tested. Computed once per process."""
    global _antidromic_passed
    if _antidromic_passed is not None:
        return _antidromic_passed
    prep = capsule_directories()['manuscript_fig_prep_dir']
    wide = _load_pickle(os.path.join(prep, 'antidromic_analysis', 'combined_antidromic_results.pkl'))
    _antidromic_passed = {}
    for i in range(len(wide)):
        key = (wide[('session', '')].iloc[i], int(wide[('unit', '')].iloc[i]))
        _antidromic_passed[key] = {}
        for focus in ANTI_FOCUSES:
            site = f'surface_{focus}'
            get = lambda m: wide[(m, site)].iloc[i] if (m, site) in wide.columns else np.nan
            responsive = get('antidromic_tier_label') in ['responsive', 'jitter', 'collision']
            reg_sharp = (get('p_antidromic') < ANTI_REG_ALPHA and get('t_antidromic') > 0
                         and get('jitter') < ANTI_REG_JITTER)
            _antidromic_passed[key][focus] = ([name for name, ok in [('responsive', responsive),
                                               ('reg_sharp_antidromic', reg_sharp)] if ok])
    return _antidromic_passed


def load_session_data(session, data_type='raw', target='soma', model_name=None):
    """Everything the per-unit page needs for one session, loaded once."""
    session_dir = session_dirs(session)
    opto_dir = session_dir[f'opto_dir_{data_type}']

    # full table (all units, spike times); the summary only holds tagged units
    unit_tbl = get_unit_tbl(session, data_type, summary=False)
    summary_tbl = get_unit_tbl(session, data_type, summary=True)
    if summary_tbl is not None:
        cols = [c for c in SUMMARY_COLS if c in summary_tbl.columns and c not in unit_tbl.columns]
        unit_tbl = unit_tbl.merge(summary_tbl[['unit_id'] + cols], on='unit_id', how='left')
    ccf_files = [os.path.join(session_dir[f'ephys_processed_dir_{data_type}'], f) for f in CCF_FILES]
    ccf_file = next((f for f in ccf_files if os.path.exists(f)), None)
    if ccf_file is not None:
        unit_tbl = unit_tbl.drop(columns=CCF_COLS, errors='ignore').merge(
            pd.read_csv(ccf_file)[['unit_id'] + CCF_COLS], on='unit_id', how='left')

    # behavior; ephys-only sessions still get the opto and CCF rows
    session_df, tblTrials = None, None
    if os.path.exists(os.path.join(session_dir['beh_fig_dir'], f'{session}.nwb')):
        session_df = makeSessionDF(session, model_name=model_name)
        tblTrials = get_session_tbl(session)
    licks = None
    if session_df is not None:
        try:
            licks = load_licks(session)
        except Exception as e:
            print(f'No licks for {session}: {type(e).__name__}: {e}')

    with open(os.path.join(session_dir['processed_dir'], f'{session}_qm.json')) as f:
        qm = json.load(f)
    with open(os.path.join(opto_dir, f'{session}_opto_info_{target}.json')) as f:
        opto_info = json.load(f)
    with open(os.path.join(opto_dir, f'{session}_waveform_params.json')) as f:
        waveform_params = json.load(f)
    opto_df = pd.read_csv(os.path.join(opto_dir, f'{session}_opto_session_{target}.csv'))

    opto_wf = _load_pickle(os.path.join(opto_dir, f'{session}_opto_waveform_metrics.pkl'))
    if opto_wf is not None:
        for col in ['unit_id', 'power', 'duration', 'spont', 'euclidean_norm', 'correlation']:
            opto_wf[col] = pd.to_numeric(opto_wf[col], errors='coerce')
    crosscorr = None
    cc_short = _load_pickle(os.path.join(opto_dir, f'{session}_correlogram_laser_short.pkl'))
    cc_long = _load_pickle(os.path.join(opto_dir, f'{session}_correlogram_laser_long.pkl'))
    if cc_short is not None and cc_long is not None:
        crosscorr = {'short': cc_short, 'long': cc_long}

    return {
        'session': session,
        'data_type': data_type,
        'unit_tbl': unit_tbl,
        'session_df': session_df,
        'tblTrials': tblTrials,
        'qm': qm,
        'opto_info': opto_info,
        'opto_df': opto_df,
        'opto_responses': _load_pickle(os.path.join(opto_dir, f'{session}_opto_responses_{target}.pkl')),
        'waveforms': _load_pickle(os.path.join(opto_dir, f'{session}_waveforms_{target}.pkl')),
        'waveform_params': waveform_params,
        'opto_wf': opto_wf,
        'opto_metrics': (_load_pickle(os.path.join(opto_dir, f'{session}_opto_tagging_metrics.pkl'))
                         or {}).get('opto_tagging_df_metrics'),
        'crosscorr': crosscorr,
        'licks': licks,
        'spike_amplitudes': load_spike_amplitudes(session, data_type),
    }


def load_spike_amplitudes(session, data_type='raw'):
    """Per-unit spike amplitudes from the postprocessed sorting analyzer (as in
    opto_tagging.opto_plotting_session); None if the sorted asset is not attached."""
    postprocessed_dir = session_dirs(session)[f'postprocessed_dir_{data_type}']
    if postprocessed_dir is None or not os.path.exists(postprocessed_dir):
        print(f'No postprocessed data for {session}: spike amplitudes skipped')
        return None
    analyzer = si.load(postprocessed_dir, load_extensions=False)
    amplitudes = analyzer.get_extension('spike_amplitudes').get_data(outputs="by_unit")[0]
    del analyzer
    return amplitudes


def load_meshes(mesh_set=DRN_MESH_SET):
    """{'drn': [(label, vertices), ...], 'AQ': vertices}, all in bregma-relative LPS mm."""
    meshes = {'drn': []}
    for label, path, frame in MESH_SETS[mesh_set]:
        if not os.path.exists(path):
            print(f'Missing mesh {path}')
            continue
        verts = np.asarray(load_mesh(path).vertices, float)
        if frame == 'ccf_pir_um':
            verts = pir_to_lps(verts / 1000) - BREGMA_LPS_MM
        meshes['drn'].append((label, verts))
    aq = os.path.join(MESH_DIR, 'AQ_bregma_lps_mm.obj')
    if os.path.exists(aq):
        meshes['AQ'] = np.asarray(load_mesh(aq).vertices, float)
    return meshes


def curate_unit_times(data, unit_id, curate_time=True):
    """Spike times and trial tables cut to the unit's stable period (as in unit_beh_analysis_JL)."""
    spike_times = data['unit_tbl'].query('unit_id == @unit_id')['spike_times'].values[0].copy()
    session_df, tblTrials = data['session_df'], data['tblTrials']
    ephys_cut = [None, None]
    unit_drift = load_drift(data['session'], unit_id, data_type=data['data_type'])
    if curate_time and unit_drift is not None:
        ephys_cut = list(unit_drift['ephys_cut'])
    elif curate_time:
        ephys_cut = nontagged_drift_cut(data['session'], unit_id, data['data_type'])
    if ephys_cut[0] is not None:
        spike_times = spike_times[spike_times >= ephys_cut[0]]
        if session_df is not None:
            session_df = session_df[session_df['go_cue_time'] >= ephys_cut[0]]
            tblTrials = tblTrials[tblTrials['goCue_start_time'] >= ephys_cut[0]]
    if ephys_cut[1] is not None:
        spike_times = spike_times[spike_times <= ephys_cut[1]]
        if session_df is not None:
            session_df = session_df[session_df['go_cue_time'] <= ephys_cut[1]]
            tblTrials = tblTrials[tblTrials['goCue_start_time'] <= ephys_cut[1]]
    return spike_times, session_df, tblTrials, ephys_cut


def nontagged_drift_cut(session, unit_id, data_type='raw'):
    """[start, end] (None = no cut) from {session}_opto_drift_tbl_nontagged.csv, the hand-placed
    cuts of the matched non-tagged units; [None, None] if the unit is not in it."""
    f = os.path.join(session_dirs(session)[f'opto_dir_{data_type}'], f'{session}_opto_drift_tbl_nontagged.csv')
    if not os.path.exists(f):
        return [None, None]
    tbl = pd.read_csv(f, index_col=0)
    row = tbl[tbl['unit_id'] == unit_id]
    if len(row) == 0:
        return [None, None]
    parts = [x.strip() for x in str(row['ephys_cut'].iloc[0]).strip('[]').split(',')]
    return [None if x.lower() in ('nan', 'none') else float(x) for x in parts]


_nontagged_selection = None


def nontagged_match(session, unit_id):
    """Row of the matched non-tagged selection for this unit (matched_tagged_unit, distance_um,
    ...), or None if the unit is not in it."""
    global _nontagged_selection
    if _nontagged_selection is None:
        _nontagged_selection = (pd.read_csv(NONTAGGED_SELECTION) if os.path.exists(NONTAGGED_SELECTION)
                                else pd.DataFrame(columns=['session', 'unit_id']))
    row = _nontagged_selection[(_nontagged_selection['session'] == session)
                               & (_nontagged_selection['unit_id'] == unit_id)]
    return row.iloc[0] if len(row) else None


_population = None


def population_metrics():
    """Firing rate, ISI violations and SNR (master unit table) and latency at the p_max
    condition (per-condition opto table, same rule as p_max_condition) for the tagged units and
    the matched non-tagged selection. Cached in POP_CACHE; rebuilt when either set changes."""
    global _population
    if _population is not None:
        return _population
    tagged = {k for k, v in criteria_passed().items() if TAGGED_CRITERION in v}
    sel = pd.read_csv(NONTAGGED_SELECTION) if os.path.exists(NONTAGGED_SELECTION) else pd.DataFrame(columns=['session', 'unit_id'])
    nontagged = set(zip(sel['session'], sel['unit_id'].astype(int)))
    keys = tagged | nontagged
    if os.path.exists(POP_CACHE):
        cache = pd.read_csv(POP_CACHE)
        if set(zip(cache['session'], cache['unit_id'])) == keys and \
                set(zip(cache.loc[cache['tagged'], 'session'], cache.loc[cache['tagged'], 'unit_id'])) == tagged:
            _population = cache
            return cache
    master = pd.read_csv(MASTER_CSV, usecols=['session', 'unit_id', 'firing_rate', 'isi_violations_ratio', 'snr'])
    pop = pd.DataFrame(sorted(keys), columns=['session', 'unit_id']).merge(master, on=['session', 'unit_id'], how='left')
    pop['tagged'] = [k in tagged for k in zip(pop['session'], pop['unit_id'])]
    pop['latency_ms'] = np.nan
    for session, rows in pop.groupby('session'):
        metrics = _load_pickle(os.path.join(session_dirs(session)['opto_dir_raw'], f'{session}_opto_tagging_metrics.pkl'))
        if metrics is None or 'opto_tagging_df_metrics' not in metrics:
            continue
        d = {'opto_metrics': metrics['opto_tagging_df_metrics']}
        for i, unit_id in zip(rows.index, rows['unit_id']):
            cond = p_max_condition(d, unit_id)
            if cond is not None and pd.notna(cond['resp_lat']):
                pop.at[i, 'latency_ms'] = 1000 * cond['resp_lat']
    pop.to_csv(POP_CACHE, index=False)
    _population = pop
    return pop


def spontaneous_licks(data, tblTrials, ephys_cut):
    """Lick-train starts per side ('L', 'R') outside the task lick period, within the recording
    and the unit's drift cut (as in F_spont_choice_lick_neuron_DRN.ipynb). None without licks."""
    licks = data['licks']
    if licks is None or tblTrials is None:
        return None
    rec_cut = data['qm']['ephys_cut']
    lo = max([c for c in [rec_cut[0], ephys_cut[0]] if c is not None], default=-np.inf)
    hi = min([c for c in [rec_cut[1], ephys_cut[1]] if c is not None], default=np.inf)
    go = np.sort(tblTrials['goCue_start_time'].values)
    spont = {}
    for side in ['L', 'R']:
        starts = np.asarray(licks[f'lick_trains_{side}']['train_starts'], dtype=float)
        starts = starts[(starts >= lo) & (starts <= hi)]
        # latest go cue at or before each start decides whether it is a task lick
        prev = np.searchsorted(go, starts - IN_TRIAL_PERIOD[0], side='right') - 1
        lag = starts - go[np.clip(prev, 0, None)] if len(go) else np.full(len(starts), np.inf)
        in_trial = (prev >= 0) & (lag >= IN_TRIAL_PERIOD[0]) & (lag <= IN_TRIAL_PERIOD[1])
        spont[side] = starts[~in_trial]
    return spont


# ---------------------------------------------------------------------
# Opto response summaries
# ---------------------------------------------------------------------

def opto_response_summary(data, unit_id):
    """Best stimulation condition, and per power the P(resp)/latency train of that power's best
    condition (site, block, ...), picked by mean P(resp) over the train.

    resp_p / resp_lat axes: power, site, num_pulses, duration, freq, pre_post, pulse
    (see opto_tagging.opto_plotting_unit). Values are raw P(resp), not baseline-subtracted.
    """
    resp_p = np.array(data['opto_responses']['resp_p'][unit_id].tolist(), dtype=float)
    resp_lat = np.array(data['opto_responses']['resp_lat'][unit_id].tolist(), dtype=float)
    p_mean = np.nanmean(resp_p, axis=-1)  # mean over pulses in the train
    powers = np.array(data['opto_info']['powers'])
    sites = np.array(data['opto_info']['sites'])

    per_power = []
    for power_ind, power in enumerate(powers):
        sub = p_mean[power_ind]
        if np.isnan(sub).all():
            continue
        cond = np.unravel_index(np.nanargmax(sub), sub.shape)
        per_power.append({'power': power, 'p_train': resp_p[power_ind][cond],
                          'lat_train': resp_lat[power_ind][cond]})

    if np.isnan(p_mean).all():
        return per_power, None
    best = np.unravel_index(np.nanargmax(p_mean), p_mean.shape)
    best_cond = {
        'power': powers[best[0]],
        'site': sites[best[1]],
        'p_train': resp_p[best],
        'lat_train': resp_lat[best],
    }
    return per_power, best_cond


def p_max_condition(data, unit_id):
    """Stimulation condition behind p_max / euc_max_p / corr_max_p in the tagging summary
    (opto_tagging_summary_DRN.opto_summary_DRN): highest baseline-subtracted P(resp) that has a
    response latency. Row of opto_tagging_df_metrics, or None."""
    metrics = data['opto_metrics']
    if metrics is None:
        return None
    rows = metrics[metrics['unit_id'] == unit_id].sort_values('resp_p_bl', ascending=False)
    if len(rows) == 0:
        return None
    with_lat = rows[rows['resp_lat'].notna()]
    rows = with_lat if len(with_lat) > 0 else rows
    # ties on resp_p_bl: the summary takes the smallest Euclidean distance among them
    top = rows[rows['resp_p_bl'] == rows['resp_p_bl'].iloc[0]]
    return top.sort_values('euclidean_norm').iloc[0]


# ---------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------

def style(ax):
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


def raster(ax, spike_times, align_times, window, scale=1, color='k'):
    df = align.to_events(spike_times, align_times, window, return_df=True)
    ax.vlines(scale * df.time, df.event_index + 0.5, df.event_index + 1.5, color=color, linewidth=0.8)
    ax.set_xlim(scale * window[0], scale * window[1])
    ax.set_ylim(0.5, len(align_times) + 0.5)
    return df


def ipsi_side(unit_row):
    """Hemisphere of the unit: ML > 0 is the right side. None without a CCF location."""
    ml = float(unit_row.get('x_ccf', np.nan)) - BREGMA_LPS_MM[ML]
    if not np.isfinite(ml):
        return None, ml
    return ('right' if ml >= 0 else 'left'), ml


def store_psth(store, key, matrix, bin_times, align_times, **info):
    """Keep one PSTH (trial x time rate matrix, mean, SEM) for saving with the page."""
    if store is None:
        return
    matrix = np.asarray(matrix, dtype=float)
    n = np.sum(np.isfinite(matrix), axis=0)
    store[key] = {'bin_times': np.asarray(bin_times), 'matrix': matrix.astype(np.float32),
                  'mean': np.nanmean(matrix, axis=0) if len(matrix) else np.full(len(bin_times), np.nan),
                  'sem': (np.nanstd(matrix, axis=0) / np.sqrt(np.maximum(n, 1))) if len(matrix)
                         else np.full(len(bin_times), np.nan),
                  'n_trials': len(matrix), 'align_times': np.asarray(align_times), **info}


def raster_psth(fig, spec, spike_times, groups, window, binSize, stepSize, title, xlabel,
                matrices=None, store=None, keys=None, info=None):
    """Raster stacked by group (trials in session order within a group) over one PSTH per group.

    groups: list of (label, align_times, color). matrices: optional list of (rate matrix,
    bin_times) per group, for rates not computed with the default binning. store/keys: dict to
    collect each group's PSTH under keys[g] (see store_psth), with info added to each entry.
    Returns the PSTH axis.
    """
    sub = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=spec, height_ratios=[3, 1], hspace=0.08)
    ax1 = fig.add_subplot(sub[0])
    ax2 = fig.add_subplot(sub[1], sharex=ax1)
    offset = 0
    for g, (label, align_times, color) in enumerate(groups):
        if len(align_times) == 0:
            continue
        if offset > 0:
            ax1.axhline(offset + 0.5, color=MUTED, ls='--', lw=0.8)
        df = align.to_events(spike_times, align_times, window, return_df=True)
        ax1.vlines(df.time, offset + df.event_index + 0.5, offset + df.event_index + 1.5,
                   color=color, linewidth=0.6)
        offset += len(align_times)
        if matrices is None:
            matrix, bin_times = get_spike_matrix(spike_times, align_times, pre_event=window[0],
                                                 post_event=window[1], binSize=binSize, stepSize=stepSize)
        else:
            matrix, bin_times = matrices[g]
        if keys is not None:
            store_psth(store, keys[g], matrix, bin_times, align_times, **(info or {}))
        plot_filled_sem(bin_times, matrix, color, ax2, f'{label} (n={len(align_times)})')
    if offset == 0:
        ax1.text(0.5, 0.5, 'No trials', ha='center', va='center', color=MUTED, transform=ax1.transAxes)
    else:
        ax2.legend(fontsize=7, frameon=False)
    for ax in (ax1, ax2):
        ax.axvline(0, color=MUTED, ls='--', lw=1, zorder=0)
        style(ax)
    ax1.set_xlim(window)
    ax1.set_ylim(0.5, max(offset, 1) + 0.5)
    ax1.tick_params(labelbottom=False)
    ax1.set_title(title, fontsize=fs)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel('Rate (Hz)')
    return ax2


def behavior_row(fig, spec, spike_times, session_df, tblTrials, spont, side, pre_event, post_event,
                 binSize, stepSize, store=None):
    has_q = session_df is not None and 'Qchosen' in session_df.columns
    gs = gridspec.GridSpecFromSubplotSpec(1, 11 if has_q else 10, subplot_spec=spec, wspace=0.45)
    if session_df is None or len(session_df) == 0:
        ax = fig.add_subplot(gs[0, :])
        ax.axis('off')
        ax.text(0.5, 0.5, 'No behavior in the stable period of this unit', ha='center', va='center',
                fontsize=fs + 2, color=MUTED, transform=ax.transAxes)
        return

    # session overview; same height split as the all-trial rasters next to it (whose heatmap
    # strip is left empty here) and the same y limits, so trial rows line up across the three
    raster_split = dict(height_ratios=[10, 1], hspace=0.04)
    ax = fig.add_subplot(gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 0],
                                                          **raster_split)[0])
    choice_history, reward_history, p_reward, autowater_offered, _ = get_history_from_nwb(session_df)
    _, axes = plot_foraging_session(choice_history, reward_history, p_reward=p_reward,
                                    autowater_offered=autowater_offered, ax=ax, vertical=True)
    for ax in axes:
        # trial k is drawn at y = k (1-based), as in raster()
        ax.set_ylim(0.5, len(session_df) + 0.5)
        # the choice legend is anchored to spill right in vertical mode; keep it over this column
        leg = ax.get_legend()
        if leg is not None and len(leg.get_texts()) > 2:
            ax.legend(fontsize=5, ncol=2, loc='lower left', bbox_to_anchor=(-0.6, 1.0), frameon=False)

    # all trials in session order: raster over a one-row heatmap of the trial-averaged rate
    go_time = session_df['go_cue_time'].values
    choice_time = session_df['choice_time'].values
    strips = []
    for col, (align_time, name) in enumerate([(go_time, 'go cue'), (choice_time, 'response')], start=1):
        sub = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, col], **raster_split)
        ax = fig.add_subplot(sub[0])
        raster(ax, spike_times, align_time, (pre_event, post_event))
        ax.axvline(0, color=MUTED, ls='--', lw=1, zorder=0)
        ax.tick_params(labelbottom=False)
        ax.set_title(f'All trials: {name}', fontsize=fs)
        if col == 1:
            ax.set_ylabel('Trial number')
        style(ax)
        matrix, bin_times = get_spike_matrix(spike_times, align_time, pre_event=pre_event,
                                             post_event=post_event, binSize=binSize, stepSize=stepSize)
        ax_s = fig.add_subplot(sub[1], sharex=ax)
        strips.append((ax_s, np.nanmean(matrix, axis=0), bin_times))
        store_psth(store, f"{name.replace(' ', '_')}_all", matrix, bin_times, align_time,
                   binSize=binSize, stepSize=stepSize, kernel=None)
        ax_s.set_xlabel(f'Time from {name} (s)')
        ax_s.set_yticks([])
    # one color scale for both strips
    vmax = max(np.nanmax(mean_rate) for _, mean_rate, _ in strips)
    for ax_s, mean_rate, bin_times in strips:
        ax_s.imshow(mean_rate[np.newaxis, :], aspect='auto', cmap=HEAT_CMAP, vmin=0, vmax=vmax,
                         interpolation='nearest',
                         extent=[bin_times[0] - 0.5 * stepSize, bin_times[-1] + 0.5 * stepSize, 0, 1])
        ax_s.axvline(0, color='white', ls='--', lw=1)
        ax_s.set_xlim(pre_event, post_event)
        # color range in place of a colorbar, which would collide with the next column
        ax_s.set_ylabel(f'0–{vmax:.0f}\nHz', rotation=0, fontsize=7, ha='right', va='center')

    # lick side relative to the unit's hemisphere; choice 1 is a right lick. Without a CCF
    # location the panels fall back to right/left.
    rewarded = session_df['outcome'].values.astype(bool)
    choice = session_df['choice'].values.astype(int)
    if side is None:
        ipsi_choice, ipsi_label, contra_label = 1, 'right', 'left'
    else:
        ipsi_choice = 1 if side == 'right' else 0
        ipsi_label = f"ipsi ({side[0].upper()})"
        contra_label = f"contra ({'L' if side == 'right' else 'R'})"
    ipsi = choice == ipsi_choice
    # ignored trials are only in the full trial table; they have no lick, so their
    # "response" alignment is the end of the response window (reward_outcome_time)
    ignored = tblTrials[tblTrials['animal_response'] == 2]

    window = (pre_event, post_event)
    psth_axes = []
    alignments = [('go_cue_time', 'goCue_start_time', 'Go cue', 'Time from go cue (s)',
                   'Time from go cue (s)'),
                  ('choice_time', 'reward_outcome_time', 'Response', 'Time from response (s)',
                   'Time from response-window end (s)')]
    for k, (col, ignore_col, name, xlabel, ignore_xlabel) in enumerate(alignments):
        t = session_df[col].values
        align_key = 'go_cue' if k == 0 else 'response'
        info = dict(binSize=binSize, stepSize=stepSize, kernel=None)
        for j, (side_mask, side_label, side_key) in enumerate([(ipsi, ipsi_label, 'ipsi'),
                                                              (~ipsi, contra_label, 'contra')]):
            groups = [('rewarded', t[side_mask & rewarded], REWARDED),
                      ('unrewarded', t[side_mask & ~rewarded], UNREWARDED)]
            psth_axes.append(raster_psth(fig, gs[0, 3 + 3 * k + j], spike_times, groups, window,
                                         binSize, stepSize, f'{name}: {side_label} licks', xlabel,
                                         store=store, info=info,
                                         keys=[f'{align_key}_{side_key}_rewarded', f'{align_key}_{side_key}_unrewarded']))
        groups = [('no lick', ignored[ignore_col].values, IGNORE)]
        # ignored trials at "response" are aligned to the end of the response window
        psth_axes.append(raster_psth(fig, gs[0, 5 + 3 * k], spike_times, groups, window,
                                     binSize, stepSize, f'{name}: ignored', ignore_xlabel,
                                     store=store, info=info, keys=[f'{align_key}_ignored']))

    # spontaneous licks. Rates follow the notebook: causal PSP kernel, and bins within 1.5 s of
    # another spontaneous train start are masked, so the matrix is computed over both sides
    # together and split afterwards.
    if spont is not None:
        ipsi_key = 'R' if ipsi_choice == 1 else 'L'
        contra_key = 'L' if ipsi_choice == 1 else 'R'
        times = np.concatenate([spont[ipsi_key], spont[contra_key]])
        is_ipsi = np.r_[np.ones(len(spont[ipsi_key])), np.zeros(len(spont[contra_key]))].astype(bool)
        order = np.argsort(times)
        times, is_ipsi = times[order], is_ipsi[order]
        matrix, bin_times = get_spike_matrix(spike_times, times, SPONT_WINDOW[0], SPONT_WINDOW[1],
                                             binSize, stepSize, avoid_overlap=True, kernel=True,
                                             tau_rise=0.01, tau_decay=0.08)
        groups = [(ipsi_label, times[is_ipsi], SPONT_IPSI), (contra_label, times[~is_ipsi], SPONT_CONTRA)]
        psth_axes.append(raster_psth(fig, gs[0, 9], spike_times, groups, SPONT_WINDOW, binSize,
                                     stepSize, 'Spontaneous licks', 'Time from lick-train start (s)',
                                     matrices=[(matrix[is_ipsi], bin_times), (matrix[~is_ipsi], bin_times)],
                                     store=store, keys=['spont_ipsi', 'spont_contra'],
                                     info=dict(kernel='psp', tau_rise=0.01, tau_decay=0.08, avoid_overlap=True)))
    else:
        ax = fig.add_subplot(gs[0, 9])
        ax.axis('off')
        ax.text(0.5, 0.5, 'No lick data', ha='center', va='center', color=MUTED, transform=ax.transAxes)
    # one rate scale across conditions so panels compare directly
    top = max(ax.get_ylim()[1] for ax in psth_axes)
    for ax in psth_axes:
        ax.set_ylim(0, top)

    if has_q:
        q = session_df['Qchosen'].values
        bins = np.quantile(q, np.linspace(0, 1, 5))
        bins[0], bins[-1] = bins[0] - 0.0001, bins[-1] + 0.0001
        _, ax1, ax2 = plot_raster_rate(spike_times, choice_time, q, bins, ['1', '2', '3', '4'], custom_cmap,
                                       fig, gs[0, 10], tb=pre_event, tf=post_event, time_bin=stepSize)
        ax1.set_title('Qchosen: response', fontsize=fs)
        ax2.set_title('')
        ax2.legend(fontsize=7, frameon=False)


def add_pulses(ax, opto_info, height, scale=1):
    for j in range(opto_info['num_pulses'][0]):
        x = j / opto_info['freqs'][0]
        ax.add_patch(patches.Rectangle((scale * x, 0), scale * opto_info['durations'][0] / 1000, height,
                                       color=LASER, alpha=0.3, linewidth=0, zorder=0))


def opto_block_raster(ax, spike_times, block, opto_info, window, title):
    """Laser-aligned raster for one stimulation block: trains grouped by power, lowest at the
    bottom, each in time order. Ticks are SPIKE_TICK rows tall so neighbouring trains separate."""
    ax.set_title(title, fontsize=fs)
    ax.set_xlabel('Time from train onset (s)')
    style(ax)
    if len(block) == 0:
        ax.text(0.5, 0.5, 'No stimulation in this block', ha='center', va='center', color=MUTED,
                transform=ax.transAxes)
        return
    powers = np.sort(block['power'].unique())
    offset, centers = 0, []
    for k, power in enumerate(powers):
        times = np.sort(block.loc[block['power'] == power, 'time'].values)
        if k > 0:
            ax.axhline(offset + 0.5, color=MUTED, ls='--', lw=0.8)
        df = align.to_events(spike_times, times, window, return_df=True)
        y = offset + df.event_index + 1
        ax.vlines(df.time, y - SPIKE_TICK / 2, y + SPIKE_TICK / 2, color='k', linewidth=0.8)
        centers.append(offset + (len(times) + 1) / 2)
        offset += len(times)
    add_pulses(ax, opto_info, offset + 1)
    ax.set_xlim(window)
    ax.set_ylim(0.5, offset + 0.5)
    ax.set_yticks(centers, [f'{p:g}' for p in powers])
    ax.set_ylabel('Power (trains in time order)')


def opto_row(fig, spec, data, unit_id, spike_times, unit_row):
    # rasters get the extra width so spikes can be read against the 4 ms pulses
    gs = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=spec, wspace=0.3,
                                          width_ratios=[2.6, 2.6, 1.2, 1, 1.6])
    opto_info, opto_df = data['opto_info'], data['opto_df']
    per_power, best = opto_response_summary(data, unit_id)
    resp_win = opto_info['resp_win']
    baseline = unit_row['firing_rate'] * resp_win  # P(spike in resp window) by chance

    # laser-aligned rasters, one per stimulation block, trains grouped by power
    block_window = (-0.05, opto_info['num_pulses'][0] / opto_info['freqs'][0] + 0.1)
    site = best['site'] if best is not None else opto_df['site'].iloc[0]
    multi_site = opto_df['site'].nunique() > 1
    for col, (pre_post, name) in enumerate([('pre', 'Pre-behavior'), ('post', 'Post-behavior')]):
        ax = fig.add_subplot(gs[0, col])
        block = opto_df[opto_df['pre_post'] == pre_post]
        if multi_site:
            block = block[block['site'] == site]
        title = f'{name} block' + (f' (site {site})' if multi_site else '')
        opto_block_raster(ax, spike_times, block, opto_info, block_window, title)

    # P(resp) and latency across the train, one line per power (plasma, low to high power;
    # stops at 0.85 so the top power stays visible on white)
    sub = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 2], hspace=0.35)
    ax_p, ax_l = fig.add_subplot(sub[0]), fig.add_subplot(sub[1])
    colors = plt.cm.plasma(np.linspace(0, 0.85, max(len(per_power), 2)))
    for color, row in zip(colors, per_power):
        pulses = np.arange(1, len(row['p_train']) + 1)
        ax_p.plot(pulses, row['p_train'], '-o', color=color, ms=4, label=f"{row['power']:g}")
        ax_l.plot(pulses, 1000 * row['lat_train'], '-o', color=color, ms=4)
    ax_p.axhline(baseline, color=MUTED, ls='--', lw=0.8)
    ax_p.set_ylim(0, 1.05)
    ax_p.set_ylabel('P(resp)')
    cond = p_max_condition(data, unit_id)
    pmax_label = ('' if cond is None else
                  f"\np_max {cond['resp_p_bl']:.2f} (bl-sub), lat {1000 * cond['resp_lat']:.1f} ms, "
                  f"power {cond['powers']:g} {cond['stim_times']}")
    ax_p.set_title('Across pulses, by power' + pmax_label, fontsize=10)
    if per_power:
        ax_p.legend(title='Power', fontsize=7, title_fontsize=7, frameon=False,
                    loc='upper left', bbox_to_anchor=(1.0, 1.0))
    ax_l.set_ylim(0, 1000 * resp_win)
    ax_l.set_xlabel('Pulse')
    ax_l.set_ylabel('Latency (ms)')
    for ax in (ax_p, ax_l):
        ax.set_xticks(np.arange(1, opto_info['num_pulses'][0] + 1))
        style(ax)

    # waveforms: spontaneous vs laser-evoked, and 2D template
    sub = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 3], hspace=0.45)
    ax = fig.add_subplot(sub[0])
    opto_wf = data['opto_wf']
    if opto_wf is not None:
        wf_unit = opto_wf[opto_wf['unit_id'] == unit_id]
        wf_laser = wf_unit[wf_unit['spont'] != 1]
        cond = p_max_condition(data, unit_id)
        if cond is not None:
            # the condition behind euc_max_p / corr_max_p, so E and C match the metrics table
            match = wf_laser[(wf_laser['site'] == cond['sites']) & (wf_laser['power'] == cond['powers'])
                             & (wf_laser['pre_post'] == cond['stim_times'])]
            if (match['duration'] == cond['durations']).any():
                match = match[match['duration'] == cond['durations']]
        elif best is not None:
            match = wf_laser[(wf_laser['site'] == best['site']) & (wf_laser['power'] == best['power'])]
        else:
            match = wf_laser.iloc[0:0]
        wf_laser = match if len(match) > 0 else wf_laser
        if len(wf_laser) > 0:
            wf_laser = wf_laser.sort_values('euclidean_norm').iloc[0]
            wf_spont = wf_unit[(wf_unit['spont'] == 1) & (wf_unit['pre_post'] == wf_laser['pre_post'])]
            if len(wf_spont) == 0:
                wf_spont = wf_unit[wf_unit['spont'] == 1]
            if len(wf_spont) > 0:
                ax.plot(wf_spont['peak_waveform'].values[0], color='k', alpha=0.6, label='spont')
            ax.plot(wf_laser['peak_waveform'], color=LASER, label='laser')
            ax.set_title(f"Euclid. dist. {wf_laser['euclidean_norm']:.2f}, corr. {wf_laser['correlation']:.2f}\n"
                         f"(power {wf_laser['power']:g}, {wf_laser['pre_post']} block)", fontsize=10)
            ax.legend(fontsize=7, frameon=False)
    ax.set_xlabel('Samples')
    style(ax)

    ax = fig.add_subplot(sub[1])
    waveform = data['waveforms'][unit_id] if data['waveforms'] is not None else None
    if waveform is not None:
        wp = data['waveform_params']
        waveform = waveform[:, 0:int(0.5 * np.shape(waveform)[1])]
        shifted_cmap = shiftedColorMap(b_w_r_cmap, np.nanmin(waveform), np.nanmax(waveform), 'shifted_b_w_r')
        cax = ax.imshow(waveform, extent=[wp['samples_to_keep'][0], wp['samples_to_keep'][1],
                                          2 * wp['y_neighbors_to_keep'] + 1, 0],
                        cmap=shifted_cmap, aspect='auto')
        fig.colorbar(cax, ax=ax)
        ax.set_xlabel('Samples')
        ax.set_ylabel('Channel')
    for spine in ax.spines.values():
        spine.set_visible(False)

    # autocorrelograms: rows short / long lag range, columns laser-free / laser periods; each row
    # shares its y scale
    sub = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[0, 4], hspace=0.45, wspace=0.15)
    crosscorr = data['crosscorr']
    for k, length in enumerate(['short', 'long']):
        ax_row = []
        for j, (period, color, name) in enumerate([('nolaser', 'k', 'no laser'), ('laser', LASER, 'laser')]):
            ax = fig.add_subplot(sub[k, j], sharey=ax_row[0] if ax_row else None)
            ax_row.append(ax)
            if crosscorr is not None:
                cc = crosscorr[length]
                ids = np.asarray(cc[f'unit_ids_{period}'])
                if unit_id in ids:
                    i = np.where(ids == unit_id)[0][0]
                    acg = cc[f'correlogram_{period}'][i, i, :].astype(float)
                    if acg.sum() > 0:
                        ax.stairs(acg / acg.sum(), cc['time_bins'], color=color, fill=True, alpha=0.7)
                ax.set_xlim(cc['time_bins'][0], cc['time_bins'][-1])
            ax.set_title(f'ACG {length}: {name}', fontsize=10)
            if j == 0:
                ax.set_ylabel('Fraction')
            else:
                ax.tick_params(labelleft=False)
            if k == 1:
                ax.set_xlabel('Lag (ms)')
            style(ax)

    return per_power, best


def plane_panel(ax, meshes, unit_tbl, session, unit_id, title, xd, yd):
    """Projection of the DRN mesh components (filled) and aqueduct (outline) with the tagged units
    of all sessions (gray), the session's units and the tagged ones among them (red)."""
    for k, (label, verts) in enumerate(meshes['drn']):
        for j, c in enumerate(project_to_plane(verts, [xd, yd], pitch=0.02, margin=0.5)):
            # later components sit under earlier ones so the DR core stays on top
            ax.fill(c[:, 0], c[:, 1], facecolor=MESH_FILLS[k % len(MESH_FILLS)], edgecolor=MESH_EDGE,
                    linewidth=0.5, zorder=-k, label=label if j == 0 else None)
    if 'AQ' in meshes:
        for c in project_to_plane(meshes['AQ'], [xd, yd], pitch=0.02, margin=0.5):
            ax.plot(c[:, 0], c[:, 1], color=MUTED, linewidth=0.6, zorder=1)

    others = tagged_ccf()
    others = others[others['session'] != session]
    X_all = others[CCF_COLS].to_numpy(float) - BREGMA_LPS_MM
    ax.scatter(X_all[:, xd], X_all[:, yd], s=10, color=ALL_TAGGED_UNIT, linewidth=0, alpha=0.8,
               zorder=2, label='tagged, all sessions')

    tbl = unit_tbl[np.isfinite(unit_tbl[CCF_COLS].to_numpy(float)).all(axis=1)]
    X = tbl[CCF_COLS].to_numpy(float) - BREGMA_LPS_MM
    tagged = np.array([is_tagged(session, u) for u in tbl['unit_id']], dtype=bool)
    is_target = (tbl['unit_id'] == unit_id).values
    other = ~tagged & ~is_target
    ax.scatter(X[other, xd], X[other, yd], s=6, color=SESSION_UNIT, linewidth=0, zorder=1.5,
               label='other units, this session')
    ax.scatter(X[tagged & ~is_target, xd], X[tagged & ~is_target, yd], s=22, color=SESSION_TAGGED_UNIT,
               edgecolor=SESSION_TAGGED_EDGE, linewidth=0.6, zorder=3, label='tagged, this session')
    ax.scatter(X[is_target, xd], X[is_target, yd], s=160, marker='*', color=TARGET_UNIT,
               edgecolor='white', linewidth=1, zorder=4, label=f'unit {unit_id}')

    all_verts = np.vstack([verts for _, verts in meshes['drn']])
    lims = np.vstack([all_verts.min(axis=0) - PLANE_PAD, all_verts.max(axis=0) + PLANE_PAD])
    if is_target.any():  # keep the unit in view even if it is outside the DR
        lims[0] = np.minimum(lims[0], X[is_target][0] - 0.1)
        lims[1] = np.maximum(lims[1], X[is_target][0] + 0.1)
    ax.set_xlim(lims[0, xd], lims[1, xd])
    ax.set_ylim(lims[0, yd], lims[1, yd])
    ax.set_aspect('equal')
    ax.set_xlabel(AXIS_LABEL[xd], fontsize=9, color=INK_2)
    ax.set_ylabel(AXIS_LABEL[yd], fontsize=9, color=INK_2)
    ax.set_title(title, loc='left', fontsize=fs, color=INK)
    style(ax)


def shade_epochs(ax, data, t0, ephys_cut):
    """Opto blocks (red), behavior (blue) and the unit's drift cut (dashed), in minutes from t0."""
    opto_df = data['opto_df']
    for pre_post in ['pre', 'post']:
        t = opto_df.query('pre_post == @pre_post')['time']
        if len(t) > 0:
            ax.axvspan((t.min() - t0) / 60, (t.max() - t0) / 60, color=LASER, alpha=0.15, linewidth=0)
    if data['tblTrials'] is not None:
        # all trials, ignored included, so the span matches the choice plot
        go = data['tblTrials']['goCue_start_time']
        ax.axvspan((go.min() - t0) / 60, (go.max() - t0) / 60, color=TARGET_UNIT, alpha=0.08, linewidth=0)
    # arrows point from each cut into the kept part of the recording
    for cut, direction in zip(ephys_cut, [1, -1]):
        if cut is not None:
            x = (cut - t0) / 60
            ax.axvline(x, color='k', ls='--', lw=1)
            ax.annotate('', xy=(x, 0.9), xycoords=ax.get_xaxis_transform(),
                        xytext=(direction * 22, 0), textcoords='offset points',
                        arrowprops=dict(arrowstyle='<-', color='k', lw=1.2))


def timeline_panels(fig, spec, unit_id, spike_times_all, data, ephys_cut, bin_len=30):
    """Firing rate, choice history and spike amplitude on one time axis (minutes from the
    unit's first spike)."""
    t0 = spike_times_all.min()
    t_end = (spike_times_all.max() - t0) / 60
    sub = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=spec, height_ratios=[1, 0.9, 1],
                                           hspace=0.45)

    # firing rate: mean per 30 s bin, peak PEAK_BIN rate within it, and candidate bursts
    ax_fr = fig.add_subplot(sub[0])
    n_sub = int(round(bin_len / PEAK_BIN))
    fine_edges = np.arange(t0, spike_times_all.max() + PEAK_BIN, PEAK_BIN)
    fine, _ = np.histogram(spike_times_all, bins=fine_edges)
    n_bins = int(np.ceil(len(fine) / n_sub))
    blocks = np.zeros(n_bins * n_sub)
    blocks[:len(fine)] = fine
    blocks = blocks.reshape(n_bins, n_sub)
    edges = t0 + bin_len * np.arange(n_bins + 1)
    x_edges = (edges - t0) / 60
    mean_rate = blocks.sum(axis=1) / bin_len
    k_max = blocks.max(axis=1)
    peak_rate = k_max / PEAK_BIN
    in_rec = fine_edges[:-1] <= spike_times_all.max()
    burst_thresh = np.percentile(fine[in_rec] / PEAK_BIN, BURST_PCT)
    floor = 0.5 / bin_len  # empty bins sit at the bottom of the log axis
    ax_fr.stairs(np.maximum(peak_rate, floor), x_edges, color=PEAK_COLOR, lw=1,
                 label=f'peak {PEAK_BIN:g} s rate')
    ax_fr.stairs(np.maximum(mean_rate, floor), x_edges, color='k', lw=1.2, label=f'mean ({bin_len:.0f} s bins)')
    burst = peak_rate > burst_thresh
    centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    ax_fr.axhline(burst_thresh, color=PEAK_COLOR, ls=':', lw=0.8)
    ax_fr.scatter(centers[burst], peak_rate[burst] * 1.35, marker='v', s=18, color=INK, zorder=4,
                  label=f'peak > unit\'s {BURST_PCT}th pct of {PEAK_BIN:g} s rates ({burst_thresh:.0f} Hz; n={int(burst.sum())})')
    shade_epochs(ax_fr, data, t0, ephys_cut)
    ax_fr.set_yscale('log')
    ax_fr.set_ylim(floor, max(peak_rate.max(), 1) * 2.5)
    ax_fr.set_xlim(0, t_end)
    ax_fr.set_ylabel('Rate (Hz)')
    ax_fr.set_title('Session firing rate (red: opto blocks, blue: behavior, dashed: drift cut)',
                    fontsize=10)
    ax_fr.legend(fontsize=7, frameon=False, loc='lower left', ncol=3)
    style(ax_fr)

    # choice history against go-cue time, all trials including ignored
    tblTrials = data['tblTrials']
    if tblTrials is not None and len(tblTrials) > 0:
        choice_history, reward_history, p_reward, autowater_offered, trial_time = get_history_from_nwb(tblTrials)
        ax = fig.add_subplot(sub[1])
        _, axes = plot_foraging_session(choice_history, reward_history, p_reward=p_reward,
                                        autowater_offered=autowater_offered,
                                        trial_time=(trial_time - t0) / 60, ax=ax, vertical=False)
        ax_choice, ax_sched = axes
        # the full legend is on the session panel in the behavior row; this one would cover
        # the firing rate above
        if ax_choice.get_legend() is not None:
            ax_choice.get_legend().remove()
        for ax in axes:
            ax.set_xlim(0, t_end)
        ax_sched.spines['bottom'].set_bounds(0, t_end)
        ax_sched.set_xlabel('')
        ax_sched.tick_params(labelbottom=False)
        ax_choice.set_title('Choices and reward probability', fontsize=10)

    # spike amplitude; y limited to mean +/- AMP_YLIM_SD so outliers don't set the scale
    ax_amp = fig.add_subplot(sub[2], sharex=ax_fr)
    amplitudes = data['spike_amplitudes']
    if amplitudes is not None and unit_id in amplitudes and len(amplitudes[unit_id]) == len(spike_times_all):
        amp = -np.asarray(amplitudes[unit_id], dtype=float)  # sign as in opto_plotting_unit
        ax_amp.scatter((spike_times_all - t0) / 60, amp, s=1, color=TARGET_UNIT, alpha=0.4,
                       linewidth=0, rasterized=True)
        mu, sd = np.nanmean(amp), np.nanstd(amp)
        ax_amp.set_ylim(mu - AMP_YLIM_SD * sd, mu + AMP_YLIM_SD * sd)
        ax_amp.axhline(0, color='k', ls='--', lw=0.5)
        shade_epochs(ax_amp, data, t0, ephys_cut)
        ax_amp.set_title(f'Spike amplitude (uV; y = mean ± {AMP_YLIM_SD:g} SD)', fontsize=10)
    else:
        ax_amp.text(0.5, 0.5, 'Spike amplitudes unavailable', ha='center', va='center',
                    color=MUTED, transform=ax_amp.transAxes)
        ax_amp.set_title('Spike amplitude (uV)', fontsize=10)
    ax_amp.set_xlim(0, t_end)
    ax_amp.set_xlabel('Time from first spike (min)')
    ax_amp.set_ylabel('Amplitude (uV)')
    style(ax_amp)


def population_panels(fig, spec, unit_row, unit_latency_ms):
    """Tagged vs matched non-tagged units for each POP_METRICS entry, with this unit marked.
    Bars are fractions of each group so the two groups compare despite different sizes; values
    outside the bin range are drawn in the edge bins."""
    pop = population_metrics()
    unit_vals = {'firing_rate': unit_row.get('firing_rate'), 'isi_violations_ratio': unit_row.get('isi_violations_ratio'),
                 'snr': unit_row.get('snr'), 'latency_ms': unit_latency_ms}
    sub = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=spec, hspace=0.75, wspace=0.3)
    for k, (col, label, scale) in enumerate(POP_METRICS):
        ax = fig.add_subplot(sub[k // 2, k % 2])
        vals = pd.to_numeric(pop[col], errors='coerce')
        u = pd.to_numeric(pd.Series([unit_vals[col]]), errors='coerce').iloc[0]
        finite = vals[np.isfinite(vals)]
        lo, hi = finite.min(), np.nanpercentile(finite, 99)
        if np.isfinite(u):
            lo, hi = min(lo, u), max(hi, u)
        if scale == 'log':
            lo = max(lo, finite[finite > 0].min())
            bins = np.logspace(np.log10(lo), np.log10(hi), 25)
            ax.set_xscale('log')
        else:
            bins = np.linspace(lo, hi, 25)
        for is_tag, color, name in [(True, TAGGED_UNIT, 'tagged'), (False, NONTAGGED_POP, 'non-tagged')]:
            x = vals[(pop['tagged'] == is_tag).values]
            x = np.clip(x[np.isfinite(x)], bins[0], bins[-1])
            if len(x) == 0:
                continue
            w = np.full(len(x), 1 / len(x))
            ax.hist(x, bins=bins, weights=w, histtype='stepfilled', color=color, alpha=0.25, linewidth=0)
            ax.hist(x, bins=bins, weights=w, histtype='step', color=color, linewidth=1.2,
                    label=f'{name} (n={len(x)})')
        text = 'n/a'
        if np.isfinite(u):
            ax.axvline(u, color=TARGET_UNIT, linewidth=2.5, zorder=5)
            text = f'{u:.3f}' if col == 'isi_violations_ratio' else f'{u:.2f}' if col != 'latency_ms' else f'{u:.1f}'
            if col == 'isi_violations_ratio' and np.isfinite(pd.to_numeric(unit_row.get('isi_violations_ratio_nonopto'), errors='coerce')):
                text += f" (laser-free {float(unit_row['isi_violations_ratio_nonopto']):.3f})"
        ax.set_title(f'{label}\nthis unit: {text}', fontsize=10)
        ax.set_xlabel(label, fontsize=9)
        if k % 2 == 0:
            ax.set_ylabel('Fraction of units', fontsize=9)
        if k == 0:
            ax.legend(fontsize=7, frameon=False, loc='upper left')
        style(ax)


# ---------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------

def plot_unit_summary(data, unit_id, meshes, curate_time=True, pre_event=-1.5, post_event=3,
                      binSize=0.2, stepSize=0.05):
    unit_row = data['unit_tbl'].query('unit_id == @unit_id').iloc[0].to_dict()
    spike_times_all = unit_row['spike_times']
    spike_times, session_df, tblTrials, ephys_cut = curate_unit_times(data, unit_id, curate_time)
    side, ml = ipsi_side(unit_row)
    spont = spontaneous_licks(data, tblTrials, ephys_cut)

    fig = plt.figure(figsize=(31, 24))
    outer = gridspec.GridSpec(3, 1, height_ratios=[1.35, 1, 1.45], hspace=0.25,
                              left=0.04, right=0.98, top=0.93, bottom=0.04)
    psth = {}
    behavior_row(fig, outer[0], spike_times, session_df, tblTrials, spont, side, pre_event, post_event,
                 binSize, stepSize, store=psth)
    # opto uses the whole recording, matching opto_plotting_session's default (no ephys cut)
    per_power, best = opto_row(fig, outer[1], data, unit_id, spike_times_all, unit_row)

    gs = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[2], wspace=0.2,
                                          width_ratios=[3, 1.6, 1.5])
    planes = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs[0, 0], height_ratios=[0.1, 1],
                                              hspace=0.05, wspace=0.35)
    # unit location over the three plane views
    X = np.array([unit_row.get(c, np.nan) for c in CCF_COLS], dtype=float) - BREGMA_LPS_MM
    depth = pd.to_numeric(unit_row.get('depth'), errors='coerce')
    ax = fig.add_subplot(planes[0, :])
    ax.axis('off')
    loc = ('no CCF location' if not np.isfinite(X).all() else
           f'ML {X[ML]:+.2f}   AP {X[AP]:.2f}   DV {X[DV]:.2f} mm from bregma')
    ax.text(0, 0.5, f'Unit {unit_id}:  {loc}   |   probe depth {depth:.0f} um', fontsize=13,
            color=INK, va='center', transform=ax.transAxes)
    if meshes['drn'] and 'x_ccf' in data['unit_tbl'].columns:
        for k, (title, xd, yd) in enumerate(PLANES):
            ax = fig.add_subplot(planes[1, k])
            plane_panel(ax, meshes, data['unit_tbl'], data['session'], unit_id, title, xd, yd)
            ax.set_anchor('N')  # equal-aspect panels sit right under the location line
            if k == 0:
                ax.legend(fontsize=7, frameon=False, loc='lower left')
    timeline_panels(fig, gs[0, 1], unit_id, spike_times_all, data, ephys_cut)
    cond = p_max_condition(data, unit_id)
    latency_ms = 1000 * cond['resp_lat'] if cond is not None and pd.notna(cond['resp_lat']) else np.nan
    population_panels(fig, gs[0, 2], unit_row, latency_ms)

    key = (data['session'], int(unit_id))
    passed = criteria_passed().get(key, [])
    tagged = is_tagged(*key)
    fig.text(0.04, 0.99, 'TAGGED' if tagged else 'NOT TAGGED', fontsize=32, fontweight='bold',
             color=TAGGED_LABEL if tagged else INK, va='top')
    fig.suptitle(f"{data['session']}  unit {unit_id}  |  criteria: {', '.join(passed) or 'none'}",
                 fontsize=18, y=0.99)
    anti = antidromic_passed().get(key)
    anti_label = ('antidromic: not tested' if anti is None else 'antidromic  ' + '  |  '.join(
        f"{focus}: {', '.join(anti[focus]) or 'none'}" for focus in ANTI_FOCUSES))
    fig.text(0.5, 0.968, anti_label, fontsize=15, ha='center', va='top')
    match = nontagged_match(*key)
    info = [f"decoder {unit_row.get('decoder_label')}", f"default qc {unit_row.get('default_qc')}",
            f"opto pass {unit_row.get('opto_pass')}",
            f'ipsi lick side {side}' if side is not None else 'no CCF: lick sides as right/left']
    if match is not None:
        info.append(f"matched non-tagged control for tagged unit {int(match['matched_tagged_unit'])} "
                    f"({match['distance_um']:.0f} um away)")
    fig.text(0.5, 0.95, '   |   '.join(info), fontsize=13, ha='center', va='top', color=INK_2)

    os.makedirs(PSTH_DIR, exist_ok=True)
    with open(os.path.join(PSTH_DIR, f"{data['session']}_unit_{unit_id}_psth.pkl"), 'wb') as f:
        pickle.dump({'session': data['session'], 'unit_id': int(unit_id), 'tagged': tagged,
                     'criteria': passed, 'ipsi_side': side,  # None: no CCF, ipsi/contra are right/left
                     'ml_mm': ml, 'ephys_cut': ephys_cut,
                     'matched_tagged_unit': None if match is None else int(match['matched_tagged_unit']),
                     'firing_rate': unit_row.get('firing_rate'), 'window': (pre_event, post_event),
                     'spont_window': SPONT_WINDOW, 'psth': psth}, f)
    return fig


def load_population_psth(psth_dir=PSTH_DIR):
    """Unit-mean PSTHs saved with the summary pages, one row per unit x condition: session,
    unit_id, tagged, condition, bin_times, mean, sem, n_trials. The per-trial matrices are in
    each unit's pickle ('psth' -> condition -> 'matrix')."""
    rows = []
    for f in sorted(os.listdir(psth_dir)):
        if not f.endswith('_psth.pkl'):
            continue
        d = _load_pickle(os.path.join(psth_dir, f))
        for cond, p in d['psth'].items():
            rows.append({'session': d['session'], 'unit_id': d['unit_id'], 'tagged': d['tagged'],
                         'ipsi_side': d['ipsi_side'], 'condition': cond, 'bin_times': p['bin_times'],
                         'mean': p['mean'], 'sem': p['sem'], 'n_trials': p['n_trials']})
    return pd.DataFrame(rows)


def plot_unit_summary_session(session, units=None, data_type='raw', target='soma', model_name=None,
                              curate_time=True, out_dir=OUT_DIR, save_format='pdf', meshes=None,
                              mesh_set=DRN_MESH_SET):
    """Write one summary page per unit; defaults to the session's tagged units that pass qc."""
    data = load_session_data(session, data_type=data_type, target=target, model_name=model_name)
    if meshes is None:
        meshes = load_meshes(mesh_set)
    if units is None:
        unit_tbl = data['unit_tbl']
        units = unit_tbl[unit_tbl['opto_pass'].astype(bool) & unit_tbl['default_qc'].astype(bool)]['unit_id'].values
    os.makedirs(out_dir, exist_ok=True)
    for unit_id in units:
        print(f'Plotting {session} unit {unit_id}')
        fig = plot_unit_summary(data, unit_id, meshes, curate_time=curate_time)
        save_fig(fig, os.path.join(out_dir, f'{session}_unit_{unit_id}_summary'), save_format=save_format)
        plt.close(fig)


def plot_unit_summary_table(selection, mesh_set=DRN_MESH_SET, **kwargs):
    """selection: DataFrame with 'session' and 'unit_id' columns (e.g. rows of the master table)."""
    meshes = load_meshes(mesh_set)
    for session, rows in selection.groupby('session'):
        try:
            plot_unit_summary_session(session, units=rows['unit_id'].tolist(), meshes=meshes, **kwargs)
        except Exception as e:
            print(f'Failed {session}: {type(e).__name__}: {e}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('session')
    parser.add_argument('units', nargs='*', type=int, help='unit ids; default: tagged units passing qc')
    parser.add_argument('--data-type', default='raw')
    parser.add_argument('--model-name', default=None)
    parser.add_argument('--save-format', default='pdf', choices=['pdf', 'svg', 'both'])
    parser.add_argument('--out-dir', default=OUT_DIR)
    parser.add_argument('--mesh-set', default=DRN_MESH_SET, choices=list(MESH_SETS))
    args = parser.parse_args()
    plot_unit_summary_session(args.session, units=args.units or None, data_type=args.data_type,
                              model_name=args.model_name, out_dir=args.out_dir,
                              save_format=args.save_format, mesh_set=args.mesh_set)

# %%
import sys
import os

curr_path = os.path.dirname(os.path.abspath(__file__))

# Add repoA to sys.path at the beginning
if curr_path not in sys.path:
    sys.path.insert(0, curr_path)
else:
    # move it to the front if it's already there
    sys.path.remove(curr_path)
    sys.path.insert(0, curr_path)

from harp.clock import decode_harp_clock, align_timestamps_to_anchor_points
from open_ephys.analysis import Session
import datetime
from aind_ephys_rig_qc.temporal_alignment import search_harp_line
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from pynwb import NWBFile, TimeSeries, NWBHDF5IO
from scipy.io import loadmat
from scipy.stats import zscore
import ast
from utils.plot_utils import combine_pdf_big

from open_ephys.analysis import Session
from pathlib import Path
import glob

import json
import seaborn as sns
from PyPDF2 import PdfMerger
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import re
from aind_dynamic_foraging_basic_analysis.plot.plot_foraging_session import plot_foraging_session
from aind_dynamic_foraging_data_utils.nwb_utils import load_nwb_from_filename
from hdmf_zarr.nwb import NWBZarrIO
from utils.beh_functions import session_dirs, parseSessionID, load_model_dv, makeSessionDF, get_session_tbl, get_unit_tbl, get_history_from_nwb
from utils.ephys_functions import*
import pandas as pd
import pickle
import scipy.stats as stats
from joblib import Parallel, delayed
from multiprocessing import Pool
from functools import partial
import time
import shutil 
from aind_ephys_utils import align

# ---------------------------------------------------------------------
# Helpers for filling missing unit_beh_analysis_JL plots from master table
# ---------------------------------------------------------------------

DEFAULT_MASTER_TABLE_PATHS = (
    Path('/root/capsule/scratch/combined/master_unit_tables/master_all_units_opto_ccf_raw_fix_260710_rebuilt_corr_from_metrics.pkl'),
    Path('/root/capsule/scratch/combined/master_unit_tables/master_all_units_opto_ccf_raw_fix_260710_rebuilt_corr_from_metrics.csv'),
    Path('/root/capsule/scratch/combined/master_unit_tables/master_all_units_opto_ccf.pkl'),
    Path('/root/capsule/scratch/combined/master_unit_tables/master_all_units_opto_ccf.csv'),
)
DEFAULT_MASTER_SEARCH_DIR = Path('/root/capsule/scratch/combined/master_unit_tables')
DEFAULT_RUN_SUMMARY_DIR = Path('/root/capsule/scratch/combined/unit_beh_analysis_JL')


def _as_bool_series(s):
    """Robustly convert bool/int/string columns to boolean."""
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)

    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).astype(float) != 0

    s_str = s.astype(str).str.strip().str.lower()
    return s_str.isin(['true', '1', '1.0', 't', 'yes', 'y'])


def _normalize_unit_id_value(unit_id):
    """Return int-like unit IDs as ints so filenames match unit_16_response.pdf."""
    if pd.isna(unit_id):
        return None

    if isinstance(unit_id, (int, np.integer)):
        return int(unit_id)

    if isinstance(unit_id, (float, np.floating)):
        if np.isfinite(unit_id) and float(unit_id).is_integer():
            return int(unit_id)
        return float(unit_id)

    unit_str = str(unit_id).strip()
    try:
        unit_float = float(unit_str)
        if np.isfinite(unit_float) and unit_float.is_integer():
            return int(unit_float)
    except Exception:
        pass

    return unit_str


def _normalize_mouse_id_value(mouse_id):
    """Normalize mouse IDs so 835444, 835444.0, and '835444' compare equal."""
    if mouse_id is None:
        return None

    try:
        if pd.isna(mouse_id):
            return None
    except Exception:
        pass

    if isinstance(mouse_id, (int, np.integer)):
        return str(int(mouse_id))

    if isinstance(mouse_id, (float, np.floating)):
        if not np.isfinite(mouse_id):
            return None
        if float(mouse_id).is_integer():
            return str(int(mouse_id))
        return str(mouse_id)

    mouse_id_str = str(mouse_id).strip()
    if mouse_id_str.lower() in ['', 'nan', 'none', '<na>']:
        return None

    # CSVs sometimes round-trip integer IDs as strings like "835444.0".
    try:
        mouse_id_float = float(mouse_id_str)
        if np.isfinite(mouse_id_float) and mouse_id_float.is_integer():
            return str(int(mouse_id_float))
    except Exception:
        pass

    return mouse_id_str


def _parse_mouse_ids(mouse_ids):
    """Parse a comma/space/semicolon separated mouse-ID string or iterable."""
    if mouse_ids is None:
        return None

    if isinstance(mouse_ids, str):
        mouse_ids = mouse_ids.strip()
        if mouse_ids.lower() in ['', 'all', '*', 'none', 'null']:
            return None
        pieces = re.split(r'[,;\s]+', mouse_ids)
    else:
        try:
            pieces = list(mouse_ids)
        except TypeError:
            pieces = [mouse_ids]

    normalized = []
    for value in pieces:
        value_norm = _normalize_mouse_id_value(value)
        if value_norm is not None:
            normalized.append(value_norm)

    if len(normalized) == 0:
        return None

    # Preserve user order while removing duplicates.
    return tuple(dict.fromkeys(normalized))


def _mouse_id_from_session(session):
    """Fallback parser for session strings like behavior_835444_2026-02-18_13-01-55."""
    if session is None:
        return None

    try:
        if pd.isna(session):
            return None
    except Exception:
        pass

    match = re.search(r'behavior_([^_]+)_', str(session))
    if match is None:
        return None
    return _normalize_mouse_id_value(match.group(1))


def _unit_id_to_filename(unit_id):
    unit_id = _normalize_unit_id_value(unit_id)
    return str(unit_id)


def _unit_beh_plot_paths(pdf_dir, unit_id, align_name):
    """Expected per-unit unit_beh_analysis_JL output paths."""
    pdf_dir = Path(pdf_dir)
    unit_id_str = _unit_id_to_filename(unit_id)
    return {
        '.pdf': pdf_dir / f'unit_{unit_id_str}_{align_name}.pdf',
        '.svg': pdf_dir / f'unit_{unit_id_str}_{align_name}.svg',
        '.png': pdf_dir / f'unit_{unit_id_str}_{align_name}.png'
    }


def _unit_beh_plot_complete(pdf_dir, unit_id, align_name, required_plot_exts=('.pdf', '.svg')):
    paths = _unit_beh_plot_paths(pdf_dir, unit_id, align_name)
    for ext in required_plot_exts:
        path = paths[ext]
        if not path.exists() or path.stat().st_size == 0:
            return False
    return True


def _resolve_master_table_path(master_path=None, search_dir=DEFAULT_MASTER_SEARCH_DIR):
    """Resolve the master unit table path, preferring explicit/default exact paths."""
    candidates = []
    if master_path is not None:
        candidates.append(Path(master_path))
    candidates.extend(DEFAULT_MASTER_TABLE_PATHS)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    search_dir = Path(search_dir)
    globbed = []
    for pattern in (
        'master_all_units_opto_ccf*.pkl',
        'master_all_units_opto_ccf*.pickle',
        'master_all_units_opto_ccf*.csv',
    ):
        globbed.extend(search_dir.glob(pattern))

    if len(globbed) == 0:
        raise FileNotFoundError(
            'Could not find a master_all_units_opto_ccf table. '
            f'Tried exact paths: {[str(p) for p in candidates]} and search dir {search_dir}'
        )

    return max(globbed, key=lambda p: p.stat().st_mtime)


def _load_master_table(master_path=None):
    master_path = _resolve_master_table_path(master_path)
    suffix = master_path.suffix.lower()

    if suffix in ['.pkl', '.pickle']:
        master = pd.read_pickle(master_path)
    elif suffix == '.csv':
        master = pd.read_csv(master_path)
    else:
        raise ValueError(f'Unsupported master table extension: {master_path}')

    return master, master_path


def _well_optotagged_mask(master):
    """
    Select well-optotagged units.

    The preferred criterion matches the strict red-dot criterion used for the
    master CCF map: opto_pass, default_qc, non-artifact decoder label,
    p_max > 0.40, p_mean > 0.10, pass_count >= 2, and
    0.007 < lat_max_p < 0.025.
    """
    required_for_strict = [
        'opto_pass',
        'default_qc',
        'decoder_label',
        'p_max',
        'lat_max_p',
        'isi_violations_ratio',
        'corr_max_p'
    ]

    missing_for_strict = [c for c in required_for_strict if c not in master.columns]

    if len(missing_for_strict) == 0:
        decoder = master['decoder_label'].astype('string').str.strip().str.lower()
        is_nonartifact = (
            decoder.notna()
            & ~decoder.isin(['noise', 'artifact', 'nan', 'none', '<na>', ''])
        )

        opto_pass_bool = _as_bool_series(master['opto_pass'])
        default_qc_bool = _as_bool_series(master['default_qc'])
        p_max = pd.to_numeric(master['p_max'], errors='coerce')
        pass_count = pd.to_numeric(master['pass_count'], errors='coerce')
        lat_max_p = pd.to_numeric(master['lat_max_p'], errors='coerce')
        isi_violations_ratio = pd.to_numeric(master['isi_violations_ratio'], errors='coerce')
        corr_max_p = pd.to_numeric(master['corr_max_p'], errors='coerce')

        mask = (
            opto_pass_bool
            & default_qc_bool
            & is_nonartifact
            & (p_max > 0.50)
            & (pass_count >= 2)
            & (lat_max_p > 0.005)
            & (lat_max_p < 0.025)
            & (isi_violations_ratio < 0.1)
            & (corr_max_p > 0.7)
        )
        return mask.fillna(False), 'candidate_strict_5ht_no_wf_recomputed'

    for fallback_col in ['candidate_strict_5ht_no_wf', 'first_pass_putative_5ht_no_wf', 'first_pass_putative_opto']:
        if fallback_col in master.columns:
            return _as_bool_series(master[fallback_col]), fallback_col

    if {'opto_pass', 'default_qc'}.issubset(master.columns):
        return (
            _as_bool_series(master['opto_pass'])
            & _as_bool_series(master['default_qc'])
        ), 'opto_pass_and_default_qc_fallback'

    raise ValueError(
        'Could not identify well-optotagged units. Missing strict columns: '
        f'{missing_for_strict}; no fallback optotag columns found.'
    )


def _prepare_master_units_for_plotting(master, data_types_to_run=None, mouse_ids_to_run=None):
    if 'unit_id' not in master.columns:
        raise ValueError('Master table must contain a unit_id column.')

    session_col = 'session' if 'session' in master.columns else 'session_id'
    if session_col not in master.columns:
        raise ValueError('Master table must contain session or session_id.')

    well_mask, criterion_name = _well_optotagged_mask(master)
    unit_tbl = master.loc[well_mask].copy()

    if 'data_type' not in unit_tbl.columns:
        unit_tbl['data_type'] = 'raw'

    session_str = unit_tbl[session_col].astype('string').str.strip()
    good_session = (
        session_str.notna()
        & ~session_str.str.lower().isin(['', 'nan', 'none', '<na>'])
    )
    unit_tbl = unit_tbl.loc[good_session].copy()
    unit_tbl['session'] = session_str.loc[good_session].astype(str)

    if 'mouse_id' not in unit_tbl.columns:
        unit_tbl['mouse_id'] = unit_tbl['session'].map(_mouse_id_from_session)
    unit_tbl['mouse_id_norm'] = unit_tbl['mouse_id'].map(_normalize_mouse_id_value)

    mouse_ids_to_run = _parse_mouse_ids(mouse_ids_to_run)
    if mouse_ids_to_run is not None:
        unit_tbl = unit_tbl[unit_tbl['mouse_id_norm'].isin(mouse_ids_to_run)].copy()

    unit_tbl['data_type'] = unit_tbl['data_type'].fillna('raw').astype(str)
    unit_tbl['unit_id_norm'] = unit_tbl['unit_id'].map(_normalize_unit_id_value)
    unit_tbl = unit_tbl[unit_tbl['unit_id_norm'].notna()].copy()

    if data_types_to_run is not None:
        data_types_to_run = {str(data_type) for data_type in data_types_to_run}
        unit_tbl = unit_tbl[unit_tbl['data_type'].isin(data_types_to_run)].copy()

    keep_cols = [
        'session',
        'data_type',
        'unit_id',
        'unit_id_norm',
        'unit_uid',
        'mouse_id',
        'mouse_id_norm',
        'decoder_label',
        'opto_pass',
        'default_qc',
        'p_max',
        'p_mean',
        'pass_count',
        'lat_max_p',
    ]
    keep_cols = [c for c in keep_cols if c in unit_tbl.columns]

    unit_tbl = unit_tbl[keep_cols].drop_duplicates(['session', 'data_type', 'unit_id_norm'])
    unit_tbl['_unit_id_sort'] = unit_tbl['unit_id_norm'].astype(str)
    unit_tbl = (
        unit_tbl
        .sort_values(['session', 'data_type', '_unit_id_sort'])
        .drop(columns=['_unit_id_sort'])
        .reset_index(drop=True)
    )

    return unit_tbl, criterion_name


def _session_has_behavior_nwb(session, session_dir):
    beh_fig_dir = session_dir.get('beh_fig_dir')
    if beh_fig_dir is None:
        return False
    return os.path.exists(os.path.join(beh_fig_dir, f'{session}.nwb'))


def _data_type_available(session_dir, data_type):
    fig_key = f'ephys_fig_dir_{data_type}'
    ephys_key = f'ephys_dir_{data_type}'
    if session_dir.get(fig_key) is None or session_dir.get(ephys_key) is None:
        return False

    # Keep the old curated guard, but do not require curated_dir_raw for raw data.
    if data_type == 'curated':
        return session_dir.get('curated_dir_curated') is not None

    return True


def run_missing_well_opto_unit_beh_plots(
    master_path=None,
    align_names=('response',),
    data_types_to_run=None,
    mouse_ids_to_run=None,
    model_name=None,
    formula='spikes ~ 1 + outcome + choice',
    pre_event_by_align=None,
    post_event=3,
    binSize=0.2,
    stepSize=0.05,
    required_plot_exts=('.pdf', '.svg'),
    curate_time_by_data_type=None,
    run_summary_dir=DEFAULT_RUN_SUMMARY_DIR,
):
    """Plot missing unit_beh_analysis_JL figures for well-optotagged master-table units.

    Set mouse_ids_to_run to restrict existence checks and plotting to specific
    mice, for example ('835444', '835451') or '835444,835451'.
    """
    if pre_event_by_align is None:
        pre_event_by_align = {
            'response': -1,
            'go_cue': -1.5,
        }

    if curate_time_by_data_type is None:
        curate_time_by_data_type = {
            'raw': False,
            'curated': True,
        }

    align_names = tuple(align_names)
    mouse_ids_to_run = _parse_mouse_ids(mouse_ids_to_run)
    master, resolved_master_path = _load_master_table(master_path)
    unit_tbl, criterion_name = _prepare_master_units_for_plotting(
        master,
        data_types_to_run=data_types_to_run,
        mouse_ids_to_run=mouse_ids_to_run,
    )

    print(f'Loaded master table: {resolved_master_path}')
    print(f'Master rows: {len(master)}')
    print(f'Well-optotagged criterion: {criterion_name}')
    if mouse_ids_to_run is None:
        print('Mouse ID filter: all mice')
    else:
        print(f"Mouse ID filter: {', '.join(mouse_ids_to_run)}")
    print(f'Well-optotagged unique units queued for existence check: {len(unit_tbl)}')
    print(f'Alignments: {align_names}')

    records = []

    for (session, data_type), session_units in unit_tbl.groupby(['session', 'data_type'], sort=True):
        session = str(session)
        data_type = str(data_type)
        unit_ids = [_normalize_unit_id_value(unit_id) for unit_id in session_units['unit_id_norm'].tolist()]
        mouse_id_lookup = {}
        if 'mouse_id_norm' in session_units.columns:
            mouse_id_lookup = {
                _normalize_unit_id_value(row['unit_id_norm']): row['mouse_id_norm']
                for _, row in session_units.iterrows()
            }
        mouse_ids_for_session = sorted(
            {mouse_id for mouse_id in mouse_id_lookup.values() if mouse_id is not None}
        )
        mouse_label = ','.join(mouse_ids_for_session) if len(mouse_ids_for_session) > 0 else 'unknown'

        print('----------------------------------')
        print(f'Checking {session} mouse_id={mouse_label} data_type={data_type}, candidate units={len(unit_ids)}')

        try:
            session_dir = session_dirs(session, model_name=model_name)
        except Exception as exc:
            print(f'Skipping {session}: session_dirs failed: {repr(exc)}')
            for align_name in align_names:
                for unit_id in unit_ids:
                    records.append({
                        'session': session,
                        'data_type': data_type,
                        'align_name': align_name,
                        'unit_id': unit_id,
                        'mouse_id': mouse_id_lookup.get(unit_id, ''),
                        'status': 'session_dirs_failed',
                        'message': repr(exc),
                    })
            continue

        if not _session_has_behavior_nwb(session, session_dir):
            print(f'Skipping {session}: behavior NWB not found in beh_fig_dir')
            for align_name in align_names:
                for unit_id in unit_ids:
                    records.append({
                        'session': session,
                        'data_type': data_type,
                        'align_name': align_name,
                        'unit_id': unit_id,
                        'mouse_id': mouse_id_lookup.get(unit_id, ''),
                        'status': 'skipped_no_behavior_nwb',
                        'message': '',
                    })
            continue

        if not _data_type_available(session_dir, data_type):
            print(f'Skipping {session}: no {data_type} ephys data according to session_dirs')
            for align_name in align_names:
                for unit_id in unit_ids:
                    records.append({
                        'session': session,
                        'data_type': data_type,
                        'align_name': align_name,
                        'unit_id': unit_id,
                        'mouse_id': mouse_id_lookup.get(unit_id, ''),
                        'status': f'skipped_no_{data_type}_data',
                        'message': '',
                    })
            continue

        for align_name in align_names:
            pdf_dir = Path(session_dir[f'ephys_fig_dir_{data_type}']) / align_name
            existing_units = []
            missing_units = []

            for unit_id in unit_ids:
                if _unit_beh_plot_complete(pdf_dir, unit_id, align_name, required_plot_exts=required_plot_exts):
                    existing_units.append(unit_id)
                else:
                    missing_units.append(unit_id)

            print(
                f'{session} {data_type} {align_name}: '
                f'{len(existing_units)} already complete, {len(missing_units)} missing'
            )

            for unit_id in existing_units:
                paths = _unit_beh_plot_paths(pdf_dir, unit_id, align_name)
                records.append({
                    'session': session,
                    'data_type': data_type,
                    'align_name': align_name,
                    'unit_id': unit_id,
                    'mouse_id': mouse_id_lookup.get(unit_id, ''),
                    'status': 'already_exists',
                    'message': '',
                    'pdf_path': str(paths['.pdf']),
                    'svg_path': str(paths['.svg']),
                })

            if len(missing_units) == 0:
                continue

            session_exc = None
            try:
                plot_unit_beh_session(
                    session=session,
                    data_type=data_type,
                    align_name=align_name,
                    curate_time=curate_time_by_data_type.get(data_type, True),
                    opto_only=False,
                    model_name=model_name,
                    formula=formula,
                    pre_event=pre_event_by_align.get(align_name, -1),
                    post_event=post_event,
                    binSize=binSize,
                    stepSize=stepSize,
                    units=missing_units,
                    clear_existing=False,
                    skip_existing=True,
                    required_plot_exts=required_plot_exts,
                )
            except Exception as exc:
                session_exc = repr(exc)
                print(f'plot_unit_beh_session failed for {session} {data_type} {align_name}: {session_exc}')

            for unit_id in missing_units:
                paths = _unit_beh_plot_paths(pdf_dir, unit_id, align_name)
                complete_now = _unit_beh_plot_complete(
                    pdf_dir,
                    unit_id,
                    align_name,
                    required_plot_exts=required_plot_exts,
                )
                records.append({
                    'session': session,
                    'data_type': data_type,
                    'align_name': align_name,
                    'unit_id': unit_id,
                    'mouse_id': mouse_id_lookup.get(unit_id, ''),
                    'status': 'created' if complete_now else ('failed_with_exception' if session_exc else 'missing_after_attempt'),
                    'message': '' if session_exc is None else session_exc,
                    'pdf_path': str(paths['.pdf']),
                    'svg_path': str(paths['.svg']),
                })

    summary = pd.DataFrame(records)
    run_summary_dir = Path(run_summary_dir)
    run_summary_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_path = run_summary_dir / f'unit_beh_analysis_JL_missing_well_opto_summary_{timestamp}.csv'
    summary.to_csv(summary_path, index=False)

    print('----------------------------------')
    print('Done checking/generating missing unit_beh_analysis_JL plots.')
    if len(summary) > 0:
        print(summary['status'].value_counts(dropna=False).to_string())
    print(f'Saved run summary: {summary_path}')

    return summary


def plot_unit_beh_session(session, data_type = 'raw', align_name = 'go_cue', curate_time=True, opto_only=True,
                        model_name = None, 
                        formula = 'spikes ~ 1 + outcome + choice + Qchosen',
                        pre_event=-1, post_event=3, binSize=0.2, stepSize=0.05,
                        units  = None, clear_existing=True, skip_existing=False,
                        required_plot_exts=('.pdf', '.svg')):
    # %%
    # load behavior data
    session_dir = session_dirs(session, model_name = model_name) 
    session_df = makeSessionDF(session, model_name = model_name)
    tblTrials = get_session_tbl(session)
    pdf_dir = os.path.join(session_dir[f'ephys_fig_dir_{data_type}'], align_name)
    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir, exist_ok=True)
    if clear_existing:
        # Historical behavior: clear all files before rebuilding a session/alignment.
        # The master-table backfill path passes clear_existing=False so existing
        # unit plots are preserved and only missing units are generated.
        for f in os.listdir(pdf_dir):
            path = os.path.join(pdf_dir, f)
            if os.path.isfile(path) or os.path.islink(path):
                os.remove(path)  # Remove files
            elif os.path.isdir(path):
                shutil.rmtree(path)  # Remove subdirectories
        
    # %%
    qm_dir = os.path.join(session_dir['processed_dir'], f'{session}_qm.json')
    with open(qm_dir, 'r') as f:
        qm = json.load(f)

    # %% load units 
    unit_tbl = get_unit_tbl(session, data_type)

    # %%
    fs = 14
    fsLegend = 8

    # %%
    colors = ["blue", "white", "red"]
    custom_cmap = LinearSegmentedColormap.from_list("blue_white_red", colors)
    
    def plot_unit(unit_id):
        unit_drift = load_drift(session, unit_id, data_type=data_type)
        spike_times = unit_tbl.query('unit_id == @unit_id')['spike_times'].values[0]
        qc_pass = unit_tbl.query('unit_id == @unit_id')['default_qc'].values[0]
        opto_pass = unit_tbl.query('unit_id == @unit_id')['opto_pass'].values[0]
        session_df_curr = session_df.copy()
        spike_times_curr = spike_times.copy()
        tblTrials_curr = tblTrials.copy()
        if curate_time:
            if unit_drift is not None:
                if unit_drift['ephys_cut'][0] is not None:
                    spike_times_curr = spike_times_curr[spike_times_curr >= unit_drift['ephys_cut'][0]]
                    session_df_curr = session_df_curr[session_df_curr['go_cue_time'] >= unit_drift['ephys_cut'][0]]
                    tblTrials_curr = tblTrials_curr[tblTrials_curr['goCue_start_time'] >= unit_drift['ephys_cut'][0]]
                if unit_drift['ephys_cut'][1] is not None:
                    spike_times_curr = spike_times_curr[spike_times_curr <= unit_drift['ephys_cut'][1]]
                    session_df_curr = session_df_curr[session_df_curr['go_cue_time'] <= unit_drift['ephys_cut'][1]]
                    tblTrials_curr = tblTrials_curr[tblTrials_curr['goCue_start_time'] <= unit_drift['ephys_cut'][1]]
        if len(session_df_curr) == 0:
            # return None and exit function
            print(f'No session data for unit {unit_id}')
            fig = plt.figure(figsize=(20, 10))
            plt.suptitle(f'Unit{str(unit_id)} Aligned to {align_name} default qc: {qc_pass} maybe opto: {opto_pass} No behavior', fontsize = 20)
        else:
            print(f'Plotting unit {unit_id}')      
            if align_name == 'go_cue':
                align_time = session_df_curr['go_cue_time'].values
                align_time_all = tblTrials_curr['goCue_start_time'].values
            elif align_name == 'response':
                align_time = session_df_curr['choice_time'].values
                align_time_all = tblTrials_curr['reward_outcome_time'].values
            spike_matrix, slide_times = get_spike_matrix(spike_times_curr, align_time, 
                                                        pre_event=pre_event, post_event=post_event, 
                                                        binSize=binSize, stepSize=stepSize)
            spike_matrix_LM, slide_times_LM = get_spike_matrix(spike_times_curr, align_time, 
                                                        pre_event=-2, post_event=2.5, 
                                                        binSize=0.5, stepSize=0.2)
            spike_matrix_all, slide_times = get_spike_matrix(spike_times_curr, align_time_all, 
                                                        pre_event=pre_event, post_event=post_event, 
                                                        binSize=binSize, stepSize=stepSize)

            fig = plt.figure(figsize=(20, 10))
            gs = gridspec.GridSpec(2, 7, height_ratios=[3, 1], wspace=0.35, hspace=0.2)
            # plot session
            ax = fig.add_subplot(gs[0, 0]) 
            choice_history, reward_history, p_reward, autowater_offered, trial_time = get_history_from_nwb(session_df_curr)
            _, axes = plot_foraging_session(  # noqa: C901
                                            choice_history,
                                            reward_history,
                                            p_reward = p_reward,  
                                            autowater_offered = autowater_offered,
                                            ax = ax,
                                            # legend=False,
                                            vertical=True,
                                            ) 
            for ax in axes:
                ax.set_ylim(0, len(session_df_curr))
            ax.set_ylim(0, len(session_df_curr))
            # from start to end
            ax = fig.add_subplot(gs[0, 1])  
            df = align.to_events(spike_times_curr, align_time, (pre_event, post_event), return_df=True)
            plt.plot([0,0],[0,df.event_index.max()],'r', zorder = 1)
            ax.scatter(df.time, df.event_index, c='k', marker= '|', s=1, zorder = 2)
            ax.set_xlim(pre_event, post_event)
            ax.set_ylabel('Trial number')
            ax.tick_params(axis='both', which='major')
            ax.set_ylim(0, len(session_df_curr))


            # waveform
            ax = fig.add_subplot(gs[1, 0])

            waveform = None
            try:
                if 'waveform_mean' in unit_tbl.columns:
                    vals = unit_tbl.query('unit_id == @unit_id')['waveform_mean'].values
                    if len(vals) > 0:
                        waveform_candidate = vals[0]
                        if isinstance(waveform_candidate, np.ndarray) and waveform_candidate.ndim == 2 and waveform_candidate.size > 0:
                            waveform = waveform_candidate
            except Exception:
                waveform = None

            if waveform is None:
                ax.text(
                    0.5, 0.5,
                    'waveform unavailable',
                    ha='center',
                    va='center',
                    transform=ax.transAxes,
                )
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                peakChannel = np.argmin(np.min(waveform, axis=0))
                peakWaveform = waveform[:, peakChannel]
                timeWF = np.array(range(len(peakWaveform))) - 90

                ax.plot(timeWF, peakWaveform, color='k')
                ax.axhline(y=0, color='r', ls='--')
                ax.set_xlabel('Time (ms)', fontsize=fs)
                ax.set_ylabel(r'$\mu$-Plot')

            # reward and no reward
            outcome_int = [int(item) for item in session_df_curr['outcome'].tolist()]
            bins = [-1, 0.5, 1.5]
            labels = ['no reward', 'reward']
            fig, ax1, ax2 = plot_raster_rate(spike_times_curr,
                                            align_time, 
                                            outcome_int, # sorted by certain value
                                            bins,
                                            labels, 
                                            custom_cmap,
                                            fig,
                                            gs[0, 2],
                                            tb=pre_event,
                                            tf=post_event,
                                            time_bin=stepSize,
                                            )
            ax1.set_title('Reward vs No Reward', fontsize = fs+2)

            # left and right
            side_int = [int(item) for item in session_df_curr['choice'].tolist()]
            bins = [-1.5, 0.5, 1.5]
            labels = ['left', 'right']
            fig, ax1, ax2 = plot_raster_rate(spike_times_curr,
                                            align_time, 
                                            side_int, # sorted by certain value
                                            bins,
                                            labels,
                                            custom_cmap,
                                            fig,
                                            gs[0, 3],
                                            tb=pre_event,
                                            tf=post_event,
                                            time_bin=stepSize,
                                            )
            ax1.set_title('Right vs Left', fontsize = fs+2)

            # rpe
            target_var = 'pe'
            bin_counts = 4
            if target_var in session_df_curr.columns.to_list():
                bins = np.quantile(session_df_curr[target_var].values, np.linspace(0, 1, bin_counts+1))
                # bins = [-1.0001, -0.5, 0, 0.5, 1.0001]
                bins[0] = bins[0] - 0.0001
                bins[-1] = bins[-1] + 0.0001
                labels = ['1', '2', '3', '4']
                
                fig, ax1, ax2 = plot_raster_rate(spike_times_curr,
                                                align_time,  
                                                session_df_curr[target_var].values, # sorted by certain value
                                                bins,
                                                labels,
                                                custom_cmap,
                                                fig,
                                                gs[0, 4],
                                                tb=pre_event,
                                                tf=post_event,
                                                time_bin=stepSize,
                                                )
                # ax.set_yticks([])
                # ax.set_ylabel(label, fontsize = fs)
                ax1.set_title(target_var, fontsize = fs+2)


                # Qchosen
                target_var = 'Qchosen'
                bin_counts = 4
                bins = np.quantile(session_df_curr[target_var].values, np.linspace(0, 1, bin_counts+1))
                # bins = [-1.0001, -0.5, 0, 0.5, 1.0001]
                bins[0] = bins[0] - 0.0001
                bins[-1] = bins[-1] + 0.0001
                labels = ['1', '2', '3', '4']
                fig, ax1, ax2 = plot_raster_rate(spike_times_curr,
                                                align_time, 
                                                session_df_curr[target_var].values, # sorted by certain value
                                                bins,
                                                labels,
                                                custom_cmap,
                                                fig,
                                                gs[0, 5],
                                                tb=pre_event,
                                                tf=post_event,
                                                time_bin=stepSize,
                                                )
                ax1.set_title(target_var, fontsize = fs+2)

            # stay vs switch
            target_var = 'svs'
            # bins = np.quantile(session_df_curr[target_var].values, np.linspace(0, 1, bin_counts+1))
            bins = [-1.0001, 0.5, 1.0001]
            labels = ['stay', 'switch']
            # ax = fig.add_subplot(gs[1, 1])
            fig, ax = plot_rate(
                                spike_matrix,
                                slide_times, 
                                session_df_curr[target_var].values,
                                bins,
                                labels,
                                custom_cmap,
                                fig,
                                gs[1, 1],
                                tb=pre_event,
                                tf=post_event,
                                )

            ax.set_yticks([])
            ax.set_title(target_var, fontsize = fs+2)

            # right rwd vs no rwd
            target_var = 'outcome'
            spike_matrix_curr = spike_matrix[session_df_curr['choice'].values == 1, :]
            focus_var = session_df_curr[session_df_curr['choice'].values == 1][target_var].values
            bins = [-1.0001, 0.5, 1.0001]
            labels = ['no rwd', 'rwd']
            fig, ax = plot_rate(
                                spike_matrix_curr,
                                slide_times,
                                focus_var,
                                bins,
                                labels,
                                custom_cmap,
                                fig,
                                gs[1, 2],
                                tb=pre_event,
                                tf=post_event,
                                )
            ax.set_title('Right: rwd nrwd', fontsize = fs+2)

            # left rwd vs no rwd
            target_var = 'outcome'
            spike_matrix_curr = spike_matrix[session_df_curr['choice'].values == 0, :]
            focus_var = session_df_curr[session_df_curr['choice'].values == 0][target_var].values
            bins = [-1.0001, 0.5, 1.0001]
            labels = ['no rwd', 'rwd']
            fig, ax = plot_rate(
                                spike_matrix_curr,
                                slide_times,
                                focus_var,
                                bins,
                                labels,
                                custom_cmap,
                                fig,
                                gs[1, 3],
                                tb=pre_event,
                                tf=post_event,
                                )
            ax.set_title('Left: rwd nrwd', fontsize = fs+2)

            # go vs miss
            map_value = tblTrials_curr['animal_response'].values!=2
            bins = [-0.5, 0.5, 1.5]
            labels = ['miss', 'go']
            fig, ax = plot_rate(
                                spike_matrix_all,
                                slide_times,
                                map_value,
                                bins,
                                labels,
                                custom_cmap,
                                fig,
                                gs[1, 4],
                                tb=pre_event,
                                tf=post_event,
                                )


            
            if len(session_df_curr) > 100 and np.sum((spike_times_curr>=session_df_curr['go_cue_time'].values[0]) & (spike_times_curr<=session_df_curr['go_cue_time'].values[-1]))/(session_df_curr['go_cue_time'].values[-1] - session_df_curr['go_cue_time'].values[0]) > 0.1:
                # rwd history
                # fr with regression with rwd history
                target_var = 'outcome'
                vector = session_df_curr[target_var].values
                align_time = session_df_curr['choice_time'].values
                _, events_id, _ = align.to_events(spike_times_curr, align_time, (0, 1.5), return_df=False) 
                spike_counts = [np.sum(events_id==curr_id) for curr_id in range(len(align_time))]
                spike_counts = stats.zscore(np.array(spike_counts))
                trials_back = [0, 2]
                ax = fig.add_subplot(gs[1, 5])
                try:
                    coeffs, pvals, tvals, conf_int = regression_rwd(spike_counts, vector, trials_back = trials_back)
                    ax.plot(range(trials_back[0], trials_back[1] + 1), coeffs, c = 'k', lw = 2)
                    ax.fill_between(range(trials_back[0], trials_back[1] + 1), conf_int[:, 0], conf_int[:, 1], color = 'k', alpha = 0.25, edgecolor = None)
                    ax.axhline(y=0, color = 'r', ls = '--')
                    ax.scatter(np.array(range(trials_back[0], trials_back[1] + 1))[pvals<0.05], coeffs[pvals<0.05], c = 'r', s = 10, zorder = 2)
                    ax.set_title('Spikes~rwd history', fontsize = fs+2)
                    ax.set_xlabel('Trials back')
                except:
                    ax.plot(range(trials_back[0], trials_back[1] + 1), np.zeros(trials_back[1]-trials_back[0]+1), c = 'k', lw = 2, label = 'failed')
                    ax.set_title('Spikes~rwd history failed', fontsize = fs+2)


                # only on left trials
                trials_back = [0, 2]
                ax = fig.add_subplot(gs[1, 6])
                ax.set_title('Spikes~rwd hist L/R', fontsize = fs+2)
                ax.set_xlabel('Trials back')
                if np.sum(session_df_curr['choice'].values == 0) >= 40:
                    try:
                        coeffs, pvals, tvals, conf_int = regression_rwd(spike_counts, vector, trials_back = trials_back, sub_selection=session_df_curr['choice'].values == 0)
                        ax.plot(range(trials_back[0], trials_back[1] + 1), coeffs, c = 'm', lw = 2, label = 'left')
                        ax.fill_between(range(trials_back[0], trials_back[1] + 1), conf_int[:, 0], conf_int[:, 1], color = 'm', alpha = 0.25, edgecolor = None)
                        ax.scatter(np.array(range(trials_back[0], trials_back[1] + 1))[pvals<0.05], coeffs[pvals<0.05], c = 'r', s = 10)
                        ax.axhline(y=0, color = 'r', ls = '--')
                        ax.legend()
                    except:
                        ax.plot(range(trials_back[0], trials_back[1] + 1), np.zeros(trials_back[1]-trials_back[0]+1), c = 'm', lw = 2, label = 'left failed')

                # only on right trials
                if np.sum(session_df_curr['choice'].values == 1) >= 40:
                    try:
                        coeffs, pvals, tvals, conf_int = regression_rwd(spike_counts, vector, trials_back = trials_back, sub_selection=session_df_curr['choice'].values == 1)
                        ax.plot(range(trials_back[0], trials_back[1] + 1), coeffs, c = 'c', lw = 2, label = 'right')
                        ax.fill_between(range(trials_back[0], trials_back[1] + 1), conf_int[:, 0], conf_int[:, 1], color = 'c', alpha = 0.25, edgecolor = None)
                        ax.scatter(np.array(range(trials_back[0], trials_back[1] + 1))[pvals<0.05], coeffs[pvals<0.05], c = 'r', s = 10)
                        ax.axhline(y=0, color = 'r', ls = '--')
                        ax.set_title('Spikes~rwd hist L/R', fontsize = fs+2)
                        ax.set_xlabel('Trials back')
                        ax.legend()
                    except:
                        ax.plot(range(trials_back[0], trials_back[1] + 1), np.zeros(trials_back[1]-trials_back[0]+1), c = 'c', lw = 2, label = 'right failed')

                # plot regresssions
                gs = gridspec.GridSpec(3, 7, height_ratios=[1, 1, 1], wspace=0.3, hspace=0.3)
                ax = fig.add_subplot(gs[0,-1])
                try: 
                    regressors, TvCurrU, PvCurrU, EvCurrU = fitSpikeModelG(session_df_curr, spike_matrix_LM, formula)
                    TvCurrUSig = TvCurrU.copy()
                    TvCurrUSig[PvCurrU>=0.05] = np.nan
                    cmap = plt.get_cmap('viridis')
                    colors = cmap(np.linspace(0, 1, len(regressors)))
                    for regress in range(1, len(regressors)):
                        ax.plot(slide_times_LM, TvCurrU[:, regress], lw = 2, color = colors[regress,], label = regressors[regress])
                        ax.plot(slide_times_LM, TvCurrUSig[:, regress], lw = 4, color = colors[regress,])
                    ax.legend(fontsize = fsLegend)
                    ax.set_xlabel(f'Time from {align_name} (s)')
                    ax.set_title('T-stats', fontsize = fs)

                    ax = fig.add_subplot(gs[1,-1])
                    for regress in range(1, len(regressors)):
                        ax.plot(slide_times_LM, -np.log10(PvCurrU[:, regress]), lw = 1, color = colors[regress,], label = regressors[regress])

                    plt.axhline(y = -np.log10(0.05), color='r', ls = '--')
                    ax.legend(fontsize = fsLegend)
                    ax.set_xlabel(f'Time from {align_name} (s)')
                    ax.set_title('p-value', fontsize = fs)
                except:
                    print(f'Failed to fit model for unit {unit_id}')
            plt.suptitle(f'Unit{str(unit_id)} Aligned to {align_name} default qc: {qc_pass} maybe opto: {opto_pass}', fontsize = 20) 
            # plt.tight_layout()  
        return fig

    log_record_file = os.path.join(session_dir[f'ephys_fig_dir_{data_type}'], align_name, f'{session}_unit_beh.log')
    # def process(unit_id): 
    #     try:
    #         fig = plot_unit(unit_id) 
    #         if fig is not None:
    #             fig.savefig(fname=os.path.join(session_dir[f'ephys_fig_dir_{data_type}'], align_name, f'unit_{unit_id}_goCue.pdf'))
    #         plt.close(fig)
    #         # write to log
    #         with open(log_record_file, 'a') as f:
    #             f.write(f'Unit {unit_id} plotted\n')
    #         # pause for 1 second
    #     except:
    #         with open(log_record_file, 'a') as f:
    #             f.write(f'Unit {unit_id} failed\n')
    #     time.sleep(1)

    def process(unit_id):
        unit_id = _normalize_unit_id_value(unit_id)

        if skip_existing and _unit_beh_plot_complete(pdf_dir, unit_id, align_name, required_plot_exts=required_plot_exts):
            print(f'Skipping unit {unit_id}: existing {align_name} plots found')
            with open(log_record_file, 'a') as f:
                f.write(f'Unit {unit_id} skipped_existing\n')
            return 'skipped_existing'

        fig = None
        try:
            fig = plot_unit(unit_id)
            if fig is not None:
                paths = _unit_beh_plot_paths(pdf_dir, unit_id, align_name)
                fig.savefig(fname=paths['.pdf'])
                fig.savefig(fname=paths['.svg'])
            with open(log_record_file, 'a') as f:
                f.write(f'Unit {unit_id} plotted\n')
            status = 'plotted'
        except Exception as exc:
            print(f'Unit {unit_id} failed: {repr(exc)}')
            with open(log_record_file, 'a') as f:
                f.write(f'Unit {unit_id} failed: {repr(exc)}\n')
            status = 'failed'
        finally:
            if fig is not None:
                plt.close(fig)
            time.sleep(1)

        return status



    # with Pool(processes=4) as pool:  # Ensures cleanup
    #     result = pool.map(process, unit_tbl['unit_id'].values)
    if units is None:
        if not opto_only:
            units= unit_tbl['unit_id'].values
        else:
            units = unit_tbl[unit_tbl['opto_pass']==True]['unit_id'].values

    units = [_normalize_unit_id_value(unit_id) for unit_id in units]
    units = [unit_id for unit_id in units if unit_id is not None]

    if skip_existing:
        units_before_skip = len(units)
        units = [
            unit_id for unit_id in units
            if not _unit_beh_plot_complete(pdf_dir, unit_id, align_name, required_plot_exts=required_plot_exts)
        ]
        print(
            f'{session} {align_name}: {units_before_skip - len(units)} units already had plots; '
            f'{len(units)} units left to plot'
        )

    # Parallel(n_jobs=12)(
    #     delayed(process)(unit_id)
    #     for unit_id in units
    # )
    for unit_id in units:
        process(unit_id)

    output_pdf = os.path.join(session_dirs(session)[f'ephys_dir_{data_type}'],f'{session}_unit_beh_{align_name}.pdf')

    if os.path.exists(pdf_dir):
        print(f'Combining {session}')
        combine_pdf_big(pdf_dir, output_pdf)
    
    plt.close('all')

def burst_analysis(session, data_type, units = None):
    print(f'Processing session {session} for data type {data_type}')
    unit_tbl = get_unit_tbl(session, data_type)
    session_df = get_session_tbl(session)
    session_dir = session_dirs(session, data_type)
    save_path = os.path.join(session_dir[f'ephys_fig_dir_{data_type}'], 'burst')
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    if units is None:
        units = unit_tbl['unit_id'].tolist()
    for unit_id in units:
        if not unit_tbl[unit_tbl['unit_id']==unit_id]['tagged_loc'].values[0]:
            continue
        spike_times = unit_tbl[unit_tbl['unit_id']==unit_id]['spike_times'].values[0]
        pre_event = -0.1
        post_event= 0.1
        session_df_curr = session_df.copy()
        spike_times_curr = spike_times.copy()
        unit_drift = load_drift(session, unit_id, data_type=data_type)

        if unit_drift is not None:
            if unit_drift['ephys_cut'][0] is not None:
                spike_times_curr = spike_times_curr[spike_times_curr >= unit_drift['ephys_cut'][0]]
                session_df_curr = session_df_curr[session_df_curr['goCue_start_time'] >= unit_drift['ephys_cut'][0]]
            if unit_drift['ephys_cut'][1] is not None:
                spike_times_curr = spike_times_curr[spike_times_curr <= unit_drift['ephys_cut'][1]]
                session_df_curr = session_df_curr[session_df_curr['goCue_start_time'] <= unit_drift['ephys_cut'][1]]
        if len(session_df_curr) <=20:
            print(f'Skipping {unit_id} due to insufficient trials after drift cut.')
            continue
        # align to go cue sorted by choice time, separate by choice or not
        # from start to end
        fig = plt.figure(figsize=(14,10))
        gs = gridspec.GridSpec(2,6)
        lick_lat = session_df_curr['reward_outcome_time'].values - session_df_curr['goCue_start_time'].values
        lick_lat[session_df_curr['animal_response'].values==2] = np.nan
        pre_event = -0.1
        post_event= 0.1
        align_time = session_df_curr['goCue_start_time'].values
        align_time_licklat_sort = align_time[np.argsort(lick_lat)]
        ax = fig.add_subplot(gs[0, 0])  
        df = align.to_events(spike_times_curr, align_time_licklat_sort, (pre_event, post_event), return_df=True)
        plt.plot([0,0],[0,df.event_index.max()],'r', zorder = 1)
        ax.scatter(df.time, df.event_index, c='k', marker= '|', s=1, zorder = 2)
        ax.set_xlim(pre_event, post_event)
        ax.set_ylabel('Lick latency sorted trials')
        ax.tick_params(axis='both', which='major')
        ax.set_ylim(0, len(session_df_curr))
        ax.axhline(len(align_time_licklat_sort)-np.sum(session_df_curr['animal_response'].values==2), color='blue', linestyle='--')
        ax.set_title('Aligned to Go Cue')

        align_time = session_df_curr['reward_outcome_time'].values
        lick_lat[session_df_curr['animal_response'].values==2] = np.nan
        align_time_licklat_sort = align_time[np.argsort(lick_lat)]
        ax = fig.add_subplot(gs[0, 1])  
        df = align.to_events(spike_times_curr, align_time_licklat_sort, (pre_event, post_event), return_df=True)
        plt.plot([0,0],[0,df.event_index.max()],'r', zorder = 1)
        ax.scatter(df.time, df.event_index, c='k', marker= '|', s=1, zorder = 2)
        ax.set_xlim(pre_event, post_event)
        ax.set_ylabel('Lick latency sorted trials')
        ax.tick_params(axis='both', which='major')
        ax.set_ylim(0, len(session_df_curr))
        ax.axhline(len(align_time_licklat_sort)-np.sum(session_df_curr['animal_response'].values==2), color='blue', linestyle='--')
        ax.set_title('Aligned to Choice')


        # align to go cue and sort by frist spike time
        align_time = session_df_curr['goCue_start_time'].values
        spike_df = align.to_events(spike_times_curr, align_time, (0, 10), return_df=True)
        # for each value in event_index, get the first spike time
        first_spike_times = np.full(len(session_df_curr), np.nan)
        for i in range(len(session_df_curr)):
            spikes_in_trial = spike_df[spike_df['event_index']==i]['time']
            if len(spikes_in_trial) > 0:
                first_spike_times[i] = spikes_in_trial.min()
        # first_spike_times = np.full(len(session_df_curr), np.nan)
        align_time_firstspike_sort = align_time[np.argsort(first_spike_times)]
        ax = fig.add_subplot(gs[0, 2])
        df = align.to_events(spike_times_curr, align_time_firstspike_sort, (pre_event, post_event), return_df=True)
        plt.plot([0,0],[0,df.event_index.max()],'r', zorder = 1)
        ax.scatter(df.time, df.event_index, c='k', marker= '|', s=1, zorder = 2)
        ax.set_xlim(pre_event, post_event)
        ax.set_ylabel('First spike time sorted trials')
        ax.tick_params(axis='both', which='major')
        ax.set_ylim(0, len(session_df_curr))
        ax.set_title('Aligned to Go Cue')

        # align to first spike time
        first_spike_times_abs = first_spike_times + session_df_curr['goCue_start_time'].values
        first_spike_times_abs_sorted = first_spike_times_abs[np.argsort(first_spike_times)]
        df = align.to_events(spike_times_curr, first_spike_times_abs_sorted, (-0.05, 0.03), return_df=True)
        plt.plot([0,0],[0,df.event_index.max()],'r', zorder = 1)
        ax = fig.add_subplot(gs[0, 3])
        ax.scatter(df.time, df.event_index, c='k', marker= '|', s=1, zorder = 2)
        ax.set_xlim(-0.05, 0.03)
        ax.set_ylabel('Sorted by First Spike Time to go cue')
        ax.tick_params(axis='both', which='major')
        ax.set_ylim(0, len(session_df_curr))
        ax.set_title('Aligned to First Spike Time')

        # isi distribution
        ax = fig.add_subplot(gs[1, 3:5])
        isi_spikes = np.log(np.diff(spike_times_curr))
        edges = np.linspace(np.nanmin(isi_spikes), np.nanmax(isi_spikes), 50)
        ax.hist(isi_spikes, bins=edges, color='k', alpha = 0.5, density=True)
        # set xlabel to log
        ax.axvline(np.log(0.002), color='k', linestyle='--')
        ax.set_xlabel('log(Inter-spike interval (s))')
        ax.set_ylabel('Density')
        ax.set_title('log(ISI) Distribution')

        # color by isi
        first_spike_times_abs = first_spike_times + session_df_curr['goCue_start_time'].values
        first_spike_times_abs_sorted = first_spike_times_abs[np.argsort(first_spike_times)]
        df = align.to_events(spike_times_curr, first_spike_times_abs_sorted, (-0.05, 0.03), return_df=True)

        isi_list = df.copy()
        isi_list['isi'] = np.nan
        # infer time interval of previous spike for each spike
        for ind, row in isi_list.iterrows():
            event_index = row['event_index']
            time = row['time']
            prev_spikes = spike_times_curr[spike_times_curr < (time + first_spike_times_abs_sorted[int(event_index)])]
            if len(prev_spikes) == 0:
                isi_list.at[ind, 'isi'] = np.nan
            else:
                isi_list.at[ind, 'isi'] = time + first_spike_times_abs_sorted[[int(event_index)]] - prev_spikes[-1]

        isi_color_code = np.log(isi_list['isi'].values)
        up_bound = np.percentile(isi_color_code[~np.isnan(isi_color_code)], 95)
        low_bound = np.percentile(isi_color_code[~np.isnan(isi_color_code)], 5)
        isi_color_code = (isi_color_code - low_bound) / (up_bound - low_bound)
        isi_color_code[isi_color_code>1] = 1
        isi_color_code[isi_color_code<0] = 0

        ax.hist(np.log(isi_list['isi'].values), bins=50, color='r', alpha=0.5, density=True)
        ax.axvline(low_bound, color='b', linestyle='--')
        ax.axvline(up_bound, color='r', linestyle='--')

        ax= fig.add_subplot(gs[0,4])
        sc = ax.scatter(df.time, df.event_index, c=isi_color_code, marker= '|', s=4, zorder = 2, cmap='Reds_r')
        ax.set_xlim(-0.02, 0.03)
        ax.set_ylabel('Sorted by First Spike Time to go cue')
        ax.tick_params(axis='both', which='major')
        ax.set_ylim(0, len(session_df_curr))
        ax.set_title('Aligned to First Spike Time')

        # add colorbar
        ax = fig.add_subplot(gs[1,5])
        cbar = plt.colorbar(sc, ax=ax, orientation='horizontal', fraction=0.046, pad=0.04)

        # align to go cue and sort by frist spike time
        align_time = session_df_curr['goCue_start_time'].values
        spike_df = align.to_events(spike_times_curr, align_time, (0, 100), return_df=True)
        # for each value in event_index, get the first spike time
        first_spike_times = spike_df.groupby('event_index')['time'].min().values
        align_time_firstspike_sort = align_time[np.argsort(first_spike_times)]
        df = align.to_events(spike_times_curr, align_time_firstspike_sort, (pre_event, post_event), return_df=True)
        isi_list = df.copy()
        isi_list['isi'] = np.nan
        # infer time interval of previous spike for each spike
        for ind, row in isi_list.iterrows():
            event_index = row['event_index']
            time = row['time']
            prev_spikes = spike_times_curr[spike_times_curr < (time + align_time_firstspike_sort[int(event_index)])]
            if len(prev_spikes) == 0:
                isi_list.at[ind, 'isi'] = np.nan
            else:
                isi_list.at[ind, 'isi'] = time + align_time_firstspike_sort[[int(event_index)]] - prev_spikes[-1]

        isi_color_code = np.log(isi_list['isi'].values)
        isi_color_code = (isi_color_code - low_bound) / (up_bound - low_bound)
        isi_color_code[isi_color_code>1] = 1
        isi_color_code[isi_color_code<0] = 0

        # plt.plot([0,0],[0,df.event_index.max()],'r', zorder = 1)
        ax = fig.add_subplot(gs[0, 5])
        ax.scatter(df.time, df.event_index, c=isi_color_code, marker= '|', s=3, zorder = 2, cmap = 'Reds_r')
        ax.set_xlim(pre_event, post_event)
        ax.set_ylabel('First spike time sorted trials')
        ax.tick_params(axis='both', which='major')
        ax.set_xlim(-0.02, 0.03)
        ax.set_title('Aligned to Go Cue')


        pre_event = -1.7
        post_event = 0.1
        align_time = session_df_curr['goCue_start_time'].values
        align_time_licklat_sort = align_time[np.argsort(lick_lat)]
        ax = fig.add_subplot(gs[1, 0])  
        df = align.to_events(spike_times_curr, align_time_licklat_sort, (pre_event, post_event), return_df=True)
        plt.plot([0,0],[0,df.event_index.max()],'r', zorder = 1)
        ax.scatter(df.time, df.event_index, c='k', marker= '|', s=3, zorder = 2)
        ax.set_xlim(pre_event, post_event)
        ax.set_ylabel('Lick latency sorted trials')
        ax.tick_params(axis='both', which='major')
        ax.set_ylim(0, len(session_df_curr))
        ax.axhline(len(align_time_licklat_sort)-np.sum(session_df_curr['animal_response'].values==2), color='blue', linestyle='--')
        ax.set_title('Aligned to Go Cue')


        align_time = session_df_curr['reward_outcome_time'].values
        lick_lat[session_df_curr['animal_response'].values==2] = np.nan
        align_time_licklat_sort = align_time[np.argsort(lick_lat)]
        ax = fig.add_subplot(gs[1, 1])  
        df = align.to_events(spike_times_curr, align_time_licklat_sort, (pre_event, post_event), return_df=True)
        plt.plot([0,0],[0,df.event_index.max()],'r', zorder = 1)
        ax.scatter(df.time, df.event_index, c='k', marker= '|', s=3, zorder = 2)
        ax.set_xlim(pre_event, post_event)
        ax.set_ylabel('Lick latency sorted trials')
        ax.tick_params(axis='both', which='major')
        ax.set_ylim(0, len(session_df_curr))
        ax.axhline(len(align_time_licklat_sort)-np.sum(session_df_curr['animal_response'].values==2), color='blue', linestyle='--')
        ax.set_title('Aligned to Choice')


        # align to go cue and sort by frist spike time
        align_time = session_df_curr['goCue_start_time'].values
        spike_df = align.to_events(spike_times_curr, align_time, (0, 100), return_df=True)
        # for each value in event_index, get the first spike time
        first_spike_times = spike_df.groupby('event_index')['time'].min().values
        align_time_firstspike_sort = align_time[np.argsort(first_spike_times)]
        ax = fig.add_subplot(gs[1, 2])
        df = align.to_events(spike_times_curr, align_time_firstspike_sort, (pre_event, post_event), return_df=True)
        plt.plot([0,0],[0,df.event_index.max()],'r', zorder = 1)
        ax.scatter(df.time, df.event_index, c='k', marker= '|', s=4, zorder = 2)
        ax.set_xlim(pre_event, post_event)
        ax.set_ylabel('First spike time sorted trials')
        ax.tick_params(axis='both', which='major')
        ax.set_ylim(0, len(session_df_curr))
        ax.axhline(len(align_time_firstspike_sort)-np.sum(session_df_curr['animal_response'].values==2), color='blue', linestyle='--')
        ax.set_title('Aligned to Go Cue') 

        plt.suptitle(f'Session {session}, Unit {unit_id}')
        plt.tight_layout()
        fig.savefig(os.path.join(save_path, f'opto_{session}{unit_id}_burst_selected.pdf'), dpi=300)
        plt.close(fig)

    print(f'{session} Combining PDFs...')
    combine_pdf_big(save_path, os.path.join(session_dir[f'ephys_fig_dir_{data_type}'], f'{session}_bursting.pdf'))
    print(f'{session} Done!')




def plot_alignments(session, data_type='curated', unit_ids=None, win_len = 0.5):
    bin_len = 0.01
    time_constant = 100
    time_window = [-1, 1.5]
    session_dir = session_dirs(session)
    unit_tbl = get_unit_tbl(session, data_type)
    model_name = 'stan_qLearning_5params'
    session_tbl_all = get_session_tbl(session)
    unit_tbl = get_unit_tbl(session, data_type=data_type)
    session_tbl = makeSessionDF(session, model_name = model_name)
    if unit_ids is None:
        unit_ids = unit_tbl[unit_tbl['opto_pass'] & unit_tbl['default_qc']]['unit_id'].tolist()
    lick_lat = session_tbl['reward_outcome_time'].values - session_tbl['goCue_start_time'].values
    lick_lat = lick_lat[session_tbl['animal_response']!=2]
    lick_lat_sort = np.argsort(lick_lat)
    outcomes = session_tbl['rewarded_historyL'] | session_tbl['rewarded_historyR']
    outcomes = outcomes[session_tbl['animal_response']!=2].values
    for unit_id in unit_ids:
        spike_times = unit_tbl[unit_tbl['unit_id']==unit_id]['spike_times'].values[0]
        session_tbl_curr = session_tbl.copy()
        session_tbl_all_curr = session_tbl_all.copy()
        spike_times_curr = spike_times.copy()
        unit_drift = load_drift(session, unit_id, data_type=data_type)
        if unit_drift is not None:
            if unit_drift['ephys_cut'][0] is not None:
                spike_times_curr = spike_times_curr[spike_times_curr >= unit_drift['ephys_cut'][0]]
                session_tbl_curr = session_tbl_curr[session_tbl_curr['go_cue_time'] >= unit_drift['ephys_cut'][0]]
                session_tbl_all_curr = session_tbl_all_curr[session_tbl_all_curr['goCue_start_time'] >= unit_drift['ephys_cut'][0]]
                # tblTrials_curr = tblTrials_curr[tblTrials_curr['goCue_start_time'] >= unit_drift['ephys_cut'][0]]
            if unit_drift['ephys_cut'][1] is not None:
                spike_times_curr = spike_times_curr[spike_times_curr <= unit_drift['ephys_cut'][1]]
                session_tbl_curr = session_tbl_curr[session_tbl_curr['go_cue_time'] <= unit_drift['ephys_cut'][1]]
                session_tbl_all_curr = session_tbl_all_curr[session_tbl_all_curr['goCue_start_time'] <= unit_drift['ephys_cut'][1]]
                # tblTrials_curr = tblTrials_curr[tblTrials_curr['goCue_start_time'] <= unit_drift['ephys_cut'][1]]
        align_time_go = session_tbl['goCue_start_time']
        align_time = session_tbl[session_tbl['animal_response']!=2]['reward_outcome_time']
        
        filtered_rate_go_cue, timestamps_go = get_spike_matrix_filter(spike_times, session_tbl['goCue_start_time'], time_window[0], time_window[1], time_constant=time_constant, stepSize=bin_len)
        filtered_response, timestamps_response = get_spike_matrix_filter(spike_times, session_tbl[session_tbl['animal_response']!=2]['reward_outcome_time'], time_window[0], time_window[1], time_constant=time_constant, stepSize=bin_len)

        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 5, height_ratios=[3, 1], wspace=0.35, hspace=0.2)
        gs_model = gridspec.GridSpec(3, 5, height_ratios=[1, 1, 1], wspace=0.3, hspace=0.3)
        colors = [[1, 1, 1], "red"]
        custom_cmap_heatmap = LinearSegmentedColormap.from_list("custom_heatmap", colors)
        colors = [[1, 0.8, 0.8], "red"]
        custom_cmap = LinearSegmentedColormap.from_list("custom_map", colors)

        ax = fig.add_subplot(gs[0, 0])
        im = ax.imshow(filtered_response[lick_lat_sort], extent=[time_window[0], time_window[1], 0, filtered_response.shape[0]], aspect='auto', origin='lower', cmap=custom_cmap_heatmap, vmin=0, vmax=filtered_response.max())
        plt.colorbar(im, label='Firing rate (Hz)', ax=ax)
        ax.set_xlabel('Time from choice (s)')
        numbins = 3

        fig, ax = plot_rate(
                            filtered_response,
                            timestamps_response, 
                            lick_lat,
                            # np.quantile(lick_lat, np.linspace(0, 0.95, numbins+1)),
                            np.linspace(np.min(lick_lat), np.quantile(lick_lat, 0.95), numbins+1),
                            range(numbins),
                            custom_cmap,
                            fig,
                            gs[1, 0],
                        )
        ax.set_xlabel('Time from choice (s)')

        ax = fig.add_subplot(gs[0, 1])

        im = ax.imshow(filtered_rate_go_cue[session_tbl['animal_response']!=2, :][lick_lat_sort], extent=[time_window[0], time_window[1], 0, filtered_rate_go_cue.shape[0]], aspect='auto', origin='lower', cmap=custom_cmap_heatmap, vmin=0, vmax=filtered_rate_go_cue.max())
        plt.colorbar(im, label='Firing rate (Hz)', ax=ax)
        numbins = 3
        fig, ax = plot_rate(
                            filtered_rate_go_cue[session_tbl['animal_response']!=2, :],
                            timestamps_go, 
                            lick_lat,
                            np.linspace(np.min(lick_lat), np.quantile(lick_lat, 0.95), numbins+1),
                            range(numbins),
                            custom_cmap,
                            fig,
                            gs[1, 1],
                        )
        ax.set_xlabel('Time from go cue (s)')

        ax = fig.add_subplot(gs[0, 2])
        outcomes_lick = 100*outcomes + lick_lat
        outcomes_lick_sort = np.argsort(outcomes_lick)
        im = ax.imshow(filtered_response[outcomes_lick_sort], extent=[time_window[0], time_window[1], 0, filtered_response.shape[0]], aspect='auto', origin='lower', cmap=custom_cmap_heatmap, vmin=0, vmax=filtered_response.max())
        plt.colorbar(im, label='Firing rate (Hz)', ax=ax)
        ax.axhline(np.sum(outcomes==0), color='black', linestyle='--', linewidth=1)
        ax.set_xlabel('Time from choice (s)')

        fig, ax = plot_rate(
                            filtered_response,
                            timestamps_response, 
                            outcomes,
                            np.array([-1, 0.5, 1.5]),
                            ['no rwd', 'rwd'],
                            custom_cmap,
                            fig,
                            gs[1, 2],
                        )
        ax.set_xlabel('Time from choice (s)')

        ax = fig.add_subplot(gs[0, 3])
        im = ax.imshow(filtered_rate_go_cue[session_tbl['animal_response']!=2, :][outcomes_lick_sort], extent=[time_window[0], time_window[1], 0, filtered_response.shape[0]], aspect='auto', origin='lower', cmap=custom_cmap_heatmap, vmin=0, vmax=filtered_rate_go_cue.max())
        plt.colorbar(im, label='Firing rate (Hz)', ax=ax)
        ax.axhline(np.sum(outcomes==0), color='black', linestyle='--', linewidth=1)
        ax.set_xlabel('Time from go cue (s)')

        fig, ax = plot_rate(
                            filtered_rate_go_cue[session_tbl['animal_response']!=2, :],
                            timestamps_go, 
                            outcomes,
                            np.array([-1, 0.5, 1.5]),
                            ['no rwd', 'rwd'],
                            custom_cmap,
                            fig,
                            gs[1, 3],
                        )

        # regresssion model
        outcome_time = session_tbl_curr[session_tbl_curr['animal_response']!=2]['reward_outcome_time'].values
        rewarded_ind = session_tbl_curr[session_tbl_curr['animal_response']!=2]['rewarded_historyL'].values | session_tbl_curr[session_tbl_curr['animal_response']!=2]['rewarded_historyR'].values
        # rewarded_ind = np.full(len(outcome_time), True, dtype=bool)
        reward_time = outcome_time[rewarded_ind] + 0.2
        if len(reward_time) > 2:
            # acf
            # compute how past activity contribute to future activity, 
            bin_len = 0.05
            lag_length = 2
            bin_num = int(np.ceil(lag_length / bin_len))

            session_start = session_tbl_curr['goCue_start_time'].values[0]-10
            session_end = session_tbl_curr['goCue_start_time'].values[-1]+20

            counts = np.histogram(spike_times_curr, bins=np.arange(session_start, session_end, bin_len))[0]
            starts = np.arange(session_start, session_end, bin_len)[:-1]
            ends = np.arange(session_start, session_end, bin_len)[1:]

            pre_time = 0
            post_time = 2
            # remove periods within session
            counts_bl = counts.copy().astype(float)
            if len(session_tbl_all_curr) > 0:
                for ind, row in session_tbl_all_curr.iterrows():
                    start_time = row['goCue_start_time'] - pre_time
                    end_time = row['goCue_start_time'] + post_time
                    # set counts in this period to np.nan
                    mask = (ends >= start_time) & (starts <= end_time)
                    if np.sum(mask) > 0:
                        counts_bl[mask] = np.nan

            # compute the lagged activity
            lagged_matrix = np.zeros((len(counts_bl), bin_num))
            for i in range(bin_num):
                lagged_matrix[:, i] = np.roll(counts_bl, i + 1)
                lagged_matrix[:i + 1, i] = np.nan  # set the first i+1 elements to np.nan

            lagged_matrix = sm.add_constant(data=lagged_matrix)

            model_bl = sm.OLS(counts_bl, lagged_matrix, missing='drop').fit()
            ci = model_bl.conf_int(alpha=0.05)  # 95% CI
            yerr = np.vstack([
                model_bl.params[1:] - ci[1:, 0],  # lower error
                ci[1:, 1] - model_bl.params[1:]   # upper error
            ])
            
            spikes_df = align.to_events(spike_times_curr, reward_time, (0, win_len), return_df=True)
            spike_counts = spikes_df.groupby('event_index').size()
            spike_counts = np.array([spike_counts[i] if i in spike_counts.index else 0 for i in range(len(reward_time))])
            spike_matrix, timestamps = get_spike_matrix(spike_times_curr, reward_time, -lag_length, win_len+bin_len, bin_len, bin_len)
            predicted_counts = np.full((spike_matrix.shape[0], spike_matrix.shape[1]-bin_num+1), np.nan)
            predicted_times = timestamps[bin_num-1:]  # Adjusted to match the predicted counts
            pre_cue_counts = np.sum(predicted_times < 0)  # Count how many predicted times are before the cue
            for i in range(spike_matrix.shape[1]-bin_num+1):
                if i <= pre_cue_counts:
                    X = sm.add_constant(spike_matrix[:, i:i+bin_num].copy())
                    predicted_counts[:, i] = model_bl.predict(X)
                else:
                    mix_spikes = np.concatenate((spike_matrix[:, i:np.sum(timestamps<0)].copy(), predicted_counts[:, :i-1]), axis=1)
                    X = sm.add_constant(mix_spikes)
                    predicted_counts[:, i] = model_bl.predict(X)
            predicted_inds = (predicted_times>=0.5* bin_len) & (predicted_times<=win_len-0.5* bin_len)
            predicted_sum_win = predicted_counts[:, predicted_inds].sum(axis=1)
            spike_counts_residual = spike_counts - predicted_sum_win

            # compare 2 models
            # fit regression, use Qchosen in session_df_curr to predict residuals vs spike counts
            if 'Qchosen' in session_tbl_curr.columns.to_list():
                X = session_tbl_curr[['Qchosen']].values[session_tbl_curr['animal_response']!=2].reshape(-1, 1)  # reshape for single feature
                X = X[rewarded_ind]
                X = sm.add_constant(X)  # add intercept term
                model_res = sm.OLS(spike_counts_residual, X).fit()  # fit model to residuals
                ci_res = model_res.conf_int(alpha=0.05)  # 90% CI for residuals model
                model_whole = sm.OLS(spike_counts, X).fit()  # fit model to spike counts
                ci_whole = model_whole.conf_int(alpha=0.05)  # 90% CI for whole model

            # plot model results
            ax = fig.add_subplot(gs_model[0, 4])
            ci = model_bl.conf_int(alpha=0.05)  # 95% CI
            yerr = np.vstack([
                model_bl.params[1:] - ci[1:, 0],  # lower error
                ci[1:, 1] - model_bl.params[1:]   # upper error
            ])
            ax.errorbar(bin_len*np.arange(1, bin_num+1), model_bl.params[1:], yerr=yerr, fmt='-o', label='Model Coefficients with 90% CI',
                        color='blue', capsize=5)
            ax.axhline(0, color='black', linestyle='--', linewidth=1)
            ax.set_xlabel('Lag (s)')
            if 'Qchosen' in session_tbl_curr.columns.to_list():
                ax = fig.add_subplot(gs_model[1, 4])
                yerr = np.vstack([
                    model_res.params[1:] - ci_res[1:, 0],  # lower error
                    ci_res[1:, 1] - model_res.params[1:]   # upper error
                ])
            
                ax.errorbar(range(1, 2), model_res.params[1:], yerr=yerr, fmt='o', label='Model Coefficients with 90% CI',
                            color='blue', capsize=5)
                yerr_whole = np.vstack([
                    model_whole.params[1:] - ci_whole[1:, 0],  # lower error
                    ci_whole[1:, 1] - model_whole.params[1:]   # upper error
                ])
                ax.errorbar(range(1), model_whole.params[1:], yerr=yerr_whole, fmt='o', label='Whole Model Coefficients with 90% CI',
                            color='red', capsize=5)
                ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
                ax.set_xlabel("Whole Outcome Qchosen ---- Res Outcome Qchosen")
                ax.set_ylabel("Coefficient value")
                ax.set_title("Linear Regression Coefficients with 95% CI")

                ax = fig.add_subplot(gs_model[2, 4])
                t_res = model_res.tvalues[1:]       
                t_whole = model_whole.tvalues[1:]

                # Plot bars for each model
                ax.bar(range(1, 2), t_res, 0.3, label='Res Model', color='blue')
                ax.bar(range(1), t_whole, 0.3, label='Whole Model', color='red')

                # Reference line at 0
                ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)

                # Labels
                ax.set_xlabel("Whole Outcome Qchosen ---- Res Outcome Qchosen")
                ax.set_ylabel("t-statistic")
                ax.set_title("Linear Regression t-Statistics")
                ax.legend()

        fig.tight_layout()
        fig.suptitle(f'Session: {session}, Unit: {unit_id}', fontsize=16)
        target_folder = os.path.join(session_dir[f'ephys_fig_dir_{data_type}'], 'go_cue_vs_response')
        os.makedirs(target_folder, exist_ok=True)
        fig.savefig(os.path.join(target_folder, f'{unit_id}_alignments.pdf'))

    # if len(unit_ids) > 0:
    #     combine_pdf_big(target_folder, os.path.join(session_dir[f'ephys_fig_dir_{data_type}'], f'alignments_compare_combined.pdf'))
    

if __name__ == '__main__':

    # Set UNIT_BEH_MASTER_TABLE=/path/to/master.csv to override. If unset,
    # the resolver first tries the fixed raw CCF master table paths, then falls
    # back to the newest master_all_units_opto_ccf*.pkl/csv in the master table dir.
    MASTER_TABLE_PATH = os.environ.get('UNIT_BEH_MASTER_TABLE') or None

    # By default this backfills the same response-aligned unit_beh plots used in
    # the previous manual calls. Add 'go_cue' here if you also want go-cue plots.
    ALIGN_NAMES = ('go_cue','response')

    # None means use the data_type column in the master table, usually 'raw' for
    # master_all_units_opto_ccf_raw_fix. To restrict, use e.g. ('raw',) or ('curated',).
    DATA_TYPES_TO_RUN = ('raw',)

    # Limit the backfill to selected mice. None means all mice. You can either
    # hard-code a tuple here, e.g. ('835444', '835451'), or set the environment
    # variable UNIT_BEH_MOUSE_IDS='835444,835451' when launching the script.
    MOUSE_IDS_TO_RUN = None
    ENV_MOUSE_IDS_TO_RUN = _parse_mouse_ids(os.environ.get('UNIT_BEH_MOUSE_IDS'))
    if ENV_MOUSE_IDS_TO_RUN is not None:
        MOUSE_IDS_TO_RUN = ENV_MOUSE_IDS_TO_RUN

    MODEL_NAME = None
    FORMULA = 'spikes ~ 1 + outcome + choice'
    PRE_EVENT_BY_ALIGN = {
        'response': -1,
        'go_cue': -1.5,
    }
    POST_EVENT = 3
    BIN_SIZE = 0.2
    STEP_SIZE = 0.05

    # A unit is treated as already complete only if both per-unit outputs exist.
    # Change to ('.pdf',) if you only care about the PDF.
    REQUIRED_PLOT_EXTS = ('.pdf',)

    run_missing_well_opto_unit_beh_plots(
        master_path=MASTER_TABLE_PATH,
        align_names=ALIGN_NAMES,
        data_types_to_run=DATA_TYPES_TO_RUN,
        mouse_ids_to_run=MOUSE_IDS_TO_RUN,
        model_name=MODEL_NAME,
        formula=FORMULA,
        pre_event_by_align=PRE_EVENT_BY_ALIGN,
        post_event=POST_EVENT,
        binSize=BIN_SIZE,
        stepSize=STEP_SIZE,
        required_plot_exts=REQUIRED_PLOT_EXTS,
    )

"""
Enhanced NWB builder that merges custom and kilosort unit tables.

This module provides a function to build an NWB file with units from both:
1. Custom pickle files (opto-tagging, CCF coordinates, etc.)
2. Kilosort NWB files (raw ephys metrics)

The columns are merged using the mappings defined in column_names_map.json.
"""
import functools
import glob
import json
import logging
import os
import tempfile
import pandas as pd
import numpy as np
from datetime import datetime
from uuid import uuid4
from dateutil.tz import tzlocal
from hdmf.spec import DatasetSpec, GroupSpec, NamespaceBuilder
from pynwb import NWBFile, TimeSeries, get_class, load_namespaces
from hdmf_zarr import NWBZarrIO
from pynwb.file import Subject

import sys
sys.path.insert(0, '/root/capsule/code/beh_ephys_analysis')
from aind_dynamic_foraging_data_utils.nwb_utils import load_nwb_from_filename
from utils.beh_functions import get_session_tbl, get_unit_tbl, session_dirs, parseSessionID
from utils.pupil_utils import load_pupil
from pathlib import Path
from hdmf.common import DynamicTable, VectorData
from aind_dynamic_foraging_behavior_video_analysis.ephys.tongue_ephys import load_intermediate_data

logger = logging.getLogger(__name__)

# Tongue movement data
TONGUE_MOVEMENT_DATA_DIR = Path('/root/capsule/data/all_tongue_movements_04022026')
TONGUE_MOVEMENT_PARQUET = TONGUE_MOVEMENT_DATA_DIR / 'all_tongue_movements_04022026.parquet'
KEYPOINT_TRACKING_DIR = Path('/root/capsule/data/keypoint_tracking_bottomview_LCrecordings_20260403')

# AIND metadata extension: the raw metadata JSON files are bundled as a single JSON
# blob in a LabMetaData container. Placeholder for now — expected to be replaced by
# properly typed metadata later.
AIND_NAMESPACE = 'aind_beh_ephys'
AIND_NAMESPACE_VERSION = '0.1.0'
AIND_NEURODATA_TYPE = 'AindMetadata'
AIND_LAB_META_DATA_KEY = 'aind_metadata'

# Load column mappings and descriptions
COLUMN_MAP_PATH = '/root/capsule/code/data_management/column_names_map.json'
COLUMN_DESC_PATH = '/root/capsule/code/data_management/column_names_description.json'

# Fiber photometry: per-subject surgery records (region -> implant hemisphere), used to
# label photometry acquisitions by brain region instead of channel index.
FP_METADATA_DIR = '/root/capsule/code/data_management/FP_metadata'
PHOTOMETRY_SIGNALS = ('G', 'Iso', 'G-Iso')

with open(COLUMN_MAP_PATH, 'r') as f:
    COLUMN_MAP = json.load(f)

with open(COLUMN_DESC_PATH, 'r') as f:
    COLUMN_DESCRIPTIONS = json.load(f)

# Known array columns (must be arrays, not scalars, even if all values are null)
KNOWN_ARRAY_COLUMNS = {
    'waveform_mean', 'waveform_sd',  # 2D waveform arrays
    'peak_of_optimized_waveform', 'peak_of_aligned_optimized_waveform',  # 1D arrays
    '2D_matrix_of_optimized_waveform', '2D_matrix_of_raw_waveform',  # 2D arrays
    '2D_matrix_of_aligned_raw_waveform', '2D_matrix_of_fake_raw_waveform',
    '2D_matrix_of_aligned_fake_raw_waveform',
    'waveform_on_peak_channel_of_raw_waveform', 'waveform_on_peak_channel_of_aligned_raw_waveform',
    'peak_waveform_fake_raw', 'peak_waveform_aligned_fake_raw',
}

@functools.cache
def aind_metadata_type():
    """
    Return the AindMetadata container class, registering its namespace on first call.

    In-code NWB extension, following the pattern in
    aind_behavior_vr_foraging_packaging.provenance:
      * build a NamespaceBuilder + GroupSpec at runtime
      * export the YAMLs to a temp directory (registration only, not persistent)
      * load_namespaces() reads them back into pynwb
      * get_class() returns the auto-generated container class

    Cached so the namespace is only registered once per Python process.
    """
    spec = GroupSpec(
        doc='AIND raw metadata files bundled as one JSON blob keyed by filename stem.',
        data_type_def=AIND_NEURODATA_TYPE,
        data_type_inc='LabMetaData',
        datasets=[
            DatasetSpec(name='json_data', doc='JSON metadata (all files, keyed by stem)', dtype='text'),
        ],
    )

    builder = NamespaceBuilder(
        doc=f'In-code extension for {AIND_NAMESPACE}',
        name=AIND_NAMESPACE,
        version=AIND_NAMESPACE_VERSION,
    )
    builder.include_namespace('core')
    builder.add_spec(f'{AIND_NAMESPACE}.extensions.yaml', spec)

    outdir = Path(tempfile.mkdtemp(prefix='aind-beh-ephys-spec-'))
    namespace_name = f'{AIND_NAMESPACE}.namespace.yaml'
    builder.export(namespace_name, outdir=str(outdir))
    load_namespaces(str(outdir / namespace_name))

    return get_class(AIND_NEURODATA_TYPE, AIND_NAMESPACE)


def load_aind_metadata(session_id):
    """
    Load the raw AIND metadata JSON files for a session.

    Reads every *.json in the session's raw data directory into one dict keyed by
    filename stem (e.g. {'subject': {...}, 'procedures': {...}, 'rig': {...}}).

    Args:
        session_id: Session identifier

    Returns:
        dict keyed by filename stem, or None if no metadata files were found.
    """
    raw_dir = session_dirs(session_id).get('raw_dir')
    if raw_dir is None or not os.path.exists(raw_dir):
        logger.warning(f"Raw data directory not found for {session_id}")
        return None

    meta_files = sorted(glob.glob(os.path.join(raw_dir, '*.json')))
    if not meta_files:
        logger.info(f"No metadata JSON files found in {raw_dir}")
        return None

    meta_dict = {}
    for path in meta_files:
        key = os.path.splitext(os.path.basename(path))[0]
        try:
            with open(path, 'r') as f:
                meta_dict[key] = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Could not read metadata file {path}: {e}")

    if not meta_dict:
        return None

    logger.info(f"Loaded {len(meta_dict)} metadata files: {', '.join(meta_dict)}")
    return meta_dict


def add_aind_metadata(nwb_file, meta_dict):
    """Attach the combined JSON metadata to nwb_file under lab_meta_data[AIND_LAB_META_DATA_KEY]."""
    nwb_file.add_lab_meta_data(
        aind_metadata_type()(
            name=AIND_LAB_META_DATA_KEY,
            json_data=json.dumps(meta_dict),
        )
    )
    return nwb_file


def photometry_channel_labels(session_id):
    """
    Build the channel index -> region label map for a session's photometry channels.

    Acquisition names in the behavior NWB carry the fiber's channel index
    ('G_1_tri-exp_mc'), which says nothing about where the fiber sat. Two metadata
    sources are combined into a 'REGION-HEMISPHERE' label (e.g. 'TH-R'):
      * index -> region: the per-session <raw_dir>/fib/*.json, same source
        photometry_utils.get_FP_data uses ('mPFC' is renamed 'PL' as it does).
        When that file is missing, fall back to the key order of the subject's
        FP_metadata file, whose keys are listed in channel order.
      * region -> hemisphere: FP_metadata/<subject>.json, the surgery record.

    Args:
        session_id: Session identifier

    Returns:
        dict of channel index (str) -> label (str). Empty when no mapping is available;
        the hemisphere is omitted from the label if the surgery record has no entry.
    """
    session_dir = session_dirs(session_id)

    hemispheres = {}
    fp_meta_path = os.path.join(FP_METADATA_DIR, f"{session_dir['aniID']}.json")
    if os.path.exists(fp_meta_path):
        with open(fp_meta_path, 'r') as f:
            hemispheres = json.load(f)
    else:
        logger.info(f"No FP surgery metadata at {fp_meta_path}")

    regions = {}
    fib_dir = os.path.join(session_dir['raw_dir'], 'fib')
    fib_jsons = sorted(glob.glob(os.path.join(fib_dir, '*.json'))) if os.path.isdir(fib_dir) else []
    if fib_jsons:
        if len(fib_jsons) > 1:
            logger.warning(f"Multiple fib metadata files in {fib_dir}, using {os.path.basename(fib_jsons[0])}")
        with open(fib_jsons[0], 'r') as f:
            regions = {str(k): v for k, v in json.load(f).items()}
        # get_FP_data renames mPFC to PL; keep the same region vocabulary
        regions = {k: ('PL' if v == 'mPFC' else v) for k, v in regions.items()}
    elif hemispheres:
        # No per-session fiber map: the surgery record lists regions in channel order
        regions = {str(i): region for i, region in enumerate(hemispheres)}
        logger.info(f"No fib metadata for {session_id}, using FP_metadata key order: {regions}")

    labels = {}
    for index, region in regions.items():
        hemisphere = hemispheres.get(region)
        labels[index] = f'{region}-{hemisphere}' if hemisphere else region
        if hemisphere is None:
            logger.warning(f"No hemisphere for region {region} in {fp_meta_path}, labeling channel {index} as {region}")
    return labels


def rename_photometry_acquisition(acq_name, channel_labels):
    """
    Replace the channel index in a photometry acquisition name with its region label.

    Photometry names are '<signal>_<channel index>[_<processing method>]', where signal
    is G, Iso or G-Iso, e.g. 'G_1_tri-exp_mc' -> 'G_TH-R_tri-exp_mc'. Names that are not
    photometry channels, or whose index has no label, are returned unchanged.

    Args:
        acq_name: acquisition name from the behavior NWB
        channel_labels: dict of channel index (str) -> region label, from
                        photometry_channel_labels()

    Returns:
        (new_name, region_label). region_label is None when nothing was renamed.
    """
    parts = acq_name.split('_')
    if len(parts) < 2 or parts[0] not in PHOTOMETRY_SIGNALS:
        return acq_name, None

    label = channel_labels.get(parts[1])
    if label is None:
        return acq_name, None

    return '_'.join([parts[0], label] + parts[2:]), label


def pupil_data_to_timeseries(pupil_data):
    """
    Convert a pupil data dict to a pynwb TimeSeries.

    Args:
        pupil_data: dict with keys 'pupil_times' (1D array, seconds) and
                    'pupil_diameter' (1D array, pixels)

    Returns:
        pynwb.TimeSeries with name 'pupil_diameter'
    """
    return TimeSeries(
        name='pupil_diameter',
        data=np.array(pupil_data['pupil_diameter'], dtype=np.float64),
        timestamps=np.array(pupil_data['pupil_times'], dtype=np.float64),
        unit='pixels',
        description='Pupil diameter measured from DLC tracking, aligned to session time.',
    )


def load_tongue_movements(session_id):
    """
    Load tongue movements for a session from the pooled parquet data asset.

    Matches session_id to the video session by animal ID and closest datetime,
    then returns all tongue movements for that session as a pynwb DynamicTable.
    The has_lick column flags which movements contain a lick contact.

    Args:
        session_id: session identifier string, e.g. 'behavior_791691_2025-06-27_13-54-30'

    Returns:
        hdmf DynamicTable named 'tongue_movements' with one row per tongue movement,
        or None if no match found. Same table name as the movs table built by
        load_keypoint_tracking, since the two are interchangeable sources.
    """
    if not TONGUE_MOVEMENT_PARQUET.exists():
        logger.warning(f"Tongue movement parquet not found at {TONGUE_MOVEMENT_PARQUET}")
        return None

    all_movements_df = pd.read_parquet(TONGUE_MOVEMENT_PARQUET)
    session_video_list = all_movements_df['session'].unique().tolist()

    animal_id, session_time, _ = parseSessionID(session_id)
    if animal_id is None:
        logger.warning(f"Could not parse session_id: {session_id}")
        return None

    candidate_sessions = [s for s in session_video_list if str(s).startswith(f'behavior_{animal_id}')]
    if not candidate_sessions:
        logger.info(f"No tongue movement data found for animal {animal_id}")
        return None

    time_diffs = [abs((parseSessionID(s)[1] - session_time).total_seconds()) for s in candidate_sessions]
    best_idx = int(np.argmin(time_diffs))
    if time_diffs[best_idx] > 60:
        logger.info(f"Closest tongue movement session is {time_diffs[best_idx]:.0f}s away — skipping")
        return None

    matched_session = candidate_sessions[best_idx]
    logger.info(f"Matched tongue movement session: {matched_session}")

    movements = all_movements_df[all_movements_df['session'] == matched_session].copy().reset_index(drop=True)
    if len(movements) == 0:
        logger.info("No tongue movement rows found for matched session")
        return None

    # Columns to include in the DynamicTable (drop session identifier)
    exclude_cols = {'session'}
    col_descriptions = COLUMN_DESCRIPTIONS.get('tongue_movement_columns', {})

    keep_cols = [c for c in movements.columns if c not in exclude_cols]

    table = DynamicTable(
        id=np.array(range(len(movements))),
        name='tongue_movements',
        description=('Tongue movements detected from video DLC tracking, one row per movement. '
                     'has_lick flags movements containing a lick contact.'),
    )

    for col in keep_cols:
        # Normalise nullable integer / boolean dtypes, string columns with
        # missing values, and ragged array columns
        data, index = _prepare_column_data(movements[col])
        table.add_column(
            name=col,
            description=_lookup_description(col_descriptions, col),
            data=data,
            index=index,
        )

    logger.info(f"Built tongue_movements DynamicTable with {len(movements)} rows "
                f"and {len(keep_cols)} columns")
    return table


def _prepare_column_data(series):
    """
    Coerce a DataFrame column into data hdmf/zarr can write.

    Handles the same two problem cases as the trials and units tables:
      - array-valued cells with varying lengths (ragged), which need a VectorIndex
      - mixed string / missing columns, where zarr infers the dataset dtype from
        the first element and then fails on later values
        (e.g. ValueError: could not convert string to float: 'right_lick_time')

    Args:
        series: pandas Series (one table column)

    Returns:
        Tuple (data, index) where data is a list of per-row values and index is
        True when the column must be written as a ragged/indexed column.
    """
    non_null = series.dropna()

    # Array-valued columns: index=True when lengths vary, same as trials/units
    if len(non_null) > 0 and isinstance(non_null.iloc[0], (list, np.ndarray)):
        sample_val = np.asarray(non_null.iloc[0])
        is_ragged = sample_val.ndim == 1 and len({len(v) for v in non_null}) > 1
        data = []
        for val in series:
            if isinstance(val, (list, np.ndarray)):
                data.append(np.asarray(val))
            elif is_ragged:
                data.append(np.array([]))
            else:
                # Keep a rectangular column rectangular: NaN-fill missing rows
                dtype = sample_val.dtype if sample_val.dtype.kind in 'fc' else np.float64
                data.append(np.full(sample_val.shape, np.nan, dtype=dtype))
        return data, is_ragged

    arr = series.to_numpy(dtype=object, na_value=np.nan)

    # Numeric (incl. bool -> 1.0/0.0) columns, keeping NaN for missing values
    try:
        return arr.astype(np.float64).tolist(), False
    except (ValueError, TypeError):
        pass

    # Anything else (strings, mixed types): write as strings, '' for missing
    return ['' if (val is None or (isinstance(val, float) and np.isnan(val))) else str(val)
            for val in arr], False


def _lookup_description(col_descriptions, col):
    """Look up a column description, falling back to the column name when unfilled."""
    description = col_descriptions.get(col, col)
    return col if description == 'to be filled' else description


def _df_to_dynamic_table(df, name, description, col_descriptions=None):
    """Convert a DataFrame to an hdmf DynamicTable, coercing nullable/ragged columns."""
    col_descriptions = col_descriptions or {}
    table = DynamicTable(id=np.array(range(len(df))), name=name, description=description)
    for col in df.columns:
        data, index = _prepare_column_data(df[col])
        table.add_column(name=col, description=_lookup_description(col_descriptions, col),
                         data=data, index=index)
    return table


def load_keypoint_tracking(session_id):
    """
    Load tongue keypoint tracking data for a session from the bottomview DLC asset.

    Matches session_id to the nearest session directory under KEYPOINT_TRACKING_DIR
    by animal ID and datetime, then calls load_intermediate_data and returns
    DynamicTables for the movement summary (movs) and per-frame kinematics (kins).

    Args:
        session_id: session identifier string

    Returns:
        Tuple (movs_table, kins_table), or (None, None) if no match found.
    """
    if not KEYPOINT_TRACKING_DIR.exists():
        logger.warning(f"Keypoint tracking directory not found: {KEYPOINT_TRACKING_DIR}")
        return None, None

    animal_id, session_time, _ = parseSessionID(session_id)
    if animal_id is None:
        logger.warning(f"Could not parse session_id: {session_id}")
        return None, None

    candidate_dirs = [d for d in KEYPOINT_TRACKING_DIR.iterdir()
                      if d.is_dir() and d.name.startswith(f'behavior_{animal_id}')]
    if not candidate_dirs:
        logger.info(f"No keypoint tracking data found for animal {animal_id}")
        return None, None

    time_diffs = [abs((parseSessionID(d.name)[1] - session_time).total_seconds()) for d in candidate_dirs]
    best_idx = int(np.argmin(time_diffs))
    if time_diffs[best_idx] > 60:
        logger.info(f"Closest keypoint session is {time_diffs[best_idx]:.0f}s away — skipping")
        return None, None

    matched_dir = candidate_dirs[best_idx]
    logger.info(f"Matched keypoint tracking session: {matched_dir.name}")

    try:
        data = load_intermediate_data(matched_dir)
    except Exception as e:
        logger.warning(f"Failed to load intermediate data from {matched_dir}: {e}")
        return None, None

    movs_table = _df_to_dynamic_table(
        data['movs'],
        name='tongue_movements',
        description='Tongue movement summary table from DLC keypoint tracking (one row per movement).',
        col_descriptions=COLUMN_DESCRIPTIONS.get('tongue_movement_columns', {}),
    )
    kins_table = _df_to_dynamic_table(
        data['kins'],
        name='tongue_kinematics',
        description='Per-frame tongue kinematics from DLC keypoint tracking (x, y, velocity, confidence).',
        col_descriptions=COLUMN_DESCRIPTIONS.get('tongue_kinematics_columns', {}),
    )

    logger.info(f"Built tongue_movements ({len(data['movs'])} rows) and tongue_kinematics ({len(data['kins'])} rows) tables")
    return movs_table, kins_table


def merge_unit_tables(session_id, data_type='curated', return_nwb=False):
    """
    Merge unit tables from custom pickle and NWB kilosort data.

    Args:
        session_id: Session identifier
        data_type: 'curated' or 'raw'
        return_nwb: If True, return (merged_df, ephys_nwb). If False, return just merged_df

    Returns:
        If return_nwb=False: Merged DataFrame with mapped column names, or None if merge fails
        If return_nwb=True: Tuple of (merged_df, ephys_nwb) or (None, None) if merge fails
    """
    # 1. Load custom unit table (use summary version)
    custom_unit_tbl = get_unit_tbl(session_id, data_type=data_type, summary=True)
    if custom_unit_tbl is None or len(custom_unit_tbl) == 0:
        logger.warning(f"No custom unit table found for {session_id}")
        return (None, None) if return_nwb else None

    logger.info(f"Loaded {len(custom_unit_tbl)} units from custom table")

    # 2. Load NWB kilosort data
    session_dir = session_dirs(session_id)
    nwb_path = session_dir.get(f'nwb_dir_{data_type}')
    if nwb_path is None or not os.path.exists(nwb_path):
        logger.warning(f"NWB file not found at {nwb_path}")
        return (None, None) if return_nwb else None

    ephys_nwb = load_nwb_from_filename(nwb_path)
    if ephys_nwb.units is None:
        logger.warning(f"No units in NWB file for {session_id}")
        return (None, None) if return_nwb else None

    nwb_unit_tbl = ephys_nwb.units.to_dataframe()
    logger.info(f"Loaded {len(nwb_unit_tbl)} units from NWB")

    # 3. Verify and align by unit_id / ks_unit_id
    custom_unit_ids = set(custom_unit_tbl['unit_id'].values)

    # Determine which ID column the NWB file uses
    if 'ks_unit_id' in nwb_unit_tbl.columns:
        nwb_id_col = 'ks_unit_id'
    elif 'unit_id' in nwb_unit_tbl.columns:
        nwb_id_col = 'unit_id'
    else:
        logger.error(f"NWB units table has neither 'ks_unit_id' nor 'unit_id'. Columns: {list(nwb_unit_tbl.columns)}")
        return None

    logger.info(f"Using NWB ID column: '{nwb_id_col}' for alignment")
    nwb_unit_ids = set(nwb_unit_tbl[nwb_id_col].values)
    common_ids = custom_unit_ids & nwb_unit_ids

    if len(common_ids) == 0:
        logger.error(f"No common units found between custom and NWB tables!")
        logger.error(f"  Custom unit_ids ({len(custom_unit_ids)}): {sorted(list(custom_unit_ids))[:10]}")
        logger.error(f"  NWB {nwb_id_col} ({len(nwb_unit_ids)}): {sorted(list(nwb_unit_ids))[:10]}")
        return None

    if len(custom_unit_tbl) != len(common_ids):
        only_custom = custom_unit_ids - nwb_unit_ids
        logger.warning(
            f"Row count mismatch: {len(custom_unit_tbl)} custom units "
            f"but only {len(common_ids)} found in NWB. "
            f"Missing from NWB: {sorted(list(only_custom))[:10]}"
        )

    # Align tables
    custom_aligned = custom_unit_tbl[custom_unit_tbl['unit_id'].isin(common_ids)].sort_values('unit_id').reset_index(drop=True)
    nwb_aligned = nwb_unit_tbl[nwb_unit_tbl[nwb_id_col].isin(common_ids)].sort_values(nwb_id_col).reset_index(drop=True)

    logger.info(f"Aligned {len(custom_aligned)} common units")

    # 4. Apply column mappings and merge
    merged_df = pd.DataFrame()
    unit_columns_custom = COLUMN_MAP['unit_columns_custom']
    unit_columns_ks = COLUMN_MAP['unit_columns_ks']

    # Add custom columns with mapped names
    for orig_col, mapped_name in unit_columns_custom.items():
        if orig_col not in custom_aligned.columns:
            continue

        # Skip duplicates (will be taken from NWB)
        if 'duplicate as' in mapped_name:
            continue

        # Handle similar columns - use descriptive name
        if 'similar to' in mapped_name:
            clean_name = mapped_name.split(';')[0].strip()
            merged_df[clean_name] = custom_aligned[orig_col].values
        else:
            merged_df[mapped_name] = custom_aligned[orig_col].values

    # Add NWB columns with mapped names
    for orig_col, mapped_name in unit_columns_ks.items():
        if orig_col not in nwb_aligned.columns:
            continue

        # Skip if already exists
        if mapped_name in merged_df.columns:
            continue

        merged_df[mapped_name] = nwb_aligned[orig_col].values

    logger.info(f"Merged table has {len(merged_df)} rows and {len(merged_df.columns)} columns")

    if return_nwb:
        return merged_df, ephys_nwb
    else:
        return merged_df


def build_combined_nwb(session_id, data_type='curated', save_file=None, add_metadata=False):
    """
    Build a complete NWB file with available data modalities.

    Combines whichever data is available:
    - Behavior trials (from session table)
    - Ephys units (merged custom + kilosort)
    - Acquisition TimeSeries (lick times, reward times, etc.)

    Args:
        session_id: Session identifier
        data_type: 'curated' or 'raw'
        save_file: Path to save NWB file (if None, returns in-memory only)
        add_metadata: If True, bundle the raw AIND metadata JSON files into a
            LabMetaData container (see add_aind_metadata). Placeholder metadata,
            expected to be replaced by properly typed metadata later.

    Returns:
        Tuple of (save_path, nwb_object, data_modalities_dict)
        data_modalities_dict has keys:
            'behavior_trials': bool - whether trial data is included
            'ephys_units': bool - whether ephys units are included
            'lick_times': bool - whether lick acquisition is included
            'reward_times': bool - whether reward acquisition is included
            'FP': bool - whether fiber photometry is included
            'pupil': bool - whether pupil diameter is included
            'tongue_movements': bool - whether the tongue_movements table is included
            'keypoint_tracking': bool - whether the tongue_kinematics table is included
            'aind_metadata': bool - whether the AIND metadata blob is included
            'beh_version': str - 'raw', 'processed', or 'none'
            'nwb_created': str - ISO timestamp when NWB object was created
            'nwb_saved': str or None - ISO timestamp when NWB was saved to file (None if not saved)
    """
    logger.info(f"Building combined NWB for {session_id}")

    # Track which data modalities are included
    data_modalities = {
        'behavior_trials': False,
        'ephys_units': False,
        'lick_times': False,
        'reward_times': False,
        'FP': False,  # Fiber photometry
        'pupil': False,
        'tongue_movements': False,
        'keypoint_tracking': False,
        'aind_metadata': False,
        'beh_version': 'none',  # 'raw', 'processed', or 'none'
        'nwb_created': None,  # Timestamp when NWB object was created
        'nwb_saved': None,  # Timestamp when NWB file was saved (if save_file provided)
    }

    # 1. Merge unit tables (optional - may not exist for all sessions)
    # Get both merged units and ephys NWB for metadata
    merge_result = merge_unit_tables(session_id, data_type, return_nwb=True)
    if merge_result[0] is None:
        logger.warning(f"No unit tables to merge for {session_id} - will create NWB with behavior/acquisition only")
        merged_units = None
        ephys_nwb = None
    else:
        merged_units, ephys_nwb = merge_result
        logger.info(f"Merged {len(merged_units)} units")
        data_modalities['ephys_units'] = True

    # 2. Load session/trial table (optional - may not exist for all sessions)
    # Try raw version first, then processed version
    session_tbl = get_session_tbl(session_id, load_raw=True)
    if session_tbl is not None and len(session_tbl) > 0:
        logger.info(f"Loaded {len(session_tbl)} trials from raw behavior NWB")
        data_modalities['beh_version'] = 'raw'
    else:
        # Try processed version
        session_tbl = get_session_tbl(session_id, load_raw=False)
        if session_tbl is not None and len(session_tbl) > 0:
            logger.info(f"Loaded {len(session_tbl)} trials from processed behavior NWB")
            data_modalities['beh_version'] = 'processed'
        else:
            logger.warning(f"No session table found for {session_id} - will create NWB without behavior trials")
            session_tbl = None
            data_modalities['beh_version'] = 'none'

    # 2b. Load behavior NWB for acquisition data
    session_dir = session_dirs(session_id)
    behavior_nwb_path = os.path.join(session_dir['beh_fig_dir'], session_id + '.nwb')
    behavior_nwb = None
    if os.path.exists(behavior_nwb_path):
        behavior_nwb = load_nwb_from_filename(behavior_nwb_path)
        logger.info(f"Loaded behavior NWB from {behavior_nwb_path}")
    else:
        logger.warning(f"Behavior NWB not found at {behavior_nwb_path}")

    # 3. Get session metadata - use from ephys NWB if available, otherwise behavior NWB
    # Priority: ephys_nwb > behavior_nwb > defaults
    source_nwb = ephys_nwb if ephys_nwb is not None else behavior_nwb

    if source_nwb is not None:
        session_description = source_nwb.session_description
        session_start_time = source_nwb.session_start_time
        source_session_id = source_nwb.session_id if hasattr(source_nwb, 'session_id') else session_id
        logger.info(f"Using metadata from {'ephys' if ephys_nwb is not None else 'behavior'} NWB")
    else:
        # Fallback to defaults
        session_description = f"Combined behavior and ephys data for {session_id}"
        session_dir = session_dirs(session_id)
        session_start_time = session_dir.get('datetime')
        if session_start_time is None:
            session_start_time = datetime.now(tzlocal())
        elif getattr(session_start_time, 'tzinfo', None) is None:
            session_start_time = session_start_time.replace(tzinfo=tzlocal())
        source_session_id = session_id
        logger.info("Using default metadata (no source NWB available)")

    # 4. Create NWB file
    creation_time = datetime.now(tzlocal())
    new_nwb = NWBFile(
        session_description=session_description,
        identifier=f"{session_id}_merged_{creation_time.strftime('%Y%m%d_%H%M%S')}",
        session_start_time=session_start_time,
        session_id=source_session_id,
        institution='Allen Institute for Neural Dynamics',
    )

    # Track creation time
    data_modalities['nwb_created'] = creation_time.isoformat()
    logger.info("Created NWB file")
    
    # 5. Add trials with descriptions (if session table exists)
    if session_tbl is not None:
        trial_df = session_tbl.reset_index(drop=True).copy()
        trial_cols = [col for col in trial_df.columns if col not in ('start_time', 'stop_time')]
        trial_descriptions = COLUMN_DESCRIPTIONS.get('behavior_trial_columns', {})

        ragged_trial_cols = set()
        for col in trial_cols:
            description = trial_descriptions.get(col, f'Trial column: {col}')
            non_null = trial_df[col].dropna()
            is_ragged = len(non_null) > 0 and isinstance(non_null.iloc[0], (list, np.ndarray))
            if is_ragged:
                ragged_trial_cols.add(col)
            new_nwb.add_trial_column(name=col, description=description, index=is_ragged)

        for _, row in trial_df.iterrows():
            start_time = float(row.get('start_time', 0.0))
            stop_time = float(row.get('stop_time', start_time))
            if stop_time < start_time:
                stop_time = start_time

            trial_kwargs = {}
            for col in trial_cols:
                if col not in row.index:
                    continue
                val = row[col]

                # Convert Python None to appropriate type (like reference behavior NWB)
                if val is None or (isinstance(val, float) and pd.isna(val)):
                    if col in ragged_trial_cols:
                        val = np.array([])
                    else:
                        val = np.nan

                trial_kwargs[col] = val

            new_nwb.add_trial(start_time=start_time, stop_time=stop_time, **trial_kwargs)

        logger.info(f"Added {len(trial_df)} trials with {len(trial_cols)} columns")
        data_modalities['behavior_trials'] = True
    else:
        logger.info("No behavior trials to add")

    # 5b. Add acquisition TimeSeries from behavior NWB (length > 1)
    if behavior_nwb and behavior_nwb.acquisition:
        # Photometry channels are renamed from channel index to brain region
        channel_labels = photometry_channel_labels(session_id)
        for acq_name, acq_data in behavior_nwb.acquisition.items():
            if hasattr(acq_data, 'timestamps') and len(acq_data.timestamps) > 1:
                new_name, region_label = rename_photometry_acquisition(acq_name, channel_labels)
                description = acq_data.description if hasattr(acq_data, 'description') else ''
                if region_label is not None:
                    # Keep the original channel index, it is the only link back to the raw data
                    description = f"{description} (fiber in {region_label}, channel {acq_name.split('_')[1]})".strip()

                # Copy TimeSeries to new NWB
                from pynwb import TimeSeries
                new_ts = TimeSeries(
                    name=new_name,
                    data=acq_data.data[:].astype(np.float64),
                    timestamps=acq_data.timestamps[:].astype(np.float64),
                    unit=acq_data.unit if hasattr(acq_data, 'unit') else 'N/A',
                    description=description
                )
                new_nwb.add_acquisition(new_ts)
                renamed_from = f" (renamed from {acq_name})" if new_name != acq_name else ''
                logger.info(f"Added acquisition TimeSeries: {new_name}{renamed_from} ({len(acq_data.timestamps)} timestamps)")

                # Track lick, reward, and fiber photometry modalities
                if 'lick' in acq_name.lower():
                    data_modalities['lick_times'] = True
                if 'reward' in acq_name.lower():
                    data_modalities['reward_times'] = True
                # Photometry channels start with 'G' or 'Iso' (but not FIP which is timing signal)
                if (acq_name.startswith('G') or acq_name.startswith('Iso')) and not acq_name.startswith('FIP'):
                    data_modalities['FP'] = True

    # 6. Add merged units with descriptions (if units exist)
    if merged_units is not None:
        unit_df = merged_units.reset_index(drop=True).copy()

        # Predefined NWB columns that should NOT be added via add_unit_column()
        predefined_cols = ['spike_times', 'electrodes', 'obs_intervals', 'electrode_group']

        # Separate custom columns from predefined ones
        unit_cols = [col for col in unit_df.columns if col not in predefined_cols]
        unit_descriptions = COLUMN_DESCRIPTIONS.get('unit_columns', {})

        # Create reverse mapping: mapped_name -> original_name for looking up descriptions
        mapped_to_original = {}
        for orig_col, mapped_name in COLUMN_MAP['unit_columns_custom'].items():
            if 'duplicate as' in mapped_name:
                continue
            if 'similar to' in mapped_name:
                clean_name = mapped_name.split(';')[0].strip()
                mapped_to_original[clean_name] = orig_col
            else:
                mapped_to_original[mapped_name] = orig_col

        for orig_col, mapped_name in COLUMN_MAP['unit_columns_ks'].items():
            if mapped_name not in mapped_to_original:
                mapped_to_original[mapped_name] = orig_col

        # Only add custom columns (not predefined ones)
        ragged_unit_cols = set()
        for col in unit_cols:
            # Look up description using original column name
            orig_col = mapped_to_original.get(col, col)
            description = unit_descriptions.get(orig_col, col)
            if description == 'to be filled':
                description = col

            non_null = unit_df[col].dropna()
            is_ragged = False
            if len(non_null) > 0 and isinstance(non_null.iloc[0], (list, np.ndarray)):
                if getattr(non_null.iloc[0], 'ndim', 1) == 1:
                    lens = {len(v) for v in non_null}
                    is_ragged = len(lens) > 1
            if is_ragged:
                ragged_unit_cols.add(col)
            new_nwb.add_unit_column(name=col, description=description, index=is_ragged)

        for idx, row in unit_df.iterrows():
            unit_kwargs = {}

            # Handle predefined columns (spike_times, electrodes, etc.)
            if 'spike_times' in unit_df.columns:
                spike_times = row['spike_times']
                if isinstance(spike_times, (list, np.ndarray)):
                    unit_kwargs['spike_times'] = np.array(spike_times, dtype=np.float64)
                else:
                    unit_kwargs['spike_times'] = np.array([], dtype=np.float64)
            else:
                unit_kwargs['spike_times'] = np.array([], dtype=np.float64)

            # Handle electrodes if present (pass as parameter, not as custom column)
            # Skip if no electrode table exists in the NWB
            if 'electrodes' in unit_df.columns and new_nwb.electrodes is not None:
                val = row['electrodes']
                if isinstance(val, pd.DataFrame):
                    unit_kwargs['electrodes'] = list(val.index)
                elif val is not None and not pd.isna(val):
                    unit_kwargs['electrodes'] = val

            # Add custom columns (convert None to appropriate type)
            for col in unit_cols:
                val = row[col]

                # Convert Python None to np.nan for scalars, or a NaN-filled array
                # of the same shape as a non-null sample for array columns.
                if val is None or (isinstance(val, float) and pd.isna(val)):
                    non_null_vals = unit_df[col].dropna()
                    if len(non_null_vals) > 0:
                        sample_val = non_null_vals.iloc[0]
                        if isinstance(sample_val, np.ndarray):
                            dtype = sample_val.dtype if sample_val.dtype.kind in 'fc' else np.float64
                            val = np.full(sample_val.shape, np.nan, dtype=dtype)
                        elif isinstance(sample_val, list):
                            val = []
                        else:
                            val = np.nan
                    elif col in KNOWN_ARRAY_COLUMNS:
                        # No non-null sample to copy a shape from, but these columns
                        # must still be array-like (pynwb type-checks waveform_mean /
                        # waveform_sd, and zarr/hdf5 need a consistent dtype), so
                        # write an empty 1D float array for every unit.
                        val = np.array([], dtype=np.float64)
                    else:
                        val = np.nan

                unit_kwargs[col] = val

            new_nwb.add_unit(**unit_kwargs)

        logger.info(f"Added {len(unit_df)} units with {len(unit_cols)} columns")
    else:
        logger.info("No units to add - behavior/acquisition only NWB")

    # 7. Add pupil data to behavior processing module (if available)
    pupil_data = load_pupil(session_id)
    if pupil_data is not None:
        if 'behavior' not in new_nwb.processing:
            new_nwb.create_processing_module(
                name='behavior',
                description='Processed behavioral data',
            )
        new_nwb.processing['behavior'].add(pupil_data_to_timeseries(pupil_data))
        data_modalities['pupil'] = True
        logger.info("Added pupil diameter TimeSeries to behavior processing module")
    else:
        logger.info("No pupil data available for this session")

    # 7b/7c. Add tongue movement and kinematics tables to behavior processing module.
    # The movement table only comes from the pooled parquet asset; sessions missing from
    # that asset simply get no movement table. The keypoint asset's own movs table is
    # deliberately ignored (it duplicates the parquet one, minus the out_* columns).
    movement_table = load_tongue_movements(session_id)
    _, kins_table = load_keypoint_tracking(session_id)

    if movement_table is not None or kins_table is not None:
        if 'behavior' not in new_nwb.processing:
            new_nwb.create_processing_module(
                name='behavior',
                description='Processed behavioral data',
            )

    if movement_table is not None:
        new_nwb.processing['behavior'].add(movement_table)
        data_modalities['tongue_movements'] = True
        logger.info(f"Added {movement_table.name} DynamicTable to behavior processing module")
    else:
        logger.info("No tongue movement data available for this session")

    if kins_table is not None:
        new_nwb.processing['behavior'].add(kins_table)
        data_modalities['keypoint_tracking'] = True
        logger.info("Added tongue_kinematics table to behavior processing module")
    else:
        logger.info("No keypoint tracking data available for this session")

    # 7d. Attach the raw AIND metadata JSON files (if requested)
    if add_metadata:
        meta_dict = load_aind_metadata(session_id)
        if meta_dict is not None:
            add_aind_metadata(new_nwb, meta_dict)
            data_modalities['aind_metadata'] = True
            logger.info(f"Added AIND metadata to lab_meta_data['{AIND_LAB_META_DATA_KEY}']")
        else:
            logger.info("No AIND metadata available for this session")

    # 8. Log data modalities included
    included_modalities = [k for k, v in data_modalities.items() if v]
    logger.info(f"Data modalities included: {', '.join(included_modalities) if included_modalities else 'none'}")

    # 9. Save if requested (zarr backend; the store is a directory, so make sure
    # the path carries the .zarr suffix rather than collide with an .nwb file)
    if save_file is not None:
        if not save_file.endswith('.zarr'):
            save_file = save_file + '.zarr'
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        # mode='w' overwrites, but a zarr store has to be a directory: drop any
        # regular file sitting at this path (e.g. left by the HDF5 backend).
        if os.path.exists(save_file) and not os.path.isdir(save_file):
            logger.warning(f"Removing non-directory file at {save_file} to make room for the zarr store")
            os.remove(save_file)
        save_time = datetime.now(tzlocal())
        with NWBZarrIO(save_file, mode='w') as io:
            io.write(new_nwb)
        data_modalities['nwb_saved'] = save_time.isoformat()
        logger.info(f"Saved combined NWB to {save_file}")
    else:
        logger.info("Generated NWB in memory only (no file written)")

    return save_file, new_nwb, data_modalities


if __name__ == '__main__':
    # Test
    logging.basicConfig(level=logging.INFO)

    sessions = [
        'behavior_754897_2025-03-13_11-20-42',
    ]

    for session in sessions:
        print(f"\n{'='*80}")
        print(f"Testing: {session}")
        print(f"{'='*80}\n")


        # Test the full build_combined_nwb function
        save_path, nwb, modalities = build_combined_nwb(session, data_type='curated', save_file=None)
        if nwb is not None:
            print(f"\n✓ Success! Combined NWB created")
            print(f"  Trials: {len(nwb.trials) if nwb.trials is not None else 0} rows")
            print(f"  Units: {len(nwb.units) if nwb.units is not None else 0} rows")
            print(f"  Modalities: {', '.join(k for k, v in modalities.items() if v)}")

            # Show sample columns
            if nwb.trials is not None:
                trials_df = nwb.trials.to_dataframe()
                print(f"\n  Trial columns ({len(trials_df.columns)}): {list(trials_df.columns)[:5]}...")
            if nwb.units is not None:
                units_df = nwb.units.to_dataframe()
                print(f"  Unit columns ({len(units_df.columns)}): {list(units_df.columns)[:5]}...")
        else:
            print("\n✗ Build failed")

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
import re
import tempfile
import pandas as pd
import numpy as np
from datetime import datetime
from uuid import uuid4
from dateutil.tz import tzlocal
from hdmf.spec import DatasetSpec, GroupSpec, NamespaceBuilder
from pynwb import NWBFile, NWBHDF5IO, TimeSeries, get_class, load_namespaces
from hdmf_zarr import NWBZarrIO
from pynwb.file import Subject

import sys
sys.path.insert(0, '/root/capsule/code/beh_ephys_analysis')
from aind_dynamic_foraging_data_utils.nwb_utils import load_nwb_from_filename
from utils.beh_functions import get_session_tbl, get_unit_tbl, session_dirs, parseSessionID
from utils.pupil_utils import load_pupil
from pathlib import Path
from hdmf.common import DynamicTable, VectorData
from data_management.build_nwb_metadata import write_session_metadata

logger = logging.getLogger(__name__)

# Tongue movement data
TONGUE_MOVEMENT_DATA_DIR = Path('/root/capsule/data/all_tongue_movements')
TONGUE_MOVEMENT_PARQUET = TONGUE_MOVEMENT_DATA_DIR / 'all_tongue_movements_04022026.parquet'
KEYPOINT_TRACKING_DIR = Path('/root/capsule/data/keypoint_tracking_bottomview_LCrecordings')

# Spike times must reach the NWB in seconds. A sorter that leaves them as sample
# indices is caught by the span, not the magnitude: seconds here carry a large
# absolute-clock offset (the earliest spike is often 1e6-1.7e7), so only the
# max-to-min distance separates the two. Real sessions span 1.6e3-7.0e3 s; an hour of
# sample indices at 30 kHz spans ~1.1e8. See spike_times_are_samples.
MAX_PLAUSIBLE_SESSION_SECONDS = 86400  # 24 h, far above any observed session
NOMINAL_SAMPLING_RATE = 30000  # only to report what the span would be as seconds

# AIND metadata extension: the raw metadata JSON files are bundled as a single JSON
# blob in a LabMetaData container. Placeholder for now — expected to be replaced by
# properly typed metadata later.
AIND_NAMESPACE = 'aind_beh_ephys'
AIND_NAMESPACE_VERSION = '0.1.0'
AIND_NEURODATA_TYPE = 'AindMetadata'
AIND_LAB_META_DATA_KEY = 'aind_metadata'

# Subject metadata. The NWB Subject is inherited from the source NWBs field by field
# (see build_subject), falling back to the AIND subject.json in the session's raw asset —
# the ephys NWBs of the older Neuralynx sessions carry a subject_id and nothing else.

SUBJECT_JSON_NAME = 'subject.json'
# The pynwb Subject fields this inherits, in the order they are logged. Everything Subject
# takes except age__reference, which no source carries and pynwb defaults to 'birth', and
# weight: the sources record a bare number whose unit is ambiguous (nwbinspector rejects
# '0.0257'), and some carry a NaN, so it is left out rather than written wrong.
SUBJECT_FIELDS = ('subject_id', 'species', 'strain', 'sex', 'date_of_birth', 'age',
                  'genotype', 'description')
# NWB writes sex as a single-letter code, AIND metadata spells it out
SEX_CODES = {'m': 'M', 'male': 'M', 'f': 'F', 'female': 'F', 'u': 'U', 'unknown': 'U',
             'o': 'O', 'other': 'O'}
SEX_UNKNOWN = 'U'
# Descriptions saying nothing subject_id does not already say, e.g. 'Animal name:754897'
UNINFORMATIVE_SUBJECT_DESCRIPTION = re.compile(r'^\s*animal\s*name\s*:', re.IGNORECASE)
# ISO 8601 duration, the format NWB wants for Subject.age. The curated ephys NWBs carry
# a stringified timedelta instead ('P237 days, 11:30:09D'), which this rejects.
ISO8601_DURATION = re.compile(
    r'^P(?!$)(\d+(?:\.\d+)?Y)?(\d+(?:\.\d+)?M)?(\d+(?:\.\d+)?W)?(\d+(?:\.\d+)?D)?'
    r'(T(?=\d)(\d+(?:\.\d+)?H)?(\d+(?:\.\d+)?M)?(\d+(?:\.\d+)?S)?)?$'
)
# Load column mappings and descriptions
COLUMN_MAP_PATH = '/root/capsule/code/data_management/column_names_map.json'
COLUMN_DESC_PATH = '/root/capsule/code/data_management/column_names_description.json'

# Fiber photometry: per-subject surgery records (region -> implant hemisphere), used to
# label photometry acquisitions by brain region instead of channel index.
FP_METADATA_DIR = '/root/capsule/code/data_management/FP_metadata'
PHOTOMETRY_SIGNALS = ('G', 'Iso', 'G-Iso')
# Signals left out of the merged NWB entirely; drop from here to start packing one again.
PHOTOMETRY_SIGNALS_SKIPPED = ('G-Iso',)

# Vocabulary for the photometry channel descriptions written by
# rename_photometry_acquisition(), one entry per token of the acquisition name
# '{signal}_{region}-{hemisphere}[_{detrending method}][_mc]'.
PHOTOMETRY_SIGNAL_DESCRIPTIONS = {
    'G': 'Green (470 nm excitation) fluorescence',
    'Iso': 'Isosbestic (415 nm excitation) control fluorescence',
    'G-Iso': "Green fluorescence referenced to the isosbestic control ('G-Iso' channel "
             "from the upstream FIP preprocessing)",
}
PHOTOMETRY_DETRENDING_DESCRIPTIONS = {
    'exp': "detrended with the 'exp' method (two exponential curves fitted to the baseline)",
    'tri-exp': "detrended with the 'tri-exp' method (three exponential curves fitted to the baseline)",
    'bright': "detrended with the 'bright' method (biphasic exponential bleaching baseline "
              "scaled by a saturating exponential brightening term)",
}
PHOTOMETRY_MOTION_CORRECTION_DESCRIPTION = 'motion corrected using the isosbestic channel'
HEMISPHERE_NAMES = {'L': 'left', 'R': 'right'}

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

# Modalities that make a session worth writing. If none of them are present the NWB
# would carry nothing but session metadata, so build_combined_nwb skips saving and
# reports NO_VALID_DATA as the path instead.
REQUIRED_MODALITIES = (
    'behavior_trials',
    'ephys_units',
    'FP',
    'pupil',
    'tongue_movements',
    'keypoint_tracking',
)
NO_VALID_DATA = 'no valid data'

# Modalities that go into the file name, mapped to the label they get there. Only these
# three are named: they are the acquisition modalities the file is filed under, while the
# rest (pupil, tongue movements, ...) are derived from them. Order matters - the labels
# are joined with '+' in this order (see nwb_file_name).
FILE_NAME_MODALITIES = (
    ('behavior_trials', 'behavior'),
    ('ephys_units', 'ecephys'),
    ('FP', 'fib'),
)

# Backends build_combined_nwb can write, mapped to the extension each one needs:
# hdf5 writes a single file, zarr writes a directory store.
NWB_BACKENDS = {
    'hdf5': ('.nwb', NWBHDF5IO),
    'zarr': ('.nwb.zarr', NWBZarrIO),
}


def nwb_save_path(save_file, backend='zarr'):
    """
    Give a save path the extension its backend needs.

    The two backends must not share a path: hdf5 writes a regular file and zarr a
    directory store, so '<name>.nwb' is reserved for hdf5 and '<name>.nwb.zarr' for
    zarr. Any existing .nwb / .nwb.zarr extension on the input is replaced, so a
    caller can pass the same path for either backend.

    Args:
        save_file: path to save to, with or without an extension
        backend: 'hdf5' or 'zarr'

    Returns:
        The path with the backend's extension, e.g.
        nwb_save_path('nwb/s_combined.nwb', 'zarr') -> 'nwb/s_combined.nwb.zarr'
    """
    if backend not in NWB_BACKENDS:
        raise ValueError(f"Unknown NWB backend '{backend}', expected one of {sorted(NWB_BACKENDS)}")
    stem = str(save_file)
    for extension in ('.zarr', '.nwb'):  # in this order, so '.nwb.zarr' comes off whole
        if stem.endswith(extension):
            stem = stem[:-len(extension)]
    return stem + NWB_BACKENDS[backend][0]


def nwb_file_name(session_id, data_modalities, backend='zarr'):
    """
    Name a combined NWB from the session and the modalities it ended up with.

    The name is 'sub-<animal_id>_ses-<modalities>-<raw_id>' plus the backend's
    extension, with the underscores of raw_id turned into dashes so that '_' stays the
    separator between the name's own fields. <modalities> is the '+'-joined labels of
    the modalities in the file (see FILE_NAME_MODALITIES); if the file has none of them
    the '<modalities>-' part is dropped.

    Args:
        session_id: Session identifier, e.g. 'behavior_669492_2023-06-26_19-14-31'
        data_modalities: the modalities dict build_combined_nwb fills in
        backend: 'hdf5' or 'zarr', which decides the extension

    Returns:
        The file name
    """
    animal_id, _, raw_id = parseSessionID(session_id)
    if animal_id is None or raw_id is None:
        raise ValueError(f"Cannot build a file name from unparseable session ID '{session_id}'")
    labels = [label for key, label in FILE_NAME_MODALITIES if data_modalities.get(key)]
    # session_label = '-'.join(filter(None, ['+'.join(labels), raw_id.replace('_', '-')]))
    session_label = '-'.join(filter(None, [raw_id.replace('_', '-'), '+'.join(labels)]))
    return nwb_save_path(f"sub-{animal_id}_ses-{session_label}", backend)


def nwb_file_name(session_id, data_modalities, backend='zarr'):
    """
    Name a combined NWB from the session and the modalities it ended up with.

    The name is 'sub-<animal_id>_ses-<modalities>-<raw_id>' plus the backend's
    extension, with the underscores of raw_id turned into dashes so that '_' stays the
    separator between the name's own fields. <modalities> is the '+'-joined labels of
    the modalities in the file (see FILE_NAME_MODALITIES); if the file has none of them
    the '<modalities>-' part is dropped.

    Args:
        session_id: Session identifier, e.g. 'behavior_669492_2023-06-26_19-14-31'
        data_modalities: the modalities dict build_combined_nwb fills in
        backend: 'hdf5' or 'zarr', which decides the extension

    Returns:
        The file name, e.g.
        'sub-669492_ses-behavior+ecephys-669492-2023-06-26-19-14-31.nwb.zarr'
    """
    animal_id, _, raw_id = parseSessionID(session_id)
    if animal_id is None or raw_id is None:
        raise ValueError(f"Cannot build a file name from unparseable session ID '{session_id}'")
    labels = [label for key, label in FILE_NAME_MODALITIES if data_modalities.get(key)]
    # session_label = '-'.join(filter(None, ['+'.join(labels), raw_id.replace('_', '-')]))
    session_label = '-'.join(filter(None, [raw_id.replace('_', '-'), '+'.join(labels)]))
    return nwb_save_path(f"sub-{animal_id}_ses-{session_label}", backend)


def nwb_file_name(session_id, data_modalities, backend='zarr'):
    """
    Name a combined NWB from the session and the modalities it ended up with.

    The name is 'sub-<animal_id>_ses-<modalities>-<raw_id>' plus the backend's
    extension, with the underscores of raw_id turned into dashes so that '_' stays the
    separator between the name's own fields. <modalities> is the '+'-joined labels of
    the modalities in the file (see FILE_NAME_MODALITIES); if the file has none of them
    the '<modalities>-' part is dropped.

    Args:
        session_id: Session identifier, e.g. 'behavior_669492_2023-06-26_19-14-31'
        data_modalities: the modalities dict build_combined_nwb fills in
        backend: 'hdf5' or 'zarr', which decides the extension

    Returns:
        The file name, e.g.
        'sub-669492_ses-behavior+ecephys-669492-2023-06-26-19-14-31.nwb.zarr'
    """
    animal_id, _, raw_id = parseSessionID(session_id)
    if animal_id is None or raw_id is None:
        raise ValueError(f"Cannot build a file name from unparseable session ID '{session_id}'")
    labels = [label for key, label in FILE_NAME_MODALITIES if data_modalities.get(key)]
    session_label = '-'.join(filter(None, ['+'.join(labels), raw_id.replace('_', '-')]))
    return nwb_save_path(f"sub-{animal_id}_ses-{session_label}", backend)


def load_intermediate_data(session_dir: Path) -> dict:
    """Load the four intermediate parquet tables for a session."""
    idir = session_dir / "intermediate_data"
    return {
        "movs":   pd.read_parquet(idir / "tongue_movs.parquet"),
        "trials": pd.read_parquet(idir / "nwb_df_trials.parquet"),
        "licks":  pd.read_parquet(idir / "nwb_df_licks.parquet"),
        "kins":   pd.read_parquet(idir / "tongue_kins.parquet"),
        "events": pd.read_parquet(idir / "nwb_df_events.parquet"),
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


def add_aind_metadata(nwb_file, meta_dict):
    """Attach the combined JSON metadata to nwb_file under lab_meta_data[AIND_LAB_META_DATA_KEY]."""
    nwb_file.add_lab_meta_data(
        aind_metadata_type()(
            name=AIND_LAB_META_DATA_KEY,
            json_data=json.dumps(meta_dict),
        )
    )
    return nwb_file


def _subject_value(value):
    """Normalise a subject field to None when it carries nothing, so a later source fills it."""
    if isinstance(value, str) and not value.strip():
        return None
    return value


def _named(value):
    """Take the name out of an AIND metadata term ({'name': 'Mus musculus', ...} or a string)."""
    if isinstance(value, dict):
        return _subject_value(value.get('name'))
    return _subject_value(value)


def _subject_fields_from_nwb(nwb):
    """Read the SUBJECT_FIELDS off a source NWB's Subject, {} when it has none."""
    subject = getattr(nwb, 'subject', None) if nwb is not None else None
    if subject is None:
        return {}
    return {field: _subject_value(getattr(subject, field, None)) for field in SUBJECT_FIELDS}


def _subject_fields_from_json(session_id):
    """
    Read the subject fields out of the AIND subject.json in the session's raw asset.

    Two aind-data-schema layouts are in circulation, and both turn up across these
    sessions: v1 keeps the subject fields at the top level, v2 nests them under
    'subject_details'. Only the fields NWB's Subject has a slot for are taken; the rest
    (registries, breeding info, housing) stays in the AIND metadata blob, see
    add_aind_metadata. Age is not in this file at all - it is computed from the date of
    birth by _subject_age. Weight is deliberately not inherited at all, see SUBJECT_FIELDS.

    Args:
        session_id: Session identifier

    Returns:
        dict of subject field -> value, empty when the file is missing or unreadable.
    """
    path = os.path.join(session_dirs(session_id)['raw_dir'], SUBJECT_JSON_NAME)
    if not os.path.exists(path):
        logger.info(f"No subject metadata at {path}")
        return {}

    try:
        with open(path, 'r') as f:
            record = json.load(f)
        details = record.get('subject_details', record)
        date_of_birth = _subject_value(details.get('date_of_birth'))
        return {
            'subject_id': _subject_value(record.get('subject_id')),
            'date_of_birth': datetime.fromisoformat(date_of_birth) if date_of_birth else None,
            'genotype': _subject_value(details.get('genotype')),
            'sex': _subject_value(details.get('sex')),
            'species': _named(details.get('species')),
            'strain': _named(details.get('strain')) or _named(details.get('background_strain')),
        }
    except Exception as e:
        logger.warning(f"Could not read subject metadata from {path}: {e}")
        return {}


def _subject_sex(sex):
    """Map a spelled-out sex to its NWB code, 'U' when it is missing or unrecognised."""
    if sex is None:
        return SEX_UNKNOWN
    code = SEX_CODES.get(str(sex).strip().lower())
    if code is None:
        logger.warning(f"Unrecognised subject sex '{sex}', writing '{SEX_UNKNOWN}'")
        return SEX_UNKNOWN
    return code


def _subject_date_of_birth(date_of_birth, session_start_time):
    """Give a date of birth the session's timezone, so hdmf does not write a naive datetime."""
    if date_of_birth is None:
        return None
    if getattr(date_of_birth, 'tzinfo', None) is None:
        tzinfo = getattr(session_start_time, 'tzinfo', None) or tzlocal()
        return date_of_birth.replace(tzinfo=tzinfo)
    return date_of_birth


def _subject_age(age, date_of_birth, session_start_time):
    """
    Age at the session as an ISO 8601 duration in whole days, e.g. 'P237D'.

    Computed from the date of birth whenever both dates are known, since the curated
    ephys NWBs carry a stringified timedelta ('P237 days, 11:30:09D') that is not a
    valid duration. Dates are compared as calendar dates, which is what the raw ephys
    NWBs' own 'P237D' counts. An inherited age is kept only if the computation is
    impossible and the string is a valid duration.
    """
    if date_of_birth is not None and session_start_time is not None:
        days = (session_start_time.date() - date_of_birth.date()).days
        if days >= 0:
            computed = f'P{days}D'
            if age is not None and age != computed:
                logger.info(f"Replacing inherited subject age '{age}' with computed {computed}")
            return computed
        logger.warning(f"Date of birth {date_of_birth.date()} is after the session, dropping age")
        return None
    if age is not None and not ISO8601_DURATION.match(str(age)):
        logger.warning(f"Subject age '{age}' is not an ISO 8601 duration and no date of birth "
                       f"is available to recompute it, dropping age")
        return None
    return age


def _subject_description(description):
    """Drop descriptions that only repeat the animal name (see UNINFORMATIVE_SUBJECT_DESCRIPTION)."""
    if description is None or UNINFORMATIVE_SUBJECT_DESCRIPTION.match(str(description)):
        return None
    return description


def build_subject(session_id, session_start_time, source_nwbs):
    """
    Build the merged NWB's Subject, inheriting each field from the first source that has it.

    The source NWBs come first, in the order given, and the AIND subject.json in the
    session's raw asset fills whatever they leave empty — for the Neuralynx sessions that
    is everything but subject_id. A few fields are then normalised rather than copied
    through, since the sources disagree with what NWB asks for: sex is coded to a single
    letter, age is recomputed from the date of birth as an ISO 8601 duration, and a
    description that only repeats the animal name is dropped (see the _subject_* helpers).
    Weight is not inherited at all, see SUBJECT_FIELDS. Fields no source has are left unset, apart
    from sex, which falls back to 'U' (unknown), and subject_id, which falls back to the
    animal ID parsed out of session_id.

    Args:
        session_id: Session identifier
        session_start_time: Session start time, used to compute age from the date of birth
        source_nwbs: List of (label, nwb) pairs in priority order; the label is only used
                     for logging and either NWB may be None

    Returns:
        pynwb.file.Subject
    """
    fields = {field: None for field in SUBJECT_FIELDS}
    sources = [(label, _subject_fields_from_nwb(nwb)) for label, nwb in source_nwbs]
    sources.append((SUBJECT_JSON_NAME, _subject_fields_from_json(session_id)))

    inherited_from = {}
    for label, candidate in sources:
        for field, value in candidate.items():
            if fields.get(field) is None and value is not None:
                fields[field] = value
                inherited_from[field] = label

    animal_id, _, raw_id = parseSessionID(session_id)
    if fields['subject_id'] is None:
        logger.warning(f"No subject_id in any source, using the animal ID from {session_id}")
        fields['subject_id'] = animal_id
    elif animal_id is not None and str(fields['subject_id']) != str(animal_id):
        logger.warning(f"Inherited subject_id '{fields['subject_id']}' does not match the animal "
                       f"ID '{animal_id}' in {session_id}")

    fields['species'] = _named(fields['species'])
    fields['strain'] = _named(fields['strain'])
    fields['sex'] = _subject_sex(fields['sex'])
    fields['date_of_birth'] = _subject_date_of_birth(fields['date_of_birth'], session_start_time)
    fields['age'] = _subject_age(fields['age'], fields['date_of_birth'], session_start_time)
    fields['description'] = _subject_description(fields['description'])

    kwargs = {field: value for field, value in fields.items() if value is not None}
    logger.info('Subject: ' + ', '.join(
        f"{field}={value!r} (from {inherited_from.get(field, 'derived')})"
        for field, value in kwargs.items()))
    missing = [field for field in SUBJECT_FIELDS if field not in kwargs]
    if missing:
        logger.info(f"Subject fields no source provided: {', '.join(missing)}")
    return Subject(**kwargs)


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
    Replace the channel index in a photometry acquisition name with its region label
    and spell the renamed name out as a description.

    Photometry names are '<signal>_<channel index>[_<detrending method>][_mc]', where
    signal is G, Iso or G-Iso, e.g. 'G_1_tri-exp_mc' -> 'G_TH-R_tri-exp_mc'. The region
    label carries the implant hemisphere, so renamed channels read
    '{signal}_{region}-{hemisphere}[_{detrending method}][_mc]' and the description
    reads back one clause per token, e.g. 'G_Gi-L_exp_mc' as green fluorescence from
    the fiber in Gi in the left hemisphere, detrended with the 'exp' method and motion
    corrected. The detrending method and the '_mc' motion correction flag are passed
    through as they are, including when the name has neither. Names that are not
    photometry channels, or whose index has no label, are returned unchanged and
    undescribed.

    Args:
        acq_name: acquisition name from the behavior NWB
        channel_labels: dict of channel index (str) -> region label, from
                        photometry_channel_labels()

    Returns:
        (new_name, region_label, description). region_label and description are None
        when nothing was renamed.
    """
    parts = acq_name.split('_')
    if len(parts) < 2 or parts[0] not in PHOTOMETRY_SIGNALS:
        return acq_name, None, None

    signal, index = parts[0], parts[1]
    label = channel_labels.get(index)
    if label is None:
        return acq_name, None, None

    # Everything after the channel index is a detrending method, optionally followed
    # by the 'mc' motion correction flag; either or both may be absent.
    processing = parts[2:]
    motion_corrected = bool(processing) and processing[-1] == 'mc'
    detrending = processing[:-1] if motion_corrected else processing

    region, _, hemisphere = label.rpartition('-')
    location = (f'{region} in the {HEMISPHERE_NAMES[hemisphere]} hemisphere'
                if hemisphere in HEMISPHERE_NAMES else label)

    clauses = [f'{PHOTOMETRY_SIGNAL_DESCRIPTIONS[signal]} from the fiber in {location}']
    clauses += [PHOTOMETRY_DETRENDING_DESCRIPTIONS.get(method, f"processed with '{method}'")
                for method in detrending]
    if motion_corrected:
        clauses.append(PHOTOMETRY_MOTION_CORRECTION_DESCRIPTION)
    # Keep the original channel index, it is the only link back to the raw data
    description = f"{', '.join(clauses)}. Photometry channel {index} in the raw data."

    return '_'.join([signal, label] + processing), label, description


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


def spike_times_are_samples(spike_times_col):
    """
    Detect spike times left as sample indices instead of seconds.

    Compares the span (latest minus earliest spike across every unit) against the
    longest session that could plausibly exist. The absolute values are not usable
    here: correct seconds are offset onto an absolute clock, so the earliest spike
    can itself be in the millions. The span is offset-free.

    Args:
        spike_times_col: iterable of per-unit spike time arrays

    Returns:
        (is_samples, span) where span is None if no unit holds any spike
    """
    lo = hi = None
    for spike_times in spike_times_col:
        if not isinstance(spike_times, (list, np.ndarray)):
            continue
        arr = np.asarray(spike_times, dtype=np.float64)
        if arr.size == 0:
            continue
        lo = arr.min() if lo is None else min(lo, arr.min())
        hi = arr.max() if hi is None else max(hi, arr.max())

    if lo is None:
        return False, None

    span = float(hi - lo)
    return span > MAX_PLAUSIBLE_SESSION_SECONDS, span


def merge_unit_tables(session_id, data_type='curated', return_nwb=False):
    """
    Merge unit tables from custom pickle and NWB kilosort data.

    Args:
        session_id: Session identifier
        data_type: 'curated' or 'raw'. If 'curated' and no curated unit table
            exists, falls back to 'raw'.
        return_nwb: If True, return (merged_df, ephys_nwb, data_type_used).
            If False, return just merged_df

    Returns:
        If return_nwb=False: Merged DataFrame with mapped column names, or None if merge fails
        If return_nwb=True: Tuple of (merged_df, ephys_nwb, data_type_used), where
            data_type_used is the version actually loaded ('curated' or 'raw'),
            or (None, None, None) if merge fails

        Counts as a failure, so the caller builds an NWB without units: spike times
        that are sample indices rather than seconds (see spike_times_are_samples).
    """
    # 1. Load custom unit table (use summary version)
    custom_unit_tbl = get_unit_tbl(session_id, data_type=data_type, summary=True)
    # commented out to make sure data is consistent with what is used in manuscript
    # if custom_unit_tbl is None and data_type == 'curated':
    #     logger.info(f"No curated unit table for {session_id} - falling back to raw")
    #     data_type = 'raw'
    #     custom_unit_tbl = get_unit_tbl(session_id, data_type=data_type, summary=True)
    if custom_unit_tbl is None or len(custom_unit_tbl) == 0:
        logger.warning(f"No custom unit table found for {session_id}")
        return (None, None, None) if return_nwb else None

    logger.info(f"Loaded {len(custom_unit_tbl)} units from custom table")

    # 2. Load NWB kilosort data
    session_dir = session_dirs(session_id)
    nwb_path = session_dir.get(f'nwb_dir_{data_type}')
    if nwb_path is None or not os.path.exists(nwb_path):
        logger.warning(f"NWB file not found at {nwb_path}")
        return (None, None, None) if return_nwb else None

    ephys_nwb = load_nwb_from_filename(nwb_path)
    if ephys_nwb.units is None:
        logger.warning(f"No units in NWB file for {session_id}")
        return (None, None, None) if return_nwb else None

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
        return (None, None, None) if return_nwb else None

    logger.info(f"Using NWB ID column: '{nwb_id_col}' for alignment")
    nwb_unit_ids = set(nwb_unit_tbl[nwb_id_col].values)
    common_ids = custom_unit_ids & nwb_unit_ids

    if len(common_ids) == 0:
        logger.error(f"No common units found between custom and NWB tables!")
        logger.error(f"  Custom unit_ids ({len(custom_unit_ids)}): {sorted(list(custom_unit_ids))[:10]}")
        logger.error(f"  NWB {nwb_id_col} ({len(nwb_unit_ids)}): {sorted(list(nwb_unit_ids))[:10]}")
        return (None, None, None) if return_nwb else None

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

    # Drop the whole table when the sorter left spike times as sample indices: rescaling
    # them here would guess at a sampling rate, and passing them through would silently
    # misalign every unit against the behavior clock. Checked before the dedup below so a
    # doomed table does not pay for a np.unique over millions of spikes.
    if 'spike_times' in merged_df.columns:
        is_samples, span = spike_times_are_samples(merged_df['spike_times'])
        if is_samples:
            logger.error(
                f"Dropping unit table for {session_id} ({data_type}): spike_times span "
                f"{span:,.1f} exceeds {MAX_PLAUSIBLE_SESSION_SECONDS}s, so they are sample "
                f"indices rather than seconds (as {NOMINAL_SAMPLING_RATE} Hz samples the span "
                f"would be {span / NOMINAL_SAMPLING_RATE:,.1f}s). Building without units."
            )
            return (None, None, None) if return_nwb else None

    # Remove duplicate spike times — equal consecutive values violate the NWB refractory-
    if 'spike_times' in merged_df.columns:
        def _clean_spike_times(st):
            if isinstance(st, (list, np.ndarray)) and len(st) > 0:
                return np.unique(np.asarray(st, dtype=np.float64))
            return st
        before = merged_df['spike_times'].apply(lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0).sum()
        merged_df['spike_times'] = merged_df['spike_times'].apply(_clean_spike_times)
        after = merged_df['spike_times'].apply(lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0).sum()
        if before != after:
            logger.warning(f"Removed {before - after} duplicate spike times across all units")

    if return_nwb:
        return merged_df, ephys_nwb, data_type
    else:
        return merged_df


def build_combined_nwb(session_id, data_type='curated', save_dir=None, add_metadata=False,
                       backend='zarr'):
    """
    Build a complete NWB file with available data modalities.

    Combines whichever data is available:
    - Behavior trials (from session table)
    - Ephys units (merged custom + kilosort)
    - Acquisition TimeSeries (lick times, reward times, etc.)

    Session and subject metadata are inherited from the source NWBs, the ephys one first
    (see build_subject for how the Subject fields are filled).

    Args:
        session_id: Session identifier
        data_type: 'curated' or 'raw'. 'curated' falls back to 'raw' when no
            curated unit table exists (the version used is reported as
            'ephys_version' in the returned modalities dict)
        save_dir: Directory to save the NWB into (if None, returns in-memory only).
            The file name is built from the session and the modalities the file ended
            up with, see nwb_file_name
        add_metadata: If True, bundle the raw AIND metadata JSON files into a
            LabMetaData container (see add_aind_metadata). Placeholder metadata,
            expected to be replaced by properly typed metadata later.
        backend: 'zarr' to write a '<name>.nwb.zarr' directory store, or 'hdf5'
            to write a single '<name>.nwb' file (see nwb_save_path)

    Returns:
        Tuple of (save_path, nwb_object, data_modalities_dict)
        save_path is the written file or store path, None if save_dir was None, or the
        string NO_VALID_DATA ('no valid data') if none of REQUIRED_MODALITIES were
        found - in that case nothing is written and 'nwb_saved' stays None.
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
            'ephys_version': str - unit table version actually used: 'curated', 'raw', or 'none'
            'nwb_created': str - ISO timestamp when NWB object was created
            'nwb_saved': str or None - ISO timestamp when NWB was saved to file (None if not saved)
    """
    logger.info(f"Building combined NWB for {session_id}")

    # Checked up front: both only matter at the very end, where the file is named and
    # written, and neither a bad backend nor an unnameable session should cost a whole
    # build before it is reported
    if backend not in NWB_BACKENDS:
        raise ValueError(f"Unknown NWB backend '{backend}', expected one of {sorted(NWB_BACKENDS)}")
    if save_dir is not None:
        nwb_file_name(session_id, {}, backend)

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
        'ephys_version': 'none',  # unit table version actually used: 'curated', 'raw', or 'none'
        'nwb_created': None,  # Timestamp when NWB object was created
        'nwb_saved': None,  # Timestamp when NWB file was saved (if save_dir provided)
    }

    # 1. Merge unit tables (optional - may not exist for all sessions)
    # Get both merged units and ephys NWB for metadata
    merge_result = merge_unit_tables(session_id, data_type, return_nwb=True)
    if merge_result[0] is None:
        logger.warning(f"No unit tables to merge for {session_id} - will create NWB with behavior/acquisition only")
        merged_units = None
        ephys_nwb = None
    else:
        merged_units, ephys_nwb, data_type = merge_result
        logger.info(f"Merged {len(merged_units)} units from the {data_type} unit table")
        data_modalities['ephys_units'] = True
        data_modalities['ephys_version'] = data_type

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
        session_description = f"Combined data for {session_id}"
        session_start_time = source_nwb.session_start_time
        source_session_id = source_nwb.session_id if hasattr(source_nwb, 'session_id') else session_id
        logger.info(f"Using metadata from {'ephys' if ephys_nwb is not None else 'behavior'} NWB")
        # remove .json from end of source_sessison_id if it exist
        if source_session_id.endswith(".json"):
            source_session_id = source_session_id[:-5]
        # Add "behavior_" to the beginning if it doesn't already exist
        if not source_session_id.startswith("behavior_") and not source_session_id.startswith("ecephys_"):
            source_session_id = "behavior_" + source_session_id
                # add 'behavior_" to start of source_session_id it it doesn't start with it
    else:
        # Fallback to defaults
        session_description = f"Combined data for {session_id}"
        session_dir = session_dirs(session_id)
        session_start_time = session_dir.get('datetime')
        if session_start_time is None:
            session_start_time = datetime.now(tzlocal())
        elif getattr(session_start_time, 'tzinfo', None) is None:
            session_start_time = session_start_time.replace(tzinfo=tzlocal())
        source_session_id = session_id
        logger.info("Using default metadata (no source NWB available)")

    # 4. Create NWB file, with the subject inherited from the source NWBs (and the raw
    # asset's subject.json for whatever they leave empty)
    creation_time = datetime.now(tzlocal())
    subject = build_subject(
        session_id,
        session_start_time,
        [('ephys NWB', ephys_nwb), ('behavior NWB', behavior_nwb)],
    )
    animal_id, _, raw_id = parseSessionID(session_id)
    new_nwb = NWBFile(
        session_description=session_description,
        subject=subject,
        identifier=f"{session_id}_merged_{creation_time.strftime('%Y%m%d_%H%M%S')}",
        session_start_time=session_start_time,
        session_id=raw_id, # use current session_id instead of inheriting from source nwbs
        institution='Allen Institute for Neural Dynamics',
        source_script='https://github.com/AllenNeuralDynamics/LC-beh-physiology-analysis/blob/pack/code/data_management/build_merged_nwb.py',
        source_script_file_name='build_merged_nwb.py'
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
            if acq_name.split('_')[0] in PHOTOMETRY_SIGNALS_SKIPPED:
                logger.info(f"Skipping acquisition TimeSeries: {acq_name} (skipped photometry signal)")
                continue
            if hasattr(acq_data, 'timestamps') and len(acq_data.timestamps) > 1:
                new_name, region_label, photometry_description = rename_photometry_acquisition(
                    acq_name, channel_labels)
                description = acq_data.description if hasattr(acq_data, 'description') else ''
                if photometry_description is not None:
                    description = photometry_description

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
        md = write_session_metadata(session_id, include_tongue=movement_table is not None, include_keypoint=True)
        if md is not None:
            add_aind_metadata(new_nwb, md.model_dump_json())
            data_modalities['aind_metadata'] = True
            logger.info(f"Added AIND metadata to lab_meta_data['{AIND_LAB_META_DATA_KEY}']")
        else:
            logger.info("No AIND metadata available for this session")

    # 8. Log data modalities included
    included_modalities = [k for k, v in data_modalities.items() if v]
    logger.info(f"Data modalities included: {', '.join(included_modalities) if included_modalities else 'none'}")

    # 9. Nothing but metadata: not worth a file, so report it in place of the path
    if not any(data_modalities[modality] for modality in REQUIRED_MODALITIES):
        logger.warning(
            f"No valid data for {session_id} (none of {', '.join(REQUIRED_MODALITIES)}) - skipping save"
        )
        return NO_VALID_DATA, new_nwb, data_modalities

    # 10. Save if requested, under a name built from the session and the modalities that
    # made it into the file, with the extension the chosen backend needs ('.nwb' for
    # hdf5, '.nwb.zarr' for the zarr directory store)
    if save_dir is not None:
        save_file = os.path.join(save_dir, nwb_file_name(session_id, data_modalities, backend))
        io_class = NWB_BACKENDS[backend][1]
        os.makedirs(save_dir, exist_ok=True)
        # mode='w' overwrites, but only in kind: a zarr store has to be a directory and
        # an hdf5 file a regular file, so drop whatever is at the path if it is neither.
        if backend == 'zarr' and os.path.exists(save_file) and not os.path.isdir(save_file):
            logger.warning(f"Removing non-directory file at {save_file} to make room for the zarr store")
            os.remove(save_file)
        if backend == 'hdf5' and os.path.isdir(save_file):
            raise IsADirectoryError(
                f"{save_file} is a directory (a zarr store?), cannot write an hdf5 file there - "
                f"remove it or build with backend='zarr'"
            )
        save_time = datetime.now(tzlocal())
        with io_class(save_file, mode='w') as io:
            io.write(new_nwb)
        data_modalities['nwb_saved'] = save_time.isoformat()
        logger.info(f"Saved combined NWB ({backend}) to {save_file}")
    else:
        save_file = None
        logger.info("Generated NWB in memory only (no file written)")

    return save_file, new_nwb, data_modalities


if __name__ == '__main__':
    # Test
    logging.basicConfig(level=logging.INFO)

    sessions = [
        'behavior_ZS062_2021-05-06_15-46-14',
        'behavior_ZS059_2021-04-29_14-02-45',
        'behavior_ZS061_2021-04-08_18-01-30',
        'behavior_781166_2025-05-13_14-04-27',
        'behavior_754897_2025-03-12_12-23-15',
        'behavior_754897_2025-03-13_11-20-42',
        'behavior_754898_2025-01-01_20-40-03',
        'behavior_749472_2025-01-09_13-56-02',
        'behavior_754896_2025-01-03_17-20-19',
    ]

    for session in sessions:
        print(f"\n{'='*80}")
        print(f"Testing: {session}")
        print(f"{'='*80}\n")


        # Test the full build_combined_nwb function
        save_path, nwb, modalities = build_combined_nwb(session, data_type='curated', save_dir=None)
        if nwb is not None:
            print(f"\n✓ Success! Combined NWB created")
            print(f"  Subject: " + ', '.join(
                f"{field}={getattr(nwb.subject, field)!r}" for field in SUBJECT_FIELDS
                if getattr(nwb.subject, field, None) is not None))
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

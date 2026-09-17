"""
Build combined NWB files for a set of sessions and write everything to /results.

Two modes:
    test        The nine hand-picked sessions from the "Test" section of
                test_nwb.ipynb. Use this to check a change end to end before
                committing to a full run.
    production  Every session listed in the three session-asset CSVs
                (session_assets.csv, hopkins_session_assets.csv,
                hopkins_FP_session_assets.csv).

Both modes drop anything listed in sessions_to_exclude.txt.

The packing itself is identical in both modes: build_one() calls
build_combined_nwb() once per session in a joblib worker and returns a flat
result row, exactly as the notebook does.

Build parameters come from a JSON file in code/data_management (created with
defaults if absent), can be overridden per run with --n-jobs / --add-metadata /
--backend, and a copy is written into the results folder so every run records
what it was given.

Outputs, all under --results-dir (default /root/capsule/results):
    nwb/<session>_combined.nwb.zarr   the zarr stores (--backend zarr, the default)
    nwb/<session>_combined.nwb        single-file NWBs instead, with --backend hdf5
    logs/<session>.log               per-session build log
    build_all_nwb.log                the full run log (same text as stdout)
    nwb_build_params.json            copy of the parameters used
    modalities_df.csv                one row per session, one column per modality

Usage:
    python build_all_nwb.py --mode test
    python build_all_nwb.py --mode production --n-jobs 8 --add-metadata
    python build_all_nwb.py --mode test --backend hdf5
"""
import argparse
import io
import json
import logging
import os
import shutil
import sys
import traceback
from datetime import datetime

import pandas as pd
from joblib import Parallel, delayed

# Resolve code/ and code/beh_ephys_analysis (the folder containing `utils`) relative
# to this file's location, so imports work no matter where the repo is checked out.
_code_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_beh_ephys_root = os.path.join(_code_root, 'beh_ephys_analysis')
for _path in (_beh_ephys_root, _code_root):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from utils.capsule_migration import CAPSULE_ROOT

# Imported through the data_management package, not flat as `build_merged_nwb`:
# that module uses a relative import for its metadata helper, which only resolves
# when it is reached via its parent package. (data_management has no __init__.py,
# but Python treats it as a namespace package, so this works as is.)
import data_management.build_merged_nwb as build_merged_nwb
from data_management.build_merged_nwb import build_combined_nwb

DEFAULT_RESULTS_DIR = '/root/capsule/results'
DEFAULT_PARAMS_PATH = os.path.join(CAPSULE_ROOT, 'code', 'data_management', 'nwb_build_params.json')

# Written out verbatim if the params file is missing, so a fresh checkout can run
# without hand-authoring the JSON first.
DEFAULT_PARAMS = {
    'build_one': {
        'data_type': 'curated',
        'add_metadata': True,
        'backend': 'zarr',
    },
    'parallel': {
        'n_jobs': 4,
        'backend': 'loky',
        'verbose': 10,
    },
}

# The "Test" section of test_nwb.ipynb.
TEST_SESSIONS = [
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

SESSION_ASSET_CSVS = [
    'session_assets.csv',
    'hopkins_session_assets.csv',
    'hopkins_FP_session_assets.csv',
]

# Sessions never to build, one ID per line. Applied to both modes.
SESSIONS_TO_EXCLUDE_FILE = os.path.join(CAPSULE_ROOT, 'code', 'data_management', 'sessions_to_exclude.txt')

FILE_LOG_FORMAT = '%(asctime)s %(levelname)s %(name)s: %(message)s'
# No timestamp: worker lines are relayed through the parent logger, which stamps
# them itself, and two timestamps per line reads badly.
RELAY_LOG_FORMAT = '%(levelname)s %(name)s: %(message)s'
LOG_DATEFMT = '%H:%M:%S'


def load_params(params_path):
    """Load build parameters, creating the file with defaults if it doesn't exist."""
    if not os.path.exists(params_path):
        os.makedirs(os.path.dirname(params_path), exist_ok=True)
        with open(params_path, 'w') as f:
            json.dump(DEFAULT_PARAMS, f, indent=2)
            f.write('\n')
        logging.info(f"No params file at {params_path}, wrote defaults")

    with open(params_path, 'r') as f:
        params = json.load(f)

    unknown = set(params) - set(DEFAULT_PARAMS)
    if unknown:
        raise ValueError(
            f"Unknown top-level keys in {params_path}: {sorted(unknown)}. "
            f"Expected only {sorted(DEFAULT_PARAMS)}."
        )
    # Missing sections fall back to defaults rather than erroring, so a params
    # file can carry only the keys it wants to change.
    return {
        'build_one': {**DEFAULT_PARAMS['build_one'], **params.get('build_one', {})},
        'parallel': {**DEFAULT_PARAMS['parallel'], **params.get('parallel', {})},
    }


def load_excluded_sessions():
    """Session IDs from sessions_to_exclude.txt, one per line."""
    if not os.path.exists(SESSIONS_TO_EXCLUDE_FILE):
        logging.warning(f"No exclusion list at {SESSIONS_TO_EXCLUDE_FILE}, excluding nothing")
        return set()
    with open(SESSIONS_TO_EXCLUDE_FILE, 'r') as f:
        return {line.strip() for line in f if line.strip() and not line.startswith('#')}


def gather_sessions(mode):
    """Return the session list for the given mode, minus the excluded sessions."""
    if mode == 'test':
        session_ids = list(TEST_SESSIONS)
    else:
        # Read CSVs with probe as string to ensure '2' not 2.0
        dfs = [
            pd.read_csv(os.path.join(CAPSULE_ROOT, 'code', 'data_management', name), dtype={'probe': str})
            for name in SESSION_ASSET_CSVS
        ]
        df = pd.concat(dfs)
        # drop rows with no raw data asset: nothing to build from
        df = df[df['raw_data'].notna()]
        session_ids = df['session_id'].values
        # filter only behavior sessions (blank rows in the CSVs come through as NaN)
        session_ids = [session_id for session_id in session_ids if isinstance(session_id, str)]

        # A session appearing in more than one CSV would otherwise be built twice.
        deduped = list(dict.fromkeys(session_ids))
        if len(deduped) != len(session_ids):
            logging.warning(f"Dropped {len(session_ids) - len(deduped)} duplicate session_id(s)")
        session_ids = deduped

    excluded = load_excluded_sessions()
    dropped = [s for s in session_ids if s in excluded]
    if dropped:
        logging.info(f"Excluded {len(dropped)} session(s) via sessions_to_exclude.txt: {dropped}")
    return [s for s in session_ids if s not in excluded]


def build_one(session, save_dir, log_dir, **kwargs):
    """Build + save one session and return a flat result row.

    The NWB object itself is not returned: it can't be pickled back from a worker,
    so only the store path and the modalities dict come home.

    This runs in a worker process, where nothing printed reaches the parent's
    stdout, so it installs its own log handlers: a per-session file (which
    survives a worker killed for memory) plus a buffer returned as 'log' for the
    parent to relay.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{session}.log")
    relay = io.StringIO()

    root = logging.getLogger()
    prev_handlers, prev_level = root.handlers[:], root.level
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setFormatter(logging.Formatter(FILE_LOG_FORMAT, datefmt=LOG_DATEFMT))
    relay_handler = logging.StreamHandler(relay)
    relay_handler.setFormatter(logging.Formatter(RELAY_LOG_FORMAT))
    root.handlers = [file_handler, relay_handler]
    root.setLevel(logging.INFO)

    started = datetime.now()
    try:
        # build_combined_nwb sets the extension from its backend, so it returns the
        # actual path written ('..._combined.nwb' or '..._combined.nwb.zarr')
        kwargs['save_file'] = os.path.join(save_dir, f"{session}_combined.nwb")
        logging.info(f"Building {session}: {kwargs}")
        nwb_path, _, modalities = build_combined_nwb(session, **kwargs)
        # flatten: each key in modalities becomes its own column
        row = {'session': session, 'error': False, 'error_msg': None, 'nwb_path': nwb_path, **modalities}
    except Exception as e:
        logging.error(f"{session} failed: {type(e).__name__}: {e}")
        logging.error(traceback.format_exc())
        row = {'session': session, 'error': True, 'error_msg': f"{type(e).__name__}: {e}", 'nwb_path': None}

    row['duration_s'] = round((datetime.now() - started).total_seconds(), 1)
    logging.info(f"{session} finished in {row['duration_s']}s (error={row['error']})")

    for handler in (file_handler, relay_handler):
        handler.flush()
        handler.close()
    root.handlers, root.level = prev_handlers, prev_level

    row['log_file'] = log_path
    row['log'] = relay.getvalue()
    return row


def setup_logging(log_file):
    """Send the run log to both stdout and a file under the results folder."""
    formatter = logging.Formatter(FILE_LOG_FORMAT, datefmt=LOG_DATEFMT)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers = [stream_handler, file_handler]
    root.setLevel(logging.INFO)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mode', required=True, choices=['test', 'production'],
                        help="'test' builds the nine notebook sessions, 'production' builds every session in the asset CSVs")
    parser.add_argument('--params', default=DEFAULT_PARAMS_PATH,
                        help='JSON file holding build_one and parallel parameters (created with defaults if missing)')
    parser.add_argument('--results-dir', default=DEFAULT_RESULTS_DIR,
                        help='where the NWB stores, logs, params copy and summary go')
    parser.add_argument('--n-jobs', type=int, default=None,
                        help='override parallel.n_jobs from the params file (memory-bound, not CPU-bound)')
    parser.add_argument('--add-metadata', action=argparse.BooleanOptionalAction, default=None,
                        help='override build_one.add_metadata from the params file: bundle the raw AIND metadata JSON into each NWB')
    parser.add_argument('--backend', choices=['zarr', 'hdf5'], default=None,
                        help="override build_one.backend from the params file: 'zarr' writes <session>_combined.nwb.zarr stores, 'hdf5' writes <session>_combined.nwb files")
    parser.add_argument('--fail-on-error', action='store_true',
                        help='exit nonzero if any session failed, instead of only reporting it')
    args = parser.parse_args(argv)

    results_dir = os.path.abspath(args.results_dir)
    nwb_dir = os.path.join(results_dir, 'nwb')
    log_dir = os.path.join(results_dir, 'logs')
    os.makedirs(nwb_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    setup_logging(os.path.join(results_dir, 'build_all_nwb.log'))
    run_started = datetime.now()
    logging.info(f"mode={args.mode} results_dir={results_dir}")

    params = load_params(args.params)
    # Record what this run was actually given, defaults filled in and the CLI overrides applied.
    if args.n_jobs is not None:
        params['parallel']['n_jobs'] = args.n_jobs
    if args.add_metadata is not None:
        params['build_one']['add_metadata'] = args.add_metadata
    if args.backend is not None:
        params['build_one']['backend'] = args.backend
    params_copy = os.path.join(results_dir, 'nwb_build_params.json')
    with open(params_copy, 'w') as f:
        json.dump(params, f, indent=2)
        f.write('\n')
    logging.info(f"params from {args.params} (copy written to {params_copy}): {params}")

    sessions = gather_sessions(args.mode)
    logging.info(f"{len(sessions)} session(s) to build")

    # n_jobs is memory-bound (each worker holds a whole NWB in memory), not CPU-bound.
    # return_as='generator' streams rows back as they finish, so each worker's log
    # is printed while the rest are still building instead of only at the end.
    parallel = Parallel(return_as='generator', **params['parallel'])
    rows = []
    for i, row in enumerate(parallel(delayed(build_one)(s, nwb_dir, log_dir, **params['build_one']) for s in sessions), 1):
        for line in row.pop('log', '').rstrip().splitlines():
            logging.info(f"[{row['session']}] {line}")
        status = 'FAILED' if row['error'] else 'ok'
        logging.info(f"({i}/{len(sessions)}) {row['session']} {status}")
        rows.append(row)

    # one row per session, one column per modality field: the modalities_df from the
    # "Test" section of test_nwb.ipynb, with this run's bookkeeping columns appended.
    modalities_df = pd.DataFrame(rows)
    modalities_df['mode'] = args.mode
    lead_cols = ['session', 'error', 'error_msg', 'nwb_path']
    run_cols = ['mode', 'duration_s', 'log_file']
    modalities_df = modalities_df[
        [c for c in lead_cols if c in modalities_df.columns]
        + [c for c in modalities_df.columns if c not in lead_cols + run_cols]
        + [c for c in run_cols if c in modalities_df.columns]
    ]
    modalities_path = os.path.join(results_dir, 'modalities_df.csv')
    modalities_df.to_csv(modalities_path, index=False)

    n_failed = int(modalities_df['error'].sum()) if len(modalities_df) else 0
    elapsed = (datetime.now() - run_started).total_seconds()
    logging.info(f"{len(modalities_df) - n_failed}/{len(modalities_df)} sessions written in {elapsed / 60:.1f} min")
    for row in rows:
        if row['error']:
            logging.error(f"  x {row['session']}: {row['error_msg']}")
    logging.info(f"modalities table written to {modalities_path}")

    if n_failed and args.fail_on_error:
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())

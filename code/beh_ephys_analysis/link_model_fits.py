"""Build the model-fit layout that session_dirs() expects from an attached data asset.

session_dirs() resolves model fits as:

    data/{aniID}_model_stan/{model_name}/{session_id}_session_model_dv.csv
    data/{aniID}_model_stan/{aniID}_session_data.csv

Assets exported from the fitting capsule instead arrive as

    data/{asset}/{aniID}/{model_name}/{session_id}_session_model_dv.csv
    data/{asset}/{aniID}/{model_name}/ani_session_data.csv

(or with the {asset} level absent, when one asset is attached per animal). This
script finds those directories anywhere under data/ and symlinks them into the
expected names. data/ itself is writable xfs; only the assets mount read-only,
so the shim lives next to them rather than inside them.

Run once after attaching or re-attaching the asset -- symlinks under data/ do not
survive a capsule reset. Not safe for postInstall, which runs before data mounts.

    python code/beh_ephys_analysis/link_model_fits.py
    python code/beh_ephys_analysis/link_model_fits.py --check
"""

import argparse
import ast
import os
import re
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.capsule_migration import capsule_directories

CURATION_NAMES = ['ani_session_data.csv', '{aniID}_session_data.csv']


def is_aniID(name):
    """Animal IDs are six digits on the AIND rigs, ZS-prefixed on the older ones."""
    return bool(re.fullmatch(r'\d{6}', name)) or name.startswith('ZS')


def find_model_dirs(data_dir):
    """Locate every {aniID}/{model_name} dir holding *_session_model_dv.csv files.

    Returns a list of (aniID, model_name, model_dir) tuples. Directories already
    living under a correctly named {aniID}_model_stan parent are skipped.
    """
    found = []
    for root, dirs, files in os.walk(data_dir, followlinks=True):
        # never descend into the big raw/sorted session assets
        dirs[:] = [d for d in dirs if not (d.endswith('_raw_data') or d.endswith('_sorted')
                                           or d.endswith('_sorted_curated'))]
        if not any(f.endswith('_session_model_dv.csv') for f in files):
            continue
        parent = os.path.dirname(root)
        aniID = os.path.basename(parent)
        if not is_aniID(aniID):
            print(f'  ! skipping {root}: parent {aniID!r} is not an animal ID')
            continue
        if os.path.basename(os.path.dirname(parent)) == f'{aniID}_model_stan':
            continue  # already in place
        found.append((aniID, os.path.basename(root), root))
    return found


def find_curation_file(aniID, model_dir):
    """The session-cut table sits either beside the dv csvs or one level up."""
    parent = os.path.dirname(model_dir)
    for directory in (model_dir, parent):
        for name in CURATION_NAMES:
            candidate = os.path.join(directory, name.format(aniID=aniID))
            if os.path.exists(candidate):
                return candidate
    return None


def link(src, dst, force):
    """Symlink src -> dst, leaving real (non-symlink) files at dst untouched."""
    if os.path.islink(dst):
        if os.readlink(dst) == src and not force:
            return 'ok'
        os.unlink(dst)
    elif os.path.exists(dst):
        print(f'  ! {dst} exists and is not a symlink, leaving it alone')
        return 'conflict'
    os.symlink(src, dst)
    return 'linked'


def build_links(data_dir, force=False):
    model_dirs = find_model_dirs(data_dir)
    if not model_dirs:
        print(f'No *_session_model_dv.csv files found under {data_dir}.')
        print('Attach the model-fit data asset first.')
        return []

    linked = []
    for aniID, model_name, model_dir in model_dirs:
        target_root = os.path.join(data_dir, f'{aniID}_model_stan')
        os.makedirs(target_root, exist_ok=True)

        n_dv = len([f for f in os.listdir(model_dir) if f.endswith('_session_model_dv.csv')])
        status = link(model_dir, os.path.join(target_root, model_name), force)
        print(f'{aniID}/{model_name}: {status} ({n_dv} sessions)')

        curation = find_curation_file(aniID, model_dir)
        if curation is None:
            print(f'  ! no session-cut table found for {aniID}; '
                  'load_model_dv() will raise when it reads session_curation_file')
        else:
            status = link(curation, os.path.join(target_root, f'{aniID}_session_data.csv'), force)
            print(f'  {os.path.basename(curation)} -> {aniID}_session_data.csv: {status}')

        linked.append((aniID, model_name, target_root))
    return linked


def check(data_dir):
    """Validate the linked fits against what load_model_dv/makeSessionDF require."""
    roots = sorted(d for d in os.listdir(data_dir) if d.endswith('_model_stan'))
    if not roots:
        print('No *_model_stan directories to check. Run without --check first.')
        return 1

    problems = 0
    for root in roots:
        aniID = root[:-len('_model_stan')]
        root_path = os.path.join(data_dir, root)
        print(f'\n{aniID}')

        curation_path = os.path.join(root_path, f'{aniID}_session_data.csv')
        if not os.path.exists(curation_path):
            print(f'  MISSING {aniID}_session_data.csv')
            problems += 1
        else:
            curation = pd.read_csv(curation_path)
            missing = [c for c in ('session_id', 'session_cut') if c not in curation.columns]
            if missing:
                print(f'  session_data.csv missing columns: {missing}')
                print(f'    has: {list(curation.columns)}')
                problems += 1
            else:
                # session_cut is ast.literal_eval'd then used as tblTrials.iloc[a:b],
                # so both entries must be plain ints -- nan/None/floats all fail.
                for _, row in curation.iterrows():
                    raw = row['session_cut']
                    try:
                        cut = ast.literal_eval(str(raw))
                    except (ValueError, SyntaxError) as err:
                        print(f'  session_cut {raw!r} ({row["session_id"]}) '
                              f'fails literal_eval: {err}')
                        problems += 1
                        continue
                    if not (isinstance(cut, (list, tuple)) and len(cut) == 2
                            and all(isinstance(v, int) for v in cut)):
                        print(f'  session_cut {raw!r} ({row["session_id"]}) is not two ints; '
                              'use e.g. "[0, 350]", and an int above the trial count for '
                              '"to the end"')
                        problems += 1

        for model_name in sorted(os.listdir(root_path)):
            model_dir = os.path.join(root_path, model_name)
            if not os.path.isdir(model_dir):
                continue
            dv_files = sorted(f for f in os.listdir(model_dir)
                              if f.endswith('_session_model_dv.csv'))
            print(f'  {model_name}: {len(dv_files)} sessions')
            for dv_file in dv_files:
                dv = pd.read_csv(os.path.join(model_dir, dv_file), index_col=0)
                missing = [c for c in ('pChoice', 'Q_l', 'Q_r') if c not in dv.columns]
                if missing:
                    print(f'    {dv_file}: missing {missing}')
                    problems += 1
                if not dv.index.equals(pd.RangeIndex(len(dv))):
                    # merged positionally on the index against the responded-trial
                    # table, with how='inner' -- a bad index silently drops rows
                    print(f'    {dv_file}: index is not 0..{len(dv) - 1}')
                    problems += 1

    print(f'\n{problems} problem(s) found.' if problems else '\nAll checks passed.')
    return 1 if problems else 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--check', action='store_true',
                        help='validate already-linked fits instead of creating links')
    parser.add_argument('--force', action='store_true',
                        help='replace existing symlinks')
    args = parser.parse_args()

    data_dir = str(capsule_directories()['data_dir'])
    if args.check:
        sys.exit(check(data_dir))
    build_links(data_dir, force=args.force)
    print('\nNow run with --check to validate the fits.')

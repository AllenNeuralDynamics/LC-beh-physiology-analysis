"""
Organize manuscript figures and CSVs into per-figure subfolders.

Creates subfolders named Figure4, Figure5, ..., FigureE15 under
capsule_dirs["manuscript_fig_dir"], then copies all files whose names start
with the corresponding figure prefix from every existing subfolder into the
appropriate destination folder.
"""

import json
import os
import re
import shutil
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.normpath(os.path.join(_here, '..', 'beh_ephys_analysis'))
if _root not in sys.path:
    sys.path.insert(0, _root)

from utils.capsule_migration import capsule_directories

PANEL_MAP_PATH = os.path.join(
    _here, '..', 'beh_ephys_analysis',
    'session_combine', 'manuscript_figures',
    'manuscript_figure_panel_map.json'
)


def main():
    capsule_dirs = capsule_directories()
    fig_dir = capsule_dirs['manuscript_fig_dir']

    with open(PANEL_MAP_PATH) as f:
        panel_map = json.load(f)

    figure_names = list(panel_map.keys())  # e.g. ['Figure4', 'Figure5', ..., 'FigureE15']

    # Create destination subfolders
    dest_dirs = {}
    for fig_name in figure_names:
        dest = os.path.join(fig_dir, fig_name)
        os.makedirs(dest, exist_ok=True)
        dest_dirs[fig_name] = dest
    print(f'Created {len(dest_dirs)} figure folders under {fig_dir}')

    # Walk existing subfolders and copy matching files
    copied = {fig: 0 for fig in figure_names}
    source_subdirs = [
        d for d in os.listdir(fig_dir)
        if os.path.isdir(os.path.join(fig_dir, d)) and d not in dest_dirs
    ]

    # Match any FigureN or FigureEN prefix in a filename
    fig_prefix_re = re.compile(r'^(Figure(?:E)?\d+)')

    unlisted_figures = set()
    for subdir_name in sorted(source_subdirs):
        subdir_path = os.path.join(fig_dir, subdir_name)
        # collect files from this level and one level deeper
        search_paths = []
        for entry in os.listdir(subdir_path):
            entry_path = os.path.join(subdir_path, entry)
            if os.path.isfile(entry_path):
                search_paths.append(entry_path)
            elif os.path.isdir(entry_path):
                for fname2 in os.listdir(entry_path):
                    fp = os.path.join(entry_path, fname2)
                    if os.path.isfile(fp):
                        search_paths.append(fp)
        for src in search_paths:
            fname = os.path.basename(src)
            m = fig_prefix_re.match(fname)
            if not m:
                continue
            fig_name = m.group(1)
            if fig_name not in dest_dirs:
                # figure prefix not in panel map — still transfer, create folder on the fly
                dest = os.path.join(fig_dir, fig_name)
                os.makedirs(dest, exist_ok=True)
                dest_dirs[fig_name] = dest
                copied[fig_name] = 0
                unlisted_figures.add(fig_name)
            dst = os.path.join(dest_dirs[fig_name], fname)
            shutil.copy2(src, dst)
            copied[fig_name] += 1

    for fig_name, count in sorted(copied.items()):
        tag = ' [not in panel map]' if fig_name in unlisted_figures else ''
        print(f'  {fig_name}: {count} files copied{tag}')

    if unlisted_figures:
        print(f'\nFigures found on disk but missing from panel map: {sorted(unlisted_figures)}')


if __name__ == '__main__':
    main()

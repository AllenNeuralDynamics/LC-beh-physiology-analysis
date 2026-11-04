# %% Summary pages for all tagged units and their matched non-tagged controls, bound into one
#    flattened PDF.
#
#   1. pages:  one PDF per unit in PAGES_DIR (sessions in parallel; existing pages are skipped,
#              so an interrupted run resumes). Each page also saves its PSTHs to PSTH_DIR.
#   2. flatten: each page rendered to a JPEG at DPI (pdftoppm), so the combined file opens fast
#   3. cover:  unit_summary_DRN_cover.py pages (tagged vs non-tagged), rebuilt from the saved
#              PSTHs after the pages are made, and bound as the first two pages
#   4. bind:   one PDF per --order: 'paired' puts each tagged unit right before its matched
#              non-tagged control; 'grouped' puts all tagged units first, then all controls
#              (in the same order as their tagged units). Re-binding in another order only
#              reads the flattened pages, so it takes a minute or two (--skip-pages).
#
#     python unit_summary_DRN_run_all.py [--jobs 6] [--overwrite] [--sessions S ...]
#                                        [--order paired grouped] [--skip-pages]

import os
import sys
import time
import argparse
import subprocess
import glob

curr_path = os.path.dirname(os.path.abspath(__file__))
if curr_path not in sys.path:
    sys.path.insert(0, curr_path)

import pandas as pd
from joblib import Parallel, delayed
from PIL import Image

from unit_summary_DRN import (OUT_DIR, NONTAGGED_SELECTION, TAGGED_CRITERION, criteria_passed,
                              plot_unit_summary_session)
from unit_summary_DRN_cover import make_cover

PAGES_DIR = os.path.join(OUT_DIR, 'pages')
FLAT_DIR = os.path.join(OUT_DIR, 'pages_flat')
COMBINED_PDF = os.path.join(OUT_DIR, 'unit_summary_DRN_all_{order}.pdf')
LOG = os.path.join(OUT_DIR, 'run_all.log')
DPI = 100       # 31 x 24 in pages -> 3100 x 2400 px; readable on screen, ~1 MB per page
JPEG_QUALITY = 85


def page_name(session, unit_id):
    return f'{session}_unit_{unit_id}_summary'


def unit_list(order='paired'):
    """Tagged units with their matched non-tagged controls: 'paired' interleaves each tagged
    unit with its control, 'grouped' lists all tagged units, then all controls."""
    tagged = sorted(k for k, v in criteria_passed().items() if TAGGED_CRITERION in v)
    sel = pd.read_csv(NONTAGGED_SELECTION).sort_values('distance_um')
    controls = {}  # a tagged unit can have two controls, nearest first
    for r in sel.itertuples():
        controls.setdefault((r.session, int(r.matched_tagged_unit)), []).append(int(r.unit_id))
    pages = []
    for session, unit_id in tagged:
        pages.append((session, unit_id, 'tagged'))
        for control in controls.get((session, unit_id), []):
            pages.append((session, control, 'non-tagged'))
    if order == 'grouped':
        pages = [p for p in pages if p[2] == 'tagged'] + [p for p in pages if p[2] != 'tagged']
    return pages


def log(msg):
    with open(LOG, 'a') as f:
        f.write(f'{time.strftime("%H:%M:%S")} {msg}\n')
    print(msg, flush=True)


def run_session(session, units):
    t0 = time.time()
    try:
        plot_unit_summary_session(session, units=units, out_dir=PAGES_DIR)
        log(f'{session}: {len(units)} pages in {time.time() - t0:.0f} s')
    except Exception as e:
        log(f'{session}: FAILED after {time.time() - t0:.0f} s, {type(e).__name__}: {e}')


def flatten(name):
    src = os.path.join(PAGES_DIR, f'{name}.pdf')
    dst = os.path.join(FLAT_DIR, name)
    if os.path.exists(src) and (not os.path.exists(f'{dst}.jpg') or os.path.getmtime(f'{dst}.jpg') < os.path.getmtime(src)):
        subprocess.run(['pdftoppm', '-jpeg', '-jpegopt', f'quality={JPEG_QUALITY}', '-r', str(DPI),
                        '-singlefile', src, dst], check=True)
    return f'{dst}.jpg' if os.path.exists(f'{dst}.jpg') else None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobs', type=int, default=6)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--sessions', nargs='*', default=None)
    parser.add_argument('--skip-pages', action='store_true', help='only flatten and bind existing pages')
    parser.add_argument('--order', nargs='+', default=['paired'], choices=['paired', 'grouped'])
    args = parser.parse_args()

    order = unit_list('paired')  # page set; the order only matters when binding
    if args.sessions:
        order = [u for u in order if u[0] in args.sessions]
    log(f'{len(order)} pages: {sum(k == "tagged" for _, _, k in order)} tagged, '
        f'{sum(k != "tagged" for _, _, k in order)} non-tagged')

    os.makedirs(PAGES_DIR, exist_ok=True)
    if not args.skip_pages:
        todo = {}
        for session, unit_id, _ in order:
            if args.overwrite or not os.path.exists(os.path.join(PAGES_DIR, f'{page_name(session, unit_id)}.pdf')):
                todo.setdefault(session, []).append(unit_id)
        log(f'pages to make: {sum(map(len, todo.values()))} in {len(todo)} sessions, {args.jobs} jobs')
        t0 = time.time()
        # largest sessions first so they don't finish last
        Parallel(n_jobs=args.jobs)(delayed(run_session)(s, u)
                                   for s, u in sorted(todo.items(), key=lambda x: -len(x[1])))
        log(f'pages done in {(time.time() - t0) / 60:.1f} min')

    t0 = time.time()
    os.makedirs(FLAT_DIR, exist_ok=True)
    names = [page_name(s, u) for s, u, _ in order]
    jpgs = Parallel(n_jobs=args.jobs * 2)(delayed(flatten)(n) for n in names)
    missing = [n for n, j in zip(names, jpgs) if j is None]
    jpgs = [j for j in jpgs if j is not None]
    log(f'flattened {len(jpgs)} pages in {time.time() - t0:.0f} s; missing pages: {len(missing)}')
    for n in missing:
        log(f'  missing: {n}')

    cover_pdf = make_cover()
    for old in glob.glob(os.path.join(FLAT_DIR, '00_cover*.jpg')):
        os.remove(old)
    subprocess.run(['pdftoppm', '-jpeg', '-jpegopt', f'quality={JPEG_QUALITY}', '-r', str(DPI),
                    cover_pdf, os.path.join(FLAT_DIR, '00_cover')], check=True)
    cover_jpgs = sorted(glob.glob(os.path.join(FLAT_DIR, '00_cover*.jpg')))
    log(f'cover pages: {cover_pdf} ({len(cover_jpgs)} pages)')

    have = set(jpgs)
    for bind_order in args.order:
        t0 = time.time()
        pages = unit_list(bind_order)
        if args.sessions:
            pages = [u for u in pages if u[0] in args.sessions]
        files = [f for f in (os.path.join(FLAT_DIR, f'{page_name(s, u)}.jpg') for s, u, _ in pages) if f in have]
        files = cover_jpgs + files
        if not files:
            continue
        out = COMBINED_PDF.format(order=bind_order)
        first, rest = Image.open(files[0]), (Image.open(j) for j in files[1:])
        first.save(out, save_all=True, append_images=rest, resolution=DPI)
        log(f'bound {len(files)} pages ({bind_order}) into {out} '
            f'({os.path.getsize(out) / 1e6:.0f} MB) in {time.time() - t0:.0f} s')

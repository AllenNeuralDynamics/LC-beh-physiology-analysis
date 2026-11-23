# %% Demo: cover pages with task PSTHs smoothed by the causal PSP kernel instead of the
#    centered 0.2 s window, so post-event spikes cannot appear before the go cue / response.
#
# The saved per-unit PSTH files keep each condition's alignment times, not spike times, so
# spike times are reloaded from the unit tables and limited to each unit's drift cut (the
# 'ephys_cut' saved with its page). Task conditions are recomputed with the kernel used for the
# spontaneous-lick panels (tau_rise 10 ms, tau_decay 80 ms, 50 ms steps), written to a separate
# PSTH folder, and the cover is built from it. The saved PSTHs and the main cover are untouched.
#
#     python unit_summary_DRN_cover_causal_demo.py

import os
import sys
import glob
import pickle

curr_path = os.path.dirname(os.path.abspath(__file__))
if curr_path not in sys.path:
    sys.path.insert(0, curr_path)

import numpy as np
from joblib import Parallel, delayed

from unit_summary_DRN import OUT_DIR, PSTH_DIR, causal_kernel_matrix as _kernel
from unit_summary_DRN_cover import make_cover, COVER_PDF
from utils.beh_functions import get_unit_tbl
from utils.ephys_functions import get_spike_matrix_filter

TAU_RISE, TAU_DECAY, STEP = 0.01, 0.08, 0.05
DEMO_PSTH_DIR = os.path.join(OUT_DIR, 'psth_causal_demo')
DEMO_COVER = COVER_PDF.replace('.pdf', '_causal_demo.pdf')
SMOOTHING = (f'causal PSP kernel, rise {1000 * TAU_RISE:.0f} ms, decay {1000 * TAU_DECAY:.0f} ms, '
             f'{1000 * STEP:.0f} ms steps; only spikes before each time point count')


def causal_kernel_matrix(spike_times, align_times, tb, tf):
    """The task-PSTH kernel of unit_summary_DRN (same values as get_spike_matrix_filter)."""
    return _kernel(spike_times, align_times, tb, tf, STEP, TAU_RISE, TAU_DECAY)


def session_demo(session, files):
    tbl = get_unit_tbl(session, 'raw', summary=False)
    for f in files:
        with open(f, 'rb') as fh:
            d = pickle.load(fh)
        spikes = np.asarray(tbl[tbl['unit_id'] == d['unit_id']]['spike_times'].values[0], float)
        lo, hi = d['ephys_cut']
        if lo is not None:
            spikes = spikes[spikes >= lo]
        if hi is not None:
            spikes = spikes[spikes <= hi]
        tb, tf = d['window']
        for key, p in d['psth'].items():
            if key.startswith('spont') or p['n_trials'] == 0:
                continue  # spontaneous licks already use the kernel
            M, t = causal_kernel_matrix(spikes, p['align_times'], tb, tf)
            n = np.sum(np.isfinite(M), axis=0)
            p.update({'bin_times': t, 'matrix': M.astype(np.float32), 'mean': M.mean(axis=0),
                      'sem': M.std(axis=0) / np.sqrt(np.maximum(n, 1)), 'kernel': 'psp',
                      'tau_rise': TAU_RISE, 'tau_decay': TAU_DECAY, 'stepSize': STEP})
        with open(os.path.join(DEMO_PSTH_DIR, os.path.basename(f)), 'wb') as fh:
            pickle.dump(d, fh)
    return len(files)


def check_against_original():
    """The vectorized kernel matches get_spike_matrix_filter on one unit and condition."""
    f = sorted(glob.glob(os.path.join(PSTH_DIR, '*_psth.pkl')))[0]
    with open(f, 'rb') as fh:
        d = pickle.load(fh)
    spikes = np.asarray(get_unit_tbl(d['session'], 'raw', summary=False)
                        .query('unit_id == @d["unit_id"]')['spike_times'].values[0], float)
    times = d['psth']['go_cue_all']['align_times'][:20]
    ref, t_ref = get_spike_matrix_filter(spikes, times, *d['window'], tau_decay=TAU_DECAY, tau_rise=TAU_RISE,
                                         step_size=STEP)
    new, t_new = causal_kernel_matrix(spikes, times, *d['window'])
    print(f'check vs get_spike_matrix_filter: max |diff| {np.max(np.abs(ref - new)):.2e} Hz, '
          f'same bins {np.allclose(t_ref, t_new)}')


if __name__ == '__main__':
    check_against_original()
    os.makedirs(DEMO_PSTH_DIR, exist_ok=True)
    by_session = {}
    for f in glob.glob(os.path.join(PSTH_DIR, '*_psth.pkl')):
        by_session.setdefault(os.path.basename(f).split('_unit_')[0], []).append(f)
    n = Parallel(n_jobs=8)(delayed(session_demo)(s, fs) for s, fs in by_session.items())
    print(f'recomputed {sum(n)} units into {DEMO_PSTH_DIR}')
    make_cover(DEMO_COVER, psth_dir=DEMO_PSTH_DIR, smoothing=SMOOTHING)

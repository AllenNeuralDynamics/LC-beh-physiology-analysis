"""Run the hand-picked rhythmicity units, saving after each session."""
import matplotlib
matplotlib.use('Agg')
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append('/root/capsule/code/beh_ephys_analysis')
from rhythmicity_DRN import (UNITS_OF_INTEREST, EXPECTED_BANDS, diagnose_session,
                             add_fdr, SUMMARY_COLS)

warnings.filterwarnings('ignore')
OUT = '/root/capsule/scratch/results/uoi_rhythmicity.pkl'
N_SURR = int(os.environ.get('N_SURR', 200))

os.makedirs(os.path.dirname(OUT), exist_ok=True)
done = pd.read_pickle(OUT) if os.path.exists(OUT) else None
seen = set(zip(done['session'], done['unit_id'])) if done is not None else set()
parts = [done] if done is not None else []

for i, (session, unit_bands) in enumerate(UNITS_OF_INTEREST.items(), 1):
    todo = {u: b for u, b in unit_bands.items() if (session, u) not in seen}
    if not todo:
        print(f'[{i}/{len(UNITS_OF_INTEREST)}] {session}: cached', flush=True)
        continue
    t0 = time.time()
    print(f'[{i}/{len(UNITS_OF_INTEREST)}] {session} units {list(todo)}', flush=True)
    try:
        m = diagnose_session(session, 'raw', unit_ids=list(todo), expected=todo,
                             n_surrogates=N_SURR, max_units=None, plot=True, save=True)
        if m is not None:
            parts.append(m)
            pd.concat(parts, ignore_index=True).to_pickle(OUT)
        print(f'    done in {time.time() - t0:.0f}s', flush=True)
    except Exception as e:
        print(f'    ERROR {type(e).__name__}: {e}', flush=True)
        plt.close('all')

metrics = pd.concat(parts, ignore_index=True)
lo = metrics['expected'].map(lambda b: EXPECTED_BANDS.get(b, (np.nan, np.nan))[0])
hi = metrics['expected'].map(lambda b: EXPECTED_BANDS.get(b, (np.nan, np.nan))[1])
metrics['agrees'] = (metrics['peak_freq'] >= lo) & (metrics['peak_freq'] <= hi)
metrics = add_fdr(metrics)
metrics.to_pickle(OUT)
metrics.to_csv(OUT.replace('.pkl', '.csv'), index=False)
print(f'\nwrote {OUT} ({len(metrics)} units)', flush=True)

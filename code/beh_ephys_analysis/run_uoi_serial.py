"""
Add the drift-detrended ISI serial-correlation statistics to the hand-picked units.

Separate from run_uoi.py because these need no surrogates: recomputing them costs seconds
per unit, against ~40 s for the autocorrelogram pass, so there is no reason to redo that.
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd

sys.path.append('/root/capsule/code/beh_ephys_analysis')
from rhythmicity_DRN import (UNITS_OF_INTEREST, spontaneous_window, unit_window)
from utils.beh_functions import get_unit_tbl
from utils.rhythmicity import isi_stats

warnings.filterwarnings('ignore')
IN = '/root/capsule/scratch/results/uoi_rhythmicity.pkl'
COLS = ['isi_serial_1', 'isi_serial_2', 'isi_serial_3', 'isi_serial_z',
        'isi_serial_1_det', 'isi_serial_2_det', 'isi_serial_3_det', 'isi_serial_z_det']

metrics = pd.read_pickle(IN)
rows = []
for session, unit_bands in UNITS_OF_INTEREST.items():
    window = spontaneous_window(session, 'raw')
    if window is None:
        continue
    unit_tbl = get_unit_tbl(session, 'raw', summary=False)
    for unit_id in unit_bands:
        row = unit_tbl[unit_tbl['unit_id'] == unit_id]
        if len(row) == 0:
            continue
        start, end = unit_window(session, unit_id, 'raw', *window)
        s = isi_stats(row['spike_times'].values[0], start, end)
        rows.append({'session': session, 'unit_id': unit_id,
                     **{c: s[c] for c in COLS}})
    print(f'{session}: {len(unit_bands)} units', flush=True)

serial = pd.DataFrame(rows)
metrics = metrics.drop(columns=[c for c in COLS if c in metrics.columns])
metrics = metrics.merge(serial, on=['session', 'unit_id'], how='left')
metrics.to_pickle(IN)
metrics.to_csv(IN.replace('.pkl', '.csv'), index=False)
print(f'\nmerged serial stats into {IN} ({len(metrics)} units)')

# %% Pick one non-tagged control unit per tagged DRN unit (option C: nearest neighbour).
#
# For each tagged unit (basic_ephys_all_DRN), take the closest non-tagged unit on the same
# insertion in CCF space, without reuse. Tagged units closer to a candidate are matched first,
# so a crowded insertion cannot take the best candidates. Candidates never respond to light
# (not tagged, not an opto candidate, not opto_pass) and are chosen by distance only, not by
# quality or activity.
#
#   tier 1: DV within the session's tagged range +/- 100 um, default_qc, ISI <= 0.1
#   tier 2: for tagged units left unmatched by tier 1: DV range +/- 200 um, default_qc
#
# Reads the per-unit flags written by unit_summary_DRN_population_options.py.
#
#     python unit_summary_DRN_select_nontagged.py

import os

import numpy as np
import pandas as pd

FLAGS_CSV = '/root/capsule/scratch/combined/unit_summary_DRN/population_unit_flags.csv'
OUT_CSV = '/root/capsule/scratch/combined/unit_summary_DRN/nontagged_matched_selection.csv'
TIERS = [  # (name, DV margin in mm, max ISI violations ratio)
    ('tier1', 0.1, 0.1),
    ('tier2', 0.2, None),
]
XYZ = ['ML', 'AP', 'DV']


def candidate_pool(m, margin, isi_max):
    rng = m[m['tagged']].groupby('session')['DV'].agg(['min', 'max'])
    in_range = ((m['DV'] >= m['session'].map(rng['min']) - margin)
                & (m['DV'] <= m['session'].map(rng['max']) + margin)).fillna(False)
    ok = (~m['decoder_label'].isin(['noise', 'artifact']) & m['default_qc'].astype(bool)
          & ~m['tagged'] & ~m['opto_candidate'] & ~m['opto_pass'].astype(bool) & in_range)
    if isi_max is not None:
        ok &= m['isi_violations_ratio'] <= isi_max
    return m[ok]


def match_session(tagged, pool, used):
    """Greedy nearest-neighbour matching without replacement, closest pairs first."""
    pool = pool[~pool.index.isin(used)]
    if len(tagged) == 0 or len(pool) == 0:
        return []
    D = np.linalg.norm(tagged[XYZ].to_numpy()[:, None] - pool[XYZ].to_numpy()[None], axis=2)
    pairs, taken = [], set()
    for i in np.argsort(D.min(axis=1)):
        for j in np.argsort(D[i]):
            if j not in taken:
                taken.add(j)
                pairs.append((tagged.index[i], pool.index[j], D[i, j]))
                break
    return pairs


if __name__ == '__main__':
    m = pd.read_csv(FLAGS_CSV)
    tagged_all = m[m['tagged']]
    rows, used, matched = [], set(), set()
    for tier, margin, isi_max in TIERS:
        pool = candidate_pool(m, margin, isi_max)
        for session, tagged in tagged_all[~tagged_all.index.isin(matched)].groupby('session'):
            for t_idx, c_idx, dist in match_session(tagged, pool[pool['session'] == session], used):
                used.add(c_idx)
                matched.add(t_idx)
                rows.append({'session': session, 'unit_id': int(m.at[c_idx, 'unit_id']),
                             'matched_tagged_unit': int(m.at[t_idx, 'unit_id']),
                             'distance_um': round(1000 * dist, 1), 'tier': tier,
                             'decoder_label': m.at[c_idx, 'decoder_label'],
                             'isi_violations_ratio': m.at[c_idx, 'isi_violations_ratio']})
    sel = pd.DataFrame(rows).sort_values(['session', 'unit_id']).reset_index(drop=True)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    sel.to_csv(OUT_CSV, index=False)

    print(f'{len(sel)} non-tagged units matched to {len(tagged_all)} tagged units '
          f'({sel["session"].nunique()} sessions)')
    print(sel.groupby('tier')['distance_um'].describe()[['count', '50%', 'max']].rename(columns={'50%': 'median'}))
    unmatched = tagged_all[~tagged_all.index.isin(matched)]
    print(f'unmatched tagged units: {len(unmatched)} in {unmatched["session"].nunique()} sessions')
    print(f'wrote {OUT_CSV}')

# %%
"""
How many opto-tagged units the tagging criteria keep when the ISI-violations bound is
read off the laser-free part of the session instead of the whole recording.

isi_nonopto_report.py answers the upstream question (does dropping the opto blocks change
the ISI metric, and does it flip default_qc). This script answers the one that decides
whether the change is worth making: how many *opto units* -- rows that survive the full
tagging criteria, not just QC -- are gained and lost, and at what ISI bound.

Criteria: session_combine/metrics/basic_ephys_low_DRN_nonopto.json, applied through
utils.combine_tools.apply_qc, so the counts are the ones the figure-prep scripts would
get. Its whole-recording twin (basic_ephys_low_DRN.json) is identical apart from the two
ISI terms, so swapping the ISI source is the only difference between the two arms and
every flip is attributable to it.

Three things this reports that a single apply_qc call does not:

  * gained/lost broken out per unit and per session, not just two totals. A flip that
    lives in one session is a session effect, not a QC effect.
  * a sweep of the ISI upper bound. Both criteria files pair an explicit bound with a
    stored boolean QC flag that has its *own* bound baked in -- 0.5 for qc_pass, 0.4 for
    qc_pass_nonopto -- so as written the nonopto file is capped at 0.4 and sweeping its
    explicit bound past that does nothing. The sweep therefore rebuilds both flags at the
    swept bound (label not noise/artifact AND ISI <= bound) so the two arms differ only in
    which ISI column they read. The as-written numbers are reported too, for reference.
  * what happens to units with no spikes outside the laser, which have no laser-free ISI
    to be judged on. Reported both ways: falling back on the whole-recording ratio (what
    utils.isi_qc.default_qc does) and failing them.

Data: the combined unit table plus the laser-free ISI columns. The combined table on disk
predates those columns, so they are merged in from the per-unit summary add_isi_nonopto.py
writes; the merge is checked against the table's own isi_violations before anything is
computed. (Regenerating the table would bake them in, but
make_combined_unit_tbl_DRN.py currently calls get_unit_tbl with a ccf_suffix argument it
does not take, so every session fails.)

Text report to stdout, tables and one figure to the manuscript fig-prep dir.
"""
# %%
import contextlib
import io
import json
import os
import sys
sys.path.append('/root/capsule/code/beh_ephys_analysis')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from utils.capsule_migration import capsule_directories
from utils.combine_tools import apply_qc

# ---- palette: the dataviz reference instance, used verbatim ----
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_2 = '#52514e'
INK_MUTED = '#8a8983'
GRID = '#e6e5e1'
# identity of the two ISI definitions: categorical slots 2 and 3
WHOLE = '#eb6834'
FREE = '#1baf7a'
# polarity of a flip: the diverging pair
GAINED = '#2a78d6'
LOST = '#e34948'

METRICS_DIR = '/root/capsule/code/beh_ephys_analysis/session_combine/metrics'
CRITERIA_NONOPTO = 'basic_ephys_low_DRN_nonopto'
CRITERIA_WHOLE = 'basic_ephys_low_DRN'
ISI_SUMMARY_CSV = '/root/capsule/scratch/combined/isi_nonopto_units.csv'

# the bounds to sweep; 0.1 / 0.2 / 0.5 are the ones asked for, the rest give the shape
BOUNDS = (0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0)
REPORT_BOUNDS = (0.1, 0.2, 0.5)
NOISE_LABELS = ('noise', 'artifact')

# column names as make_combined_unit_tbl writes them, which is what the criteria files
# key on; the summary CSV uses the raw spikeinterface-style names
SIDECAR_RENAME = {
    'isi_violations_ratio_nonopto': 'isi_violations_nonopto',
    'isi_violations_ratio_opto': 'isi_violations_opto',
    'isi_violations_ratio_window': 'isi_violations_window',
    'default_qc_nonopto': 'qc_pass_nonopto',
}

# the two ISI sources being compared. 'fallback' is the column used where the arm's own
# ISI is NaN; None fails those units instead.
ARMS = {
    'whole': {'isi_col': 'isi_violations', 'fallback': None,
              'label': 'whole recording'},
    'free': {'isi_col': 'isi_violations_nonopto', 'fallback': 'isi_violations',
             'label': 'laser-free'},
    'free_strict': {'isi_col': 'isi_violations_nonopto', 'fallback': None,
                    'label': 'laser-free, no fallback'},
}

capsule_dirs = capsule_directories()
OUT_DIR = os.path.join(str(capsule_dirs['manuscript_fig_prep_dir']), 'isi_nonopto')


# ------------------------------------------------------------------ data


def load_criteria(name):
    with open(os.path.join(METRICS_DIR, f'{name}.json')) as f:
        return json.load(f)


def load_units():
    """
    Combined unit table with the laser-free ISI columns attached.

    Raises if the merged laser-free rows do not line up with the table's own
    isi_violations -- a stale summary CSV would otherwise silently compare two different
    unit sets.
    """
    pkl = os.path.join(str(capsule_dirs['manuscript_fig_prep_dir']),
                       'combined_unit_tbl', 'combined_unit_tbl.pkl')
    units = pd.read_pickle(pkl).reset_index(drop=True)

    sidecar = pd.read_csv(ISI_SUMMARY_CSV)
    sidecar = sidecar.rename(columns=SIDECAR_RENAME)
    keep = ['session', 'unit_id', 'isi_violations_ratio', 'num_spikes_nonopto',
            'num_spikes_window', 'frac_nonopto'] + list(SIDECAR_RENAME.values())
    sidecar = sidecar[[col for col in keep if col in sidecar.columns]]

    units = units.drop(columns=[col for col in sidecar.columns
                                if col not in ('session', 'unit_id')], errors='ignore')
    units = units.merge(sidecar, left_on=['session', 'unit'], right_on=['session', 'unit_id'],
                        how='left')

    missing = units['isi_violations_ratio'].isna()
    if missing.any():
        raise RuntimeError(f'{int(missing.sum())} units have no laser-free ISI row; '
                           f'rerun add_isi_nonopto.py for '
                           f'{sorted(units.loc[missing, "session"].unique())}')
    drift = (units['isi_violations_ratio'] - units['isi_violations']).abs()
    if drift.max() > 1e-6 * max(1.0, units['isi_violations'].max()):
        raise RuntimeError(f'{ISI_SUMMARY_CSV} disagrees with the combined table on '
                           f'isi_violations (max |diff| {drift.max():.3g}); it is stale')
    units = units.drop(columns=['isi_violations_ratio', 'unit_id'])

    units['label_ok'] = ~units['decoder'].isin(NOISE_LABELS)
    # a unit with no spikes left outside the laser has no laser-free ISI to judge it on
    units['has_nonopto'] = units['num_spikes_nonopto'] > 1
    return units


def arm_isi(units, arm):
    """The ISI values one arm judges each unit on, after its NaN fallback."""
    cfg = ARMS[arm]
    isi = pd.to_numeric(units[cfg['isi_col']], errors='coerce')
    if cfg['fallback'] is not None:
        isi = isi.fillna(pd.to_numeric(units[cfg['fallback']], errors='coerce'))
    return isi


# ------------------------------------------------------------------ criteria runs


def other_constraints(criteria):
    """Everything in a criteria file except the two ISI terms."""
    return {col: cfg for col, cfg in criteria.items()
            if not col.startswith('isi_violations') and not col.startswith('qc_pass')}


def run(frame, criteria):
    """
    apply_qc, silently, returning its three masks.

    apply_qc writes the masks onto the frame it is handed and draws a figure of every
    constrained column; neither is wanted here, so it gets a copy and the figure is closed.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        _, labelled, fig, _ = apply_qc(frame.copy(), criteria, plot_all=False)
    plt.close(fig)
    return (labelled['selected'].astype(bool),
            labelled['selected_no_opto'].astype(bool),
            labelled['selected_opto_only'].astype(bool))


def run_arm(units, criteria, arm, bound):
    """
    One arm at one ISI bound, with the QC flag rebuilt at that bound.

    The rebuilt flag is default_qc with the swept bound substituted -- decoder label not
    noise/artifact, and the arm's ISI within bound -- so the arms differ only in which ISI
    column they read, and the bound is not silently capped by a flag computed elsewhere.
    """
    frame = units.copy()
    frame['isi_arm'] = arm_isi(units, arm)
    frame['qc_arm'] = units['label_ok'] & (frame['isi_arm'] <= bound)
    criteria = {'isi_arm': {'bounds': [0.0, bound]}, 'qc_arm': {'items': [True]},
                **other_constraints(criteria)}
    return run(frame, criteria)


# ------------------------------------------------------------------ reports


def report_inputs(units):
    print(f'{len(units)} units, {units["session"].nunique()} sessions '
          f'(the combined unit table, i.e. units near the tagging site)')
    print(f'decoder labels: {units["decoder"].value_counts().to_dict()}')
    frac = units.groupby('session')['frac_nonopto'].first()
    print(f'laser-free fraction of the session window: median {frac.median():.2f}, '
          f'range {frac.min():.2f}-{frac.max():.2f}')
    print(f'units with no spikes outside the laser: {int((~units["has_nonopto"]).sum())} '
          f'(no laser-free ISI; the "laser-free" arm falls back on the whole-recording '
          f'ratio for them, "no fallback" fails them)')


def report_as_written(units, criteria_free, criteria_whole):
    """
    What the two criteria files give as they stand, stored flags and all.

    This is the number a figure-prep script gets today. It is reported separately from the
    sweep because the stored flags carry their own ISI bound, which is not the bound
    written in the file.
    """
    print('\n--- the criteria files as written (stored qc flags) ---')
    rows = []
    for name, criteria, flag in ((CRITERIA_WHOLE, criteria_whole, 'qc_pass'),
                                 (CRITERIA_NONOPTO, criteria_free, 'qc_pass_nonopto')):
        isi_col = next(col for col in criteria if col.startswith('isi_violations'))
        selected, no_opto, _ = run(units, criteria)
        # the bound the stored flag enforces, read back off the data: it sits between the
        # largest ISI the flag still passes and the smallest it fails, among units the
        # label term does not already exclude
        isi = pd.to_numeric(units[isi_col], errors='coerce')
        flagged = units[flag] == True                                    # noqa: E712
        rows.append({'criteria': name, 'isi_col': isi_col,
                     'bound_in_file': criteria[isi_col]['bounds'][1],
                     'flag': flag,
                     'flag_bound_from': isi[units['label_ok'] & flagged].max(),
                     'flag_bound_to': isi[units['label_ok'] & ~flagged].min(),
                     'opto_units': int(selected.sum()),
                     'qc_only_units': int(no_opto.sum())})
    tbl = pd.DataFrame(rows)
    print(tbl.to_string(index=False, float_format=lambda v: f'{v:.3g}'))
    print('  flag_bound_from/to bracket the ISI bound the stored flag enforces; where that '
          'is tighter than bound_in_file, the flag is what decides')
    return tbl


def sweep(units, criteria):
    """Opto-unit counts for every arm at every bound, plus the flips between the arms."""
    masks = {}
    opto_cond = None
    rows = []
    for bound in BOUNDS:
        for arm in ARMS:
            selected, no_opto, cond = run_arm(units, criteria, arm, bound)
            masks[(arm, bound)] = selected
            # the opto conditions do not involve ISI, so they must not move; if they do,
            # the flips below are not attributable to the ISI source
            if opto_cond is None:
                opto_cond = cond
            elif not cond.equals(opto_cond):
                raise RuntimeError('the opto-condition mask changed between arms')
            rows.append({'isi_max': bound, 'arm': arm, 'opto_units': int(selected.sum()),
                         'qc_only_units': int(no_opto.sum())})
    counts = pd.DataFrame(rows)

    flips = []
    for bound in BOUNDS:
        base = masks[('whole', bound)]
        for arm in ('free', 'free_strict'):
            other = masks[(arm, bound)]
            gained, lost = other & ~base, base & ~other
            flips.append({
                'isi_max': bound, 'arm': arm,
                'opto_whole': int(base.sum()), 'opto_arm': int(other.sum()),
                'gained': int(gained.sum()), 'lost': int(lost.sum()),
                'net': int(other.sum()) - int(base.sum()),
                'sessions_gaining': units.loc[gained, 'session'].nunique(),
                'sessions_losing': units.loc[lost, 'session'].nunique(),
                'sessions_total': units['session'].nunique(),
            })
    return counts, pd.DataFrame(flips), masks, opto_cond


def report_sweep(counts, flips):
    print('\n--- opto units vs the ISI upper bound (qc flag rebuilt at each bound) ---')
    wide = counts.pivot(index='isi_max', columns='arm', values='opto_units')
    wide = wide[list(ARMS)].rename(columns={arm: ARMS[arm]['label'] for arm in ARMS})
    free = flips[flips['arm'] == 'free'].set_index('isi_max')
    wide['gained'] = free['gained']
    wide['lost'] = free['lost']
    wide['net'] = free['net']
    wide['sessions gaining/losing'] = (free['sessions_gaining'].astype(str) + '/'
                                       + free['sessions_losing'].astype(str)
                                       + f' of {free["sessions_total"].iloc[0]}')
    print(wide.to_string())
    print('  gained/lost/net compare the laser-free arm with the whole-recording arm at '
          'the same bound')

    qc = counts.pivot(index='isi_max', columns='arm', values='qc_only_units')
    qc = qc[list(ARMS)].rename(columns={arm: ARMS[arm]['label'] for arm in ARMS})
    print('\n--- units passing the non-opto criteria only, same sweep ---')
    print(qc.to_string())
    return wide


def flip_units(units, masks, bound, arm='free'):
    """The units that change opto status between the two arms at one bound."""
    base, other = masks[('whole', bound)], masks[(arm, bound)]
    frame = units.loc[base | other].copy()
    frame['flip'] = np.where(other[base | other] & ~base[base | other], 'gained',
                             np.where(base[base | other] & ~other[base | other],
                                      'lost', 'kept'))
    frame['isi_max'] = bound
    cols = ['isi_max', 'flip', 'session', 'unit', 'decoder', 'isi_violations',
            'isi_violations_nonopto', 'isi_violations_opto', 'frac_nonopto',
            'has_nonopto', 'num_spikes_nonopto', 'num_spikes_window',
            'p_max', 'lat_max_p', 'eu', 'corr']
    return frame[[col for col in cols if col in frame.columns]]


def report_flips(units, masks):
    """Per-unit detail at the bounds asked for, and the caveat that comes with a gain."""
    tables = []
    for bound in REPORT_BOUNDS:
        frame = flip_units(units, masks, bound)
        tables.append(frame)
        gained = frame[frame['flip'] == 'gained']
        lost = frame[frame['flip'] == 'lost']
        print(f'\n--- ISI <= {bound}: {int((frame["flip"] != "lost").sum())} opto units '
              f'on the laser-free metric, {int((frame["flip"] != "gained").sum())} on the '
              f'whole-recording metric ---')
        print(f'  +{len(gained)} gained, -{len(lost)} lost '
              f'(net {len(gained) - len(lost):+d})')
        for name, sub in (('gained', gained), ('lost', lost)):
            if not len(sub):
                continue
            per_session = sub['session'].value_counts()
            print(f'  {name}: {sub["decoder"].value_counts().to_dict()}, '
                  f'{sub["session"].nunique()} sessions, '
                  f'{per_session.iloc[0]} of {len(sub)} from {per_session.index[0]}')
            print(f'    whole-recording ISI median {sub["isi_violations"].median():.3g}, '
                  f'laser-free median {sub["isi_violations_nonopto"].median():.3g}, '
                  f'inside the laser median {sub["isi_violations_opto"].median():.3g}')
        if len(lost):
            # the ratio is rate-normalized, so cutting a clean high-rate laser epoch can
            # push it up even though nothing was added
            up = int((lost['isi_violations_opto'] < lost['isi_violations']).sum())
            print(f'    {up}/{len(lost)} lost units violate *less* inside the laser than '
                  f'overall -- removing that high-rate epoch raised their rate-normalized '
                  f'ratio')
        if len(gained):
            # a unit rescued by dropping the opto blocks is being judged on a window its
            # opto response was not measured in
            still_bad = int((gained['isi_violations_opto'] > bound).sum())
            print(f'    {still_bad}/{len(gained)} gained units still violate above '
                  f'{bound} inside the laser epochs -- their tagging evidence comes from '
                  f'the window the laser-free metric excludes')
            no_free = int((~gained['has_nonopto']).sum()) if 'has_nonopto' in gained else 0
            if no_free:
                print(f'    {no_free} of them have no laser-free spikes at all')
    return pd.concat(tables, ignore_index=True)


def report_funnel(units, criteria, bound):
    """Where each arm loses units, criterion by criterion."""
    print(f'\n--- what each criterion costs at ISI <= {bound} ---')
    rows = []
    for arm in ('whole', 'free'):
        isi = arm_isi(units, arm)
        stage = pd.Series(True, index=units.index)
        rows.append({'arm': ARMS[arm]['label'], 'stage': 'all units',
                     'units': int(stage.sum())})
        for name, keep in (('ISI bound', isi <= bound),
                           ('qc flag (label)', units['label_ok']),
                           ('peak', units['peak'].between(*criteria['peak']['bounds'])),
                           ('opto conditions', None)):
            if keep is None:
                _, _, cond = run_arm(units, criteria, arm, bound)
                keep = cond
            stage = stage & keep.fillna(False)
            rows.append({'arm': ARMS[arm]['label'], 'stage': name,
                         'units': int(stage.sum())})
    tbl = pd.DataFrame(rows)
    print(tbl.pivot(index='stage', columns='arm', values='units')
          .reindex(['all units', 'ISI bound', 'qc flag (label)', 'peak',
                    'opto conditions']).to_string())
    return tbl


def report_no_free_data(units, masks, bound):
    """The units the laser leaves nothing outside of, and what the fallback does to them."""
    sub = units[~units['has_nonopto']]
    print(f'\n--- {len(sub)} units with no laser-free spikes, at ISI <= {bound} ---')
    if not len(sub):
        return
    idx = sub.index
    print(f'  {int(masks[("whole", bound)][idx].sum())} are opto units on the '
          f'whole-recording metric, '
          f'{int(masks[("free", bound)][idx].sum())} with the fallback, '
          f'{int(masks[("free_strict", bound)][idx].sum())} without it')
    print(f'  median whole-recording ISI {sub["isi_violations"].median():.3g}, '
          f'labels {sub["decoder"].value_counts().to_dict()}')


# ------------------------------------------------------------------ figure


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def panel_counts(ax, counts):
    for arm, color, offset in (('whole', WHOLE, (7, -11)), ('free', FREE, (-7, 7))):
        sub = counts[counts['arm'] == arm].sort_values('isi_max')
        ax.plot(sub['isi_max'], sub['opto_units'], color=color, lw=1.8, marker='o', ms=6,
                mec=SURFACE, mew=1.2, label=ARMS[arm]['label'], zorder=3)
        # selective direct labels: only the bounds the report is written around
        for _, row in sub[sub['isi_max'].isin(REPORT_BOUNDS)].iterrows():
            ax.annotate(f'{int(row["opto_units"])}',
                        (row['isi_max'], row['opto_units']),
                        textcoords='offset points', xytext=offset,
                        ha='left' if arm == 'whole' else 'right',
                        fontsize=8, color=INK_2)
    ax.set_xscale('log')
    style(ax)
    ax.set_xlabel('ISI violations ratio, upper bound', color=INK_2, fontsize=9)
    ax.set_ylabel('opto units passing the criteria', color=INK_2, fontsize=9)
    ax.set_title('the bound costs far more units than the ISI source does',
                 color=INK, fontsize=9.5, loc='left')
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc='lower right')


def panel_flips(ax, flips):
    sub = flips[flips['arm'] == 'free'].sort_values('isi_max')
    y = np.arange(len(sub))
    ax.barh(y, sub['gained'], color=GAINED, height=0.62, zorder=3, label='gained')
    ax.barh(y, -sub['lost'], color=LOST, height=0.62, zorder=3, label='lost')
    for yi, (g, l) in enumerate(zip(sub['gained'], sub['lost'])):
        if g:
            ax.annotate(f'+{g}', (g, yi), xytext=(4, 0), textcoords='offset points',
                        va='center', fontsize=8, color=INK_2)
        if l:
            ax.annotate(f'-{l}', (-l, yi), xytext=(-4, 0), textcoords='offset points',
                        va='center', ha='right', fontsize=8, color=INK_2)
    ax.set_yticks(y)
    ax.set_yticklabels([f'{b:g}' for b in sub['isi_max']], fontsize=8)
    ax.axvline(0, color=INK, lw=0.8, zorder=4)
    style(ax)
    ax.set_ylabel('ISI upper bound', color=INK_2, fontsize=9)
    ax.set_xlabel('opto units changed by the laser-free metric', color=INK_2, fontsize=9)
    ax.set_title('what switching the ISI source actually moves', color=INK, fontsize=9.5,
                 loc='left')
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc='upper right')


def panel_scatter(ax, units, masks, opto_cond, bound):
    """
    The two ISI ratios against each other, for the units the flip can happen to.

    Restricted to units that pass every non-ISI criterion: for anything else the ISI
    source cannot change the outcome, so plotting it only hides the units that matter.
    """
    eligible = opto_cond & units['label_ok'] & units['peak'].notna()
    sub = units[eligible].copy()
    base, free = masks[('whole', bound)], masks[('free', bound)]
    # 'unchanged' is both arms agreeing, whether they both pass or both fail; the
    # threshold lines say which
    flip = pd.Series('unchanged', index=sub.index)
    flip[free[eligible] & ~base[eligible]] = 'gained'
    flip[base[eligible] & ~free[eligible]] = 'lost'
    lims = [1e-3, 1e2]
    for name, color, size in (('unchanged', INK_MUTED, 12), ('lost', LOST, 34),
                              ('gained', GAINED, 34)):
        mask = flip == name
        if not mask.any():
            continue
        ax.scatter(sub.loc[mask, 'isi_violations'].clip(*lims),
                   arm_isi(sub, 'free')[mask].clip(*lims),
                   s=size, c=color, alpha=0.8, lw=0.6, edgecolors=SURFACE,
                   label=f'{name} ({int(mask.sum())})',
                   zorder=3 if name == 'unchanged' else 4)
    ax.plot(lims, lims, color=INK, lw=0.8, zorder=2)
    ax.axhline(bound, color=FREE, lw=1.0, ls='--', zorder=2)
    ax.axvline(bound, color=WHOLE, lw=1.0, ls='--', zorder=2)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(lims); ax.set_ylim(lims)
    style(ax)
    ax.set_xlabel('ISI ratio, whole recording (clipped)', color=INK_2, fontsize=9)
    ax.set_ylabel('ISI ratio, laser-free (clipped)', color=INK_2, fontsize=9)
    ax.set_title(f'units that clear every other criterion (bound {bound:g})',
                 color=INK, fontsize=9.5, loc='left')
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc='upper left')


def panel_funnel(ax, funnel):
    stages = ['all units', 'ISI bound', 'qc flag (label)', 'peak', 'opto conditions']
    wide = funnel.pivot(index='stage', columns='arm', values='units').reindex(stages)
    y = np.arange(len(stages))
    height = 0.36
    for offset, arm, color in ((-height / 2 - 0.01, ARMS['whole']['label'], WHOLE),
                               (height / 2 + 0.01, ARMS['free']['label'], FREE)):
        ax.barh(y + offset, wide[arm], height=height, color=color, zorder=3, label=arm)
        for yi, value in zip(y + offset, wide[arm]):
            ax.annotate(f'{int(value)}', (value, yi), xytext=(4, 0),
                        textcoords='offset points', va='center', fontsize=7.5, color=INK_2)
    ax.set_yticks(y)
    ax.set_yticklabels(stages, fontsize=8)
    ax.invert_yaxis()
    style(ax)
    ax.set_xlabel('units remaining', color=INK_2, fontsize=9)
    ax.set_title('the criteria in order', color=INK, fontsize=9.5, loc='left')
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc='lower right')


def make_figure(units, counts, flips, masks, opto_cond, funnel, bound):
    fig = plt.figure(figsize=(10.5, 8), facecolor=SURFACE)
    gs = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.28,
                  left=0.085, right=0.965, top=0.87, bottom=0.085)
    panel_counts(fig.add_subplot(gs[0, 0]), counts)
    panel_flips(fig.add_subplot(gs[0, 1]), flips)
    panel_scatter(fig.add_subplot(gs[1, 0]), units, masks, opto_cond, bound)
    panel_funnel(fig.add_subplot(gs[1, 1]), funnel)
    fig.suptitle('Opto units kept when the ISI bound is read off laser-free data',
                 color=INK, fontsize=12, fontweight='bold', x=0.085, ha='left', y=0.955)
    fig.text(0.085, 0.915,
             f'{CRITERIA_NONOPTO}.json through apply_qc, QC flag rebuilt at each bound so '
             f'the two arms differ only in the ISI column',
             color=INK_2, fontsize=8.5, ha='left')

    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ('pdf', 'png'):
        path = os.path.join(OUT_DIR, f'isi_nonopto_opto_counts.{ext}')
        fig.savefig(path, dpi=200, facecolor=SURFACE)
        print(f'wrote {path}')
    plt.close(fig)


# %%
if __name__ == '__main__':
    criteria_free = load_criteria(CRITERIA_NONOPTO)
    criteria_whole = load_criteria(CRITERIA_WHOLE)
    units = load_units()

    report_inputs(units)
    as_written = report_as_written(units, criteria_free, criteria_whole)
    counts, flips, masks, opto_cond = sweep(units, criteria_free)
    wide = report_sweep(counts, flips)
    flip_tbl = report_flips(units, masks)
    funnel = report_funnel(units, criteria_free, REPORT_BOUNDS[0])
    report_no_free_data(units, masks, REPORT_BOUNDS[0])
    make_figure(units, counts, flips, masks, opto_cond, funnel, REPORT_BOUNDS[0])

    os.makedirs(OUT_DIR, exist_ok=True)
    as_written.to_csv(os.path.join(OUT_DIR, 'opto_counts_as_written.csv'), index=False)
    counts.to_csv(os.path.join(OUT_DIR, 'opto_counts_sweep.csv'), index=False)
    wide.to_csv(os.path.join(OUT_DIR, 'opto_counts_sweep_wide.csv'))
    flips.to_csv(os.path.join(OUT_DIR, 'opto_counts_flips.csv'), index=False)
    flip_tbl.to_csv(os.path.join(OUT_DIR, 'opto_counts_flip_units.csv'), index=False)
    funnel.to_csv(os.path.join(OUT_DIR, 'opto_counts_funnel.csv'), index=False)
    print(f'\ntables written to {OUT_DIR}')

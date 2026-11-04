# %% Two cover pages for the unit summary PDF: tagged vs matched non-tagged DRN units.
#
#   page 1: how both groups were chosen (counts computed from the current selection); unit
#           metrics (firing rate, ISI violations, SNR, latency); unit locations in three planes
#   page 2: for every condition on the unit pages, the population PSTH, tagged (red) vs
#           non-tagged (gray), over heatmaps of each unit's trace (viridis), one per group
#   page 3: every unit's normalized waveform, tagged and non-tagged grids, ordered by the score
#           on the first waveform principal component (as the grid in F_waveform_space_DRN_v2)
#
# PSTHs come from the per-unit files written with the summary pages (PSTH_DIR), so the page
# fills in as pages are made; only units in the current tagged / non-tagged sets are used.
#
#     python unit_summary_DRN_cover.py [--out PDF]

import os
import sys
import glob
import json
import pickle
import argparse
import textwrap

curr_path = os.path.dirname(os.path.abspath(__file__))
if curr_path not in sys.path:
    sys.path.insert(0, curr_path)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

from matplotlib.backends.backend_pdf import PdfPages

from unit_summary_DRN import (OUT_DIR, NONTAGGED_SELECTION, PSTH_DIR, TAGGED_CRITERION, CRITERIA_DIR,
                              PLANES, AXIS_LABEL, MESH_FILLS, MESH_EDGE, MUTED, INK, INK_2,
                              TAGGED_UNIT, NONTAGGED_POP, POP_METRICS, criteria_passed,
                              population_metrics, load_meshes, project_to_plane, style)
from utils.plot_utils import shiftedColorMap  # noqa: F401  (keeps import order as unit_summary_DRN)

COVER_PDF = os.path.join(OUT_DIR, 'cover', 'unit_summary_DRN_cover.pdf')
FLAGS_CSV = os.path.join(OUT_DIR, 'population_unit_flags.csv')
REVIEW_DIR = os.path.join(OUT_DIR, 'drift_nontagged_review')
NONTAGGED_RING = '#5c5b57'  # darker than NONTAGGED_POP so open rings read on the light mesh
MIN_TRIALS = 5  # a unit enters a condition's average with at least this many trials
TASK_BASELINE = (-1.5, 0)  # s before the go cue, from the unit's all-trials go-cue PSTH
SPONT_BASELINE = (-5, -2)  # s before the lick-train start, from the same spontaneous PSTH
# 'delta': rate minus baseline, in Hz. 'zscore': that divided by the trial-to-trial SD of the
# unit's baseline rate (per-trial mean over the baseline window); units whose baseline SD is
# below MIN_BASELINE_SD (e.g. no baseline spikes) are left out of the z-scored averages.
NORMALIZE = 'delta'
MIN_BASELINE_SD = 0.1  # Hz
# heatmap rows: units sorted, within each group, by their response in SORT_WINDOW after the go
# cue (all-trials go-cue trace), strongest on top; the order is the same in every condition.
# Colour limits at these percentiles of all heatmap values, one scale for every condition.
SORT_WINDOW = (0, 0.5)
HEAT_PCT = (2, 98)
HEAT_CMAP = plt.get_cmap('viridis').copy()
HEAT_CMAP.set_bad('#d9d9d9')  # units without enough trials in the condition
# waveform page: peak-channel slice of each unit's template (waveform_mean), 30 samples before
# to 60 after the alignment sample, i.e. the combined table's 'wf'; the notebook's 'wf_raw' was
# recomputed only for opto-tagging candidates, so the template is used for both groups.
# Normalized as in F_waveform_space_DRN_v2: minus the mean of the first 3 samples, divided by
# the trough (sample 30); the notebook's bad-shape rule drops waveforms whose first sample
# exceeds the trough-normalized value 1. One PCA is fit on both groups.
WF_CACHE = os.path.join(OUT_DIR, 'population_waveforms.pkl')
N_COVER_PAGES = 3
WF_ALIGN = 90  # alignment sample of waveform_mean
WF_PRE, WF_POST = 30, 60
WF_FS = 30000
NONTAGGED_WF = '#52514e'
# rows of the PSTH grid: (key, title); keys as saved by unit_summary_DRN.plot_unit_summary
PSTH_ROWS = [
    [('go_cue_all', 'Go cue: all trials'), ('go_cue_ipsi_rewarded', 'Go cue: ipsi, rewarded'),
     ('go_cue_ipsi_unrewarded', 'Go cue: ipsi, unrewarded'), ('go_cue_contra_rewarded', 'Go cue: contra, rewarded'),
     ('go_cue_contra_unrewarded', 'Go cue: contra, unrewarded'), ('go_cue_ignored', 'Go cue: ignored'),
     ('spont_ipsi', 'Spontaneous lick: ipsi')],
    [('response_all', 'Response: all trials'), ('response_ipsi_rewarded', 'Response: ipsi, rewarded'),
     ('response_ipsi_unrewarded', 'Response: ipsi, unrewarded'), ('response_contra_rewarded', 'Response: contra, rewarded'),
     ('response_contra_unrewarded', 'Response: contra, unrewarded'), ('response_ignored', 'Response window end: ignored'),
     ('spont_contra', 'Spontaneous lick: contra')],
]


def unit_sets():
    tagged = {k for k, v in criteria_passed().items() if TAGGED_CRITERION in v}
    sel = pd.read_csv(NONTAGGED_SELECTION)
    return tagged, set(zip(sel['session'], sel['unit_id'].astype(int))), sel


def load_psths(tagged, nontagged):
    """{(session, unit): saved page dict} for units in either current set."""
    out = {}
    for f in glob.glob(os.path.join(PSTH_DIR, '*_psth.pkl')):
        with open(f, 'rb') as fh:
            d = pickle.load(fh)
        key = (d['session'], int(d['unit_id']))
        if key in tagged or key in nontagged:
            out[key] = d
    return out


def baseline(d, key):
    """(mean, trial-to-trial SD) of the unit's baseline rate for condition key: pre-go-cue rate
    of the all-trials go-cue PSTH for task conditions, early pre-lick rate of the same PSTH for
    spontaneous licks."""
    if key.startswith('spont'):
        p, win = d['psth'].get(key), SPONT_BASELINE
    else:
        p, win = d['psth'].get('go_cue_all'), TASK_BASELINE
    if p is None:
        return np.nan, np.nan
    t = p['bin_times']
    in_win = (t >= win[0]) & (t < win[1])
    per_trial = np.nanmean(np.asarray(p['matrix'], float)[:, in_win], axis=1)
    per_trial = per_trial[np.isfinite(per_trial)]
    sd = np.std(per_trial, ddof=1) if len(per_trial) > 1 else np.nan
    return np.nanmean(p['mean'][in_win]), sd


def unit_trace(d, key, normalize=NORMALIZE):
    """(bin_times, normalized mean trace) of one unit for condition key, or None if the unit has
    fewer than MIN_TRIALS trials there (or, for z-scores, no usable baseline SD)."""
    p = d['psth'].get(key)
    if p is None or p['n_trials'] < MIN_TRIALS:
        return None
    mu, sd = baseline(d, key)
    if normalize == 'zscore':
        if not (np.isfinite(sd) and sd >= MIN_BASELINE_SD):
            return None
        return p['bin_times'], (p['mean'] - mu) / sd
    return p['bin_times'], p['mean'] - mu


def unit_order(psths, keys, normalize=NORMALIZE):
    """keys sorted by response in SORT_WINDOW of the all-trials go-cue trace, strongest first;
    units without that trace go last."""
    def score(k):
        tr = unit_trace(psths[k], 'go_cue_all', normalize)
        if tr is None:
            return -np.inf
        t, y = tr
        return np.nanmean(y[(t >= SORT_WINDOW[0]) & (t < SORT_WINDOW[1])])
    return sorted(keys, key=score, reverse=True)


def condition_matrix(psths, order, key, normalize=NORMALIZE):
    """(bin_times, units x time matrix) for condition key, rows in `order`, NaN rows for units
    without the condition."""
    traces = [unit_trace(psths[k], key, normalize) for k in order]
    t = next((tr[0] for tr in traces if tr is not None), None)
    if t is None:
        return None, np.full((len(order), 1), np.nan)
    return t, np.vstack([tr[1] if tr is not None else np.full(len(t), np.nan) for tr in traces])


def psth_panel(ax, psths, tagged, key, title, normalize=NORMALIZE):
    curves = {}
    for is_tag, color, name in [(False, NONTAGGED_POP, 'non-tagged'), (True, TAGGED_UNIT, 'tagged')]:
        traces = [unit_trace(d, key, normalize) for k, d in psths.items() if (k in tagged) == is_tag]
        traces = [tr for tr in traces if tr is not None]
        if not traces:
            continue
        t = traces[0][0]
        rows = [tr[1] for tr in traces]
        M = np.vstack(rows)
        mean = np.nanmean(M, axis=0)
        sem = np.nanstd(M, axis=0) / np.sqrt(np.sum(np.isfinite(M), axis=0).clip(1))
        ax.fill_between(t, mean - sem, mean + sem, color=color, alpha=0.25, linewidth=0)
        ax.plot(t, mean, color=color, linewidth=1.8, label=f'{name} (n={len(rows)})')
        curves[name] = len(rows)
    ax.axvline(0, color=MUTED, ls='--', lw=1, zorder=0)
    ax.axhline(0, color=MUTED, lw=0.6, zorder=0)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=7, frameon=False, loc='upper left')
    style(ax)
    return curves


def hist_panel(ax, tagged_vals, nontagged_vals, label, scale='linear', integer=False):
    vals = np.concatenate([tagged_vals, nontagged_vals])
    vals = vals[np.isfinite(vals)]
    if integer:
        bins = np.arange(vals.min() - 0.5, vals.max() + 1.5)
    elif scale == 'log':
        pos = vals[vals > 0]
        bins = np.logspace(np.log10(pos.min()), np.log10(np.nanpercentile(pos, 99.5)), 25)
        ax.set_xscale('log')
    else:
        bins = np.linspace(vals.min(), np.nanpercentile(vals, 99.5), 25)
    for x, color, name in [(tagged_vals, TAGGED_UNIT, 'tagged'), (nontagged_vals, NONTAGGED_POP, 'non-tagged')]:
        x = np.asarray(x, float)
        x = np.clip(x[np.isfinite(x)], bins[0], bins[-1])
        if len(x) == 0:
            continue
        w = np.full(len(x), 1 / len(x))
        ax.hist(x, bins=bins, weights=w, histtype='stepfilled', color=color, alpha=0.25, linewidth=0)
        ax.hist(x, bins=bins, weights=w, histtype='step', color=color, linewidth=1.3,
                label=f'{name} (n={len(x)}, median {np.median(x):.3g})')
    ax.set_xlabel(label, fontsize=10)
    ax.set_ylabel('Fraction of units', fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc='upper right')
    style(ax)


def plane_map(ax, meshes, units, xd, yd, title):
    for k, (label, verts) in enumerate(meshes['drn']):
        for j, c in enumerate(project_to_plane(verts, [xd, yd], pitch=0.02, margin=0.5)):
            ax.fill(c[:, 0], c[:, 1], facecolor=MESH_FILLS[k % len(MESH_FILLS)], edgecolor=MESH_EDGE,
                    linewidth=0.5, zorder=-k, label=label if j == 0 else None)
    if 'AQ' in meshes:
        for c in project_to_plane(meshes['AQ'], [xd, yd], pitch=0.02, margin=0.5):
            ax.plot(c[:, 0], c[:, 1], color=MUTED, linewidth=0.6, zorder=1)
    cols = ['ML', 'AP', 'DV']
    # controls sit within tens of um of their tagged unit, so tagged units are filled dots and
    # non-tagged units open rings drawn on top; both stay visible where they overlap
    u = units[units['tagged']]
    ax.scatter(u[cols[xd]], u[cols[yd]], s=12, color=TAGGED_UNIT, linewidth=0, alpha=0.85, zorder=2,
               label=f'tagged (n={len(u)})')
    u = units[~units['tagged']]
    ax.scatter(u[cols[xd]], u[cols[yd]], s=26, facecolor='none', edgecolor=NONTAGGED_RING, linewidth=0.8,
               zorder=3, label=f'non-tagged (n={len(u)})')
    all_verts = np.vstack([v for _, v in meshes['drn']])
    pts = np.vstack([all_verts, units[cols].to_numpy(float)])
    ax.set_xlim(pts[:, xd].min() - 0.1, pts[:, xd].max() + 0.1)
    ax.set_ylim(pts[:, yd].min() - 0.1, pts[:, yd].max() + 0.1)
    ax.set_aspect('equal')
    ax.set_anchor('N')
    ax.set_xlabel(AXIS_LABEL[xd], fontsize=9, color=INK_2)
    ax.set_ylabel(AXIS_LABEL[yd], fontsize=9, color=INK_2)
    ax.set_title(title, loc='left', fontsize=12, color=INK)
    style(ax)


def methods_text(tagged, sel, flags, normalize=NORMALIZE):
    with open(os.path.join(CRITERIA_DIR, f'{TAGGED_CRITERION}.json')) as f:
        c = json.load(f)
    b = lambda k: c[k]['bounds']
    mice = lambda keys: len({s.split('_')[1] for s, _ in keys})
    sel_keys = set(zip(sel['session'], sel['unit_id']))
    per_tagged = sel.groupby(['session', 'matched_tagged_unit']).size()
    sheets = sorted(glob.glob(os.path.join(REVIEW_DIR, 'drift_cut_nontagged_units_*.csv')))
    n_excl = 0
    if sheets:
        sh = pd.read_csv(sheets[-1], encoding='utf-8-sig')
        n_excl = int(sh['exclude'].astype(str).str.strip().str.upper().eq('TRUE').sum())
    removed = [pd.read_csv(f, index_col=0) for f in
               glob.glob('/root/capsule/scratch/*/behavior_*/ephys/opto/raw/*_opto_drift_tbl_nontagged_removed.csv')]
    removed = pd.concat(removed) if removed else pd.DataFrame(columns=['removed_reason'])
    n_fr = int(removed['removed_reason'].astype(str).str.contains('firing_rate').sum())
    n_isi = int(removed['removed_reason'].astype(str).str.contains('isi').sum())
    tier = sel['tier'].value_counts()

    tagged_par = (
        f"Tagged units ({len(tagged)} units, {len({s for s, _ in tagged})} sessions, {mice(tagged)} mice) pass "
        f"the {TAGGED_CRITERION} criteria, applied to the combined opto-tagging candidate table with apply_qc "
        f"as in basic_ephys_generation.py: ISI violations ratio <= {c['isi_violations']['conditional'][1]['bounds'][1]} "
        f"(<= {c['isi_violations']['conditional'][0]['bounds'][1]} for antidromic tier-1 units), qc_pass, waveform "
        f"peak within {b('peak')}, and for at least one stimulation condition baseline-subtracted P(response) "
        f"{b('p_max')[0]}-{b('p_max')[1]}, latency {1000 * b('lat_max_p')[0]:.0f}-{1000 * b('lat_max_p')[1]:.0f} ms, "
        f"laser vs spontaneous waveform Euclidean distance <= {b('eu')[1]} and correlation >= {b('corr')[0]}, "
        f"and significant responses on 5/5 pulses.")
    nontagged_par = (
        f"Non-tagged units ({len(sel)} units, {sel['session'].nunique()} sessions, {mice(sel_keys)} mice) are "
        f"location-matched controls. Candidates are units on the same insertion as a tagged unit that never "
        f"responded to light (not opto_pass, not among the tagging candidates), are not noise or artifact, pass "
        f"default_qc, have ISI violations ratio <= 0.1 and firing rate > 0.5 Hz, and lie within the session's "
        f"tagged DV range +/- 100 um (tier 1) or +/- 200 um (tier 2, used only when tier 1 had no candidate). "
        f"Each tagged unit was matched to its nearest candidate in CCF space without reuse, closest pairs first; "
        f"later rounds added controls by level (a first control for tagged units with none, then a second), so "
        f"{len(per_tagged)} of {len(tagged)} tagged units have a control ({int((per_tagged == 1).sum())} with one, "
        f"{int((per_tagged == 2).sum())} with two). Units were chosen by distance only, never by activity. "
        f"Median match distance {sel['distance_um'].median():.0f} um; tier 1 {int(tier.get('tier1', 0))}, "
        f"tier 2 {int(tier.get('tier2', 0))}. Every candidate's drift summary was reviewed by hand: cuts limit "
        f"each unit to its stable period, and {n_excl} candidates with drift throughout were excluded; "
        f"{n_fr} were replaced or removed for firing rate <= 0.5 Hz and {n_isi} removed for ISI > 0.1.")
    scale = ('' if normalize == 'delta' else
             f", divided by the trial-to-trial SD of its baseline rate (z-score; units with baseline SD "
             f"< {MIN_BASELINE_SD} Hz left out)")
    psth_par = (
        f"PSTHs (page 2): each unit's mean rate (0.2 s bins, 0.05 s steps; spontaneous licks with the causal PSP kernel of "
        f"F_spont_choice_lick_neuron_DRN) minus its baseline (pre-go-cue {TASK_BASELINE[0]}-{TASK_BASELINE[1]} s for task "
        f"conditions, {SPONT_BASELINE[0]} to {SPONT_BASELINE[1]} s before lick-train start for spontaneous licks){scale}, "
        f"averaged over units with >= {MIN_TRIALS} trials in the condition; shading is SEM across units. Ipsi/contra are "
        f"relative to each unit's hemisphere; behavior uses each unit's drift-cut stable period. Heatmaps show "
        f"each unit's trace on the same scale, one row per unit, sorted within each group by the response "
        f"{SORT_WINDOW[0]}-{SORT_WINDOW[1]} s after the go cue (all trials), strongest on top, in the same order in "
        f"every condition; gray rows have fewer than {MIN_TRIALS} trials. Colour limits are the {HEAT_PCT[0]}th-"
        f"{HEAT_PCT[1]}th percentiles of all heatmap values, shared by every condition.")
    return [tagged_par, nontagged_par, psth_par]


def heatmap_panel(ax, t, M, vmin, vmax, label, show_x):
    if t is None:
        ax.axis('off')
        return None
    step = np.median(np.diff(t))
    im = ax.imshow(M, aspect='auto', cmap=HEAT_CMAP, vmin=vmin, vmax=vmax, interpolation='nearest',
                   extent=[t[0] - step / 2, t[-1] + step / 2, len(M) + 0.5, 0.5])
    ax.axvline(0, color='white', ls='--', lw=0.8)
    ax.set_ylabel(f'{label}\n(n={len(M)})', fontsize=9)
    ax.set_yticks([])
    ax.tick_params(labelbottom=show_x)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


def _session_waveforms(session, unit_ids):
    from utils.beh_functions import get_unit_tbl
    tbl = get_unit_tbl(session, 'raw', summary=False)
    out = {}
    for u in unit_ids:
        m = tbl[tbl['unit_id'] == u]
        if len(m):
            w = np.asarray(m['waveform_mean'].values[0], float)
            ch = int(np.argmax(np.abs(w[WF_ALIGN, :])))  # peak channel as in opto_tagging_summary_DRN
            out[(session, u)] = w[WF_ALIGN - WF_PRE:WF_ALIGN + WF_POST, ch]
    return out


def population_waveforms(keys):
    """{(session, unit): peak-channel waveform slice}; cached in WF_CACHE, rebuilt when the
    unit set changes."""
    keys = set(keys)
    if os.path.exists(WF_CACHE):
        with open(WF_CACHE, 'rb') as f:
            cache = pickle.load(f)
        if set(cache) == keys:
            return cache
    from joblib import Parallel, delayed
    by_session = {}
    for s, u in keys:
        by_session.setdefault(s, []).append(u)
    parts = Parallel(n_jobs=8)(delayed(_session_waveforms)(s, u) for s, u in by_session.items())
    cache = {k: v for part in parts for k, v in part.items()}
    with open(WF_CACHE, 'wb') as f:
        pickle.dump(cache, f)
    return cache


def normalize_waveform(w):
    """F_waveform_space_DRN_v2 normalization; None if the denominator is ~0 or the shape fails
    the notebook's bad-shape rule."""
    base = np.nanmean(w[:3])
    denom = w[WF_PRE] - base
    if not np.isfinite(denom) or abs(denom) < 1e-10:
        return None
    if w[WF_PRE] != 0 and w[0] / w[WF_PRE] > 1:
        return None
    x = (w - base) / denom
    return x if np.isfinite(x).all() else None


def page_waveforms(tagged, nontagged, psths):
    """Page 3: all normalized waveforms, one grid per group, ordered by waveform PC1."""
    from sklearn.decomposition import PCA
    wfs = population_waveforms(tagged | nontagged)
    keys = [k for k in sorted(wfs) if normalize_waveform(wfs[k]) is not None]
    # trough-normalized waveforms are +1 at the trough; plot them trough-down
    X = -np.vstack([normalize_waveform(wfs[k]) for k in keys])
    pca = PCA(n_components=3, random_state=0).fit(X)
    pc1 = dict(zip(keys, pca.transform(X)[:, 0]))
    t = 1000 * (np.arange(X.shape[1]) - WF_PRE) / WF_FS

    fig = plt.figure(figsize=(31, 24))
    page_titles(fig, tagged, nontagged, psths, 3)
    outer = gridspec.GridSpec(2, 2, height_ratios=[0.13, 1], width_ratios=[1, 1], hspace=0.08, wspace=0.06,
                              left=0.03, right=0.98, top=0.94, bottom=0.03)
    # what PC1 captures: mean waveform +/- 2 SD along PC1
    ax = fig.add_subplot(outer[0, 0])
    sd1 = np.sqrt(pca.explained_variance_[0])
    for sign, color, lab in [(-2, '#1c5cab', 'low PC1 (-2 SD)'), (0, INK, 'mean'), (2, '#e07b00', 'high PC1 (+2 SD)')]:
        ax.plot(t, pca.mean_ + sign * sd1 * pca.components_[0], color=color, lw=1.8, label=lab)
    ax.set_title(f'Waveform PC1 ({100 * pca.explained_variance_ratio_[0]:.0f}% of variance; PC2 '
                 f'{100 * pca.explained_variance_ratio_[1]:.0f}%, PC3 {100 * pca.explained_variance_ratio_[2]:.0f}%), '
                 f'one PCA over both groups', fontsize=12, loc='left')
    ax.set_xlabel('Time from trough (ms)', fontsize=9)
    ax.legend(fontsize=9, frameon=False, loc='lower right')
    style(ax)
    ax = fig.add_subplot(outer[0, 1])
    ax.axis('off')
    n_bad = len(tagged | nontagged) - len(keys)
    ax.text(0, 0.9, '\n'.join(textwrap.wrap(
        'Each cell is one unit\'s mean template on its peak channel (waveform_mean, the combined table\'s "wf"), '
        f'{1000 * WF_PRE / WF_FS:.0f} ms before to {1000 * WF_POST / WF_FS:.0f} ms after the trough, baseline-subtracted and '
        'divided by the trough as in F_waveform_space_DRN_v2, all on one y scale. Units are ordered by waveform PC1 '
        'score, low (top left) to high (bottom right), separately within each group. '
        f'{n_bad} units with malformed waveforms are left out.', 130)), fontsize=12, va='top', color=INK_2,
        transform=ax.transAxes)

    lo, hi = np.percentile(X, 0.5), np.percentile(X, 99.5)
    for c, (group, color, name) in enumerate([(tagged, TAGGED_UNIT, 'tagged'), (nontagged, NONTAGGED_WF, 'non-tagged')]):
        ks = sorted([k for k in keys if k in group], key=lambda k: pc1[k])
        n = int(np.ceil(np.sqrt(len(ks))))
        grid = gridspec.GridSpecFromSubplotSpec(n, n, subplot_spec=outer[1, c], hspace=0.05, wspace=0.05)
        for i, k in enumerate(ks):
            ax = fig.add_subplot(grid[i])
            ax.axhline(0, color='#e4e3df', lw=0.5)
            ax.plot(t, -normalize_waveform(wfs[k]), color=color, lw=0.9)
            ax.set_ylim(lo, hi)
            ax.set_xlim(t[0], t[-1])
            ax.axis('off')
        fig.text(outer[1, c].get_position(fig).x0, outer[1, c].get_position(fig).y1 + 0.004,
                 f'{name} (n={len(ks)}), ordered by waveform PC1', fontsize=14, color=INK, va='bottom')
    return fig


def page_titles(fig, tagged, nontagged, psths, page):
    fig.text(0.04, 0.985, f'DRN units: 5-HT tagged vs non-tagged  ({page}/{N_COVER_PAGES})', fontsize=30,
             fontweight='normal', color=INK, va='top')
    n_psth = (sum(k in tagged for k in psths), sum(k in nontagged for k in psths))
    fig.text(0.98, 0.985, f'{len(tagged)} tagged, {len(nontagged)} non-tagged units; '
                          f'PSTHs from {n_psth[0]} tagged and {n_psth[1]} non-tagged unit pages',
             fontsize=14, color=INK_2, ha='right', va='top')


def page_selection(tagged, nontagged, sel, psths, pop, units, normalize):
    """Page 1: selection criteria, unit metrics, unit locations."""
    fig = plt.figure(figsize=(31, 24))
    page_titles(fig, tagged, nontagged, psths, 1)
    outer = gridspec.GridSpec(2, 1, height_ratios=[1, 1.25], hspace=0.15,
                              left=0.04, right=0.98, top=0.94, bottom=0.04)
    top = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], width_ratios=[1.15, 1], wspace=0.08)
    ax = fig.add_subplot(top[0])
    ax.axis('off')
    y = 1.0
    for head, par in zip(['Tagged unit selection criteria:', 'Non tagged unit selection criteria:', 'Population PSTHs:'],
                         methods_text(tagged, sel, None, normalize)):
        ax.text(0, y, head, fontsize=15, fontweight='normal', color=INK, va='top', transform=ax.transAxes)
        lines = textwrap.wrap(par, 140)
        ax.text(0, y - 0.06, '\n'.join(lines), fontsize=12, color=INK, va='top', linespacing=1.4,
                transform=ax.transAxes)
        y -= 0.1 + 0.05 * len(lines)
    hists = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=top[1], hspace=0.4, wspace=0.25)
    for k, (col, label, scale) in enumerate(POP_METRICS):
        hist_panel(fig.add_subplot(hists[k // 2, k % 2]),
                   pop.loc[pop['tagged'], col].to_numpy(float), pop.loc[~pop['tagged'], col].to_numpy(float),
                   label, scale)
    meshes = load_meshes()
    maps = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[1], wspace=0.15)
    for k, (title, xd, yd) in enumerate(PLANES):
        ax = fig.add_subplot(maps[k])
        plane_map(ax, meshes, units, xd, yd, title)
        if k == 0:
            ax.legend(fontsize=9, frameon=False, loc='lower left')
    return fig


def page_psth(tagged, nontagged, psths, normalize):
    """Page 2: per condition, population PSTH over unit heatmaps (tagged, non-tagged)."""
    fig = plt.figure(figsize=(31, 24))
    page_titles(fig, tagged, nontagged, psths, 2)
    orders = {g: unit_order(psths, [k for k in psths if (k in tagged) == (g == 'tagged')], normalize)
              for g in ['tagged', 'non-tagged']}
    # one colour scale for every heatmap
    mats = {}
    for row in PSTH_ROWS:
        for key, _ in row:
            for g in orders:
                mats[(key, g)] = condition_matrix(psths, orders[g], key, normalize)
    vals = np.concatenate([M[np.isfinite(M)] for _, M in mats.values()] or [np.array([0.0])])
    lims = np.percentile(vals, HEAT_PCT)

    unit_label = 'Rate - baseline (Hz)' if normalize == 'delta' else 'z-score (baseline SD)'
    outer = gridspec.GridSpec(2, 8, width_ratios=[1] * 7 + [0.06], hspace=0.16, wspace=0.28,
                              left=0.04, right=0.97, top=0.94, bottom=0.04)
    psth_axes = {'task': [], 'spont': []}
    image = None
    for r, row in enumerate(PSTH_ROWS):
        for c, (key, title) in enumerate(row):
            kind = 'spont' if key.startswith('spont') else 'task'
            cell = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer[r, c], hspace=0.08,
                                                    height_ratios=[1, 1.25, 1.3])
            ax = fig.add_subplot(cell[0])
            psth_panel(ax, psths, tagged, key, title, normalize)
            ax.tick_params(labelbottom=False)
            if c == 0:
                ax.set_ylabel(unit_label, fontsize=10)
            psth_axes[kind].append(ax)
            for j, g in enumerate(['tagged', 'non-tagged']):
                t, M = mats[(key, g)]
                ax_h = fig.add_subplot(cell[1 + j], sharex=ax)
                im = heatmap_panel(ax_h, t, M, *lims, g, show_x=(j == 1))
                image = im if im is not None else image
                if j == 1:
                    ax_h.set_xlabel('Time from lick-train start (s)' if kind == 'spont' else
                                    'Time from go cue (s)' if r == 0 else
                                    'Time from response-window end (s)' if key.endswith('ignored')
                                    else 'Time from response (s)', fontsize=9)
    for axes in psth_axes.values():
        if axes:
            lo = min(a.get_ylim()[0] for a in axes)
            hi = max(a.get_ylim()[1] for a in axes)
            for a in axes:
                a.set_ylim(lo, hi)
    if image is not None:
        cax = fig.add_subplot(gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer[:, 7],
                                                               height_ratios=[1, 2, 1])[1])
        cb = fig.colorbar(image, cax=cax)
        cb.set_label(f'{unit_label} (heatmaps, {HEAT_PCT[0]}-{HEAT_PCT[1]}th percentile range)', fontsize=11)
    return fig


def make_cover(out=COVER_PDF, normalize=NORMALIZE):
    """Write both cover pages to one PDF; returns its path."""
    tagged, nontagged, sel = unit_sets()
    psths = load_psths(tagged, nontagged)
    pop = population_metrics()
    flags = pd.read_csv(FLAGS_CSV)
    key = list(zip(flags['session'], flags['unit_id']))
    units = flags[[k in tagged or k in nontagged for k in key]].copy()
    units['tagged'] = [k in tagged for k in zip(units['session'], units['unit_id'])]
    units = units[np.isfinite(units[['ML', 'AP', 'DV']].to_numpy(float)).all(axis=1)]

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with PdfPages(out) as pdf:
        for fig in [page_selection(tagged, nontagged, sel, psths, pop, units, normalize),
                    page_psth(tagged, nontagged, psths, normalize),
                    page_waveforms(tagged, nontagged, psths)]:
            pdf.savefig(fig)
            plt.close(fig)
    print(f'wrote {out}')
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=None, help='default: COVER_PDF, with _zscore for --normalize zscore')
    parser.add_argument('--normalize', default=NORMALIZE, choices=['delta', 'zscore'])
    args = parser.parse_args()
    out = args.out or (COVER_PDF if args.normalize == 'delta' else COVER_PDF.replace('.pdf', '_zscore.pdf'))
    make_cover(out, args.normalize)

# %%
# %%
import os
import sys
sys.path.append('/root/capsule/code/beh_ephys_analysis')
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import json
from harp.clock import decode_harp_clock, align_timestamps_to_anchor_points
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import re
from utils.beh_functions import parseSessionID, session_dirs, get_unit_tbl
from utils.plot_utils import shiftedColorMap, template_reorder, get_gradient_colors
from utils.opto_utils import opto_metrics
from opto_waveforms_preprocessing import re_filter_opto_waveforms
from open_ephys.analysis import Session
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.postprocessing as spost
import spikeinterface.widgets as sw
from aind_dynamic_foraging_basic_analysis.licks.lick_analysis import load_nwb
from aind_ephys_utils import align
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
from matplotlib import colormaps
from aind_dynamic_foraging_data_utils.nwb_utils import load_nwb_from_filename
from spikeinterface.core.sorting_tools import random_spikes_selection
import pickle
import datetime
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface.postprocessing as spost
from tqdm import tqdm
import spikeinterface.widgets as sw
import seaborn as sns
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages


def opto_summary_DRN(session, data_type, target, save=True):
    session_dir = session_dirs(session)
    we = si.load(session_dir[f'postprocessed_dir_{data_type}'], load_extensions=False)

    unit_tbl = get_unit_tbl(session, data_type, summary=True)
    opto_tag = opto_metrics(session, data_type)
    unit_ids = unit_tbl['unit_id'].values.tolist()
    unit_ids = [int(unit_id) for unit_id in unit_ids]

    unit_locations = we.get_extension('unit_locations').get_data(outputs='by_unit')
    y_loc = [unit_locations[unit_id][1] for unit_id in unit_ids]
    # %%
    p_max = []
    p_mean = []
    lat_max_p = []
    lat_mean = []
    euc_max_p = []
    corr_max_p = []
    bl_max_p = []
    pass_count = [] 
    # check if this session has a unit opto tagged at all by checking if resp_lat exists
    for unit_id in unit_ids:
        print(f'Processing unit {unit_id}')
        curr_opto = opto_tag.load_unit(unit_id)
        # find ones with respond latencies
        sort_inds = np.argsort(curr_opto['resp_p_bl'].values)[::-1]
        for curr_ind in sort_inds:
            curr_max_p = curr_opto['resp_p_bl'].values[curr_ind]
            if not curr_opto[curr_opto['resp_p_bl'] == curr_max_p]['resp_lat'].isna().all():
                break
        if curr_opto[curr_opto['resp_p_bl'] == curr_max_p]['resp_lat'].isna().all():
            curr_max_p = curr_opto['resp_p_bl'].max()

        p_max.append(curr_max_p)
        p_max_ind = curr_opto['resp_p_bl'].idxmax()
        max_conditions = curr_opto[curr_opto['resp_p_bl']==curr_max_p]
        # count number of cases that passes the threshold, group by power, pre-post and duration
        count = curr_opto.groupby(['powers', 'stim_times', 'durations']).agg({'resp_p_bl': list})
        count = count.reset_index()
        count = pd.DataFrame(count)
        count_curr_all = []
        for row in count.iterrows():
            count_curr = np.sum(np.array(row[1]['resp_p_bl']) >= 0.3)
            count_curr_all.append(count_curr)
        pass_count.append(np.max(count_curr_all))

        p_mean.append(np.nanmax(curr_opto[curr_opto['resp_p_bl'] == curr_max_p]['mean_p'].values))

        lat_max_p.append(np.nanmin(curr_opto[curr_opto['resp_p_bl'] == curr_max_p]['resp_lat'].values))
        lat_mean.append(curr_opto['resp_lat'].mean(skipna=True))
        if np.all(curr_opto[curr_opto['resp_p_bl'] == curr_max_p]['euclidean_norm'].values == None) or np.all(np.isnan(curr_opto[curr_opto['resp_p_bl'] == curr_max_p]['euclidean_norm'].values)):
            euc_max_p.append(np.nan)
            corr_max_p.append(np.nan)
        else:
            euc_max_p.append(np.nanmin(curr_opto[curr_opto['resp_p_bl'] == curr_max_p]['euclidean_norm'].values))
            corr_max_p.append(np.nanmin(curr_opto[curr_opto['resp_p_bl'] == curr_max_p]['correlation'].values))
        bl_max_p.append(curr_opto.loc[p_max_ind]['resp_p'] - curr_opto.loc[p_max_ind]['resp_p_bl'])
    peak_channel = [np.argmax(np.abs(temp[90,:])) for temp in unit_tbl['waveform_mean']]
    peak_index = [np.argmin(temp[:, peakC]) for temp, peakC in zip(unit_tbl['waveform_mean'], peak_channel)]
    peak_wf = [temp[90-30:90+60, peak_C] for temp, peak_C in zip(unit_tbl['waveform_mean'], peak_channel)]
    peak_wf_aligned = [
        np.concatenate((
            np.full(max(30 - peak_ind, 0), np.nan),
            temp[max(peak_ind - 30, 0) : min(peak_ind + 60, temp.shape[0]), peak_C],
            np.full(max((peak_ind + 60) - temp.shape[0], 0), np.nan)
        ))
        for temp, peak_ind, peak_C in zip(unit_tbl['waveform_mean'], peak_index, peak_channel)
    ]

    # recompute 2d waveform
    # load wfs
    waveform_info_file = os.path.join(session_dir[f'opto_dir_{data_type}'], f'{session}_waveform_params.json')
    with open(waveform_info_file) as f:
        waveform_info = json.load(f)
    y_neighbors_to_keep = waveform_info['y_neighbors_to_keep']
    samples_to_keep = waveform_info['samples_to_keep']
    orginal_loc = True
    analyzer = si.load(session_dir[f'postprocessed_dir_{data_type}'], load_extensions=False)
    extreme_channel_indices = si.get_template_extremum_channel(analyzer, mode = "at_index", outputs = "index")
    unit_locations = analyzer.get_extension("unit_locations").get_data(outputs="by_unit")
    channel_locations = analyzer.get_channel_locations()
    right_left = channel_locations[:, 0]<20 
    all_channels = analyzer.sparsity.channel_ids
    if all_channels[0].startswith('AP'):
        all_channels_int = np.array([int(channel.split('AP')[-1]) for channel in all_channels])
    else:
        all_channels_int = np.array([int(channel.split('CH')[-1]) for channel in all_channels])

    # temp_ext = analyzer.get_extension("templates")
    # temps = temp_ext.get_templates(operator='average')
    # wf_2d = [template_reorder(curr_temp, right_left, all_channels_int, 
    #                                 sample_to_keep = samples_to_keep, y_neighbors_to_keep = y_neighbors_to_keep, orginal_loc = orginal_loc, 
    #                                 peak_ind=curr_peak) 
    #                                 for curr_temp, curr_peak in zip(list(temps), list(extreme_channel_indices.values()))]
    temp_ext = analyzer.get_extension("templates")
    temps = temp_ext.get_templates(operator='average')

    # map templates by unit_id
    analyzer_unit_ids = list(analyzer.unit_ids)
    unit_to_temp = dict(zip(analyzer_unit_ids, list(temps)))

    wf_2d = [
        template_reorder(
            unit_to_temp[unit_id],
            right_left,
            all_channels_int,
            sample_to_keep=samples_to_keep,
            y_neighbors_to_keep=y_neighbors_to_keep,
            orginal_loc=orginal_loc,
            peak_ind=extreme_channel_indices[unit_id],
        )
        for unit_id in unit_ids
    ]   
                                 
    amp = [np.max(temp[:, peak_C]) - np.min(temp[:, peak_C]) for temp, peak_C in zip(unit_tbl['waveform_mean'], peak_channel)]
    peak = [temp[90,peak_C] for temp, peak_C in zip(unit_tbl['waveform_mean'], peak_channel)]
    label = unit_tbl['decoder_label'].values
    real_unit = label != 'artifact'

    # print('len(unit_ids):', len(unit_ids))
    # print('len(bl_max_p):', len(bl_max_p))
    # print('len(p_max):', len(p_max))
    # print('len(p_mean):', len(p_mean))
    # print('len(lat_max_p):', len(lat_max_p))
    # print('len(lat_mean):', len(lat_mean))
    # print('len(euc_max_p):', len(euc_max_p))
    # print('len(corr_max_p):', len(corr_max_p))
    # print('len(unit_tbl[opto_pass]):', len(unit_tbl['opto_pass'].values))
    # print('len(amp):', len(amp))
    # print('len(peak):', len(peak))
    # print('len(real_unit):', len(real_unit))
    # print('len(y_loc):', len(y_loc))
    # print('len(pass_count):', len(pass_count))

    opto_tag_tbl = pd.DataFrame({'unit_id': unit_ids, 
                                'bl_max_p': bl_max_p,
                                'p_max': p_max, 
                                'p_mean': p_mean, 
                                'lat_max_p': lat_max_p, 
                                'lat_mean': lat_mean, 
                                'euc_max_p': euc_max_p, 
                                'corr_max_p': corr_max_p, 
                                'opto_pass': unit_tbl['opto_pass'].values,
                                'amp': amp, 
                                'peak': peak,
                                'real_unit': real_unit,
                                'y_loc': y_loc, 
                                'pass_count': pass_count,
                                })
    
    unit_tbl_tmp = unit_tbl.copy()
    unit_tbl_tmp.drop(columns=opto_tag_tbl.columns.difference(['unit_id']), inplace=True, errors='ignore')
    unit_tbl_tmp['peak_wf'] = peak_wf
    unit_tbl_tmp['peak_wf_aligned'] = peak_wf_aligned
    unit_tbl_tmp['wf_2d'] = wf_2d
    opto_tag_tbl_summary = pd.merge(opto_tag_tbl, unit_tbl_tmp, on='unit_id')

    # add 2D waveform:


    # %%
    pairplot = sns.pairplot(opto_tag_tbl, hue='real_unit', corner=True, diag_kind='kde', plot_kws={'alpha': 0.25})
    plt.legend() 

    # %%
    fig = plt.figure(figsize=(20, 10))
    gs_probe = gridspec.GridSpec(2, 8, figure=fig, height_ratios=[1, 20])
    # p_resp
    ax = fig.add_subplot(gs_probe[1, 0])
    colors, norm, cmap = get_gradient_colors(opto_tag_tbl['p_max'].values, ceiling=0.8)
    color_dict = dict(zip(unit_ids, colors))
    w = sw.plot_unit_locations(we, backend="matplotlib", unit_ids=unit_ids, unit_colors=color_dict, ax = ax)
    w.ax.set_ylim(-200, 2000)
    w.ax.set_title('All units')

    ax = fig.add_subplot(gs_probe[1, 1])
    unit_ids_filtered = pd.Series(unit_ids)[opto_tag_tbl['real_unit'] & (opto_tag_tbl['lat_mean']>=0.004) & (opto_tag_tbl['corr_max_p']>=0.85)].values.tolist()
    w = sw.plot_unit_locations(we, backend="matplotlib", unit_ids=unit_ids_filtered, unit_colors=color_dict, ax = ax)
    w.ax.set_ylim(-200, 2000)
    w.ax.set_title('Real units')

    # Define the color map (red gradient)
    cmap = plt.get_cmap("Reds")  # Use Reds colormap

    # Normalize between 0 and 1
    norm = mcolors.Normalize(vmin=0, vmax=0.8)

    # Create a colorbar
    ax = fig.add_subplot(gs_probe[0, 0:3])
    cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='horizontal', aspect=60, shrink=2, pad=0.05)
    cbar.set_label("max(p(resp)-bl)")
    ax.clear()
    ax.axis('off')
    ax.set_title("p_resp")

    # p_resp_mean
    colors, norm, cmap = get_gradient_colors(opto_tag_tbl['p_mean'].values, ceiling=0.8)
    color_dict = dict(zip(unit_ids, colors))

    ax = fig.add_subplot(gs_probe[1, 3])
    w = sw.plot_unit_locations(we, backend="matplotlib", unit_ids=unit_ids_filtered, unit_colors=color_dict, ax = ax)
    w.ax.set_ylim(-200, 2000)
    w.ax.set_title('Real units')

    # Define the color map (red gradient)
    cmap = plt.get_cmap("Reds")  # Use Reds colormap

    # Normalize between 0 and 1
    norm = mcolors.Normalize(vmin=0, vmax=0.8)

    # Create a colorbar
    ax = fig.add_subplot(gs_probe[0, 3])
    cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='horizontal', aspect=20, shrink=2, pad=0.05)
    cbar.set_label("mean(p(resp)-bl)")
    ax.clear()
    ax.axis('off')
    ax.set_title("p_resp_mean")

    # amp
    ax = fig.add_subplot(gs_probe[1, 4])
    colors, norm, cmap = get_gradient_colors(opto_tag_tbl['amp'].values, floor = 30, ceiling=150, cmap_name='Blues')
    color_dict = dict(zip(unit_ids, colors))
    w = sw.plot_unit_locations(we, backend="matplotlib", unit_ids=unit_ids, unit_colors=color_dict, ax = ax)
    w.ax.set_ylim(-200, 2000)
    w.ax.set_title('All units')

    ax = fig.add_subplot(gs_probe[1, 5])
    unit_ids_filtered = pd.Series(unit_ids)[opto_tag_tbl['real_unit'] & (opto_tag_tbl['lat_mean']>=0.005) & (opto_tag_tbl['corr_max_p']>=0.85)].values.tolist()
    w = sw.plot_unit_locations(we, backend="matplotlib", unit_ids=unit_ids_filtered, unit_colors=color_dict, ax = ax)
    w.ax.set_ylim(-200, 2000)
    w.ax.set_title('Real units')

    # Define the color map (blue gradient)
    cmap = plt.get_cmap("Blues")  # Use
    norm = mcolors.Normalize(vmin=30, vmax=150)
    ax = fig.add_subplot(gs_probe[0, 4:6])
    cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='horizontal', aspect=40, shrink=2, pad=0.05)
    cbar.set_label("amp")
    ax.clear()
    ax.axis('off')
    ax.set_title("Amp")

    # peak, presp

    # depth, presp
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.5)
    # top unit pass presp
    thresh = 0.2
    top = opto_tag_tbl[opto_tag_tbl['p_max']>0.6]['y_loc'].max()
    # bottom = opto_tag_tbl[opto_tag_tbl['p_max']>thresh]['y_loc'].min()

    ax = fig.add_subplot(gs[0, 3])
    focus = (opto_tag_tbl['real_unit']) & (opto_tag_tbl['lat_mean']>0.005)
    ax.scatter(opto_tag_tbl[focus & unit_tbl['default_qc']]['p_max'], opto_tag_tbl[focus& unit_tbl['default_qc']]['lat_max_p'], label='pass', alpha=0.5, facecolors='none', edgecolors='k', linewidths=2)
    ax.scatter(opto_tag_tbl[focus & ~unit_tbl['default_qc']]['p_max'], opto_tag_tbl[focus& ~unit_tbl['default_qc']]['lat_max_p'], label='fail', alpha=0.5, facecolors='none', edgecolors='r', linewidths=2)
    ax.set_xlabel('max(p-bl)')
    ax.set_ylabel('lat_max_p/(s)')
    ax.set_title('All real units')
    ax.legend()


    ax = fig.add_subplot(gs[1, 3])
    focus = (opto_tag_tbl['real_unit']) & (opto_tag_tbl['y_loc']<=top) & (opto_tag_tbl['p_max']>thresh)
    ax.scatter(opto_tag_tbl[focus & unit_tbl['default_qc']]['p_max'], opto_tag_tbl[focus& unit_tbl['default_qc']]['y_loc'], label='pass', alpha=0.5, facecolors='none', edgecolors='k', linewidths=2)
    ax.scatter(opto_tag_tbl[focus & ~unit_tbl['default_qc']]['p_max'], opto_tag_tbl[focus& ~unit_tbl['default_qc']]['y_loc'], label='fail', alpha=0.5, facecolors='none', edgecolors='r', linewidths=2)
    ax.set_ylabel('y_location')
    ax.set_xlabel('p_max')
    ax.set_title('All real units in DRN with low bar')

    ax = fig.add_subplot(gs[2, 3])
    # ax.scatter(opto_tag_tbl[focus]['p_max'], opto_tag_tbl[focus]['amp'], c=opto_tag_tbl[focus]['default_qc'])
    ax.scatter(opto_tag_tbl[focus & unit_tbl['default_qc']]['p_max'], opto_tag_tbl[focus& unit_tbl['default_qc']]['amp'], label='pass', alpha=0.5, facecolors='none', edgecolors='k', linewidths=2)
    ax.scatter(opto_tag_tbl[focus & ~unit_tbl['default_qc']]['p_max'], opto_tag_tbl[focus& ~unit_tbl['default_qc']]['amp'], label='fail', alpha=0.5, facecolors='none', edgecolors='r', linewidths=2)
    ax.set_ylabel('amp')
    ax.set_xlabel('p_max')
    ax.set_title('All real units in DRN with low bar')

    low_thresh = 0.3
    high_thresh = 0.5
    mid_thresh = 0.4

    unit_tag = (
        (opto_tag_tbl['p_max'] > mid_thresh)
        & (opto_tag_tbl['p_mean'] > 0.1)
        & (opto_tag_tbl['pass_count'] >= 2)
        & (opto_tag_tbl['lat_max_p'] < 0.025)
        & (opto_tag_tbl['lat_max_p'] > 0.005)
        & (opto_tag_tbl['bl_max_p'] > 0.5 * 0.02)
        & (opto_tag_tbl['bl_max_p'] < 20 * 0.02)
        & (opto_tag_tbl['real_unit'])
        & (opto_tag_tbl['corr_max_p'] >= 0.8)
    )

    # No location gating for DRN currently 3/20/26 JL
    unit_tag_loc = unit_tag

    opto_tag_tbl_summary['DRN_range_top'] = None
    opto_tag_tbl_summary['DRN_range_bottom'] = None
    opto_tag_tbl_summary['tagged_loc'] = unit_tag_loc
    opto_tag_tbl_summary['tagged'] = unit_tag

    plt.suptitle(f'{session} {data_type} {target} {np.sum(unit_tag)} maybe, {np.sum(unit_tag_loc)} with location info')


    # Save figures to a single PDF
    pdf_filename = os.path.join(session_dir[f'opto_dir_{data_type}'], f"{session}_{data_type}_{target}_opto_summary.pdf")
    with PdfPages(pdf_filename) as pdf:
        pdf.savefig(fig)  # Save Figure 1
        pdf.savefig(pairplot.fig)  # Save Figure 2

        # Add metadata (optional)
        pdf.infodict().update({
            "Title": "Opto Summary",
        })

    print(f"PDF saved as {pdf_filename}")

    
    if save:
        with open(os.path.join(session_dir[f'opto_dir_{data_type}'], f'{session}_{data_type}_{target}_opto_tagging_summary.pkl'), 'wb') as f:
            pickle.dump(opto_tag_tbl_summary, f)
        print(f"Opto tagging summary saved as {session}_{data_type}_{target}_opto_tagging_summary.pkl")

    return opto_tag_tbl_summary


if __name__ == "__main__":
    import warnings
    import traceback
    from tqdm import tqdm

    warnings.filterwarnings("ignore", category=UserWarning, module="spikeinterface")

    data_type = "raw"
    load_sorting_analyzer = True
    opto_only = True

    session_assets = pd.read_csv("/root/capsule/code/data_management/session_assets.csv")
    session_list = session_assets["session_id"].values

    def process(session):
        print(f"\n===== Processing {session} =====")
        session_dir = session_dirs(session)

        if session_dir.get("opto_dir_raw") is None:
            print(f"Session {session}: no opto_dir_raw, skip.")
            return

        unit_tbl = get_unit_tbl(session, data_type, summary=True)
        if unit_tbl is None:
            print(f"Session {session}: no raw summary unit table, skip.")
            return

        if "peak_waveform_raw_fake_aligned" in unit_tbl.columns:
            print(f"Session {session}: already processed, skip.")
            return

        print(f"Session {session}: running re_filter_opto_waveforms on raw")
        re_filter_opto_waveforms(
            session,
            data_type,
            opto_only=opto_only,
            load_sorting_analyzer=load_sorting_analyzer,
        )
        print(f"Session {session}: done.")

    failed_sessions = []

    for session in tqdm(session_list, desc="Re-filtering opto waveforms"):
        try:
            process(session)
        except Exception as e:
            print(f"Session {session}: FAILED")
            print(f"Error: {e}")
            traceback.print_exc()
            failed_sessions.append(session)

    print("\n===== Finished =====")
    print(f"Failed sessions ({len(failed_sessions)}):")
    for session in failed_sessions:
        print(session)





import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors

import utils.analysis_funcs as af
import utils.plotting_funcs as pf

import seaborn as sns

def shiftedColorMap(cmap, min_val, max_val, name):
    '''Function to offset the "center" of a colormap. Useful for data with a negative min and positive max and you want the middle of the colormap's dynamic range to be at zero. Adapted from https://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib
    Input
    -----
      cmap : The matplotlib colormap to be altered.
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower ofset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax/(vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highets point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          `midpoint` and 1.0.'''
    epsilon = 0.001
    start, stop = 0.0, 1.0
    min_val, max_val = min(0.0, min_val), max(0.0, max_val) # Edit #2
    midpoint = 1.0 - max_val/(max_val + abs(min_val))
    cdict = {'red': [], 'green': [], 'blue': [], 'alpha': []}
    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)
    # shifted index to match the data
    shift_index = np.hstack([np.linspace(0.0, midpoint, 128, endpoint=False), np.linspace(midpoint, 1.0, 129, endpoint=True)])
    for ri, si in zip(reg_index, shift_index):
        if abs(si - midpoint) < epsilon:
            r, g, b, a = cmap(0.5) # 0.5 = original midpoint.
        else:
            r, g, b, a = cmap(ri)
        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))
    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)

    matplotlib.colormaps.register(cmap=newcmap, force=True)
    return newcmap


# def raster_plot(event_locked_spike_times, time_range, cond_each_trial=None, raster=None, color='k', cond_colors=None, trial_start=0, ax=None, **kwargs):
#     '''
#     :param event_locked_spike_times: spike timestamps each trial relative to an event
#     :param cond_each_trial: (OPTIONAL) some sort of label for each trial so that trials with the same parameters can be grouped together.
#     :param ax: matplotlib axis to plot on
#     :return: a cool raster plot
#     '''
#     import matplotlib.pyplot as plt
#     if raster is None:
#         raster = []

#     if ax is None:
#         ax = plt.gca()

#     if cond_each_trial is not None:
#         conds = np.unique(cond_each_trial)

#         if type(color) == str:
#             color = np.tile(color, len(conds))
#         if cond_colors is None:
#             cond_colors = np.tile(['0.5', '0.75'], int(np.ceil(len(conds)/2)))

#         total_trials = 0
#         cond_lines = []
#         cond_bars = []

#         for indcond, cond in enumerate(conds):
#             this_event_locked_spike_times = np.array(event_locked_spike_times, dtype=object)[cond_each_trial == cond]
#             raster, none_cond_lines, none_cond_bars = raster_plot(this_event_locked_spike_times, time_range, raster=raster, color=color[indcond], trial_start=total_trials, ax=ax)
#             total_trials += len(this_event_locked_spike_times)

#             xpos = [time_range[0]-0.03*(time_range[1]-time_range[0]),time_range[0]]
#             ybot = [total_trials-len(this_event_locked_spike_times), total_trials-len(this_event_locked_spike_times)]
#             ytop = [total_trials, total_trials]
#             cond_bar = ax.fill_between(xpos, ybot, ytop,ec='none',fc=cond_colors[indcond], clip_on=False)
#             cond_bars.append(cond_bar)

#         trials_per_cond = total_trials/len(conds)
#         ax.set_yticks(np.arange(trials_per_cond/2, total_trials, trials_per_cond))
#         ax.set_yticklabels([f'{cond}' for cond in conds])
#         ax.tick_params('y', length=0, pad=8)

#     else:
#         Ntrials = len(event_locked_spike_times)
#         for trial in range(Ntrials):
#             this_raster = ax.plot(event_locked_spike_times[trial],
#                                    (trial + 1 + trial_start) * np.ones(len(event_locked_spike_times[trial])),
#                                    '.', color=color, rasterized=False, ms=1, **kwargs)
#             raster.append(this_raster)

#         cond_lines = None
#         cond_bars = None
#         ax.set_ylim(0,Ntrials+2+trial_start)

#     ax.set_xlim(time_range)
#     return raster, cond_lines, cond_bars

def raster_plot(event_locked_spike_times, time_range, cond_each_trial=None, raster=None,
                color='k', cond_colors=None, trial_start=0, ax=None,
                spike_height=0.8, rasterized=False, **kwargs):
    """
    Raster plot with vertical bars for spikes instead of dots.
    Each spike is a short line segment centered on the trial index.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if raster is None:
        raster = []

    if ax is None:
        ax = plt.gca()

    if cond_each_trial is not None:
        conds = np.unique(cond_each_trial)
        if isinstance(color, str):
            color = np.tile(color, len(conds))
        if cond_colors is None:
            cond_colors = np.tile(['0.5', '0.75'], int(np.ceil(len(conds) / 2)))

        total_trials = 0
        cond_lines = []
        cond_bars = []

        for indcond, cond in enumerate(conds):
            this_event_locked_spike_times = np.array(event_locked_spike_times, dtype=object)[cond_each_trial == cond]
            raster, none_cond_lines, none_cond_bars = raster_plot(
                this_event_locked_spike_times, time_range,
                raster=raster, color=color[indcond],
                trial_start=total_trials, ax=ax,
                spike_height=spike_height, rasterized=rasterized
            )
            total_trials += len(this_event_locked_spike_times)

            xpos = [time_range[0] - 0.03 * (time_range[1] - time_range[0]), time_range[0]]
            ybot = [total_trials - len(this_event_locked_spike_times), total_trials - len(this_event_locked_spike_times)]
            ytop = [total_trials, total_trials]
            cond_bar = ax.fill_between(xpos, ybot, ytop, ec='none', fc=cond_colors[indcond], clip_on=False)
            cond_bars.append(cond_bar)

        trials_per_cond = total_trials / len(conds)
        ax.set_yticks(np.arange(trials_per_cond / 2, total_trials, trials_per_cond))
        ax.set_yticklabels([f'{cond}' for cond in conds])
        ax.tick_params('y', length=0, pad=8)

    else:
        Ntrials = len(event_locked_spike_times)
        for trial in range(Ntrials):
            y_bottom = trial + 1 + trial_start - spike_height / 2
            y_top = trial + 1 + trial_start + spike_height / 2
            for spike_time in event_locked_spike_times[trial]:
                ax.vlines(spike_time, y_bottom, y_top,
                          color=color, linewidth=0.5, rasterized=rasterized, **kwargs)
            # (Alternatively: use LineCollection for performance if many spikes)
        cond_lines = None
        cond_bars = None
        ax.set_ylim(0, Ntrials + 2 + trial_start)

    ax.set_xlim(time_range)
    return raster, cond_lines, cond_bars




def raster_plot_demo(event_locked_spike_times, time_range, cond_each_trial=None, raster=None, color='k', cond_colors = None, trial_start=0, **kwargs):
    '''
    :param event_locked_spike_times: spike timestamps each trial relative to an event
    :param cond_each_trial: (OPTIONAL) some sort of label for each trial so that trials with the same parameters can be grouped together.
    :return: a cool raster plot
    '''
    if raster is None:
        raster = []

    if cond_each_trial is not None:
        conds = np.unique(cond_each_trial)

        if type(color) == str:
            color = np.tile(color, len(conds))
        if cond_colors is None:
            cond_colors = np.tile(['0.5', '0.75'], int(np.ceil(len(conds)/2)))

        total_trials = 0
        cond_lines = []
        cond_bars = []

        for indcond, cond in enumerate(conds):
            this_event_locked_spike_times = np.array(event_locked_spike_times, dtype=object)[cond_each_trial == cond]
            raster, none_cond_lines, none_cond_bars = raster_plot(this_event_locked_spike_times, time_range, raster=raster, color=color[indcond], trial_start=total_trials)
            total_trials += len(this_event_locked_spike_times)

            cond_line = plt.axhline(total_trials, color='0.7', zorder=-100)
            cond_lines.append(cond_line)

            xpos = [time_range[0]-0.03*(time_range[1]-time_range[0]),time_range[0]]            
            ybot = [total_trials-len(this_event_locked_spike_times), total_trials-len(this_event_locked_spike_times)]
            ytop = [total_trials, total_trials]
            cond_bar = plt.fill_between(xpos, ybot, ytop,ec='none',fc=cond_colors[indcond], clip_on=False)
            cond_bars.append(cond_bar)


        trials_per_cond = total_trials/len(conds)
        # plt.yticks(np.arange(trials_per_cond/2, total_trials, trials_per_cond), [f'{cond}' for cond in conds])
        # plt.gca().tick_params('y', length=0, pad=8)

    else:
        Ntrials = len(event_locked_spike_times)
        for trial in range(Ntrials):
            this_raster = plt.plot(event_locked_spike_times[trial], 
                                   (trial + 1 + trial_start) * np.ones(len(event_locked_spike_times[trial])),
                                   ".", color=color, rasterized=False, ms=1) #, **kwargs)
            raster.append(this_raster)

        cond_lines = None
        cond_bars = None
        plt.ylim(0,Ntrials+2+trial_start)


        #zline = plt.axvline(0, color='0.8', zorder=-100)

    plt.xlim(time_range)
    return raster, cond_lines, cond_bars


def sorted_psths(event_locked_spike_times, cond_each_trial, time_range, **kwargs):
    conds = np.unique(cond_each_trial)

    psths = []

    for indcond, cond in enumerate(conds):
        this_event_locked_spike_times = np.array(event_locked_spike_times, dtype=object)[cond_each_trial == cond]
        this_psth, bins = psth(this_event_locked_spike_times, time_range, **kwargs)
        psths.append(this_psth)

    return psths, bins


def psth(event_locked_spike_times, time_range, bin_size=0.0005, smooth_window_size=3):
    '''Actually just makes the inputs for a psth (FR at each time point) that can just be plotted with plt.plot'''
    Ntrials = len(event_locked_spike_times)
    bins = np.arange(time_range[0], time_range[1], bin_size)

    spike_counts = af.timestamps_to_spike_counts(event_locked_spike_times, bins)

    psth = np.mean(spike_counts, axis=0) / bin_size  # convert spike counts to spikes/sec
    
    if smooth_window_size == 0:
        smooth_psth = psth
    else:
        window_shape = np.concatenate((np.zeros(smooth_window_size), np.ones(smooth_window_size)))  # square (causal)
        window_shape = window_shape / np.sum(window_shape)
        smooth_psth = np.convolve(psth, window_shape, mode='same')

    return psth, smooth_psth, bins[1:]-bin_size/2


def umap_plot(umap_df, ax, colors, colormap=None, xlim=None, ylim=None):
    ax.scatter(umap_df['x'].tolist(), umap_df['y'].tolist(),
               marker='o', c=colors, cmap=colormap, s=32, edgecolor='w',
               linewidth=0.5, clip_on=False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    xlim = np.array(plt.xlim())
    ylim = np.array(plt.ylim())

    xlen = xlim[1] - xlim[0]
    ylen = ylim[1] - ylim[0]

    ax.arrow(xlim[0], ylim[0], 0, ylen * 0.09, width=xlen * 0.01, shape="full", ec="none", fc="black")
    ax.arrow(xlim[0], ylim[0], xlen * 0.09, 0, width=ylen * 0.01, shape="full", ec="none", fc="black")

    ax.text(xlim[0] + 0.05 * xlen, ylim[0] - ylen * 0.05, "UMAP 1", va="center")
    ax.text(xlim[0] - 0.05 * xlen, ylim[0] + 0.05 * ylen, "UMAP 2", rotation=90, ha="left", va="bottom")


def hist_plot(data, ax, bins=50, color='k', xlabel='', ylabel='', title='', xlim=None, ylim=None):
    sns.histplot(data, bins=bins, color=color, edgecolor='w', linewidth=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
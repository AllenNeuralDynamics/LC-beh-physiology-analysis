"""
Python port of optoIDTT.m — plots opto-tagging rasters and raw traces for tetrode data.

Reads Neuralynx binary files (.nev, .ntt, .ncs) directly without MATLAB I/O bridges.
"""

import os
import struct
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import butter, filtfilt

HEADER_SIZE = 16384  # bytes — all Neuralynx binary files share this header size


# ---------------------------------------------------------------------------
# Neuralynx file readers
# ---------------------------------------------------------------------------

def _parse_header_field(header_str: str, field: str) -> str | None:
    for line in header_str.split('\n'):
        if field in line:
            return line.strip()
    return None


def read_nev(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (timestamps_us, ttl_values) from a Neuralynx .nev file."""
    timestamps, ttl_values = [], []
    with open(filepath, 'rb') as f:
        f.seek(HEADER_SIZE)
        while True:
            rec = f.read(184)
            if len(rec) < 184:
                break
            ts  = struct.unpack_from('<q', rec, 6)[0]
            ttl = struct.unpack_from('<H', rec, 16)[0]
            timestamps.append(ts)
            ttl_values.append(ttl)
    return np.array(timestamps), np.array(ttl_values)


def read_ncs(filepath: str) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Return (timestamps_us, samples_flat, ad2uv) from a Neuralynx .ncs file.

    timestamps_us are the record-level timestamps (one per 512-sample block).
    samples_flat is a 1-D array of all ADC samples in order.
    """
    RECORD_SIZE = 1044  # 8 + 4 + 4 + 4 + 512*2
    timestamps, all_samples = [], []
    ad2uv = None

    with open(filepath, 'rb') as f:
        header_raw = f.read(HEADER_SIZE).decode('latin-1', errors='replace')
        line = _parse_header_field(header_raw, 'ADBitVolts')
        if line:
            ad2uv = float(line.split()[-1]) * 1e6  # V → µV

        while True:
            rec = f.read(RECORD_SIZE)
            if len(rec) < RECORD_SIZE:
                break
            ts      = struct.unpack_from('<q', rec, 0)[0]
            n_valid = struct.unpack_from('<I', rec, 16)[0]
            samples = np.frombuffer(rec[20:20 + 512 * 2], dtype='<i2')
            timestamps.append(ts)
            all_samples.append(samples[:n_valid] if n_valid < 512 else samples)

    return np.array(timestamps), np.concatenate(all_samples), ad2uv or 1.0


def read_ntt(filepath: str) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Return (timestamps_us, waveforms, ad2uv) from a Neuralynx .ntt file.

    waveforms has shape (n_spikes, 4_channels, 32_samples).
    """
    RECORD_SIZE = 304  # 8 + 4 + 4 + 8*4 + 4*32*2
    timestamps, waveforms = [], []
    ad2uv = None

    with open(filepath, 'rb') as f:
        header_raw = f.read(HEADER_SIZE).decode('latin-1', errors='replace')
        line = _parse_header_field(header_raw, 'ADBitVolts')
        if line:
            # may list one value per channel, e.g. "0.000 0.000 0.000 0.000"
            nums = [x for x in line.split() if re.match(r'^[\d.eE+-]+$', x)]
            if nums:
                ad2uv = float(nums[0]) * 1e6

        while True:
            rec = f.read(RECORD_SIZE)
            if len(rec) < RECORD_SIZE:
                break
            ts = struct.unpack_from('<q', rec, 0)[0]
            # snData is stored sample-major: snData[32][4] in C → reshape (32,4) then transpose
            wf = np.frombuffer(rec[48:48 + 4 * 32 * 2], dtype='<i2').reshape(32, 4).T
            timestamps.append(ts)
            waveforms.append(wf)

    wf_arr = np.stack(waveforms) if waveforms else np.empty((0, 4, 32), dtype='<i2')
    return np.array(timestamps), wf_arr, ad2uv or 1.0


# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------

def _interpolate_timestamps(ts_records: np.ndarray, n_samples: int,
                             samples_per_record: int, samp_freq: float) -> np.ndarray:
    """Build a per-sample timestamp array from block-level NCS timestamps."""
    t_per_sample_us = 1e6 / samp_freq
    ts_interp = np.full(n_samples, np.nan)
    for k, t0 in enumerate(ts_records):
        start = samples_per_record * k
        end   = start + samples_per_record
        ts_interp[start:end] = t0 + t_per_sample_us * np.arange(samples_per_record)
    return ts_interp


def _bandpass(signal: np.ndarray, lowcut: float, samp_freq: float,
              order: int = 2) -> np.ndarray:
    wn = lowcut / (samp_freq / 2)
    b, a = butter(order, wn, btype='low')
    return filtfilt(b, a, signal)


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def opto_id_tt(
    opto_dir: str,
    unit_ids: list[str] | None = None,
    save_path: str | None = None,
    session_label: str = "",
    n_pulses: int = 10,
    n_trains: int = 10,
    pulse_width_ms: float = 10.0,
    response_window_us: float = 20000.0,
    samp_freq: float = 32000.0,
    shutter_offset_ms: float = 0.8,
) -> list[plt.Figure]:
    """
    Reproduce optoIDTT.m: raster + raw trace + latency/probability + waveform figure
    for every sorted unit found in *opto_dir*.

    Parameters
    ----------
    opto_dir          : path containing Events.nev, TT*.ntt, CSC*.ncs, and *TT*_SS*.txt files
    save_path         : directory for saved PDFs; defaults to opto_dir/figures
    session_label     : string prefix used in figure titles and filenames
    n_pulses          : pulses per train
    n_trains          : number of trains
    pulse_width_ms    : laser pulse duration (ms)
    response_window_us: window after each pulse onset for counting evoked spikes (µs)
    samp_freq         : ADC sampling frequency (Hz)
    shutter_offset_ms : mechanical shutter delay (ms) added to laser onset times
    """
    opto_dir = Path(opto_dir)
    if save_path is None:
        save_path = opto_dir / "figures"
    # Path(save_path).mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load laser events
    # -----------------------------------------------------------------------
    ts_ev, ttl_ev = read_nev(str(opto_dir / "Events.nev"))

    # In this rig TTL=0 is laser-on, TTL=laser_raw_ttl is laser-off.
    # Laser onset = TTL=0 events where the *next* event is laser_raw_ttl (0→high transition),
    # mirroring the MATLAB: laserInd = (biEv(2:end)==1 & biEv(1:end-1)==0) which takes
    # the index of the 0 side of a 0→1 transition.
    # if event contains 1024
    laser = 4
    laser_raw_ttl = 1024

    ttl_ev[ttl_ev == laser_raw_ttl] = laser

    is_laser = (ttl_ev == laser).astype(int)
    onset_mask = (is_laser[:-1] == 0) & (is_laser[1:] == 1)
    laser_on_us = ts_ev[:-1][onset_mask] + shutter_offset_ms * 1000.0  # µs


    pulse_freq_hz = round(1e6 / np.min(np.diff(laser_on_us)))

    t_before_us = 500_000.0
    t_after_us  = 500_000.0
    pulse_inds  = np.arange(0, n_pulses * n_trains, n_pulses)  # first pulse of each train
    resp_win_us = response_window_us
    pw_us       = pulse_width_ms * 1000.0

    # "sham" control offsets relative to train start (µs), outside stimulation window
    half = n_pulses // 2
    sham_offsets = np.concatenate([
        np.linspace(-3_000_000, -t_before_us, half),
        np.linspace(n_pulses * 1e6 / pulse_freq_hz,
                    n_pulses * 1e6 / pulse_freq_hz + 3_000_000, half),
    ])

    # -----------------------------------------------------------------------
    # Find sorted unit files
    # -----------------------------------------------------------------------
    sorted_files = sorted(opto_dir.glob("*TT*_SS*.txt"))
    if not sorted_files:
        print(f"No sorted unit files found in {opto_dir}")
        return []
    if unit_ids is not None:
        sorted_files = [sf for sf in sorted_files if any(uid in sf.name for uid in unit_ids)]
        if not sorted_files:
            print(f"No matching unit files found for {unit_ids} in {opto_dir}")
            return []

    # NCS header AD→µV (use CSC1 as reference — same gain for all channels in session)
    csc1_path = opto_dir / "CSC1.ncs"
    _, _, csc1_ad2uv = read_ncs(str(csc1_path)) if csc1_path.exists() else (None, None, 1.0)

    prev_tt_name = None
    tt_ts_global = None
    tt_wf_global = None
    ntt_ad2uv    = 1.0

    figs = []
    for sf in sorted_files:
        cell_name = sf.stem  # e.g. "TT4_SS_02"

        # unit spike times in µs — cast to int64 to match ntt timestamps exactly
        spike_times = np.loadtxt(str(sf)).ravel().astype(np.int64)
        if spike_times.ndim == 0:
            spike_times = spike_times.reshape(1)

        # -----------------------------------------------------------------------
        # Baseline firing rate (5 s before first laser pulse)
        # -----------------------------------------------------------------------
        pre_window = min(5_000_000.0, laser_on_us[0] - spike_times[0])
        spont_mask = (spike_times > laser_on_us[0] - 5_000_000) & (spike_times < laser_on_us[0])
        spont_freq = 1e6 * spont_mask.sum() / max(pre_window, 1.0)

        # -----------------------------------------------------------------------
        # Build raster per train
        # -----------------------------------------------------------------------
        spike_rast = []
        for j in range(n_trains):
            t0 = laser_on_us[pulse_inds[j]]
            t_last = laser_on_us[pulse_inds[j] + n_pulses - 1]
            win_mask = (spike_times > t0 - t_before_us) & (spike_times < t_last + pw_us + t_after_us)
            rel = spike_times[win_mask] - t0
            spike_rast.append(rel)

        # -----------------------------------------------------------------------
        # Latency / probability per pulse per train
        # -----------------------------------------------------------------------
        spike_lat      = np.full((n_trains, n_pulses), np.nan)
        spike_lat_sham = np.full((n_trains, n_pulses), np.nan)
        spike_num      = np.zeros((n_trains, n_pulses))
        spike_num_sham = np.zeros((n_trains, n_pulses))
        light_spike_times = []

        for j in range(n_trains):
            t0 = laser_on_us[pulse_inds[j]]
            for k in range(n_pulses):
                pulse_t   = laser_on_us[pulse_inds[j] + k]
                sham_t    = sham_offsets[k] + t0

                resp_real = spike_times[(spike_times > pulse_t) &
                                        (spike_times < pulse_t + resp_win_us)]
                resp_sham = spike_times[(spike_times > sham_t) &
                                        (spike_times < sham_t + resp_win_us)]

                if resp_real.size:
                    spike_lat[j, k]  = resp_real[0] - pulse_t
                    spike_num[j, k]  = resp_real.size
                    light_spike_times.extend(resp_real.tolist())
                if resp_sham.size:
                    spike_lat_sham[j, k] = resp_sham[0] - sham_t
                    spike_num_sham[j, k] = resp_sham.size

        light_spike_times = np.array(light_spike_times)
        spont_spike_times = spike_times[~np.isin(spike_times, light_spike_times)]

        avg_lat      = np.nanmean(spike_lat, axis=0)
        sem_lat      = np.nanstd(spike_lat, axis=0) / np.sqrt(n_trains)
        spike_prob   = np.mean(~np.isnan(spike_lat), axis=0)
        spike_prob_s = np.mean(~np.isnan(spike_lat_sham), axis=0)

        # -----------------------------------------------------------------------
        # Load tetrode waveforms (cache if same TT as previous unit)
        # -----------------------------------------------------------------------
        tt_name  = "_".join(cell_name.split("_")[:1])  # e.g. "TT4"
        ntt_path = opto_dir / f"{tt_name}.ntt"

        if tt_name != prev_tt_name and ntt_path.exists():
            tt_ts_global, tt_wf_arr, ntt_ad2uv = read_ntt(str(ntt_path))
            # low-pass at 6 kHz — shape (n_spikes, 4_ch, 32_samp)
            fc_norm = 6000.0 / (samp_freq / 2.0)
            b_lp, a_lp = butter(2, fc_norm, btype='low')
            tt_wf_global = np.zeros_like(tt_wf_arr, dtype=float)
            for ch in range(4):
                for sp_i in range(tt_wf_arr.shape[0]):
                    tt_wf_global[sp_i, ch, :] = filtfilt(b_lp, a_lp,
                                                          tt_wf_arr[sp_i, ch, :].astype(float))
            prev_tt_name = tt_name

        # waveforms for light-evoked vs spontaneous spikes
        if tt_ts_global is not None:
            light_mask = np.isin(tt_ts_global, light_spike_times)
            spont_mask_wf = np.isin(tt_ts_global, spont_spike_times)
            light_wf = ntt_ad2uv * tt_wf_global[light_mask]  # (n, 4, 32)
            spont_wf = ntt_ad2uv * tt_wf_global[spont_mask_wf]
        else:
            light_wf = np.empty((0, 4, 32))
            spont_wf = np.empty((0, 4, 32))

        # -----------------------------------------------------------------------
        # Determine best CSC channel from mean spontaneous waveform
        # -----------------------------------------------------------------------
        tt_num    = int(re.search(r'\d+', tt_name).group())
        csc_nums  = list(range((tt_num - 1) * 4 + 1, tt_num * 4 + 1))
        if spont_wf.shape[0] > 0:
            ch_peaks = [spont_wf[:, ch, :].mean(axis=0).max() for ch in range(4)]
            best_ch  = csc_nums[int(np.argmax(ch_peaks))]
        else:
            best_ch = csc_nums[0]

        # -----------------------------------------------------------------------
        # Load raw CSC trace for best channel
        # -----------------------------------------------------------------------
        csc_path = opto_dir / f"CSC{best_ch}.ncs"
        raw_traces, raw_times = [], []
        if csc_path.exists():
            ts_recs, samp_flat, csc_ad2uv = read_ncs(str(csc_path))
            ts_interp = _interpolate_timestamps(ts_recs, len(samp_flat), 512, samp_freq)
            samp_uv   = csc_ad2uv * samp_flat.astype(float)

            for j in range(n_trains):
                t0    = laser_on_us[pulse_inds[j]]
                t_end = laser_on_us[pulse_inds[j] + n_pulses - 1]
                mask  = (ts_interp > t0 - t_before_us) & (ts_interp < t_end + t_after_us)
                raw_traces.append(samp_uv[mask])
                raw_times.append((ts_interp[mask] - t0) / 1e6)  # seconds relative to train start

        # -----------------------------------------------------------------------
        # Build figure
        # -----------------------------------------------------------------------
        pulse_x_s = np.linspace(0, (n_pulses - 1) * 1000.0 / pulse_freq_hz, n_pulses) / 1000.0
        pulse_x_end_s = pulse_x_s + pw_us / 1e6

        fig = plt.figure(figsize=(18, 12))
        gs  = fig.add_gridspec(4, 3, hspace=0.45, wspace=0.35)

        title = f"{session_label}_{cell_name}"

        # --- Panel 1: spike raster ---
        ax_rast = fig.add_subplot(gs[0, :])
        ax_rast.set_title(title, fontsize=9)
        ax_rast.set_xlabel("Time (µs)")
        ax_rast.set_ylabel("Trial")
        for trial_i, spikes in enumerate(spike_rast):
            if spikes.size:
                ax_rast.vlines(spikes, trial_i, trial_i + 0.8, color='k', linewidth=0.8)
        for px in pulse_x_s:
            ax_rast.axvspan(px * 1e6, (px + pw_us / 1e6) * 1e6, color='b', alpha=0.2)
        ax_rast.set_xlim(-t_before_us, (n_pulses * 1e6 / pulse_freq_hz) + t_after_us)
        ax_rast.set_ylim(n_trains, 0)  # trial 1 at top
        ax_rast.text(0, 0.25, f"baseline {spont_freq:.1f} Hz", fontsize=7,
                     transform=ax_rast.get_xaxis_transform())

        # --- Panel 2: raw traces ---
        ax_raw = fig.add_subplot(gs[1, :])
        if raw_traces:
            row_h = 1.2 * np.ptp(raw_traces[0]) if raw_traces[0].size > 1 else 1000.0
            for j, (tr, tm) in enumerate(zip(raw_traces, raw_times)):
                offset = row_h * j
                ax_raw.plot(tm, tr + offset, color='k', linewidth=0.5)
                for px, px_end in zip(pulse_x_s, pulse_x_end_s):
                    ax_raw.plot([px, px_end],
                                [tr.max() + offset, tr.max() + offset],
                                color='b', linewidth=2)
            # scale bar
            x0 = raw_times[0][0]
            ax_raw.plot([x0, x0], [raw_traces[0].min() - 700, raw_traces[0].min() - 200],
                        'k', linewidth=2)
            ax_raw.plot([x0, x0 + 0.25], [raw_traces[0].min() - 700, raw_traces[0].min() - 700],
                        'k', linewidth=2)
            ax_raw.text(x0 - 0.05, raw_traces[0].min() - 450, "500 µV", ha='right', fontsize=7)
            ax_raw.text(x0 + 0.125, raw_traces[0].min() - 800, "0.25 s", ha='center', fontsize=7)
            ax_raw.set_xlim(raw_times[0][[0, -1]])
        ax_raw.axis('off')

        # --- Panel 3: latency ---
        ax_lat = fig.add_subplot(gs[2, 0])
        ax_lat.errorbar(np.arange(1, n_pulses + 1), avg_lat / 1000.0,
                        yerr=sem_lat / 1000.0, fmt='-b', linewidth=2)
        ax_lat.set_xlabel("Pulse #")
        ax_lat.set_ylabel("Latency (ms)")
        ax_lat.set_xlim([0, n_pulses + 1])
        ax_lat.set_ylim([0, resp_win_us / 1000.0])

        # --- Panel 4: spike probability ---
        ax_prob = fig.add_subplot(gs[2, 1])
        ax_prob.plot(np.arange(1, n_pulses + 1), spike_prob, '-b', linewidth=2, label='laser')
        ax_prob.plot(np.arange(1, n_pulses + 1), spike_prob_s, '-k', linewidth=2, label='control')
        ax_prob.set_xlabel("Pulse #")
        ax_prob.set_ylabel("P(spike)")
        ax_prob.set_xlim([0, n_pulses + 1])
        ax_prob.set_ylim([-0.1, 1.1])
        ax_prob.legend(fontsize=7)

        # --- Panel 5: spike count ---
        ax_cnt = fig.add_subplot(gs[2, 2])
        ax_cnt.plot(np.arange(1, n_pulses + 1), spike_num.mean(axis=0), '-b', linewidth=2)
        ax_cnt.plot(np.arange(1, n_pulses + 1), spike_num_sham.mean(axis=0), '-k', linewidth=2)
        ax_cnt.set_xlabel("Pulse #")
        ax_cnt.set_ylabel("Spike count")
        ax_cnt.set_xlim([0, n_pulses + 1])

        # --- Waveform helper ---
        def _plot_waveforms(ax, wf_array, color):
            if wf_array.shape[0] == 0:
                return
            x_base = np.arange(32)
            for ch in range(4):
                x = x_base + 32 * ch
                if wf_array.shape[0] > 1:
                    mean = wf_array[:, ch, :].mean(axis=0)
                    std  = wf_array[:, ch, :].std(axis=0)
                    se = std / np.sqrt(wf_array.shape[0])
                    ax.fill_between(x, mean - se, mean + se, color=color, alpha=0.3)
                    ax.plot(x, mean, color=color, linewidth=1)
                else:
                    ax.plot(x, wf_array[0, ch, :], color=color, linewidth=1)

        def _decorate_wf_xaxis(ax):
            # ticks at the start of each channel block; label = channel number
            # 32 samples @ 32 kHz = 1 ms per channel
            ms_per_sample = 1000.0 / samp_freq           # 0.03125 ms
            tick_positions = [32 * ch for ch in range(4)]
            tick_labels    = [f'Ch{ch + 1}\n0 ms' for ch in range(4)]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, fontsize=6)
            # add a mid-channel time mark (0.5 ms into each channel)
            half = int(0.5 / ms_per_sample)              # 16 samples
            # for ch in range(4):
            #     ax.axvline(32 * ch + half, color='gray', linewidth=0.4, linestyle=':')
            #     ax.text(32 * ch + half, ax.get_ylim()[0],
            #             '0.5', fontsize=5, ha='center', va='top', color='gray')
            # channel dividers
            for ch in range(1, 4):
                ax.axvline(32 * ch, color='gray', linewidth=0.6, linestyle='--')
            ax.set_xlabel("Channel  (1 ms / channel)", fontsize=7)

        # --- Panel 6: spontaneous waveforms ---
        ax_sp = fig.add_subplot(gs[3, 0])
        _plot_waveforms(ax_sp, spont_wf, 'k')
        ax_sp.set_ylabel("Amplitude (µV)")
        ax_sp.set_title("Spontaneous", fontsize=8)
        _decorate_wf_xaxis(ax_sp)

        # --- Panel 7: light-evoked waveforms ---
        ax_lt = fig.add_subplot(gs[3, 1])
        _plot_waveforms(ax_lt, light_wf, 'b')
        ax_lt.set_ylabel("Amplitude (µV)")
        ax_lt.set_title("Light-evoked", fontsize=8)
        _decorate_wf_xaxis(ax_lt)

        # --- Panel 8: overlay ---
        ax_ov = fig.add_subplot(gs[3, 2])
        _plot_waveforms(ax_ov, light_wf, 'b')
        _plot_waveforms(ax_ov, spont_wf, 'k')
        ax_ov.set_ylabel("Amplitude (µV)")
        ax_ov.set_title("Overlay", fontsize=8)
        _decorate_wf_xaxis(ax_ov)

        fig.tight_layout()
        figs.append(fig)
        # out_path = Path(save_path) / f"{session_label}_{cell_name}_optoID.pdf"
        # fig.savefig(str(out_path), bbox_inches='tight')
        # plt.close(fig)
        # print(f"Saved {out_path.name}")
    return figs

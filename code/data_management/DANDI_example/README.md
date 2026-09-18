# DANDI Example — Dandiset 001950

This folder contains a download script and example notebook for working with the publicly released NWB data from this study.

- **Dandiset**: [DANDI:001950](https://dandiarchive.org/dandiset/001950)
- **Manuscript**: https://www.nature.com/articles/s41586-026-11026-0

Each NWB file contains one session and includes behavioral trials, spike times, waveform features, fiber photometry signals, pupil diameter, and tongue movement data where available.

---

## Files

| File | Purpose |
|------|---------|
| [`download_dandiset_001950.py`](download_dandiset_001950.py) | Download all NWB files from DANDI to a local directory |
| [`DANDI_load_example.ipynb`](DANDI_load_example.ipynb) | Example notebook showing how to load and plot behavior, ephys, pupil, and photometry data |

---

## Step 1 — Download the dandiset

Install the DANDI client if you haven't already:

```bash
pip install dandi
```

Then run the download script, pointing `--data_dir` to wherever you want the files saved:

```bash
python download_dandiset_001950.py --data_dir /path/to/your/data
```

This will download the full dandiset into a `001950/` subfolder inside the directory you specify. The download can take a while depending on how many sessions you need — you can also download individual files directly from the [DANDI web interface](https://dandiarchive.org/dandiset/001950).

---

## Step 2 — Run the example notebook

Open [`DANDI_load_example.ipynb`](DANDI_load_example.ipynb). The notebook walks through four data modalities with example plots:

1. **Behavior** — trial table, smoothed choice/reward, block structure
2. **Pupil** — raw pupil diameter trace + go-cue-aligned PSTH (go vs. no-go)
3. **Ephys** — raw spike raster + firing rate with events, go-cue-aligned raster sorted by first-spike latency
4. **Fiber photometry** — raw ΔF/F trace + go-cue-aligned PSTH (reward vs. no-reward)

### Parameters to update before running

**Cell 2 — `data_dir`**: set this to the directory you passed to `--data_dir` above.

```python
data_dir = "/path/to/your/data"   # ← update this
```

The notebook constructs all file paths as `{data_dir}/001950/{subject_id}/{session_filename}`, so this is the only path you need to change.

---

**Cells 3–6 — example filenames**: the notebook pre-selects four representative sessions. You can replace any of these with any other filename from the dandiset.

```python
example_beh   = 'behavior_ZS061_2021-04-08_18-01-30_combined.nwb'
example_pupil = 'behavior_ZS062_2021-05-06_15-46-14_combined.nwb'
example_FP    = 'behavior_754896_2025-01-03_17-20-19_combined.nwb'
example_ephys = 'behavior_754897_2025-03-13_11-20-42_combined.nwb'
```

---

# aind-beh-physiology-analysis

### Table of Contents
- aind-beh-physiology-analysis
    - Table of Contents
    - Overview
- Code
    - Data Preparation Pipeline
    - Analysis Notebooks
        - Behavior
        - Behavior and Electrophysiology
        - Pupil
        - Electrophysiology
        - Waveform and Spatial Organization
        - Behavior and Photometry
- Data
    - NWB Data on DANDI (001950)
    - Electrophysiology Recordings with and without behavior
    - Anatomical Registration and Spatial Mapping
    - Behavior Video Tracking
    - MERFISH Spatial Transcriptomics
    - Retrograde Tracing
- Running on Code Ocean
    - Reproducible Run
    - Re-attaching Data
    - Optional Run Flags
    - Customizing Directory Paths



### Overview

This capsule contains analysis code for a study of physiology of LC NE neurons and behavior in a dynamic foraging task, focusing on the distribution of neuron properties across space. 

- **Manuscript**: https://www.nature.com/articles/s41586-026-11026-0
- **Github Repository**: https://github.com/AllenNeuralDynamics/aind-beh-ephys-analysis
- **Code Ocean Capsule**: NEED TO ADD

---

# Code

The code is organized into two sections, a data preparation pipeline, and then dedicated notebooks for each analysis. The data preparation pipeline aggregates and preprocesses data and generates combined tables and metrics. The analysis notebooks in `code/beh_ephys_analysis/session_combine/manuscript_figures/` use those aggregated results to produce the figures in the manuscript.

## Data Preparation Pipeline

Before running any analysis notebook, the **figure preparation scripts** in [`code/beh_ephys_analysis/session_combine/figure_preparation`](code/beh_ephys_analysis/session_combine/figure_preparation) must be executed first. These scripts aggregate and preprocess data across all sessions and animals, generating combined tables and derived metrics that the analysis notebooks depend on. 

> [!CAUTION] 
> The scripts must be run **in the exact order** specified in the [`sequence.txt`](code/beh_ephys_analysis/session_combine/figure_preparation/sequence.txt) file, as later steps depend on outputs from earlier ones. These scripts must be run before any analysis notebook. 


**Estimated run time for each script:**
1. [`make_combined_unit_tbl.py`](code/beh_ephys_analysis/session_combine/figure_preparation/make_combined_unit_tbl.py) - 10.3 min
2. [`antidromic_generation.py`](code/beh_ephys_analysis/session_combine/figure_preparation/antidromic_generation.py) - < 1 min
3. [`waveform_generation_np.py`](code/beh_ephys_analysis/session_combine/figure_preparation/waveform_generation_np.py) - < 1 min
4. [`waveform_generation_tt.py`](code/beh_ephys_analysis/session_combine/figure_preparation/waveform_generation_tt.py) - < 1 min
5. [`basic_ephys_generation.py`](code/beh_ephys_analysis/session_combine/figure_preparation/basic_ephys_generation.py) - 10.8 min
6. [`behavior_metrics_generation.py`](code/beh_ephys_analysis/session_combine/figure_preparation/behavior_metrics_generation.py) - 8.9 min
7. [`acg_generation.py`](code/beh_ephys_analysis/session_combine/figure_preparation/acg_generation.py) - 2.1 min
8. [`response_tstats_generation.py`](code/beh_ephys_analysis/session_combine/figure_preparation/response_tstats_generation.py) - 4.7 min
9. [`outcome_window_generation_parallel.py`](code/beh_ephys_analysis/session_combine/figure_preparation/outcome_window_generation_parallel.py) - 5.6 min
10. [`beh_combined_outcome_generation.py`](code/beh_ephys_analysis/session_combine/figure_preparation/beh_combined_outcome_generation.py) - 12.9 min
11. [`photometry_generation.py`](code/beh_ephys_analysis/session_combine/figure_preparation/photometry_generation.py) - 12.2 min
12. [`licks_prep.py`](code/beh_ephys_analysis/session_combine/figure_preparation/licks_prep.py) - 10.7 min

**Total estimated time:** ~80 min

> [!NOTE]
> These run times were measured on the current Code Ocean capsule's compute resources and will vary on other machines. Reference specs: Intel Xeon Platinum 8259CL @ 2.50 GHz, 8 cores / 16 threads, 124 GiB RAM, no GPU.



## Behavior Analysis

### Lick Examples
**Notebook:** [`F_example_licks.ipynb`](code/beh_ephys_analysis/session_combine/manuscript_figures/F_example_licks.ipynb)\
Generates example lick-raster and lick-rate PSTH figures for representative sessions, combining video-detected and behavior-sensor lick detection, split by in-trial reward outcome.\
**Run time:** ~1 min\
**Manuscript figure panels:** Fig. E9 b,c_top,c_bottom,g_bottom\
**Prerequisites:**
- Per-session behavioral and video-based lick detection data

### Hit and Miss
**Notebook:** [`F_hit_miss.ipynb`](code/beh_ephys_analysis/session_combine/manuscript_figures/F_hit_miss.ipynb)\
Analyzes behavioral and neural factors underlying hit vs. miss responses to the go cue. Fits a logistic regression GLM predicting upcoming hit/miss from reward history across multiple lags, both per-session and per-animal, and at the population level with confidence intervals.\
**Run time:** ~2.5 min\
**Manuscript figure panels:** Fig. 5\
**Prerequisites:**
- `combined_beh_sessions.pkl` (from [`behavior_metrics_generation.py`](code/beh_ephys_analysis/session_combine/figure_preparation/behavior_metrics_generation.py))

### Choice Analysis
**Notebook:** [`F_behavior_w_FP.ipynb`](code/beh_ephys_analysis/session_combine/manuscript_figures/F_behavior_w_FP.ipynb)\
Characterizes choice-history dependent behavior (logistic regression GLM over reward/choice history, up to 15-trial lags) restricted to the subset of sessions with simultaneous fiber-photometry recordings. Applies behavioral quality-control filtering (`beh_only.json`) and visualizes which sessions pass or fail each quality criterion.\
**Run time:** ~4 min\
**Manuscript figure panels:** Fig. 4 c; Fig. 5 e; Fig. 6 b(left); Fig. E9 a,l,m\
**Prerequisites:**
- `combined_beh_sessions.pkl` filtered for photometry sessions (from [`behavior_metrics_generation.py`](code/beh_ephys_analysis/session_combine/figure_preparation/behavior_metrics_generation.py) / [`photometry_generation.py`](code/beh_ephys_analysis/session_combine/figure_preparation/photometry_generation.py))

### Lick-Train Statistics
**Notebook:** [`F_lick_train_analysis.ipynb`](code/beh_ephys_analysis/session_combine/manuscript_figures/F_lick_train_analysis.ipynb)\
Analyzes lick-train statistics across sessions: stay-vs-switch lick-latency distributions, spontaneous lick-side preference versus task reward ratio, and logistic-regression z-statistics relating reward-ratio deviation to spontaneous licking.\
**Run time:** ~2 min\
**Manuscript figure panels:** Fig. E9 d,h,i,j\
**Prerequisites:**
- Session behavioral tables (`session_assets.csv` / Hopkins session assets) and video-based lick detection


## Behavior and Electrophysiology Analysis

### Single Unit Examples
**Notebook:** [`F_ephys_behavior_examples.ipynb`](code/beh_ephys_analysis/session_combine/manuscript_figures/F_ephys_behavior_examples.ipynb)\
Generates curated single-unit example raster + PSTH figures (silicon probe and tetrode recordings) for stay-vs-switch, respond-vs-ignore, hit-vs-miss, RPE-scaling, and antidromic-stimulation behavioral splits, plus lick-bout-aligned examples for task and spontaneous licks.\
**Run time:** ~7 min\
**Manuscript figure panels:** Fig. 4 b,f; Fig. 5 a; Fig. 6 b(right),d(left,right); Fig. E10 j(top,bottom); Fig. E12 d(left,right),o,p\
**Prerequisites:**
- `combined_unit_tbl.pkl` (from [`make_combined_unit_tbl.py`](code/beh_ephys_analysis/session_combine/figure_preparation/make_combined_unit_tbl.py))
- Per-session spike data

### Single Neuron encoding
**Notebook:** [`F_ephys_behavior_action&outcome.ipynb`](code/beh_ephys_analysis/session_combine/manuscript_figures/F_ephys_behavior_action&outcome.ipynb)\
The primary neural coding notebook. Analyzes how single units encode behavioral variables related to actions and outcomes. Performs GLM-based analysis to identify neurons tuned to task variables including outcome, chosen action (Q-value), and policy updates, maps functionally-defined neurons in CCF space, and compares neural encoding with photometry signals.\
**Run time:** ~20 min\
**Manuscript figure panels:** Fig. 4 g; Fig. 5 b,c,f,g,k(left,right); Fig. 6 a,e,f,g,h; Fig. E12 a,b,c,e,f(top,bottom); Fig. E14 i(left,right)\
**Prerequisites:**
- Combined unit table with quality control applied, waveform features, basic ephys metrics, and GLM model results for behavioral-variable encoding (see [`make_combined_unit_tbl.py`](code/beh_ephys_analysis/session_combine/figure_preparation/make_combined_unit_tbl.py), [`outcome_window_generation_parallel.py`](code/beh_ephys_analysis/session_combine/figure_preparation/outcome_window_generation_parallel.py))

### Task vs. Spontaneous Licking Neural Responses
**Notebook:** [`F_spont_choice_lick_neuron.ipynb`](code/beh_ephys_analysis/session_combine/manuscript_figures/F_spont_choice_lick_neuron.ipynb)\
Compares neural activity during in-trial (task-related) and out-of-trial (spontaneous) licking events, testing whether neurons that respond to licks during task performance show similar responses during spontaneous licking and whether these response properties are spatially organized in the brain.\
**Run time:** ~9 min\
**Manuscript figure panels:** Fig. E12 q\
**Prerequisites:**
- Combined unit table with quality control metrics, behavioral session data, lick detection from video and behavioral data, CCF coordinate registration


## Pupil Analysis

### Pupil-Neural Examples
**Notebook:** [`F_pupil_examples.ipynb`](code/beh_ephys_analysis/session_combine/manuscript_figures/F_pupil_examples.ipynb)\
Shows example-session illustrations of the relationship between pupil dilation dynamics and neural activity, including raw traces, spike-pupil cross-correlations, and trial-aligned spike-rate and pupil-dilation PSTHs split by task variables.\
**Run time:** ~3 min\
**Manuscript figure panels:** Fig. E13 b,c,e(top/bottom,left/right),g(left),i,k,m\
**Prerequisites:**
- Preprocessed per-session pupil data, unit quality metrics and behavioral regression results, waveform features and basic ephys characterization, session-level behavioral performance metrics

### Pupil-Neural Coupling
**Notebook:** [`F_pupil_beh_ephys.ipynb`](code/beh_ephys_analysis/session_combine/manuscript_figures/F_pupil_beh_ephys.ipynb)\
Analyzes the relationship between pupil dilation dynamics and neural activity during the task. Computes spike-pupil cross-correlations and pupil auto-correlations, fits exponential decay to characterize coupling and intrinsic timescales, and relates pupil features to behavioral encoding, waveform features, and task response properties.\
**Run time:** ~3 min\
**Manuscript figure panels:** Fig. E13 d,f,g(right),h,j,l,n\
**Prerequisites:**
- Preprocessed per-session pupil data, unit quality metrics and behavioral regression results, waveform features and basic ephys characterization, session-level behavioral performance metrics


## Electrophysiology Analysis

### Opto-Tagging Examples
**Notebook:** [`F_ephys_opto_examples.ipynb`](code/beh_ephys_analysis/session_combine/manuscript_figures/F_ephys_opto_examples.ipynb)\
Generates single-unit example rasters, PSTHs, and raw voltage traces for opto-tagging and antidromic-stimulation experiments, illustrating light-evoked spiking and collision tests for identified LC-NE and projection neurons.\
**Run time:** ~8 min\
**Manuscript figure panels:** Fig. E10 c,d(left,right),i\
**Prerequisites:**
- Per-session spike and raw opto-stimulation trace data

### Antidromic stimulation
**Notebook:** [`F_antidromic_combined.ipynb`](code/beh_ephys_analysis/session_combine/manuscript_figures/F_antidromic_combined.ipynb)\
Identifies and characterizes antidromically-activated projection neurons across sessions. Applies a tiered classification system (tier 1: jitter, collision test, and antidromic response criteria; tier 2: looser thresholds) to classify PrL → subcortical projection neurons.\
**Run time:** ~1 min\
**Manuscript figure panels:** Fig. E10 k,l,m\
**Prerequisites:**
- `combined_unit_tbl.pkl` (from [`make_combined_unit_tbl.py`](code/beh_ephys_analysis/session_combine/figure_preparation/make_combined_unit_tbl.py))
- `combined_antidromic_tbl.pkl` (from [`antidromic_generation.py`](code/beh_ephys_analysis/session_combine/figure_preparation/antidromic_generation.py))

### Comprehensive characterization of single units
**Notebook:** [`F_basic_ephys.ipynb`](code/beh_ephys_analysis/session_combine/manuscript_figures/F_basic_ephys.ipynb)\
Comprehensive characterization of electrophysiological unit properties across all recorded neurons. Analyzes baseline and response firing rates, burst properties (ACG fit parameters), waveform features, and opto-tagging quality. Fits OLS models examining how intrinsic properties predict the degree of outcome vs. action coding.\
**Run time:** ~1 min\
**Manuscript figure panels:** Fig. 4 h; Fig. E10 e,f(top/bottom,left/right)\
**Prerequisites:**
- `combined_unit_tbl.pkl` (from [`make_combined_unit_tbl.py`](code/beh_ephys_analysis/session_combine/figure_preparation/make_combined_unit_tbl.py))
- `combined_basic_ephys.pkl` (from [`basic_ephys_generation.py`](code/beh_ephys_analysis/session_combine/figure_preparation/basic_ephys_generation.py))
- `combined_acg.pkl` (from [`acg_generation.py`](code/beh_ephys_analysis/session_combine/figure_preparation/acg_generation.py))

### Cross-Correlation
**Notebook:** [`F_cross_correlation.ipynb`](code/beh_ephys_analysis/session_combine/manuscript_figures/F_cross_correlation.ipynb)\
Analyzes spike train temporal structure using auto-correlations and cross-correlations. Computes pairwise cross-correlations between neurons (including across PrL and S1) to assess functional connectivity, and visualizes correlation structure mapped to CCF coordinates.\
**Run time:** ~7 min\
**Manuscript figure panels:** Fig. 4 e; Fig. E12 r(left,mid,right),s,t\
**Prerequisites:**
- `combined_unit_tbl.pkl` (from [`make_combined_unit_tbl.py`](code/beh_ephys_analysis/session_combine/figure_preparation/make_combined_unit_tbl.py))
- Per-session spike data


## Waveform and Spatial Organization

### Action Potential Waveforms
**Notebook:** [`F_waveform_space.ipynb`](code/beh_ephys_analysis/session_combine/manuscript_figures/F_waveform_space.ipynb)\
Characterizes action potential waveform morphology across the unit population (silicon probe recordings). Extracts waveform shape features, reduces via PCA, maps onto CCF coordinates with brain mesh overlays, and tests spatial dependence statistics. Opto-tagged units are overlaid to reveal waveform-type identity.\
**Run time:** ~8 min\
**Manuscript figure panels:** Fig. E11 b,c,d,e,f\
**Prerequisites:**
- `combined_unit_tbl.pkl` (from [`make_combined_unit_tbl.py`](code/beh_ephys_analysis/session_combine/figure_preparation/make_combined_unit_tbl.py))
- `combined_waveform_NP.pkl` (from [`waveform_generation_np.py`](code/beh_ephys_analysis/session_combine/figure_preparation/waveform_generation_np.py))

### Tetrode recorded units 
**Notebook:** [`F_waveform_space_tetrode.ipynb`](code/beh_ephys_analysis/session_combine/manuscript_figures/F_waveform_space_tetrode.ipynb)\
Identical waveform analysis applied exclusively to tetrode-recorded units, producing the same spatial feature maps for the tetrode recording subset.\
**Run time:** ~1 min\
**Manuscript figure panels:** Fig. E11 g,h,i\
**Prerequisites:**
- `combined_unit_tbl.pkl` (from [`make_combined_unit_tbl.py`](code/beh_ephys_analysis/session_combine/figure_preparation/make_combined_unit_tbl.py))
- `combined_waveform_TT.pkl` (from [`waveform_generation_tt.py`](code/beh_ephys_analysis/session_combine/figure_preparation/waveform_generation_tt.py))

### Spatial Axis
**Notebook:** [`F_spatial-axis-comparison.ipynb`](code/beh_ephys_analysis/session_combine/manuscript_figures/F_spatial-axis-comparison.ipynb)\
Integrates three datasets to compare cellular organization axes: electrophysiology waveform features, MERFISH spatial transcriptomics (~2,200 cells), and retrograde tracing. Fits a linear spatial axis to each dataset and compares principal spatial gradients across data modalities.\
**Run time:** ~13 min\
**Manuscript figure panels:** Fig. 5 d,h; Fig. E15 a,b,c,d\
**Prerequisites:**
- `combined_waveform_NP.pkl` (from [`waveform_generation_np.py`](code/beh_ephys_analysis/session_combine/figure_preparation/waveform_generation_np.py))
- MERFISH data, retrograde tracing data

## Behavior and Photometry Analysis

### Photometry Examples
**Notebook:** [`F_photometry_examples.ipynb`](code/beh_ephys_analysis/session_combine/manuscript_figures/F_photometry_examples.ipynb)\
Plots example single-session fiber photometry traces, including raw signal processing steps, motion-corrected ΔF/F, GCaMP/isosbestic power spectra, and go-cue/lick-aligned PSTHs.\
**Run time:** ~2 min\
**Manuscript figure panels:** Fig. 5 i(right); Fig. E9 e; Fig. E14 c,d,e,f,h(left),j\
**Prerequisites:**
- Per-session photometry data

### Photometry PSTHs
**Notebook:** [`F_photometry_tuning_psth.ipynb`](code/beh_ephys_analysis/session_combine/manuscript_figures/F_photometry_tuning_psth.ipynb)\
Computes tuning curves and PSTHs for fiber photometry signals by binning the signal into 6 prediction error (PE) levels aligned to choice time. Reveals how the PrL photometry signal scales as a function of reward prediction error.\
**Run time:** ~4 min\
**Manuscript figure panels:** Fig. 5 j; Fig. 6 i,j,k,l; Fig. E14 g,h(right)\
**Prerequisites:**
- Photometry GLM results (from [`photometry_generation.py`](code/beh_ephys_analysis/session_combine/figure_preparation/photometry_generation.py))
- Per-session photometry data

### Fiber Placement Location
**Notebook:** [`F_fiber_location.ipynb`](code/beh_ephys_analysis/session_combine/manuscript_figures/F_fiber_location.ipynb)\
Reconstructs and visualizes fiber-photometry optic-fiber placement locations in prelimbic cortex (PL) across mice, registering fiber-tip coordinates to the Allen CCF and rendering them with brain-region meshes via brainrender/BrainGlobe.\
**Run time:** <1 min\
**Manuscript figure panels:** Fig. E14 b\
**Prerequisites:**
- Fiber CCF coordinates (`/data/fiber_ccf/PL_ccf_coordinates_pir.csv`), BrainGlobe atlas

---

# Data Organization

### NWB Data on DANDI (001950)

All physiology and behavioral session data are packaged as [NWB](https://www.nwb.org/) files, one file per session, and publicly available on DANDI:

- **Dandiset**: [DANDI:001950](https://dandiarchive.org/dandiset/001950)

**Downloading the dandiset:**

Install the DANDI client, then run:

```bash
python code/data_management/DANDI_example/download_dandiset_001950.py --data_dir /path/to/save
```

See [`download_dandiset_001950.py`](code/data_management/DANDI_example/download_dandiset_001950.py) for details.

**Loading NWB data:**

See [`DANDI_load_example.ipynb`](code/data_management/DANDI_example/DANDI_load_example.ipynb) for an example of how to stream or load NWB files from the dandiset and access spike times, behavioral events, pupil data, photometry signal and metadata.

### Electrophysiology Recordings with and without behavior
- `all 'raw' data`: raw electrophysiology data with and without behavior，include high speed video for bottom and side of the face and whole body. 
- `all 'sorted' data`: Kilosort data, containing single neuron activity, cluster quality and probe drift estimation.
- `all 'sorted_curated' data`: Kilosort data after manual curation. Single neuron activity here is used for analysis.
- `scratch_data`: Animal/session-structured derived data generated by the session processing pipeline in [`beh_ephys_processing.ipynb`](code/beh_ephys_analysis/ani_session_processing/beh_ephys_processing.ipynb). Organized as `{animal_id}/{session_id}/` with the following outputs:
  - **Alignment**: `qm.json` files with behavior/sound/ephys stream alignment metrics and session cut points
  - **Behavior analysis plots**: Choice/reward patterns, GLM fits, lick analysis (rasters, feature space, video-based lick detection), lick trains by side
  - **Ephys processing** (raw and curated): Spike times, waveform analyzers, drift trial tables, and quality control figures
  - **Opto-tagging analysis** (raw and curated): Opto response dataframes (`opto_session.csv`), waveform similarity metrics, correlation matrices (laser-triggered auto/cross-correlograms), antidromic response statistics, and per-unit tagging figures
  - **Behavior + ephys analysis plots**: Neural activity aligned to behavioral events (go-cue, response), unit PSTHs split by choice/outcome, burst analysis, LFP power spectra, waveform quality checks with behavioral alignment
  - **Animal-level summaries**: GLM choice-history models, lick statistics across sessions

### Anatomical Registration and Spatial Mapping
- `alignment_fix`: Single neuron location in CCF space inferred using IBL gui.
- `dorsal_edges`: Dorsal edge of LC, labeled by hand and converted to CCF locations.
- `LC_percentile_meshs`: LC mesh generated by Drew Friedmann.

### Behavioral Video Tracking
- `all_tongue_movements`: Tongue movement data derived from pose tracking of high speed video. Generated by Matt Becker.

### MERFISH Spatial Transcriptomics
- `merfish_data`: Merfish data used to compare spatial distribution of gene expression with other modalities. Generated by Shuonan Chen.

### Retrograde Tracing
- `LC_retro`: Retrograde labeling data with location in CCF space. Generated by Polina Kosillo.

---

# Running on Code Ocean

## Reproducible Run

The full pipeline is executed by clicking **"Reproducible Run"** on the Code Ocean capsule. This triggers [`code/run`](code/run), a bash script that calls:

```bash
python -u code/run_capsule.py
```

[`run_capsule.py`](code/run_capsule.py) orchestrates the complete pipeline in three phases:

**Phase 1 — Figure preparation scripts** (~80 min)\
All scripts listed in [`sequence.txt`](code/beh_ephys_analysis/session_combine/figure_preparation/sequence.txt) are executed in order using the Python interpreter. Each script aggregates and preprocesses data across sessions, writing output files that downstream notebooks depend on. See the [Data Preparation Pipeline](#data-preparation-pipeline) section for the full list and per-script run times.

**Phase 2 — Manuscript figure notebooks** (~102 min)\
All notebooks listed in [`fig_notebook_list.txt`](code/beh_ephys_analysis/session_combine/manuscript_figures/fig_notebook_list.txt) are executed in order via `nbconvert --execute --inplace`. Notebooks are run with no timeout (`--ExecutePreprocessor.timeout=-1`). See the [Analysis Notebooks](#behavior-analysis) sections for per-notebook run times.

**Phase 3 — Output reorganization**\
After all notebooks complete, [`figure_csv_organization.py`](code/data_management/figure_csv_organization.py) reorganizes the figure and CSV outputs in the results directory.

> [!NOTE]
> Total estimated run time is ~3 hours. Run times were measured on Code Ocean compute resources: Intel Xeon Platinum 8259CL @ 2.50 GHz, 8 cores / 16 threads, 124 GiB RAM, no GPU.

## Re-attaching Data

If the capsule's data assets need to be re-attached before running (e.g., after a data update), uncomment the `run_data_attachment` call near the top of the `run()` function in [`run_capsule.py`](code/run_capsule.py):

```python
# run_data_attachment(check_only=check_only)  # ← uncomment this line
```

This runs [`attach_all_data_capsule.py`](code/data_management/attach_all_data_capsule.py), which reads the session asset lists (`session_assets.csv`, `hopkins_session_assets.csv`, `hopkins_FP_session_assets.csv`, `combined_assets.csv`) and attaches all required raw data, sorted, sorted-curated, and model assets to the capsule via the Code Ocean API. It requires the `CO_CAPSULE_ID` and `API_SECRET` environment variables to be set.

## Optional Run Flags

`run_capsule.py` accepts two optional flags, which can be passed through `code/run` or invoked directly:

| Flag | Effect |
|------|--------|
| `--check-only` | Validates that every script and notebook listed in the sequence files exists, without executing anything. Equivalent to setting `dry_run=1` in `code/run`. |
| `--update-timing` | Writes per-step timing to `timing_report.csv` in each directory after every script/notebook completes. Omit this flag for standard reproducible runs so that files under `code/` are not modified. |

Example — validate the sequence without running:
```bash
python -u code/run_capsule.py --check-only
```

## Customizing Directory Paths

All path resolution is centralized in [`code/beh_ephys_analysis/utils/capsule_migration.py`](code/beh_ephys_analysis/utils/capsule_migration.py). The `capsule_root()` function resolves the repository root in the following order:

1. `$CAPSULE_ROOT` environment variable (if set)
2. `/root/capsule` (Code Ocean default)
3. Repository root derived from the file's own location

The `capsule_directories()` function returns all standard paths used by scripts and notebooks:

| Key | Default path |
|-----|-------------|
| `output_dir` | `<root>/scratch/results` |
| `manuscript_fig_dir` | `<root>/scratch/results/manuscript/figures` |
| `manuscript_fig_prep_dir` | `<root>/scratch/results/manuscript/prep` |
| `derived_dir` | `<root>/data/scratch_data` |
| `data_dir` | `<root>/data` |

To redirect outputs or data, either set `CAPSULE_ROOT` or edit the `capsule_directories()` function directly. All scripts and notebooks that call `capsule_directories()` will pick up the change automatically.

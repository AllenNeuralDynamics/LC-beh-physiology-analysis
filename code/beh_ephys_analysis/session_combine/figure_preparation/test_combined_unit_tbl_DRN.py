from make_combined_unit_tbl_DRN import process_session, df
import pandas as pd
from utils.opto_utils import opto_metrics
from utils.beh_functions import session_dirs, get_session_tbl, get_unit_tbl
from opto_waveforms_preprocessing import re_filter_opto_waveforms
import numpy as np
import os
import pickle
import spikeinterface as si
from utils.ephys_functions import load_drift

# session = "behavior_844920_2026-06-10_11-30-14" #not working - because missing eu/corr? Ok after rerunning opto_tagging_summary
# #session = "behavior_835444_2026-02-18_13-01-55" #working - because has eu/corr?
# row = df.loc[df["session_id"] == session].iloc[0]

# result = process_session(
#     row["session_id"],
#     row["behavior"],
#     row["side"],
#     row["probe"],
#     None,
# )

# print(result.keys())

# result_df = pd.DataFrame(result)
# pd.set_option('display.max_columns', None)
# print(result_df.shape)
# print(result_df.head())


# pkl_path = "/root/capsule/scratch/results/manuscript/prep/combined_unit_tbl/combined_unit_tbl.pkl"
# df = pd.read_pickle(pkl_path)

# print(df.shape)
# print(df.columns.tolist())

# df = pd.read_pickle(pkl_path)

# print("Shape:", df.shape)

# print("Combined table")
# print("units:", len(df))
# print("sessions:", df["session"].nunique())


# affected_sessions = [
#     "behavior_808655_2025-09-16_10-53-22",
#     "behavior_826164_2026-01-29_11-09-03",
#     "behavior_835444_2026-02-18_13-01-55",
#     "behavior_838332_2026-03-13_12-29-22",
#     "behavior_841861_2026-04-02_13-12-47",
#     "behavior_841861_2026-04-03_13-18-22",
#     "behavior_841859_2026-04-08_11-03-18",
#     "behavior_843661_2026-04-28_09-49-53",
#     "behavior_841598_2026-05-01_14-15-09",
#     "behavior_844920_2026-06-10_11-30-14",
#     "behavior_844920_2026-06-12_11-03-09",
# ]
#"None" wf unit: (behavior_841599_2026-05-13_13-19-27, unit 147)

# for session in affected_sessions:
#     print(f"\n===== {session} =====")

#     try:
#         re_filter_opto_waveforms(
#             session,
#             "raw",
#             opto_only=True,
#             load_sorting_analyzer=True,
#         )
#         print("DONE")

#     except Exception as e:
#         print(f"FAILED: {e}")


session = "behavior_838332_2026-03-13_12-29-22"
unit_id = 125

unit_tbl = get_unit_tbl(session, "raw")
session_dir = session_dirs(session)

row = unit_tbl.loc[
    unit_tbl["unit_id"] == unit_id
].iloc[0]

spike_times = row["spike_times"]

unit_drift = load_drift(
    session,
    unit_id,
    data_type="raw"
)

print("Total spikes:", len(spike_times))
print("Drift info:", unit_drift["ephys_cut"])

start, end = unit_drift["ephys_cut"]

keep = (
    (spike_times >= start)
    & (spike_times < end)
)

print("Spikes after drift cut:", keep.sum())
print("Valid duration:", end - start, "sec")
print("Valid duration:", (end - start) / 60, "min")
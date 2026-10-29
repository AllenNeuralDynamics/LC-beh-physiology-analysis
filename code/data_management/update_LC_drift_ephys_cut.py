"""
Propagate updated ephys_cut values from scratch/combined/drift back to each
session's LC_drift_tbl.csv in opto/curated, and fix [None, None] -> [nan, nan]
in all session LC_drift_tbl files.

Writes LC_curated.txt listing every session whose ephys_cut was changed.
"""

import os
import pandas as pd

SCRATCH        = "/root/capsule/scratch"
COMBINED_DRIFT = os.path.join(SCRATCH, "combined", "drift")
DATA_MGMT      = os.path.dirname(os.path.abspath(__file__))


def session_file_path(session):
    animal = session.split("_")[1]
    return os.path.join(
        SCRATCH, animal, session, "ephys", "opto", "curated",
        f"{session}_LC_drift_tbl.csv",
    )


def fix_none(path):
    """Replace [None, None] with [nan, nan] at the text level."""
    with open(path) as f:
        text = f.read()
    fixed = text.replace("[None, None]", "[nan, nan]")
    if fixed != text:
        with open(path, "w") as f:
            f.write(fixed)
        return True
    return False


updated_sessions = []

for fname in sorted(os.listdir(COMBINED_DRIFT)):
    if not fname.endswith("_LC_drift_tbl.csv"):
        continue

    session      = fname.replace("_LC_drift_tbl.csv", "")
    combined_path = os.path.join(COMBINED_DRIFT, fname)
    sess_path    = session_file_path(session)

    if not os.path.exists(sess_path):
        print(f"[SKIP] session file missing: {sess_path}")
        continue

    # combined: unit_id is the row index; session: unit_id is a regular column
    combined_df = pd.read_csv(combined_path, index_col=0)
    sess_df     = pd.read_csv(sess_path,     index_col=0)

    if "unit_id" not in sess_df.columns:
        print(f"[SKIP] no unit_id column in session file: {session}")
        continue

    # Build unit_id -> ephys_cut map from combined (unit_id is the index there)
    if combined_df.index.name == "unit_id":
        combined_map = combined_df["ephys_cut"].to_dict()
    elif "unit_id" in combined_df.columns:
        combined_map = combined_df.set_index("unit_id")["ephys_cut"].to_dict()
    else:
        print(f"[SKIP] cannot find unit_id in combined file: {session}")
        continue

    # Update ephys_cut in session file from combined, matched by unit_id
    changed = False
    for uid, new_cut in combined_map.items():
        mask = sess_df["unit_id"] == uid
        if not mask.any():
            continue
        old_cut = sess_df.loc[mask, "ephys_cut"].values[0]
        if str(old_cut) != str(new_cut):
            sess_df.loc[mask, "ephys_cut"] = new_cut
            changed = True

    if changed:
        sess_df.to_csv(sess_path)
        updated_sessions.append(session)
        print(f"[UPDATED] {session}")

    # Fix [None, None] -> [nan, nan] in session file
    fix_none(sess_path)

# Fix [None, None] in sessions that weren't otherwise changed
for fname in sorted(os.listdir(COMBINED_DRIFT)):
    if not fname.endswith("_LC_drift_tbl.csv"):
        continue
    session   = fname.replace("_LC_drift_tbl.csv", "")
    sess_path = session_file_path(session)
    if session not in updated_sessions and os.path.exists(sess_path):
        if fix_none(sess_path):
            print(f"[FIXED None] {session}")

# Write LC_curated.txt
lc_curated = os.path.join(DATA_MGMT, "LC_curated.txt")
with open(lc_curated, "w") as f:
    for s in sorted(updated_sessions):
        f.write(s + "\n")

print(f"\nDone. {len(updated_sessions)} sessions updated -> {lc_curated}")

# %% Plot non-noise/non-artifact units in CCF 3D space;
#    strict optotagged/candidate units are red, non-tagged units are gray,
#    and one requested session/unit is highlighted in yellow.
#
# Example notebook use:
#     from ccf_highlight_unit_k3d import plot_ccf_highlight_unit
#     plot, html_path, target_row = plot_ccf_highlight_unit(
#         "behavior_835444_2026-02-18_13-01-55", 83
#     )
#
# Example terminal use:
#     /opt/conda/bin/python ccf_highlight_unit_k3d.py \
#         behavior_835444_2026-02-18_13-01-55 83

import argparse
import glob
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------

MASTER_PATH = "/root/capsule/scratch/combined/master_unit_tables/master_all_units_opto_ccf_raw_fix.pkl"

# If MASTER_PATH does not exist, this will search here for the newest master pkl.
MASTER_SEARCH_DIR = "/root/capsule/scratch/combined/master_unit_tables"

OUT_DIR = Path("/root/capsule/scratch/combined/ccf_maps/master_table_plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Your previous CCF plotting code subtracts this bregma_LPS_mm before plotting.
# Set False if you want to plot raw x_ccf/y_ccf/z_ccf values directly.
USE_BREGMA_RELATIVE = True
BREGMA_LPS_MM = np.array([-5.70, 5.40, -0.45], dtype=float)

POINT_SIZE = 0.045
NON_OPTO_COLOR = 0x999999  # gray
OPTO_COLOR = 0xFF0000      # red
HIGHLIGHT_COLOR = 0xFFFF00 # yellow

# Optional Allen CCF structure mesh overlay.
# These paths assume you uploaded the locally generated *_bregma_lps_mm.obj files
# into the capsule at this location. These meshes are already in the same
# bregma-relative plotting frame used below, so do not divide by 1000 or subtract
# BREGMA_LPS_MM again.
SHOW_STRUCTURE_MESHES = True
STRUCTURE_MESH_DIR = Path("/root/capsule/data/ccf_meshes")

STRUCTURE_MESH_FILES = {
    "PAG": STRUCTURE_MESH_DIR / "PAG_bregma_lps_mm.obj",
    "AQ":  STRUCTURE_MESH_DIR / "AQ_bregma_lps_mm.obj",
    "DR":  STRUCTURE_MESH_DIR / "DR_bregma_lps_mm.obj",
    "CS":  STRUCTURE_MESH_DIR / "CS_bregma_lps_mm.obj",
}

# Avoid red/yellow for structures because those are used for units.
STRUCTURE_COLORS = {
    "PAG": 0x8C564B,  # brown
    "AQ":  0x1F77B4,  # blue
    "DR":  0x9467BD,  # purple
    "CS":  0x2CA02C,  # green
}

STRUCTURE_OPACITIES = {
    "PAG": 0.16,
    "AQ":  0.55,
    "DR":  0.42,
    "CS":  0.42,
}


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def as_bool_series(s):
    """
    Robustly convert bool/int/string opto_pass columns to boolean.
    """
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)

    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).astype(float) != 0

    s_str = s.astype(str).str.strip().str.lower()
    return s_str.isin(["true", "1", "1.0", "t", "yes", "y"])


def get_master_path(master_path=MASTER_PATH, search_dir=MASTER_SEARCH_DIR):
    """
    Use master_path if present; otherwise use newest master_all_units_opto_ccf*.pkl.
    """
    if master_path is not None and os.path.exists(master_path):
        return master_path

    candidates = sorted(
        glob.glob(os.path.join(search_dir, "master_all_units_opto_ccf*.pkl")),
        key=os.path.getmtime,
    )

    if len(candidates) == 0:
        raise FileNotFoundError(
            f"Could not find {master_path} or any master_all_units_opto_ccf*.pkl in {search_dir}"
        )

    print("MASTER_PATH not found. Using newest candidate:")
    print(candidates[-1])
    return candidates[-1]


def load_master_table(master_path):
    """
    Load a master table from .pkl/.pickle or .csv.
    """
    master_path = str(master_path)
    lower = master_path.lower()
    if lower.endswith((".pkl", ".pickle")):
        return pd.read_pickle(master_path)
    if lower.endswith(".csv"):
        return pd.read_csv(master_path)
    raise ValueError(f"Unsupported master table extension: {master_path}")


def _load_mesh_for_k3d(mesh_path):
    """
    Load an OBJ as a Trimesh. Handles OBJs that trimesh loads as a Scene.
    """
    import trimesh
    from trimesh import load_mesh

    mesh_path = Path(mesh_path)
    mesh = load_mesh(str(mesh_path), process=False)

    if isinstance(mesh, trimesh.Scene):
        geometries = tuple(
            geom for geom in mesh.geometry.values()
            if hasattr(geom, "vertices") and hasattr(geom, "faces")
        )
        if len(geometries) == 0:
            raise ValueError(f"No mesh geometry found in {mesh_path}")
        mesh = trimesh.util.concatenate(geometries)

    if not hasattr(mesh, "vertices") or not hasattr(mesh, "faces"):
        raise TypeError(f"Loaded object is not a triangular mesh: {mesh_path}")

    return mesh


def add_structure_mesh_overlay(plot, k3d_module=None):
    """
    Add PAG/AQ/DR/CS meshes to an existing K3D plot.

    This assumes STRUCTURE_MESH_FILES points to *_bregma_lps_mm.obj files,
    matching the pts = x_ccf/y_ccf/z_ccf - BREGMA_LPS_MM coordinate frame.
    """
    if k3d_module is None:
        import k3d as k3d_module

    loaded_any = False

    for acronym, mesh_path in STRUCTURE_MESH_FILES.items():
        mesh_path = Path(mesh_path)

        if not mesh_path.exists():
            print(f"Skipping {acronym}: mesh file not found: {mesh_path}")
            continue

        mesh = _load_mesh_for_k3d(mesh_path)
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        indices = np.asarray(mesh.faces, dtype=np.uint32)

        if vertices.size == 0 or indices.size == 0:
            print(f"Skipping {acronym}: empty mesh in {mesh_path}")
            continue

        plot += k3d_module.mesh(
            vertices=vertices,
            indices=indices,
            color=STRUCTURE_COLORS.get(acronym, 0x888888),
            opacity=STRUCTURE_OPACITIES.get(acronym, 0.25),
            wireframe=False,
            name=f"{acronym}_mesh",
        )

        print(
            f"Added {acronym} mesh: {mesh_path} "
            f"({len(vertices)} vertices, {len(indices)} faces)"
        )
        loaded_any = True

    if not loaded_any:
        print("No structure meshes were loaded. Units will still be plotted.")

    return plot


def _safe_token(x):
    """Return a filesystem-safe token for output filenames."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(x)).strip("_")


def _unit_key(x):
    """
    Normalize unit IDs so 83, 83.0, "83", and "83.0" compare equal.
    """
    if pd.isna(x):
        return ""
    text = str(x).strip()
    try:
        value = float(text)
        if np.isfinite(value) and value.is_integer():
            return str(int(value))
    except Exception:
        pass
    return text


def _session_col(master):
    for col in ("session", "session_id"):
        if col in master.columns:
            return col
    raise ValueError("Master table must contain either a 'session' or 'session_id' column.")


def _build_plot_table(master, use_bregma_relative=USE_BREGMA_RELATIVE):
    """
    Reproduce the existing CCF plot filtering and strict optotagged/candidate label.
    Returns plot_tbl, pts, coord_label.
    """
    required_cols = ["decoder_label", "opto_pass", "x_ccf", "y_ccf", "z_ccf"]
    missing = [c for c in required_cols if c not in master.columns]
    if missing:
        raise ValueError(
            f"Master table is missing required columns: {missing}\n"
            f"Available columns include:\n{master.columns.tolist()}"
        )

    decoder = master["decoder_label"].astype("string").str.strip().str.lower()

    non_noise_artifact_mask = (
        decoder.notna()
        & ~decoder.isin(["noise", "artifact", "nan", "none", "<na>", ""])
    )

    has_ccf_mask = master[["x_ccf", "y_ccf", "z_ccf"]].notna().all(axis=1)

    plot_tbl = master.loc[
        non_noise_artifact_mask & has_ccf_mask
    ].copy()

    required_for_strict = [
        "opto_pass",
        "default_qc",
        "decoder_label",
        "p_max",
        "p_mean",
        "pass_count",
        "lat_max_p",
    ]

    missing_for_strict = [c for c in required_for_strict if c not in plot_tbl.columns]
    if len(missing_for_strict) > 0:
        raise ValueError(
            "Cannot compute candidate_strict_5ht_no_wf because these columns are missing: "
            f"{missing_for_strict}"
        )

    decoder = plot_tbl["decoder_label"].astype("string").str.strip().str.lower()
    is_nonartifact = (
        decoder.notna()
        & ~decoder.isin(["noise", "artifact", "nan", "none", "<na>", ""])
    )

    opto_pass_bool = as_bool_series(plot_tbl["opto_pass"])
    default_qc_bool = as_bool_series(plot_tbl["default_qc"])

    p_max = pd.to_numeric(plot_tbl["p_max"], errors="coerce")
    p_mean = pd.to_numeric(plot_tbl["p_mean"], errors="coerce")
    pass_count = pd.to_numeric(plot_tbl["pass_count"], errors="coerce")
    lat_max_p = pd.to_numeric(plot_tbl["lat_max_p"], errors="coerce")

    plot_tbl["candidate_strict_5ht_no_wf"] = (
        opto_pass_bool
        & default_qc_bool
        & is_nonartifact
        & (p_max > 0.40)
        & (p_mean > 0.10)
        & (pass_count >= 2)
        & (lat_max_p > 0.007)
        & (lat_max_p < 0.025)
    )

    # Keep the same red-dot criterion as the existing CCF plot.
    plot_tbl["opto_pass_bool"] = plot_tbl["candidate_strict_5ht_no_wf"]

    if len(plot_tbl) == 0:
        raise ValueError("No units left to plot after decoder_label and CCF filtering.")

    pts = plot_tbl[["x_ccf", "y_ccf", "z_ccf"]].to_numpy(dtype=float)
    if use_bregma_relative:
        pts = pts - BREGMA_LPS_MM
        coord_label = "bregma_relative"
    else:
        coord_label = "raw_ccf"

    plot_tbl["plot_x"] = pts[:, 0]
    plot_tbl["plot_y"] = pts[:, 1]
    plot_tbl["plot_z"] = pts[:, 2]

    return plot_tbl, pts, coord_label


def _target_mask(plot_tbl, session_id, unit_id, data_type=None):
    """
    Find the requested target unit in plot_tbl.
    """
    session_col = _session_col(plot_tbl)
    session_mask = plot_tbl[session_col].astype(str).str.strip().eq(str(session_id).strip())
    unit_mask = plot_tbl["unit_id"].map(_unit_key).eq(_unit_key(unit_id))
    mask = session_mask & unit_mask

    if data_type is not None and "data_type" in plot_tbl.columns:
        mask = mask & plot_tbl["data_type"].astype(str).str.strip().eq(str(data_type).strip())

    return mask


def plot_ccf_highlight_unit(
    session_id,
    unit_id,
    master_path=MASTER_PATH,
    output_html=None,
    out_dir=OUT_DIR,
    data_type=None,
    show_structure_meshes=SHOW_STRUCTURE_MESHES,
    use_bregma_relative=USE_BREGMA_RELATIVE,
    display_plot=True,
    save_html=True,
):
    """
    Build the non-hover K3D CCF plot and highlight one requested unit in yellow.

    Parameters
    ----------
    session_id : str
        Session ID to highlight, e.g. "behavior_835444_2026-02-18_13-01-55".
    unit_id : int or str
        Unit ID to highlight.
    master_path : str or Path
        Master table path. Supports .pkl/.pickle and .csv.
    output_html : str or Path or None
        Optional output HTML path. If None, an informative filename is created.
    out_dir : str or Path
        Directory for output HTML and plotted-unit CSV.
    data_type : str or None
        Optional disambiguation if the master table has a data_type column.
    show_structure_meshes : bool
        Keep PAG/AQ/DR/CS mesh overlay.
    use_bregma_relative : bool
        Use the same bregma-relative coordinate transform as the existing plot.
    display_plot : bool
        Call plot.display() for notebook use.
    save_html : bool
        Save a standalone K3D snapshot HTML.

    Returns
    -------
    plot : k3d.Plot
        The K3D plot object.
    html_path : pathlib.Path or None
        Saved HTML path, or None if save_html=False.
    target_row : pandas.Series
        The matched row for the highlighted unit.
    """
    import k3d

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    master_path = get_master_path(master_path)
    master = load_master_table(master_path)

    print(f"Loaded master table: {master_path}")
    print(f"Master shape: {master.shape}")

    plot_tbl, pts, coord_label = _build_plot_table(
        master,
        use_bregma_relative=use_bregma_relative,
    )

    target_mask = _target_mask(plot_tbl, session_id, unit_id, data_type=data_type)
    n_target = int(target_mask.sum())
    if n_target == 0:
        session_col = _session_col(plot_tbl)
        same_session = plot_tbl[
            plot_tbl[session_col].astype(str).str.strip().eq(str(session_id).strip())
        ]
        available_units = same_session["unit_id"].map(_unit_key).tolist()[:50]
        raise ValueError(
            "Target unit was not found among plotted non-noise/non-artifact units with valid CCF coordinates.\n"
            f"session_id={session_id!r}, unit_id={unit_id!r}, data_type={data_type!r}\n"
            f"Rows in same session after plotting filter: {len(same_session)}\n"
            f"First available unit IDs in that session: {available_units}"
        )
    if n_target > 1:
        print(f"Warning: matched {n_target} rows; highlighting all matched rows.")

    # Exclude the target from the gray/red layers so it is only yellow.
    opto_mask = plot_tbl["opto_pass_bool"].to_numpy(dtype=bool)
    target_mask_arr = target_mask.to_numpy(dtype=bool)
    non_mask = (~opto_mask) & (~target_mask_arr)
    red_mask = opto_mask & (~target_mask_arr)

    pts_non = pts[non_mask].astype(np.float32)
    pts_opto = pts[red_mask].astype(np.float32)
    pts_highlight = pts[target_mask_arr].astype(np.float32)

    target_row = plot_tbl.loc[target_mask].iloc[0].copy()

    plot_tbl["highlighted_unit"] = target_mask_arr
    plotted_csv = out_dir / f"plotted_units_highlight_{_safe_token(session_id)}_unit_{_safe_token(unit_id)}_{coord_label}.csv"
    cols_to_save = [
        "mouse_id",
        "session",
        "session_id",
        "unit_id",
        "unit_uid",
        "data_type",
        "decoder_label",
        "default_qc",
        "opto_pass",
        "opto_pass_bool",
        "candidate_strict_5ht_no_wf",
        "highlighted_unit",
        "first_pass_putative_opto",
        "x_ccf",
        "y_ccf",
        "z_ccf",
        "loc_along_probe",
        "plot_x",
        "plot_y",
        "plot_z",
    ]
    plot_tbl[[c for c in cols_to_save if c in plot_tbl.columns]].to_csv(plotted_csv, index=False)

    print("\nPlot table:")
    print(f"  total units in master:              {len(master)}")
    print(f"  non-noise/non-artifact with CCF:    {len(plot_tbl)}")
    print(f"  opto/candidate red units:           {int(plot_tbl['opto_pass_bool'].sum())}")
    print(f"  highlighted yellow units:           {len(pts_highlight)}")
    print(f"  gray units after highlight split:   {len(pts_non)}")
    print(f"  red units after highlight split:    {len(pts_opto)}")
    print("\nHighlighted unit:")
    print(f"  session: {session_id}")
    print(f"  unit_id: {unit_id}")
    print(f"  xyz used for plotting: {pts_highlight.tolist()}")
    print(f"\nSaved plotted-unit table: {plotted_csv}")

    print("\nCoordinate ranges used for plotting:")
    print(pd.DataFrame(pts, columns=["x", "y", "z"]).agg(["min", "max"]))

    plot = k3d.plot(camera_auto_fit=True)

    # Add structure meshes first so unit points are drawn on top.
    if show_structure_meshes:
        if use_bregma_relative:
            plot = add_structure_mesh_overlay(plot, k3d_module=k3d)
        else:
            print(
                "Skipping structure mesh overlay because use_bregma_relative=False. "
                "The configured OBJ files are *_bregma_lps_mm.obj meshes."
            )

    if len(pts_non) > 0:
        plot += k3d.points(
            positions=pts_non,
            point_size=POINT_SIZE,
            color=NON_OPTO_COLOR,
            opacity=0.35,
            shader="3d",
            name=f"non_opto_passed_nonartifact_n={len(pts_non)}",
        )

    if len(pts_opto) > 0:
        plot += k3d.points(
            positions=pts_opto,
            point_size=POINT_SIZE * 1.35,
            color=OPTO_COLOR,
            opacity=0.95,
            shader="3d",
            name=f"opto_passed_nonartifact_n={len(pts_opto)}",
        )

    # Yellow halo plus core point makes the target visible even around transparent meshes.
    plot += k3d.points(
        positions=pts_highlight,
        point_size=POINT_SIZE * 3.0,
        color=HIGHLIGHT_COLOR,
        opacity=0.40,
        shader="3d",
        name=f"highlighted_unit_yellow_halo_session={session_id}_unit={unit_id}",
    )
    plot += k3d.points(
        positions=pts_highlight,
        point_size=POINT_SIZE * 1.85,
        color=HIGHLIGHT_COLOR,
        opacity=1.0,
        shader="3d",
        name=f"highlighted_unit_yellow_session={session_id}_unit={unit_id}",
    )

    # Optional: add a small origin/bregma marker if using bregma-relative coords.
    if use_bregma_relative:
        plot += k3d.points(
            positions=np.array([[0, 0, 0]], dtype=np.float32),
            point_size=POINT_SIZE * 1.8,
            color=0x0000FF,
            opacity=1.0,
            shader="3d",
            name="bregma_origin",
        )

    html_path = None
    if save_html:
        mesh_suffix = "_with_PAG_AQ_DR_CS" if show_structure_meshes else ""
        if output_html is None:
            output_html = out_dir / (
                f"non_noise_nonartifact_units_highlight_"
                f"{_safe_token(session_id)}_unit_{_safe_token(unit_id)}_"
                f"{coord_label}{mesh_suffix}.html"
            )
        html_path = Path(output_html)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        with open(html_path, "w") as f:
            f.write(plot.get_snapshot())
        print(f"Saved interactive K3D HTML: {html_path}")

    print("K3D object list:")
    print("  gray   = non-opto_passed, non-noise/non-artifact units")
    print("  red    = opto_passed/candidate_strict_5ht_no_wf units")
    print("  yellow = requested highlighted unit")
    print("  blue   = bregma origin")
    if show_structure_meshes:
        print("  brown  = PAG mesh")
        print("  cyan/blue = AQ mesh")
        print("  purple = DR mesh")
        print("  green  = CS mesh")

    if display_plot:
        plot.display()

    return plot, html_path, target_row


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Plot CCF units and highlight one session/unit in yellow."
    )
    parser.add_argument(
        "session_id",
        nargs="?",
        default=os.environ.get("HIGHLIGHT_SESSION_ID"),
        help="Session ID to highlight. Can also use HIGHLIGHT_SESSION_ID env var.",
    )
    parser.add_argument(
        "unit_id",
        nargs="?",
        default=os.environ.get("HIGHLIGHT_UNIT_ID"),
        help="Unit ID to highlight. Can also use HIGHLIGHT_UNIT_ID env var.",
    )
    parser.add_argument(
        "--master-path",
        default=os.environ.get("MASTER_PATH", MASTER_PATH),
        help="Master table path (.pkl/.pickle/.csv).",
    )
    parser.add_argument(
        "--output-html",
        default=os.environ.get("HIGHLIGHT_OUTPUT_HTML"),
        help="Optional output HTML path.",
    )
    parser.add_argument(
        "--data-type",
        default=os.environ.get("HIGHLIGHT_DATA_TYPE"),
        help="Optional data_type filter if the master table has a data_type column.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Save HTML without calling plot.display(). Useful in terminal runs.",
    )
    parser.add_argument(
        "--no-meshes",
        action="store_true",
        help="Do not add PAG/AQ/DR/CS meshes.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.session_id is None or args.unit_id is None:
        raise SystemExit(
            "Provide session_id and unit_id, e.g.\n"
            "  /opt/conda/bin/python ccf_highlight_unit_k3d.py "
            "behavior_835444_2026-02-18_13-01-55 83\n"
            "or set HIGHLIGHT_SESSION_ID and HIGHLIGHT_UNIT_ID."
        )

    plot_ccf_highlight_unit(
        session_id=args.session_id,
        unit_id=args.unit_id,
        master_path=args.master_path,
        output_html=args.output_html,
        data_type=args.data_type,
        show_structure_meshes=not args.no_meshes,
        display_plot=not args.no_display,
    )

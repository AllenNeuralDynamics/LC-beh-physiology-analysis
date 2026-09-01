#!/usr/bin/env python3
"""
Build a standalone Plotly 3D CCF unit map with click-locked unit info panels
and local PDF links.

Default behavior matches the existing CCF/hover notebook logic:
- plot non-noise/non-artifact units with valid CCF coordinates
- color strict well-optotagged/candidate units red
- color all other plotted units gray
- add PAG/AQ/DR/CS anatomical meshes as Plotly wireframes so points remain clickable
- on unit click, open a persistent HTML panel with optotagging metrics and PDF links
- when a red optotagged/candidate unit is clicked, overlay it with a yellow highlight until the panel is closed
- optional left-side criteria panel lets you tune optotagging thresholds and toggle mouse IDs in the browser
- use equal 3D axis scaling so ML/AP/DV millimeters are displayed to scale

Recommended local layout:
    /some/local/folder/unit_map_clickable.html
    /some/local/folder/unit_pdfs/<session>_unit_<unit_id>_go_cue.pdf
    /some/local/folder/unit_pdfs/<session>_unit_<unit_id>_response.pdf
    /some/local/folder/unit_drift_pdfs/<session>_unit_<unit_id>_drift.pdf
    /some/local/folder/unit_opto_pdfs/<session>_unit_<unit_id>_opto.pdf

Then use pdf_dir="unit_pdfs", drift_pdf_dir="unit_drift_pdfs",
opto_pdf_dir="unit_opto_pdfs", and link_mode="relative". Relative links are
the most portable when you copy the HTML + PDF folders together.

Workflow B / capsule-to-local mode:
By default this script trusts expected PDF filenames and creates links even
when the capsule cannot see the PDFs. Place the downloaded HTML next to the
unit_pdfs, unit_drift_pdfs, and unit_opto_pdfs folders locally. Use
--check-pdf-exists or trust_expected_pdf_names=False if you want generation-time
existence checks instead.
"""

from __future__ import annotations

import argparse
import glob
import html
import json
import os
import re
from pathlib import Path
from typing import Iterable, Mapping, Sequence
from urllib.parse import quote

import numpy as np
import pandas as pd


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


# ---------------------------------------------------------------------
# Defaults matching the existing CCF plotting notebook/script
# ---------------------------------------------------------------------

MASTER_PATH = "/root/capsule/scratch/combined/master_unit_tables/master_all_units_opto_ccf_raw_fix.pkl"
MASTER_SEARCH_DIR = "/root/capsule/scratch/combined/master_unit_tables"
OUT_DIR = Path("/root/capsule/scratch/combined/ccf_maps/master_table_plots")
PDF_DIR = Path(os.environ.get("UNIT_PDF_DIR", "unit_pdfs"))
DRIFT_PDF_DIR = Path(os.environ.get("UNIT_DRIFT_PDF_DIR", "unit_drift_pdfs"))
OPTO_PDF_DIR = Path(os.environ.get("UNIT_OPTO_PDF_DIR", "unit_opto_pdfs"))
# Workflow B: create clickable links from expected PDF names without checking
# that the PDFs exist in the environment generating the HTML. This is now the
# default because the generated HTML is intended to be downloaded and opened next
# to local PDF folders.
TRUST_EXPECTED_PDF_NAMES = _env_flag("TRUST_EXPECTED_PDF_NAMES", True)
INTERACTIVE_CRITERIA_CONTROLS = _env_flag("INTERACTIVE_CRITERIA_CONTROLS", True)

USE_BREGMA_RELATIVE = True
BREGMA_LPS_MM = np.array([-5.70, 5.40, -0.45], dtype=float)

NON_OPTO_COLOR = 0x999999  # gray
OPTO_COLOR = 0xFF0000      # red
CENTER_ON_MESH = os.environ.get("CENTER_ON_MESH", "DR").strip() or None
CENTER_VIEW_ON_MESH = os.environ.get("CENTER_VIEW_ON_MESH", "1").strip().lower() not in {"0", "false", "no", "n"}
CENTER_VIEW_PADDING_FRACTION = float(os.environ.get("CENTER_VIEW_PADDING_FRACTION", "0.05"))

STRUCTURE_MESH_DIR = Path("/root/capsule/data/ccf_meshes")
STRUCTURE_MESH_FILES = {
    "PAG": STRUCTURE_MESH_DIR / "PAG_bregma_lps_mm.obj",
    "AQ": STRUCTURE_MESH_DIR / "AQ_bregma_lps_mm.obj",
    "DR": STRUCTURE_MESH_DIR / "DR_bregma_lps_mm.obj",
    "CS": STRUCTURE_MESH_DIR / "CS_bregma_lps_mm.obj",
}
STRUCTURE_COLORS = {
    "PAG": 0x8C564B,  # brown
    "AQ": 0x1F77B4,   # blue
    "DR": 0x9467BD,   # purple
    "CS": 0x2CA02C,   # green
}
STRUCTURE_OPACITIES = {
    "PAG": 0.55,
    "AQ": 0.75,
    "DR": 0.70,
    "CS": 0.70,
}

DEFAULT_METRICS = (
    "lat_max_p",
    "corr_max_p",
    "eu_max_p",
    "isi_violations_ratio",
    "p_max",
)

METRIC_ALIASES: Mapping[str, Sequence[str]] = {
    "lat_max_p": ("lat_max_p", "latency_max_p", "latency_at_max_p"),
    "corr_max_p": ("corr_max_p", "correlation_max_p", "waveform_corr_max_p"),
    "eu_max_p": (
        "eu_max_p",
        "euc_max_p",
        "euclidean_max_p",
        "euclidean_norm_max_p",
        "waveform_euclidean_max_p",
        "waveform_euc_max_p",
    ),
    "isi_violations_ratio": (
        "isi_violations_ratio",
        "isi_violation_ratio",
        "isi_violations",
        "isi_violations_rate",
    ),
    "p_max": ("p_max", "max_p"),
}

# Extra customdata columns appended after [click_panel_html, hover_text].
# The browser-side criteria tuner uses these to move units between gray/red traces
# without rebuilding the HTML. plot_x/plot_y/plot_z are appended after these
# fields as a browser-side fallback for Plotly.py 6 binary coordinate arrays.
CRITERIA_CUSTOMDATA_FIELDS = (
    "opto_pass_raw_bool",
    "default_qc_bool",
    "p_max",
    "p_mean",
    "pass_count",
    "lat_max_p",
    "isi_violations_ratio",
    "corr_max_p",
    "eu_max_p",
    "mouse_id_norm",
)

DEFAULT_CRITERIA = {
    "require_opto_pass": True,
    "require_default_qc": True,
    "p_max_min": 0.40,
    "p_mean_min": 0.10,
    "pass_count_min": 2,
    "lat_max_p_min_ms": 7.0,
    "lat_max_p_max_ms": 25.0,
    "isi_violations_ratio_max": 0.10,
    "corr_max_p_min": 0.70,
    "eu_max_p_max": 0.30,
}


# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------

def _parse_csv_list(value: str | Iterable[str] | None, default: Sequence[str]) -> tuple[str, ...]:
    if value is None:
        return tuple(default)
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
        return tuple(parts) if parts else tuple(default)
    return tuple(str(p).strip() for p in value if str(p).strip())


def _hex_to_css_rgb(color_int: int) -> str:
    return f"rgb({(color_int >> 16) & 255},{(color_int >> 8) & 255},{color_int & 255})"


def as_bool_series(s: pd.Series) -> pd.Series:
    """Robustly convert bool/int/string columns to boolean."""
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).astype(float) != 0
    s_str = s.astype(str).str.strip().str.lower()
    return s_str.isin(["true", "1", "1.0", "t", "yes", "y"])


def get_master_path(master_path: str | os.PathLike = MASTER_PATH,
                    search_dir: str | os.PathLike = MASTER_SEARCH_DIR) -> str:
    """Use master_path if present; otherwise use newest master_all_units_opto_ccf*.pkl/csv."""
    master_path = str(master_path)
    if master_path and os.path.exists(master_path):
        return master_path

    candidates: list[str] = []
    search_dir = str(search_dir)
    if search_dir and os.path.exists(search_dir):
        candidates.extend(glob.glob(os.path.join(search_dir, "master_all_units_opto_ccf*.pkl")))
        candidates.extend(glob.glob(os.path.join(search_dir, "master_all_units_opto_ccf*.csv")))

    candidates = sorted(set(candidates), key=os.path.getmtime)
    if not candidates:
        raise FileNotFoundError(
            f"Could not find {master_path!r} or master_all_units_opto_ccf*.pkl/csv in {search_dir!r}"
        )
    print("MASTER_PATH not found. Using newest candidate:")
    print(candidates[-1])
    return candidates[-1]


def load_master_table(master_path: str | os.PathLike) -> pd.DataFrame:
    suffix = Path(master_path).suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(master_path)
    if suffix == ".csv":
        return pd.read_csv(master_path)
    raise ValueError(f"Unsupported master table extension: {master_path}")


def get_session_column(df: pd.DataFrame) -> str:
    if "session" in df.columns:
        return "session"
    if "session_id" in df.columns:
        return "session_id"
    raise ValueError("Need either a 'session' or 'session_id' column.")


def normalize_unit_id(unit_id) -> str:
    if pd.isna(unit_id):
        return ""
    try:
        val = float(unit_id)
        if np.isfinite(val) and val.is_integer():
            return str(int(val))
    except Exception:
        pass
    return str(unit_id).strip()


def normalize_mouse_id(mouse_id) -> str:
    """Normalize mouse IDs so 835444, 835444.0, and "835444" all match."""
    if pd.isna(mouse_id):
        return ""
    text = str(mouse_id).strip()
    if text.lower() in {"", "nan", "none", "<na>"}:
        return ""
    try:
        val = float(text)
        if np.isfinite(val) and val.is_integer():
            return str(int(val))
    except Exception:
        pass
    return text


def mouse_id_from_session(session_id) -> str:
    """Parse mouse ID from session strings like behavior_835444_2026-02-18_13-01-55."""
    if pd.isna(session_id):
        return ""
    text = str(session_id).strip()
    m = re.search(r"behavior_(\d+)_", text)
    if m:
        return m.group(1)
    return ""


def add_mouse_id_columns(plot_tbl: pd.DataFrame) -> pd.DataFrame:
    """
    Add mouse_id_norm / mouse_id_display helper columns for browser-side filtering.

    Uses the master table's mouse_id column when available, and falls back to
    parsing the mouse ID out of the session string. These columns are intentionally
    strings so JavaScript checkbox filters do not have to handle 835444 vs 835444.0.
    """
    session_col = get_session_column(plot_tbl)
    parsed = plot_tbl[session_col].map(mouse_id_from_session).map(normalize_mouse_id)

    if "mouse_id" in plot_tbl.columns:
        from_col = plot_tbl["mouse_id"].map(normalize_mouse_id)
        mouse = from_col.where(from_col.astype(str).str.len() > 0, parsed)
    else:
        mouse = parsed
        plot_tbl["mouse_id"] = mouse

    mouse = mouse.fillna("").astype(str)
    mouse_display = mouse.where(mouse.str.len() > 0, "unknown")
    plot_tbl["mouse_id_norm"] = mouse_display
    plot_tbl["mouse_id_display"] = mouse_display
    return plot_tbl


def _resolve_metric_column(df: pd.DataFrame, metric_name: str) -> str | None:
    aliases = METRIC_ALIASES.get(metric_name, (metric_name,))
    for col in aliases:
        if col in df.columns:
            return col
    return None


def _format_metric(metric_name: str, value) -> str:
    try:
        val = float(value)
    except Exception:
        return "n/a"
    if not np.isfinite(val):
        return "n/a"

    # lat_max_p is stored in seconds in the current master table; ms is easier to read.
    if metric_name.startswith("lat"):
        return f"{val * 1000:.2f} ms"
    if metric_name in {"p_max", "p_mean", "corr_max_p", "eu_max_p", "euc_max_p", "isi_violations_ratio"}:
        return f"{val:.4g}"
    return f"{val:.4g}"


def prepare_plot_table(master: pd.DataFrame,
                       use_bregma_relative: bool = USE_BREGMA_RELATIVE,
                       use_strict_opto_red: bool = True) -> tuple[pd.DataFrame, str]:
    """
    Filter to plotted units and compute the red-vs-gray mask.

    With use_strict_opto_red=True, red points match the previous hover notebook's
    candidate_strict_5ht_qc_wf criterion.
    """
    required_cols = ["decoder_label", "opto_pass", "x_ccf", "y_ccf", "z_ccf", "unit_id"]
    session_col = get_session_column(master)
    missing = [c for c in required_cols + [session_col] if c not in master.columns]
    if missing:
        raise ValueError(f"Master table is missing required columns: {missing}")

    decoder = master["decoder_label"].astype("string").str.strip().str.lower()
    non_noise_artifact_mask = (
        decoder.notna()
        & ~decoder.isin(["noise", "artifact", "nan", "none", "<na>", ""])
    )
    has_ccf_mask = master[["x_ccf", "y_ccf", "z_ccf"]].notna().all(axis=1)
    plot_tbl = master.loc[non_noise_artifact_mask & has_ccf_mask].copy()
    plot_tbl = add_mouse_id_columns(plot_tbl)

    # Browser-side interactive criteria tuning needs raw opto/default_qc bools
    # and numeric metric values for all plotted units, not only the initially red units.
    plot_tbl["opto_pass_raw_bool"] = as_bool_series(plot_tbl["opto_pass"])
    if "default_qc" in plot_tbl.columns:
        plot_tbl["default_qc_bool"] = as_bool_series(plot_tbl["default_qc"])
    else:
        plot_tbl["default_qc_bool"] = False

    for _metric_col in (
        "p_max",
        "p_mean",
        "pass_count",
        "lat_max_p",
        "isi_violations_ratio",
        "corr_max_p",
        "eu_max_p",
    ):
        _source_col = _resolve_metric_column(plot_tbl, _metric_col)
        if _source_col is not None:
            plot_tbl[_metric_col] = pd.to_numeric(plot_tbl[_source_col], errors="coerce")
        else:
            plot_tbl[_metric_col] = np.nan

    if use_strict_opto_red:
        required_for_strict = [
            "opto_pass", "default_qc", "decoder_label",
            "p_max", "p_mean", "pass_count", "lat_max_p",
            "isi_violations_ratio", "corr_max_p", "eu_max_p",
        ]
        missing_for_strict = [c for c in required_for_strict if c not in plot_tbl.columns]
        if missing_for_strict:
            raise ValueError(
                "Cannot compute candidate_strict_5ht_qc_wf because these columns are missing: "
                f"{missing_for_strict}"
            )
        decoder_plot = plot_tbl["decoder_label"].astype("string").str.strip().str.lower()
        is_nonartifact = (
            decoder_plot.notna()
            & ~decoder_plot.isin(["noise", "artifact", "nan", "none", "<na>", ""])
        )
        opto_pass_bool = as_bool_series(plot_tbl["opto_pass"])
        default_qc_bool = as_bool_series(plot_tbl["default_qc"])
        p_max = pd.to_numeric(plot_tbl["p_max"], errors="coerce")
        p_mean = pd.to_numeric(plot_tbl["p_mean"], errors="coerce")
        pass_count = pd.to_numeric(plot_tbl["pass_count"], errors="coerce")
        lat_max_p = pd.to_numeric(plot_tbl["lat_max_p"], errors="coerce")
        isi_violations_ratio = pd.to_numeric(plot_tbl["isi_violations_ratio"], errors="coerce")
        corr_max_p = pd.to_numeric(plot_tbl["corr_max_p"], errors="coerce")
        eu_max_p = pd.to_numeric(plot_tbl["eu_max_p"], errors="coerce")

        plot_tbl["candidate_strict_5ht_qc_wf"] = (
            opto_pass_bool
            & default_qc_bool
            & is_nonartifact
            & (p_max > 0.40)
            & (p_mean > 0.10)
            & (pass_count >= 2)
            & (lat_max_p > 0.007)
            & (lat_max_p < 0.025)
            & (isi_violations_ratio < 0.10)
            & (corr_max_p > 0.70)
            & (eu_max_p < 0.30)
        )
        plot_tbl["opto_pass_bool"] = plot_tbl["candidate_strict_5ht_qc_wf"]
        red_label = "candidate_strict_5ht_qc_wf"
    else:
        plot_tbl["opto_pass_bool"] = as_bool_series(plot_tbl["opto_pass"])
        red_label = "opto_pass"

    pts = plot_tbl[["x_ccf", "y_ccf", "z_ccf"]].to_numpy(dtype=float)
    if use_bregma_relative:
        pts = pts - BREGMA_LPS_MM
    plot_tbl["plot_x"] = pts[:, 0]
    plot_tbl["plot_y"] = pts[:, 1]
    plot_tbl["plot_z"] = pts[:, 2]
    return plot_tbl, red_label


# ---------------------------------------------------------------------
# PDF link + click-panel helpers
# ---------------------------------------------------------------------

def _href_for_pdf(pdf_path: Path, html_path: Path, link_mode: str = "relative") -> str:
    """
    Return an href for a PDF path.

    link_mode="relative" is recommended when the HTML and PDF folder will be copied
    together. link_mode="absolute_file" creates file:/// links, which are often blocked
    when the HTML is served over http:// rather than opened as a local file.
    """
    pdf_path = Path(pdf_path)
    html_path = Path(html_path)

    if link_mode == "relative":
        # Preserve already-relative paths. This is important for Workflow B:
        # generating HTML in the capsule with pdf_dir="unit_pdfs" should create
        # links like unit_pdfs/<filename>.pdf for the downloaded local copy.
        if not pdf_path.is_absolute():
            rel = pdf_path.as_posix()
        else:
            try:
                rel = os.path.relpath(str(pdf_path), start=str(html_path.parent))
            except Exception:
                rel = str(pdf_path)
        return quote(Path(rel).as_posix(), safe="/:#?&=%@+,.()_-~")

    if link_mode == "absolute_file":
        return pdf_path.resolve().as_uri()

    if link_mode == "path":
        return quote(str(pdf_path), safe="/:#?&=%@+,.()_-~")

    raise ValueError("link_mode must be one of: relative, absolute_file, path")


def _pdf_exists_for_generation(pdf_path: Path, html_path: Path) -> bool:
    """Check absolute paths directly and relative paths next to the output HTML."""
    pdf_path = Path(pdf_path)
    html_path = Path(html_path)
    if pdf_path.is_absolute():
        return pdf_path.exists()
    return pdf_path.exists() or (html_path.parent / pdf_path).exists()


def _pdf_links_html(session: str,
                    unit_id_text: str,
                    pdf_dir: Path,
                    html_path: Path,
                    align_names: Sequence[str],
                    filename_template: str,
                    link_mode: str,
                    include_missing_paths: bool = True,
                    trust_expected_pdf_names: bool = TRUST_EXPECTED_PDF_NAMES) -> tuple[str, int]:
    """
    Build an HTML fragment with one link per alignment PDF.

    When trust_expected_pdf_names=True, links are created from the expected
    filenames even if the PDFs are not visible to this Python process.
    """
    pdf_dir = Path(pdf_dir)
    rows: list[str] = []
    existing_count = 0

    for align_name in align_names:
        filename = filename_template.format(
            session=session,
            unit_id=unit_id_text,
            unit=unit_id_text,
            align=align_name,
            alignment=align_name,
        )
        pdf_path = pdf_dir / filename
        label = html.escape(str(align_name))
        exists_here = _pdf_exists_for_generation(pdf_path, html_path)
        if exists_here or trust_expected_pdf_names:
            href = _href_for_pdf(pdf_path, html_path, link_mode=link_mode)
            title = "" if exists_here else ' title="Expected filename; existence was not checked in this environment"'
            rows.append(
                f'<li><a href="{href}" target="_blank" rel="noopener noreferrer"{title}>{label} PDF</a></li>'
            )
            existing_count += 1
        elif include_missing_paths:
            rows.append(
                f'<li><span style="color:#777">{label} PDF not found: '
                f'{html.escape(str(pdf_path))}</span></li>'
            )

    if not rows:
        return '<div style="color:#777">No PDF links configured.</div>', 0

    return '<ul style="margin:0.25em 0 0 1.2em; padding:0">' + "".join(rows) + "</ul>", existing_count


def _single_pdf_link_html(session: str,
                          unit_id_text: str,
                          pdf_dir: Path,
                          html_path: Path,
                          filename_template: str,
                          label: str,
                          link_mode: str,
                          include_missing_paths: bool = True,
                          trust_expected_pdf_names: bool = TRUST_EXPECTED_PDF_NAMES) -> tuple[str, int]:
    """Build an HTML fragment for one expected unit-level PDF.

    Available template fields are {session}, {unit_id}, {unit}, and {label}.
    When trust_expected_pdf_names=True, the link is emitted even if the PDF is
    not visible to this Python process.
    """
    pdf_dir = Path(pdf_dir)
    filename = filename_template.format(
        session=session,
        unit_id=unit_id_text,
        unit=unit_id_text,
        label=label,
    )
    pdf_path = pdf_dir / filename
    exists_here = _pdf_exists_for_generation(pdf_path, html_path)
    safe_label = html.escape(label)

    if exists_here or trust_expected_pdf_names:
        href = _href_for_pdf(pdf_path, html_path, link_mode=link_mode)
        title = "" if exists_here else ' title="Expected filename; existence was not checked in this environment"'
        return (
            '<ul style="margin:0.25em 0 0 1.2em; padding:0">'
            f'<li><a href="{href}" target="_blank" rel="noopener noreferrer"{title}>{safe_label} PDF</a></li>'
            '</ul>'
        ), 1

    if include_missing_paths:
        return (
            '<ul style="margin:0.25em 0 0 1.2em; padding:0">'
            f'<li><span style="color:#777">{safe_label} PDF not found: '
            f'{html.escape(str(pdf_path))}</span></li>'
            '</ul>'
        ), 0

    return '', 0


def add_click_panel_columns(plot_tbl: pd.DataFrame,
                            pdf_dir: str | os.PathLike,
                            out_html: str | os.PathLike,
                            align_names: Sequence[str] = ("go_cue", "response"),
                            filename_template: str = "{session}_unit_{unit_id}_{align}.pdf",
                            drift_pdf_dir: str | os.PathLike | None = DRIFT_PDF_DIR,
                            opto_pdf_dir: str | os.PathLike | None = OPTO_PDF_DIR,
                            drift_filename_template: str = "{session}_unit_{unit_id}_drift.pdf",
                            opto_filename_template: str = "{session}_unit_{unit_id}_opto.pdf",
                            include_drift_pdf_link: bool = True,
                            include_opto_pdf_link: bool = True,
                            metrics: Sequence[str] = DEFAULT_METRICS,
                            link_mode: str = "relative",
                            include_missing_pdf_paths: bool = True,
                            trust_expected_pdf_names: bool = TRUST_EXPECTED_PDF_NAMES) -> pd.DataFrame:
    """Add hover_text, click_panel_html, and pdf_link_count columns."""
    session_col = get_session_column(plot_tbl)
    plot_tbl = plot_tbl.copy()
    out_html = Path(out_html)
    pdf_dir = Path(pdf_dir)
    drift_pdf_dir = None if drift_pdf_dir is None or str(drift_pdf_dir).strip() == "" else Path(drift_pdf_dir)
    opto_pdf_dir = None if opto_pdf_dir is None or str(opto_pdf_dir).strip() == "" else Path(opto_pdf_dir)

    resolved_metric_cols = {
        metric: _resolve_metric_column(plot_tbl, metric)
        for metric in metrics
    }

    def build_for_row(row: pd.Series) -> pd.Series:
        session = str(row[session_col]).strip()
        unit_id_text = normalize_unit_id(row["unit_id"])
        mouse_id_text = str(row.get("mouse_id_display", row.get("mouse_id_norm", "unknown"))).strip() or "unknown"
        is_red = bool(row.get("opto_pass_bool", False))
        unit_kind = "strict optotagged/candidate" if is_red else "non-opto/non-candidate"

        metric_rows = []
        for metric in metrics:
            col = resolved_metric_cols.get(metric)
            value = row[col] if col is not None else np.nan
            metric_rows.append(
                "<tr>"
                f"<td style='padding-right:12px; color:#444'>{html.escape(metric)}</td>"
                f"<td style='font-family:monospace'>{html.escape(_format_metric(metric, value))}</td>"
                "</tr>"
            )
        metric_table = (
            "<table style='border-collapse:collapse; margin-top:0.25em'>"
            + "".join(metric_rows)
            + "</table>"
        )

        links_html, n_behavior_links = _pdf_links_html(
            session=session,
            unit_id_text=unit_id_text,
            pdf_dir=pdf_dir,
            html_path=out_html,
            align_names=align_names,
            filename_template=filename_template,
            link_mode=link_mode,
            include_missing_paths=include_missing_pdf_paths,
            trust_expected_pdf_names=trust_expected_pdf_names,
        )

        drift_links_html = ""
        n_drift_links = 0
        if include_drift_pdf_link and drift_pdf_dir is not None:
            drift_links_html, n_drift_links = _single_pdf_link_html(
                session=session,
                unit_id_text=unit_id_text,
                pdf_dir=drift_pdf_dir,
                html_path=out_html,
                filename_template=drift_filename_template,
                label="drift",
                link_mode=link_mode,
                include_missing_paths=include_missing_pdf_paths,
                trust_expected_pdf_names=trust_expected_pdf_names,
            )

        opto_links_html = ""
        n_opto_links = 0
        if include_opto_pdf_link and opto_pdf_dir is not None:
            opto_links_html, n_opto_links = _single_pdf_link_html(
                session=session,
                unit_id_text=unit_id_text,
                pdf_dir=opto_pdf_dir,
                html_path=out_html,
                filename_template=opto_filename_template,
                label="opto",
                link_mode=link_mode,
                include_missing_paths=include_missing_pdf_paths,
                trust_expected_pdf_names=trust_expected_pdf_names,
            )

        aux_links_html = ""
        if drift_links_html or opto_links_html:
            aux_links_html = (
                '<hr style="border:none; border-top:1px solid #ddd; margin:8px 0" />'
                '<div style="font-weight:bold">Drift / optotagging PDFs</div>'
                + drift_links_html
                + opto_links_html
            )

        n_links = n_behavior_links + n_drift_links + n_opto_links

        title_color = "#cc0000" if is_red else "#555"
        panel = f"""
<div style="line-height:1.35">
  <button id="unit-info-close" title="Close" style="float:right; border:1px solid #aaa; background:#f7f7f7; border-radius:3px; cursor:pointer">x</button>
  <div style="font-weight:bold; color:{title_color}; font-size:14px">Unit details</div>
  <div><b>mouse ID:</b> {html.escape(mouse_id_text)}</div>
  <div><b>session:</b> {html.escape(session)}</div>
  <div><b>unit ID:</b> {html.escape(unit_id_text)}</div>
  <div><b>class:</b> {html.escape(unit_kind)}</div>
  <hr style="border:none; border-top:1px solid #ddd; margin:8px 0" />
  <div style="font-weight:bold">Optotagging / QC metrics</div>
  {metric_table}
  <hr style="border:none; border-top:1px solid #ddd; margin:8px 0" />
  <div style="font-weight:bold">Behavior-aligned PDFs</div>
  {links_html}
  {aux_links_html}
  <div style="font-size:11px; color:#777; margin-top:8px">Click another unit to update this panel. Press Esc to close.</div>
</div>
""".strip()

        metric_preview = []
        for metric in metrics:
            col = resolved_metric_cols.get(metric)
            value = row[col] if col is not None else np.nan
            metric_preview.append(f"{metric}: {_format_metric(metric, value)}")
        hover = (
            f"<b>{html.escape(session)}</b><br>"
            f"mouse ID: {html.escape(mouse_id_text)}<br>"
            f"unit ID: {html.escape(unit_id_text)}<br>"
            + "<br>".join(html.escape(x) for x in metric_preview)
            + "<br><i>click for locked panel + PDF link</i>"
        )

        return pd.Series({
            "hover_text": hover,
            "click_panel_html": panel,
            "pdf_link_count": n_links,
            "behavior_pdf_link_count": n_behavior_links,
            "drift_pdf_link_count": n_drift_links,
            "opto_pdf_link_count": n_opto_links,
        })

    extras = plot_tbl.apply(build_for_row, axis=1)
    for col in extras.columns:
        plot_tbl[col] = extras[col]

    missing_cols = [metric for metric, col in resolved_metric_cols.items() if col is None]
    if missing_cols:
        print("Metric columns not found; shown as n/a:", ", ".join(missing_cols))
    if trust_expected_pdf_names:
        print(
            f"PDF links generated from expected names: {int(plot_tbl['pdf_link_count'].sum())} "
            f"links across {len(plot_tbl)} plotted units"
        )
    else:
        print(f"PDF links found: {int(plot_tbl['pdf_link_count'].sum())} total links across {len(plot_tbl)} plotted units")
    if {"behavior_pdf_link_count", "drift_pdf_link_count", "opto_pdf_link_count"}.issubset(plot_tbl.columns):
        print(
            "  by category: "
            f"behavior={int(plot_tbl['behavior_pdf_link_count'].sum())}, "
            f"drift={int(plot_tbl['drift_pdf_link_count'].sum())}, "
            f"opto={int(plot_tbl['opto_pdf_link_count'].sum())}"
        )
    return plot_tbl


# ---------------------------------------------------------------------
# OBJ mesh helpers: pure-Python parser, no trimesh dependency required
# ---------------------------------------------------------------------

def load_obj_tri_mesh(obj_path: str | os.PathLike) -> tuple[np.ndarray, np.ndarray]:
    """
    Load vertices and triangular faces from a simple OBJ file.

    Supports v lines and f lines. Polygons with >3 vertices are fan-triangulated.
    Texture/normal indices like f 1/2/3 are accepted; only vertex indices are used.
    """
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    obj_path = Path(obj_path)

    with obj_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "v" and len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f" and len(parts) >= 4:
                idxs: list[int] = []
                for token in parts[1:]:
                    raw = token.split("/")[0]
                    if not raw:
                        continue
                    idx = int(raw)
                    if idx < 0:
                        idx = len(vertices) + idx
                    else:
                        idx = idx - 1
                    idxs.append(idx)
                if len(idxs) >= 3:
                    for j in range(1, len(idxs) - 1):
                        faces.append([idxs[0], idxs[j], idxs[j + 1]])

    v = np.asarray(vertices, dtype=float)
    f = np.asarray(faces, dtype=np.int64)
    if v.size == 0 or f.size == 0:
        raise ValueError(f"OBJ has no usable vertices/faces: {obj_path}")
    return v, f


def mesh_center_of_mass(vertices: np.ndarray, faces: np.ndarray | None = None) -> np.ndarray:
    """
    Estimate mesh center of mass from vertices/faces without requiring trimesh.

    For a closed consistently oriented triangular surface, this uses the standard
    signed-tetrahedron volume centroid. If the mesh is not closed/oriented or the
    volume estimate degenerates, it falls back to the vertex centroid.
    """
    vertices = np.asarray(vertices, dtype=float)
    if vertices.size == 0:
        raise ValueError("Cannot compute mesh center from an empty vertex array")

    if faces is None or len(faces) == 0:
        return np.nanmean(vertices, axis=0)

    try:
        faces = np.asarray(faces, dtype=np.int64)
        a = vertices[faces[:, 0]]
        b = vertices[faces[:, 1]]
        c = vertices[faces[:, 2]]
        vol6 = np.einsum("ij,ij->i", a, np.cross(b, c))
        vol6_sum = np.sum(vol6)
        if not np.isfinite(vol6_sum) or abs(vol6_sum) < 1e-12:
            return np.nanmean(vertices, axis=0)
        centers = (a + b + c) / 4.0
        center = np.sum(centers * vol6[:, None], axis=0) / vol6_sum
        if not np.all(np.isfinite(center)):
            return np.nanmean(vertices, axis=0)
        return center
    except Exception:
        return np.nanmean(vertices, axis=0)


def get_mesh_center(mesh_files: Mapping[str, str | os.PathLike], acronym: str | None) -> np.ndarray | None:
    """Return the center of mass for one configured mesh acronym, e.g. 'DR'."""
    if not acronym:
        return None
    key = str(acronym).strip().upper()
    mesh_path = None
    for candidate_key, candidate_path in mesh_files.items():
        if str(candidate_key).strip().upper() == key:
            mesh_path = Path(candidate_path)
            break
    if mesh_path is None:
        print(f"Centering skipped: no mesh configured for {acronym!r}")
        return None
    if not mesh_path.exists():
        print(f"Centering skipped: mesh file not found for {acronym}: {mesh_path}")
        return None
    vertices, faces = load_obj_tri_mesh(mesh_path)
    center = mesh_center_of_mass(vertices, faces)
    print(
        f"Centering view on {acronym} mesh center of mass: "
        f"x={center[0]:.4f}, y={center[1]:.4f}, z={center[2]:.4f}"
    )
    return center


def _symmetric_range_around_center(values: np.ndarray, center: float, padding_fraction: float = 0.05) -> list[float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0 or not np.isfinite(center):
        return []
    half_width = np.nanmax(np.abs(values - center))
    if not np.isfinite(half_width) or half_width == 0:
        half_width = 0.5
    half_width *= (1.0 + max(0.0, float(padding_fraction)))
    return [float(center - half_width), float(center + half_width)]


def scene_ranges_centered_on(
    center: np.ndarray,
    plot_tbl: pd.DataFrame,
    mesh_files: Mapping[str, str | os.PathLike] | None = None,
    padding_fraction: float = 0.05,
) -> dict:
    """
    Build equal-scale Plotly scene axis ranges centered on a mesh center.

    Plotly 3D scenes can visually stretch one axis when each axis is allowed to
    auto-range independently. To keep ML/AP/DV millimeters represented to scale,
    this function gives x/y/z the same numeric span and the layout uses
    aspectmode='cube'. Empty space is added to shorter dimensions instead of
    stretching the longer dimension.
    """
    center = np.asarray(center, dtype=float)
    coords = [plot_tbl[["plot_x", "plot_y", "plot_z"]].to_numpy(dtype=float)]

    if mesh_files:
        for acronym, mesh_path in mesh_files.items():
            mesh_path = Path(mesh_path)
            if not mesh_path.exists():
                continue
            try:
                vertices, _faces = load_obj_tri_mesh(mesh_path)
                coords.append(vertices)
            except Exception as exc:
                print(f"Could not include {acronym} mesh in centered scene bounds: {exc!r}")

    all_coords = np.vstack(coords)
    finite_rows = np.all(np.isfinite(all_coords), axis=1)
    all_coords = all_coords[finite_rows]
    if all_coords.size == 0 or not np.all(np.isfinite(center)):
        return {"x": [], "y": [], "z": []}

    axis_half_widths = np.nanmax(np.abs(all_coords - center[None, :]), axis=0)
    half_width = float(np.nanmax(axis_half_widths))
    if not np.isfinite(half_width) or half_width == 0:
        half_width = 0.5
    half_width *= (1.0 + max(0.0, float(padding_fraction)))

    return {
        "x": [float(center[0] - half_width), float(center[0] + half_width)],
        "y": [float(center[1] - half_width), float(center[1] + half_width)],
        "z": [float(center[2] - half_width), float(center[2] + half_width)],
    }


def _mesh_faces_to_unique_edges(faces: np.ndarray,
                                max_edges: int = 45000,
                                edge_stride: int = 1) -> tuple[np.ndarray, int]:
    faces = np.asarray(faces, dtype=np.int64)
    if faces.size == 0:
        return np.empty((0, 2), dtype=np.int64), 0
    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges.sort(axis=1)
    edges = np.unique(edges, axis=0)
    original_edge_count = len(edges)
    if edge_stride > 1:
        edges = edges[::edge_stride]
    if max_edges and len(edges) > max_edges:
        step = int(np.ceil(len(edges) / float(max_edges)))
        edges = edges[::step][:max_edges]
    return edges, original_edge_count


def _mesh_edges_to_plotly_line_coords(vertices: np.ndarray,
                                      edges: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    line_vertices = vertices[edges]
    n_edges = line_vertices.shape[0]
    x = np.full(n_edges * 3, np.nan, dtype=float)
    y = np.full(n_edges * 3, np.nan, dtype=float)
    z = np.full(n_edges * 3, np.nan, dtype=float)
    x[0::3] = line_vertices[:, 0, 0]
    x[1::3] = line_vertices[:, 1, 0]
    y[0::3] = line_vertices[:, 0, 1]
    y[1::3] = line_vertices[:, 1, 1]
    z[0::3] = line_vertices[:, 0, 2]
    z[1::3] = line_vertices[:, 1, 2]
    return x, y, z


def add_wireframe_meshes(fig,
                         mesh_files: Mapping[str, str | os.PathLike] = STRUCTURE_MESH_FILES,
                         max_edges_per_structure: int = 45000,
                         line_width: int = 1) -> int:
    """Add non-hovering Plotly Scatter3d wireframe traces for PAG/AQ/DR/CS."""
    import plotly.graph_objects as go

    loaded = 0
    for acronym, mesh_path in mesh_files.items():
        mesh_path = Path(mesh_path)
        if not mesh_path.exists():
            print(f"Skipping {acronym}: mesh file not found: {mesh_path}")
            continue
        try:
            vertices, faces = load_obj_tri_mesh(mesh_path)
            edges, original_edge_count = _mesh_faces_to_unique_edges(
                faces,
                max_edges=max_edges_per_structure,
                edge_stride=1,
            )
            if len(edges) == 0:
                print(f"Skipping {acronym}: no edges after simplification")
                continue
            x, y, z = _mesh_edges_to_plotly_line_coords(vertices, edges)
            fig.add_trace(
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="lines",
                    line=dict(
                        color=_hex_to_css_rgb(STRUCTURE_COLORS.get(acronym, 0x888888)),
                        width=line_width,
                    ),
                    opacity=STRUCTURE_OPACITIES.get(acronym, 0.65),
                    name=f"{acronym} wireframe",
                    hoverinfo="skip",
                    showlegend=True,
                )
            )
            print(
                f"Added {acronym} wireframe: {mesh_path} "
                f"({len(edges)} edges; original {original_edge_count} edges)"
            )
            loaded += 1
        except Exception as exc:
            print(f"Could not add {acronym} mesh {mesh_path}: {exc!r}")
    return loaded


# ---------------------------------------------------------------------
# Plotly HTML builder
# ---------------------------------------------------------------------

def _click_panel_post_script() -> str:
    """JavaScript installed into the exported HTML by Plotly write_html(post_script=...)."""
    return r"""
(function() {
  var gd = document.getElementById('{plot_id}');
  if (!gd) { return; }

  var host = gd.parentElement || gd;
  if (window.getComputedStyle(host).position === 'static') {
    host.style.position = 'relative';
  }

  var panel = document.createElement('div');
  panel.id = 'unit-click-info-panel';
  panel.style.position = 'absolute';
  panel.style.right = '14px';
  panel.style.top = '54px';
  panel.style.width = '390px';
  panel.style.maxWidth = 'calc(100% - 28px)';
  panel.style.maxHeight = '70vh';
  panel.style.overflowY = 'auto';
  panel.style.background = 'rgba(255,255,255,0.97)';
  panel.style.border = '1px solid #777';
  panel.style.borderRadius = '7px';
  panel.style.boxShadow = '0 3px 14px rgba(0,0,0,0.28)';
  panel.style.padding = '10px 12px';
  panel.style.zIndex = '10000';
  panel.style.fontFamily = 'Arial, Helvetica, sans-serif';
  panel.style.fontSize = '13px';
  panel.style.display = 'none';
  host.appendChild(panel);

  function findUnitHighlightTraceIndex() {
    for (var i = 0; i < gd.data.length; i++) {
      var meta = gd.data[i] && gd.data[i].meta;
      if (meta && meta.role === 'unit_highlight') { return i; }
    }
    return -1;
  }

  function clearUnitHighlight() {
    var highlightIndex = findUnitHighlightTraceIndex();
    if (highlightIndex >= 0 && window.Plotly) {
      window.Plotly.restyle(gd, {x: [[]], y: [[]], z: [[]], visible: [false]}, [highlightIndex]);
    }
  }

  function setUnitHighlight(point) {
    clearUnitHighlight();
    if (!point || !point.data || !point.data.meta || point.data.meta.role !== 'unit_red') {
      return;
    }
    var highlightIndex = findUnitHighlightTraceIndex();
    if (highlightIndex < 0 || !window.Plotly) { return; }

    var cd = point.customdata || [];
    var x = Number.isFinite(point.x) ? point.x : Number(cd[12]);
    var y = Number.isFinite(point.y) ? point.y : Number(cd[13]);
    var z = Number.isFinite(point.z) ? point.z : Number(cd[14]);
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) { return; }

    window.Plotly.restyle(
      gd,
      {x: [[x]], y: [[y]], z: [[z]], visible: [true]},
      [highlightIndex]
    );
  }

  window.__unitClickClearHighlight = clearUnitHighlight;

  function closePanel() {
    panel.style.display = 'none';
    clearUnitHighlight();
  }

  function installCloseButton() {
    var close = panel.querySelector('#unit-info-close');
    if (close) {
      close.onclick = function(evt) {
        if (evt) { evt.stopPropagation(); }
        closePanel();
      };
    }
  }

  gd.on('plotly_click', function(data) {
    if (!data || !data.points || data.points.length === 0) { return; }
    var point = data.points[0];
    if (!point.customdata || !point.customdata[0]) { return; }
    panel.innerHTML = point.customdata[0];
    panel.style.display = 'block';
    installCloseButton();
    setUnitHighlight(point);
  });

  document.addEventListener('keydown', function(evt) {
    if (evt.key === 'Escape') {
      closePanel();
    }
  });
})();
"""



def _criteria_tuner_post_script(
    default_criteria: Mapping[str, object] = DEFAULT_CRITERIA,
    red_color: int = OPTO_COLOR,
    gray_color: int = NON_OPTO_COLOR,
    initial_mouse_ids: Sequence[str] | None = None,
) -> str:
    """
    Browser-side optotag criteria tuner.

    The tuner uses numeric/boolean fields stored in each point's customdata and
    moves units between the gray and red Plotly traces with Plotly.restyle().
    It does not call Python and does not rebuild the HTML.
    """
    js = r"""
(function() {
  var gd = document.getElementById('{plot_id}');
  if (!gd) { return; }

  var defaults = __DEFAULTS_JSON__;
  var redColor = __RED_COLOR_JSON__;
  var grayColor = __GRAY_COLOR_JSON__;
  var initialMouseIds = __INITIAL_MOUSE_IDS_JSON__;
  var initialMouseSet = {};
  (initialMouseIds || []).forEach(function(id) { initialMouseSet[normalizeMouseId(id)] = true; });
  var IDX = {
    panel: 0,
    hover: 1,
    optoPass: 2,
    defaultQc: 3,
    pMax: 4,
    pMean: 5,
    passCount: 6,
    latMaxP: 7,
    isiViolationsRatio: 8,
    corrMaxP: 9,
    euMaxP: 10,
    mouseId: 11,
    x: 12,
    y: 13,
    z: 14
  };

  function normalizeMouseId(value) {
    if (value === null || typeof value === 'undefined') { return 'unknown'; }
    var text = String(value).trim();
    if (!text || text.toLowerCase() === 'nan' || text.toLowerCase() === 'none' || text.toLowerCase() === '<na>') {
      return 'unknown';
    }
    var numeric = Number(text);
    if (Number.isFinite(numeric) && Math.floor(numeric) === numeric) {
      return String(numeric);
    }
    return text;
  }

  function mouseIdSort(a, b) {
    if (a === 'unknown' && b !== 'unknown') { return 1; }
    if (b === 'unknown' && a !== 'unknown') { return -1; }
    var na = Number(a);
    var nb = Number(b);
    if (Number.isFinite(na) && Number.isFinite(nb)) { return na - nb; }
    return String(a).localeCompare(String(b));
  }

  function traceHasRole(trace, role) {
    return trace && trace.meta && trace.meta.role === role;
  }

  var grayTraceIndex = gd.data.findIndex(function(trace) { return traceHasRole(trace, 'unit_gray'); });
  var redTraceIndex = gd.data.findIndex(function(trace) { return traceHasRole(trace, 'unit_red'); });
  if (grayTraceIndex < 0 || redTraceIndex < 0) {
    console.warn('Optotag criteria tuner disabled: unit gray/red traces not found.');
    return;
  }

  var allUnits = [];

  function arrayLength(value) {
    return (value && typeof value.length === 'number') ? value.length : 0;
  }

  function arrayValue(value, index) {
    return (value && typeof value.length === 'number') ? value[index] : undefined;
  }

  function finiteOrFallback(primary, fallback) {
    var val = Number(primary);
    if (Number.isFinite(val)) { return val; }
    val = Number(fallback);
    return Number.isFinite(val) ? val : NaN;
  }

  function collectUnits(traceIndex) {
    var trace = gd.data[traceIndex];
    var xs = trace.x || [];
    var ys = trace.y || [];
    var zs = trace.z || [];
    var cds = trace.customdata || [];

    // Plotly.py 6 can serialize pandas/numpy coordinate arrays as compact
    // binary objects (for example {dtype: ..., bdata: ...}). Those objects do
    // not have .length, so trace.x.length can look like zero units to custom
    // JavaScript. Coordinates are therefore also stored in customdata, and the
    // collector uses customdata length as the reliable unit count.
    var n = Math.max(arrayLength(xs), arrayLength(cds));
    for (var i = 0; i < n; i++) {
      var cd = cds[i] || [];
      var x = finiteOrFallback(arrayValue(xs, i), cd[IDX.x]);
      var y = finiteOrFallback(arrayValue(ys, i), cd[IDX.y]);
      var z = finiteOrFallback(arrayValue(zs, i), cd[IDX.z]);
      if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) { continue; }
      allUnits.push({
        x: x,
        y: y,
        z: z,
        cd: cd,
        mouseId: normalizeMouseId(cd[IDX.mouseId])
      });
    }
  }
  collectUnits(grayTraceIndex);
  collectUnits(redTraceIndex);

  var host = gd.parentElement || gd;
  if (window.getComputedStyle(host).position === 'static') {
    host.style.position = 'relative';
  }

  var panel = document.createElement('div');
  panel.id = 'opto-criteria-tuner-panel';
  panel.style.position = 'absolute';
  panel.style.left = '14px';
  panel.style.top = '54px';
  panel.style.width = '330px';
  panel.style.maxWidth = 'calc(100% - 28px)';
  panel.style.maxHeight = '72vh';
  panel.style.overflowY = 'auto';
  panel.style.background = 'rgba(255,255,255,0.96)';
  panel.style.border = '1px solid #777';
  panel.style.borderRadius = '7px';
  panel.style.boxShadow = '0 3px 14px rgba(0,0,0,0.24)';
  panel.style.padding = '10px 12px';
  panel.style.zIndex = '9999';
  panel.style.fontFamily = 'Arial, Helvetica, sans-serif';
  panel.style.fontSize = '12px';
  panel.innerHTML = '' +
    '<div style="display:flex; align-items:center; justify-content:space-between; gap:8px; margin-bottom:6px">' +
      '<div style="font-weight:bold; font-size:13px">Optotag criteria tuner</div>' +
      '<button type="button" data-action="collapse" title="Collapse" style="border:1px solid #aaa; background:#f7f7f7; border-radius:3px; cursor:pointer">-</button>' +
    '</div>' +
    '<div data-section="body">' +
      '<div style="color:#555; margin-bottom:8px">Change thresholds to move units between gray and red traces. Click a unit for the locked PDF panel.</div>' +
      '<div style="display:grid; grid-template-columns: 1fr auto; row-gap:5px; column-gap:8px; align-items:center">' +
        '<label><input type="checkbox" data-control="showRed" checked> show red dots</label><span></span>' +
        '<label><input type="checkbox" data-control="showGray" checked> show gray units</label><span></span>' +
        '<label><input type="checkbox" data-control="requireOpto"> require opto_pass</label><span></span>' +
        '<label><input type="checkbox" data-control="requireDefaultQc"> require default_qc</label><span></span>' +
        '<label for="pMaxMin">p_max &gt;</label><input id="pMaxMin" data-control="pMaxMin" type="number" step="0.01" style="width:72px">' +
        '<label for="pMeanMin">p_mean &gt;</label><input id="pMeanMin" data-control="pMeanMin" type="number" step="0.01" style="width:72px">' +
        '<label for="passCountMin">pass_count &ge;</label><input id="passCountMin" data-control="passCountMin" type="number" step="1" style="width:72px">' +
        '<label for="latMinMs">lat_max_p min, ms &gt;</label><input id="latMinMs" data-control="latMinMs" type="number" step="0.5" style="width:72px">' +
        '<label for="latMaxMs">lat_max_p max, ms &lt;</label><input id="latMaxMs" data-control="latMaxMs" type="number" step="0.5" style="width:72px">' +
        '<label for="isiMax">ISI violations &lt;</label><input id="isiMax" data-control="isiMax" type="number" step="0.01" style="width:72px">' +
        '<label for="corrMin">corr_max_p &gt;</label><input id="corrMin" data-control="corrMin" type="number" step="0.01" style="width:72px">' +
        '<label for="euMax">eu_max_p &lt;</label><input id="euMax" data-control="euMax" type="number" step="0.01" style="width:72px">' +
      '</div>' +
      '<div style="border-top:1px solid #ddd; margin-top:9px; padding-top:8px">' +
        '<div style="font-weight:bold; margin-bottom:5px">Mouse ID filter</div>' +
        '<div style="display:flex; gap:6px; align-items:center; margin-bottom:5px">' +
          '<button type="button" data-action="selectMouseAll" style="border:1px solid #aaa; background:#f7f7f7; border-radius:4px; cursor:pointer">All</button>' +
          '<button type="button" data-action="selectMouseNone" style="border:1px solid #aaa; background:#f7f7f7; border-radius:4px; cursor:pointer">None</button>' +
          '<input data-control="mouseSearch" type="text" placeholder="search mouse" style="width:105px; margin-left:auto">' +
        '</div>' +
        '<div data-mouse-summary style="font-family:monospace; color:#555; margin-bottom:4px"></div>' +
        '<div data-mouse-list style="max-height:132px; overflow-y:auto; border:1px solid #ddd; border-radius:4px; padding:4px; background:#fff"></div>' +
      '</div>' +
      '<div style="display:flex; gap:6px; align-items:center; margin-top:9px">' +
        '<button type="button" data-action="apply" style="border:1px solid #777; background:#eee; border-radius:4px; cursor:pointer">Apply</button>' +
        '<button type="button" data-action="reset" style="border:1px solid #aaa; background:#f7f7f7; border-radius:4px; cursor:pointer">Reset</button>' +
        '<label style="margin-left:auto"><input type="checkbox" data-control="live" checked> live</label>' +
      '</div>' +
      '<div data-count style="font-family:monospace; margin-top:8px; color:#333"></div>' +
      '<div style="font-size:11px; color:#777; margin-top:6px">Red criterion uses strict inequalities matching the Python defaults: p/lat/corr use &gt; or &lt;; pass_count uses &ge;.</div>' +
    '</div>';
  host.appendChild(panel);

  // Prevent mouse events inside the control panel from rotating/zooming the 3D plot.
  ['mousedown', 'mouseup', 'click', 'dblclick', 'wheel', 'touchstart', 'touchmove'].forEach(function(eventName) {
    panel.addEventListener(eventName, function(evt) { evt.stopPropagation(); }, { passive: false });
  });

  function control(name) {
    return panel.querySelector('[data-control="' + name + '"]');
  }
  function readNumber(name, fallback) {
    var el = control(name);
    if (!el) { return fallback; }
    var val = Number(el.value);
    return Number.isFinite(val) ? val : fallback;
  }
  function asNumber(v) {
    var val = Number(v);
    return Number.isFinite(val) ? val : NaN;
  }
  function asBool(v) {
    if (v === true) { return true; }
    if (v === false || v === null || typeof v === 'undefined') { return false; }
    if (typeof v === 'number') { return Number.isFinite(v) && v !== 0; }
    if (typeof v === 'string') {
      var s = v.trim().toLowerCase();
      return ['true', '1', '1.0', 't', 'yes', 'y'].indexOf(s) >= 0;
    }
    return false;
  }
  function getMouseCounts() {
    var counts = {};
    for (var i = 0; i < allUnits.length; i++) {
      var id = normalizeMouseId(allUnits[i].mouseId);
      counts[id] = (counts[id] || 0) + 1;
    }
    return counts;
  }
  function updateMouseSummary() {
    var summary = panel.querySelector('[data-mouse-summary]');
    if (!summary) { return; }
    var boxes = panel.querySelectorAll('[data-mouse-toggle]');
    var selected = 0;
    for (var i = 0; i < boxes.length; i++) {
      if (boxes[i].checked) { selected += 1; }
    }
    summary.textContent = selected + ' / ' + boxes.length + ' mice selected';
  }
  function renderMouseControls() {
    var list = panel.querySelector('[data-mouse-list]');
    if (!list) { return; }
    var counts = getMouseCounts();
    var mouseIds = Object.keys(counts).sort(mouseIdSort);
    list.innerHTML = '';
    for (var i = 0; i < mouseIds.length; i++) {
      var id = mouseIds[i];
      var row = document.createElement('label');
      row.style.display = 'block';
      row.style.whiteSpace = 'nowrap';
      row.style.margin = '2px 0';
      row.setAttribute('data-mouse-row', '1');
      row.setAttribute('data-mouse-label', String(id).toLowerCase());
      var cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.value = id;
      cb.checked = (initialMouseIds.length === 0 || !!initialMouseSet[normalizeMouseId(id)]);
      cb.setAttribute('data-mouse-toggle', '1');
      row.appendChild(cb);
      row.appendChild(document.createTextNode(' ' + id + ' (' + counts[id] + ')'));
      list.appendChild(row);
    }
    updateMouseSummary();
  }
  function readAllowedMouseSet() {
    var boxes = panel.querySelectorAll('[data-mouse-toggle]');
    var allowed = {};
    if (boxes.length === 0) { return allowed; }
    for (var i = 0; i < boxes.length; i++) {
      if (boxes[i].checked) {
        allowed[normalizeMouseId(boxes[i].value)] = true;
      }
    }
    return allowed;
  }
  function setAllMouseCheckboxes(checked) {
    var boxes = panel.querySelectorAll('[data-mouse-toggle]');
    for (var i = 0; i < boxes.length; i++) {
      boxes[i].checked = checked;
    }
    updateMouseSummary();
    applyCriteria();
  }
  function filterMouseList() {
    var input = control('mouseSearch');
    var q = input ? String(input.value).trim().toLowerCase() : '';
    var rows = panel.querySelectorAll('[data-mouse-row]');
    for (var i = 0; i < rows.length; i++) {
      var label = rows[i].getAttribute('data-mouse-label') || '';
      rows[i].style.display = (!q || label.indexOf(q) >= 0) ? 'block' : 'none';
    }
  }
  function setDefaults() {
    control('showRed').checked = true;
    control('showGray').checked = true;
    control('requireOpto').checked = !!defaults.require_opto_pass;
    control('requireDefaultQc').checked = !!defaults.require_default_qc;
    control('pMaxMin').value = defaults.p_max_min;
    control('pMeanMin').value = defaults.p_mean_min;
    control('passCountMin').value = defaults.pass_count_min;
    control('latMinMs').value = defaults.lat_max_p_min_ms;
    control('latMaxMs').value = defaults.lat_max_p_max_ms;
    control('isiMax').value = defaults.isi_violations_ratio_max;
    control('corrMin').value = defaults.corr_max_p_min;
    control('euMax').value = defaults.eu_max_p_max;
    var search = control('mouseSearch');
    if (search) { search.value = ''; }
    var boxes = panel.querySelectorAll('[data-mouse-toggle]');
    for (var i = 0; i < boxes.length; i++) {
      boxes[i].checked = (initialMouseIds.length === 0 || !!initialMouseSet[normalizeMouseId(boxes[i].value)]);
    }
    filterMouseList();
    updateMouseSummary();
  }
  function passesCriteria(unit) {
    var cd = unit.cd || [];
    if (control('requireOpto').checked && !asBool(cd[IDX.optoPass])) { return false; }
    if (control('requireDefaultQc').checked && !asBool(cd[IDX.defaultQc])) { return false; }

    var pMax = asNumber(cd[IDX.pMax]);
    var pMean = asNumber(cd[IDX.pMean]);
    var passCount = asNumber(cd[IDX.passCount]);
    var latMaxP = asNumber(cd[IDX.latMaxP]);
    var isi = asNumber(cd[IDX.isiViolationsRatio]);
    var corr = asNumber(cd[IDX.corrMaxP]);
    var eu = asNumber(cd[IDX.euMaxP]);

    if (!(pMax > readNumber('pMaxMin', defaults.p_max_min))) { return false; }
    if (!(pMean > readNumber('pMeanMin', defaults.p_mean_min))) { return false; }
    if (!(passCount >= readNumber('passCountMin', defaults.pass_count_min))) { return false; }
    if (!(latMaxP > readNumber('latMinMs', defaults.lat_max_p_min_ms) / 1000.0)) { return false; }
    if (!(latMaxP < readNumber('latMaxMs', defaults.lat_max_p_max_ms) / 1000.0)) { return false; }
    if (!(isi < readNumber('isiMax', defaults.isi_violations_ratio_max))) { return false; }
    if (!(corr > readNumber('corrMin', defaults.corr_max_p_min))) { return false; }
    if (!(eu < readNumber('euMax', defaults.eu_max_p_max))) { return false; }
    return true;
  }
  function splitUnits() {
    var red = [];
    var gray = [];
    var showRed = control('showRed').checked;
    var allowedMouseSet = readAllowedMouseSet();
    var mouseFilteredTotal = 0;
    for (var i = 0; i < allUnits.length; i++) {
      var mouseId = normalizeMouseId(allUnits[i].mouseId);
      if (!allowedMouseSet[mouseId]) { continue; }
      mouseFilteredTotal += 1;
      if (showRed && passesCriteria(allUnits[i])) {
        red.push(allUnits[i]);
      } else {
        gray.push(allUnits[i]);
      }
    }
    if (!control('showGray').checked) {
      gray = [];
    }
    return { red: red, gray: gray, mouseFilteredTotal: mouseFilteredTotal };
  }
  function restyleUnits(traceIndex, units, name, color, size, opacity) {
    var xs = [];
    var ys = [];
    var zs = [];
    var cds = [];
    for (var i = 0; i < units.length; i++) {
      xs.push(units[i].x);
      ys.push(units[i].y);
      zs.push(units[i].z);
      cds.push(units[i].cd);
    }
    Plotly.restyle(gd, {
      x: [xs],
      y: [ys],
      z: [zs],
      customdata: [cds],
      name: [name],
      hovertemplate: ['%{customdata[1]}<extra></extra>'],
      'marker.color': [color],
      'marker.size': [size],
      'marker.opacity': [opacity]
    }, [traceIndex]);
  }
  function applyCriteria() {
    if (window.__unitClickClearHighlight) { window.__unitClickClearHighlight(); }
    var split = splitUnits();
    restyleUnits(grayTraceIndex, split.gray, 'gray outside tuned criteria n=' + split.gray.length, grayColor, 2.0, 0.28);
    restyleUnits(redTraceIndex, split.red, 'red tuned optotag criteria n=' + split.red.length, redColor, 4.2, 0.96);
    var countEl = panel.querySelector('[data-count]');
    if (countEl) {
      countEl.textContent = split.red.length + ' red / ' + split.mouseFilteredTotal + ' mouse-filtered / ' + allUnits.length + ' plotted units';
      updateMouseSummary();
    }
  }
  var applyTimer = null;
  function scheduleApply() {
    if (!control('live').checked) { return; }
    if (applyTimer !== null) { window.clearTimeout(applyTimer); }
    applyTimer = window.setTimeout(function() {
      applyTimer = null;
      applyCriteria();
    }, 100);
  }

  renderMouseControls();
  setDefaults();
  applyCriteria();

  panel.addEventListener('input', function(evt) {
    if (!evt.target || !evt.target.matches('input')) { return; }
    if (evt.target.getAttribute('data-control') === 'mouseSearch') {
      filterMouseList();
      return;
    }
    scheduleApply();
  });
  panel.addEventListener('change', function(evt) {
    if (!evt.target || !evt.target.matches('input')) { return; }
    if (evt.target.getAttribute('data-mouse-toggle') === '1') {
      updateMouseSummary();
    }
    scheduleApply();
  });
  var applyButton = panel.querySelector('[data-action="apply"]');
  if (applyButton) { applyButton.onclick = applyCriteria; }
  var resetButton = panel.querySelector('[data-action="reset"]');
  if (resetButton) {
    resetButton.onclick = function() {
      setDefaults();
      applyCriteria();
    };
  }
  var selectMouseAllButton = panel.querySelector('[data-action="selectMouseAll"]');
  if (selectMouseAllButton) {
    selectMouseAllButton.onclick = function() { setAllMouseCheckboxes(true); };
  }
  var selectMouseNoneButton = panel.querySelector('[data-action="selectMouseNone"]');
  if (selectMouseNoneButton) {
    selectMouseNoneButton.onclick = function() { setAllMouseCheckboxes(false); };
  }
  var collapseButton = panel.querySelector('[data-action="collapse"]');
  if (collapseButton) {
    collapseButton.onclick = function() {
      var body = panel.querySelector('[data-section="body"]');
      if (!body) { return; }
      var collapsed = body.style.display === 'none';
      body.style.display = collapsed ? 'block' : 'none';
      collapseButton.textContent = collapsed ? '-' : '+';
      collapseButton.title = collapsed ? 'Collapse' : 'Expand';
    };
  }
})();
"""
    return (
        js
        .replace("__DEFAULTS_JSON__", json.dumps(dict(default_criteria)))
        .replace("__RED_COLOR_JSON__", json.dumps(_hex_to_css_rgb(red_color)))
        .replace("__GRAY_COLOR_JSON__", json.dumps(_hex_to_css_rgb(gray_color)))
        .replace("__INITIAL_MOUSE_IDS_JSON__", json.dumps([normalize_mouse_id(x) for x in (initial_mouse_ids or []) if normalize_mouse_id(x)]))
    )


def build_clickable_unit_pdf_plotly_html(
    master_path: str | os.PathLike = MASTER_PATH,
    pdf_dir: str | os.PathLike = PDF_DIR,
    drift_pdf_dir: str | os.PathLike | None = DRIFT_PDF_DIR,
    opto_pdf_dir: str | os.PathLike | None = OPTO_PDF_DIR,
    out_html: str | os.PathLike | None = None,
    align_names: Sequence[str] = ("go_cue", "response"),
    filename_template: str = "{session}_unit_{unit_id}_{align}.pdf",
    drift_filename_template: str = "{session}_unit_{unit_id}_drift.pdf",
    opto_filename_template: str = "{session}_unit_{unit_id}_opto.pdf",
    include_drift_pdf_link: bool = True,
    include_opto_pdf_link: bool = True,
    metrics: Sequence[str] = DEFAULT_METRICS,
    link_mode: str = "relative",
    use_bregma_relative: bool = USE_BREGMA_RELATIVE,
    use_strict_opto_red: bool = True,
    include_meshes: bool = True,
    mesh_files: Mapping[str, str | os.PathLike] = STRUCTURE_MESH_FILES,
    max_edges_per_structure: int = 45000,
    mesh_line_width: int = 1,
    center_view_on_mesh: str | None = (CENTER_ON_MESH if CENTER_VIEW_ON_MESH else None),
    center_view_padding_fraction: float = CENTER_VIEW_PADDING_FRACTION,
    include_missing_pdf_paths: bool = True,
    trust_expected_pdf_names: bool = TRUST_EXPECTED_PDF_NAMES,
    interactive_criteria_controls: bool = INTERACTIVE_CRITERIA_CONTROLS,
    initial_mouse_ids: Sequence[str] | None = None,
    include_plotlyjs: bool | str = True,
) -> tuple[Path, pd.DataFrame]:
    """
    Build and save a standalone Plotly HTML file.

    Returns
    -------
    out_html : Path
        Path to the saved HTML.
    plot_tbl : pandas.DataFrame
        The plotted table with click_panel_html/pdf_link_count columns added.
    """
    import plotly.graph_objects as go

    master_path = get_master_path(master_path)
    master = load_master_table(master_path)
    print(f"Loaded master table: {master_path}")
    print(f"Master shape: {master.shape}")

    if out_html is None:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        mesh_suffix = "_with_wireframe_meshes" if include_meshes else ""
        out_html = OUT_DIR / f"non_noise_nonartifact_units_click_locked_pdf_links{mesh_suffix}.html"
    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)

    align_names = tuple(align_names)
    metrics = tuple(metrics)

    plot_tbl, red_label = prepare_plot_table(
        master,
        use_bregma_relative=use_bregma_relative,
        use_strict_opto_red=use_strict_opto_red,
    )
    plot_tbl = add_click_panel_columns(
        plot_tbl,
        pdf_dir=pdf_dir,
        out_html=out_html,
        align_names=align_names,
        filename_template=filename_template,
        drift_pdf_dir=drift_pdf_dir,
        opto_pdf_dir=opto_pdf_dir,
        drift_filename_template=drift_filename_template,
        opto_filename_template=opto_filename_template,
        include_drift_pdf_link=include_drift_pdf_link,
        include_opto_pdf_link=include_opto_pdf_link,
        metrics=metrics,
        link_mode=link_mode,
        include_missing_pdf_paths=include_missing_pdf_paths,
        trust_expected_pdf_names=trust_expected_pdf_names,
    )

    print("\nPlot table:")
    print(f"  total units in master:              {len(master)}")
    print(f"  non-noise/non-artifact with CCF:    {len(plot_tbl)}")
    print(f"  red units ({red_label}):            {int(plot_tbl['opto_pass_bool'].sum())}")
    print(f"  gray units:                         {int((~plot_tbl['opto_pass_bool']).sum())}")
    print(f"  behavior PDF directory:             {Path(pdf_dir)}")
    print(f"  drift PDF directory:                {Path(drift_pdf_dir) if drift_pdf_dir is not None else 'disabled'}")
    print(f"  opto PDF directory:                 {Path(opto_pdf_dir) if opto_pdf_dir is not None else 'disabled'}")
    print(f"  behavior filename template:         {filename_template}")
    print(f"  drift filename template:            {drift_filename_template}")
    print(f"  opto filename template:             {opto_filename_template}")
    print(f"  link mode:                          {link_mode}")
    print(f"  trust expected PDF names:           {trust_expected_pdf_names}")
    print(f"  interactive criteria controls:      {interactive_criteria_controls}")
    print(f"  red criteria extras:                isi_violations_ratio < 0.1, corr_max_p > 0.7, eu_max_p < 0.3")
    mouse_ids_available = sorted(plot_tbl["mouse_id_norm"].dropna().astype(str).unique(), key=lambda x: (x == "unknown", float(x) if str(x).replace(".", "", 1).isdigit() else float("inf"), str(x)))
    print(f"  center view on mesh:                {center_view_on_mesh if center_view_on_mesh else 'off'}")
    print(f"  mouse IDs available:                {len(mouse_ids_available)}")
    if initial_mouse_ids:
        print(f"  initial mouse IDs selected:         {tuple(initial_mouse_ids)}")

    fig = go.Figure()

    # Wireframe meshes first, points after, so points remain clickable.
    if include_meshes:
        if use_bregma_relative:
            add_wireframe_meshes(
                fig,
                mesh_files=mesh_files,
                max_edges_per_structure=max_edges_per_structure,
                line_width=mesh_line_width,
            )
        else:
            print("Skipping meshes because use_bregma_relative=False; configured meshes are bregma-relative.")

    center_point = None
    scene_axis_ranges = None
    scene_center_label = "data centroid"
    if use_bregma_relative and center_view_on_mesh:
        center_point = get_mesh_center(mesh_files, center_view_on_mesh)
        if center_point is not None:
            scene_center_label = str(center_view_on_mesh)
            scene_axis_ranges = scene_ranges_centered_on(
                center_point,
                plot_tbl,
                mesh_files=mesh_files if include_meshes else None,
                padding_fraction=center_view_padding_fraction,
            )

    # Fallback equal-scale scene even if the requested center mesh is unavailable
    # or centering on a mesh was disabled. This prevents ML/AP/DV axes from
    # being stretched independently by browser autorange.
    if scene_axis_ranges is None:
        coords_for_center = plot_tbl[["plot_x", "plot_y", "plot_z"]].to_numpy(dtype=float)
        finite_rows = np.all(np.isfinite(coords_for_center), axis=1)
        if np.any(finite_rows):
            center_point = np.nanmean(coords_for_center[finite_rows], axis=0)
            scene_axis_ranges = scene_ranges_centered_on(
                center_point,
                plot_tbl,
                mesh_files=mesh_files if (include_meshes and use_bregma_relative) else None,
                padding_fraction=center_view_padding_fraction,
            )
            print(
                "Using equal-scale scene ranges centered on plotted-unit centroid: "
                f"x={center_point[0]:.4f}, y={center_point[1]:.4f}, z={center_point[2]:.4f}"
            )

    opto_mask = plot_tbl["opto_pass_bool"].to_numpy(dtype=bool)
    non_tbl = plot_tbl.loc[~opto_mask]
    opto_tbl = plot_tbl.loc[opto_mask]

    def add_unit_trace(
        tbl: pd.DataFrame,
        color: int,
        size: float,
        opacity: float,
        name: str,
        role: str,
        always_add: bool = False,
    ) -> None:
        if len(tbl) == 0 and not always_add:
            return

        # customdata[0] = full HTML panel; customdata[1] = lightweight hover text.
        # customdata[2:] = numeric/boolean fields used by the browser-side criteria tuner.
        if len(tbl) == 0:
            customdata = []
            x = []
            y = []
            z = []
        else:
            customdata_columns = [
                tbl["click_panel_html"].to_numpy(dtype=object),
                tbl["hover_text"].to_numpy(dtype=object),
                tbl["opto_pass_raw_bool"].to_numpy(dtype=object),
                tbl["default_qc_bool"].to_numpy(dtype=object),
                tbl["p_max"].to_numpy(dtype=object),
                tbl["p_mean"].to_numpy(dtype=object),
                tbl["pass_count"].to_numpy(dtype=object),
                tbl["lat_max_p"].to_numpy(dtype=object),
                tbl["isi_violations_ratio"].to_numpy(dtype=object),
                tbl["corr_max_p"].to_numpy(dtype=object),
                tbl["eu_max_p"].to_numpy(dtype=object),
                tbl["mouse_id_norm"].to_numpy(dtype=object),
                tbl["plot_x"].to_numpy(dtype=object),
                tbl["plot_y"].to_numpy(dtype=object),
                tbl["plot_z"].to_numpy(dtype=object),
            ]
            customdata = np.column_stack(customdata_columns)
            # Use plain Python lists to avoid Plotly.py 6 compact binary array
            # serialization for pandas/numpy coordinates. The browser-side tuner
            # also stores coordinates in customdata as a fallback.
            x = tbl["plot_x"].astype(float).tolist()
            y = tbl["plot_y"].astype(float).tolist()
            z = tbl["plot_z"].astype(float).tolist()

        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                marker=dict(size=size, color=_hex_to_css_rgb(color), opacity=opacity),
                name=name,
                customdata=customdata,
                hovertemplate="%{customdata[1]}<extra></extra>",
                meta={"role": role},
            )
        )

    add_unit_trace(
        non_tbl,
        color=NON_OPTO_COLOR,
        size=2.0,
        opacity=0.28,
        name=f"gray non-opto/non-candidate n={len(non_tbl)}",
        role="unit_gray",
        always_add=interactive_criteria_controls,
    )
    add_unit_trace(
        opto_tbl,
        color=OPTO_COLOR,
        size=4.2,
        opacity=0.96,
        name=f"red {red_label} n={len(opto_tbl)}",
        role="unit_red",
        always_add=interactive_criteria_controls,
    )

    # Invisible selection overlay. Browser-side click JS places one yellow marker
    # here when a red unit is clicked, then clears it when the locked panel closes.
    fig.add_trace(
        go.Scatter3d(
            x=[],
            y=[],
            z=[],
            mode="markers",
            marker=dict(
                size=9.5,
                color="rgb(255,230,0)",
                opacity=1.0,
                line=dict(color="rgb(40,40,0)", width=2),
            ),
            name="clicked red unit highlight",
            hoverinfo="skip",
            showlegend=False,
            visible=False,
            meta={"role": "unit_highlight"},
        )
    )

    # Do not plot the bregma origin marker; anatomical centering is handled by scene ranges.

    title_suffix = "bregma-relative" if use_bregma_relative else "raw CCF"
    scene_layout = dict(
        xaxis_title="x / ML-like",
        yaxis_title="y / AP-like",
        zaxis_title="z / DV-like",
        # Equal numeric x/y/z ranges plus cube aspect keeps ML/AP/DV mm to scale.
        aspectmode="cube",
        aspectratio=dict(x=1, y=1, z=1),
    )
    if scene_axis_ranges is not None:
        scene_layout["xaxis"] = dict(range=scene_axis_ranges["x"])
        scene_layout["yaxis"] = dict(range=scene_axis_ranges["y"])
        scene_layout["zaxis"] = dict(range=scene_axis_ranges["z"])

    fig.update_layout(
        title=f"Unit map with click-locked PDF links ({title_suffix}; centered on {scene_center_label})",
        scene=scene_layout,
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, b=0, t=42),
        hoverlabel=dict(align="left"),
    )

    post_script = _click_panel_post_script()
    if interactive_criteria_controls:
        post_script += "\n" + _criteria_tuner_post_script(initial_mouse_ids=initial_mouse_ids)

    fig.write_html(
        str(out_html),
        include_plotlyjs=include_plotlyjs,
        full_html=True,
        post_script=post_script,
    )
    print(f"\nSaved clickable Plotly HTML: {out_html}")

    audit_csv = out_html.with_suffix(".plotted_units.csv")
    save_cols = [
        "mouse_id", "mouse_id_norm", "mouse_id_display", get_session_column(plot_tbl), "unit_id", "unit_uid", "decoder_label",
        "default_qc", "opto_pass", "opto_pass_bool", "candidate_strict_5ht_qc_wf",
        "p_max", "p_mean", "lat_max_p", "lat_mean", "corr_max_p", "euc_max_p", "eu_max_p",
        "isi_violations_ratio", "pass_count", "x_ccf", "y_ccf", "z_ccf",
        "plot_x", "plot_y", "plot_z", "pdf_link_count",
        "behavior_pdf_link_count", "drift_pdf_link_count", "opto_pdf_link_count",
    ]
    plot_tbl[[c for c in save_cols if c in plot_tbl.columns]].to_csv(audit_csv, index=False)
    print(f"Saved plotted-unit audit CSV: {audit_csv}")
    return out_html, plot_tbl


def build_clickable_unit_pdf_plotly_html_with_criteria_tuner(*args, **kwargs) -> tuple[Path, pd.DataFrame]:
    """
    Convenience wrapper that always includes the browser-side optotag criteria tuner.

    Use this from notebooks when you want to adjust p_max, p_mean, pass_count,
    lat_max_p, isi_violations_ratio, corr_max_p, eu_max_p, opto_pass, and default_qc after
    the HTML has loaded. No Python callback is required; the saved HTML remains
    standalone.
    """
    kwargs["interactive_criteria_controls"] = True
    return build_clickable_unit_pdf_plotly_html(*args, **kwargs)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a clickable Plotly 3D CCF unit map with local PDF links."
    )
    parser.add_argument("--master-path", default=os.environ.get("MASTER_PATH", MASTER_PATH))
    parser.add_argument("--pdf-dir", default=os.environ.get("UNIT_PDF_DIR", str(PDF_DIR)))
    parser.add_argument("--drift-pdf-dir", default=os.environ.get("UNIT_DRIFT_PDF_DIR", str(DRIFT_PDF_DIR)))
    parser.add_argument("--opto-pdf-dir", default=os.environ.get("UNIT_OPTO_PDF_DIR", str(OPTO_PDF_DIR)))
    parser.add_argument("--out-html", default=os.environ.get("OUT_HTML", ""))
    parser.add_argument("--align-names", default=os.environ.get("ALIGN_NAMES", "go_cue,response"))
    parser.add_argument(
        "--filename-template",
        default=os.environ.get("PDF_FILENAME_TEMPLATE", "{session}_unit_{unit_id}_{align}.pdf"),
        help="Behavior-aligned PDF Python format string. Available fields: {session}, {unit_id}, {unit}, {align}, {alignment}",
    )
    parser.add_argument(
        "--drift-filename-template",
        default=os.environ.get("DRIFT_PDF_FILENAME_TEMPLATE", "{session}_unit_{unit_id}_drift.pdf"),
        help="Drift PDF Python format string. Available fields: {session}, {unit_id}, {unit}, {label}",
    )
    parser.add_argument(
        "--opto-filename-template",
        default=os.environ.get("OPTO_PDF_FILENAME_TEMPLATE", "{session}_unit_{unit_id}_opto.pdf"),
        help="Optotagging PDF Python format string. Available fields: {session}, {unit_id}, {unit}, {label}",
    )
    parser.add_argument("--metrics", default=os.environ.get("METRICS", ",".join(DEFAULT_METRICS)))
    parser.add_argument(
        "--link-mode",
        default=os.environ.get("LINK_MODE", "relative"),
        choices=["relative", "absolute_file", "path"],
    )
    parser.add_argument("--no-meshes", action="store_true", help="Do not include PAG/AQ/DR/CS wireframe meshes.")
    parser.add_argument("--max-edges-per-structure", type=int, default=int(os.environ.get("MAX_EDGES_PER_STRUCTURE", "45000")))
    parser.add_argument("--mesh-line-width", type=int, default=int(os.environ.get("MESH_LINE_WIDTH", "1")))
    _center_env_enabled = os.environ.get("CENTER_VIEW_ON_MESH", "1").strip().lower() not in {"0", "false", "no", "n"}
    parser.add_argument(
        "--center-on-mesh",
        default=(os.environ.get("CENTER_ON_MESH", "DR") if _center_env_enabled else ""),
        help="Mesh acronym used to center the 3D scene ranges. Default: DR. Use empty string to disable.",
    )
    parser.add_argument(
        "--center-view-padding-fraction",
        type=float,
        default=float(os.environ.get("CENTER_VIEW_PADDING_FRACTION", "0.05")),
    )
    parser.add_argument("--broad-opto-red", action="store_true", help="Color broad opto_pass units red instead of strict candidate units.")
    trust_group = parser.add_mutually_exclusive_group()
    trust_group.add_argument(
        "--trust-expected-pdf-names",
        dest="trust_expected_pdf_names",
        action="store_true",
        default=TRUST_EXPECTED_PDF_NAMES,
        help=(
            "Create clickable links from the expected PDF filename template even if "
            "the PDFs are not visible to this Python environment. Use this for "
            "capsule-generated HTML that will be opened next to local PDFs."
        ),
    )
    trust_group.add_argument(
        "--check-pdf-exists",
        dest="trust_expected_pdf_names",
        action="store_false",
        help="Only create links for PDFs that exist where this script is running.",
    )
    criteria_group = parser.add_mutually_exclusive_group()
    criteria_group.add_argument(
        "--criteria-panel",
        dest="interactive_criteria_controls",
        action="store_true",
        default=INTERACTIVE_CRITERIA_CONTROLS,
        help="Include the browser-side optotag criteria tuner panel. This is the default.",
    )
    criteria_group.add_argument(
        "--no-criteria-panel",
        dest="interactive_criteria_controls",
        action="store_false",
        help="Disable the browser-side optotag criteria tuner panel.",
    )
    parser.add_argument(
        "--initial-mouse-ids",
        default=os.environ.get("INITIAL_MOUSE_IDS", ""),
        help="Optional comma-separated mouse IDs checked by default in the browser filter. Empty means all mice checked.",
    )
    parser.add_argument("--hide-missing-pdf-paths", action="store_true", help="Do not list missing expected PDF paths in the click panel.")
    parser.add_argument("--no-drift-pdf-link", action="store_true", help="Do not add drift PDF links to the click panel.")
    parser.add_argument("--no-opto-pdf-link", action="store_true", help="Do not add optotagging PDF links to the click panel.")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    out_html = args.out_html.strip() or None
    build_clickable_unit_pdf_plotly_html(
        master_path=args.master_path,
        pdf_dir=args.pdf_dir,
        drift_pdf_dir=args.drift_pdf_dir,
        opto_pdf_dir=args.opto_pdf_dir,
        out_html=out_html,
        align_names=_parse_csv_list(args.align_names, ("go_cue", "response")),
        filename_template=args.filename_template,
        drift_filename_template=args.drift_filename_template,
        opto_filename_template=args.opto_filename_template,
        include_drift_pdf_link=not args.no_drift_pdf_link,
        include_opto_pdf_link=not args.no_opto_pdf_link,
        metrics=_parse_csv_list(args.metrics, DEFAULT_METRICS),
        link_mode=args.link_mode,
        use_strict_opto_red=not args.broad_opto_red,
        include_meshes=not args.no_meshes,
        max_edges_per_structure=args.max_edges_per_structure,
        mesh_line_width=args.mesh_line_width,
        center_view_on_mesh=args.center_on_mesh.strip() or None,
        center_view_padding_fraction=args.center_view_padding_fraction,
        include_missing_pdf_paths=not args.hide_missing_pdf_paths,
        trust_expected_pdf_names=args.trust_expected_pdf_names,
        interactive_criteria_controls=args.interactive_criteria_controls,
        initial_mouse_ids=_parse_csv_list(args.initial_mouse_ids, ()) if args.initial_mouse_ids.strip() else None,
    )


if __name__ == "__main__":
    main()

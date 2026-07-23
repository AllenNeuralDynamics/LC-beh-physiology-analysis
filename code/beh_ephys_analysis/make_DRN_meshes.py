#!/usr/bin/env python3
"""
Create separate Allen CCF OBJ meshes for PAG, AQ, DR, and CS, plus an optional
K3D HTML viewer in which each structure has its own color.

Outputs, for each acronym:
    <ACRONYM>_ccf_ml_ap_dv_um.obj
        Vertices are Allen corner-origin CCF coordinates in microns, ordered [ML, AP, DV].
    <ACRONYM>_bregma_lps_mm.obj
        Vertices are already transformed into the bregma-relative LPS-style mm frame
        used by ccf_generation.py. Load directly into K3D with no /1000, sign flip,
        or bregma subtraction.

Install dependencies if needed:
    pip install allensdk scikit-image trimesh scipy k3d

Example:
    python make_pag_aq_dr_cs_separate_colored_meshes.py \
        --resolution-um 25 \
        --out-dir /root/capsule/data/ccf_meshes/PAG_AQ_DR_CS_separate \
        --html /root/capsule/scratch/combined/ccf_maps/PAG_AQ_DR_CS_colored.html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import trimesh
from scipy import ndimage as ndi
from skimage import measure

try:
    from allensdk.core.reference_space_cache import ReferenceSpaceCache
except ImportError as exc:
    raise SystemExit("Missing dependency: allensdk. Install with `pip install allensdk`.") from exc


DEFAULT_ACRONYMS = ("PAG", "AQ", "DR", "CS")
BREGMA_LPS_MM = np.array([-5.70, 5.40, -0.45], dtype=float)

# K3D colors as 0xRRGGBB integers.
DEFAULT_COLORS = {
    "PAG": 0x8C564B,  # brown
    "AQ":  0x1F77B4,  # blue
    "DR":  0xD62728,  # red
    "CS":  0x2CA02C,  # green
}


def get_structure_id_set(tree, acronym: str, include_descendants: bool = True) -> tuple[set[int], dict]:
    """Return Allen structure IDs for one acronym, optionally including descendants."""
    hits = tree.get_structures_by_acronym([acronym])
    if not hits:
        raise ValueError(f"Could not find acronym {acronym!r} in the Allen structure tree.")

    node = hits[0]
    sid = int(node["id"])
    ids_here = {sid}

    if include_descendants:
        try:
            ids_here.update(int(x) for x in tree.descendant_ids([sid])[0])
        except Exception:
            # Fallback: use structure_id_path membership.
            for n in tree.nodes():
                if sid in n.get("structure_id_path", []):
                    ids_here.add(int(n["id"]))

    return ids_here, node


def make_exclusive_id_sets(id_sets: dict[str, set[int]]) -> dict[str, set[int]]:
    """Assign overlapping annotation IDs to the most specific selected acronym.

    This prevents a child structure from being rendered twice if it is also a
    descendant of another selected structure. The most specific structure is
    approximated as the selected acronym with the smallest descendant ID set.
    """
    all_ids = sorted(set().union(*id_sets.values()))
    exclusive = {acronym: set() for acronym in id_sets}

    for sid in all_ids:
        owners = [acronym for acronym, ids in id_sets.items() if sid in ids]
        if not owners:
            continue
        winner = min(owners, key=lambda ac: len(id_sets[ac]))
        exclusive[winner].add(sid)

    return exclusive


def crop_mask(mask: np.ndarray, pad_voxels: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Crop a 3D mask to its bounding box plus padding.

    Returns cropped mask and crop origin in annotation axis order [AP, DV, ML].
    """
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("Mask is empty.")

    lo = np.maximum(coords.min(axis=0) - pad_voxels, 0)
    hi = np.minimum(coords.max(axis=0) + pad_voxels + 1, np.array(mask.shape))
    slices = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    return mask[slices], lo.astype(float)


def make_mesh_from_ids(
    annotation: np.ndarray,
    structure_ids: set[int],
    resolution_um: float,
    closing_iterations: int = 0,
    pad_voxels: int = 2,
) -> trimesh.Trimesh:
    """Create a mesh from selected Allen annotation IDs.

    AllenSDK annotation array axes are treated as [AP, DV, ML]. Marching cubes
    returns vertices in that same axis order. This function exports vertices in
    [ML, AP, DV] microns with Allen corner-origin signs.
    """
    if not structure_ids:
        raise ValueError("No structure IDs supplied.")

    mask = np.isin(annotation, list(structure_ids))
    if not np.any(mask):
        raise ValueError("The selected IDs produced an empty voxel mask.")

    if closing_iterations > 0:
        mask = ndi.binary_closing(mask, iterations=closing_iterations)

    mask_crop, origin_vox_ap_dv_ml = crop_mask(mask, pad_voxels=pad_voxels)

    verts_ap_dv_ml_um, faces, _normals, _values = measure.marching_cubes(
        mask_crop.astype(np.uint8),
        level=0.5,
        spacing=(resolution_um, resolution_um, resolution_um),
        allow_degenerate=False,
    )
    verts_ap_dv_ml_um += origin_vox_ap_dv_ml * resolution_um

    # Convert annotation/marching-cubes axis order [AP, DV, ML] to [ML, AP, DV].
    verts_ml_ap_dv_um = verts_ap_dv_ml_um[:, [2, 0, 1]]

    mesh = trimesh.Trimesh(vertices=verts_ml_ap_dv_um, faces=faces, process=True)
    mesh.remove_unreferenced_vertices()
    return mesh


def to_bregma_lps_mm(mesh_ccf_ml_ap_dv_um: trimesh.Trimesh) -> trimesh.Trimesh:
    """Convert [ML, AP, DV] corner-origin um to bregma-shifted LPS-style mm."""
    v = np.asarray(mesh_ccf_ml_ap_dv_um.vertices, dtype=float) / 1000.0
    v_lps_mm = v.copy()
    v_lps_mm[:, 0] *= -1.0
    v_lps_mm[:, 2] *= -1.0
    v_bregma_lps_mm = v_lps_mm - BREGMA_LPS_MM
    return trimesh.Trimesh(
        vertices=v_bregma_lps_mm,
        faces=mesh_ccf_ml_ap_dv_um.faces.copy(),
        process=False,
    )


def write_k3d_html(meshes_bregma: dict[str, trimesh.Trimesh], colors: dict[str, int], html_path: Path) -> None:
    """Write a standalone interactive K3D HTML viewer."""
    try:
        import k3d
    except ImportError as exc:
        raise SystemExit("Missing dependency: k3d. Install with `pip install k3d`.") from exc

    plot = k3d.plot(camera_auto_fit=True)

    for acronym, mesh in meshes_bregma.items():
        plot += k3d.mesh(
            vertices=np.asarray(mesh.vertices, dtype=np.float32),
            indices=np.asarray(mesh.faces, dtype=np.uint32),
            color=int(colors.get(acronym, 0x999999)),
            opacity=0.35,
            wireframe=False,
            name=acronym,
        )

    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(plot.get_snapshot())
    print(f"Wrote K3D HTML viewer: {html_path}")


def parse_color_arg(color_items: list[str] | None) -> dict[str, int]:
    """Parse --color ACRONYM=RRGGBB or ACRONYM=0xRRGGBB items."""
    colors = dict(DEFAULT_COLORS)
    if not color_items:
        return colors

    for item in color_items:
        if "=" not in item:
            raise ValueError(f"Bad color item {item!r}; expected ACRONYM=RRGGBB")
        acronym, value = item.split("=", 1)
        value = value.strip().lower().replace("#", "")
        if value.startswith("0x"):
            colors[acronym.strip()] = int(value, 16)
        else:
            colors[acronym.strip()] = int(value, 16)
    return colors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution-um", type=int, default=25, choices=(10, 25, 50, 100))
    parser.add_argument("--out-dir", type=Path, default=Path("/root/capsule/data/ccf_meshes/PAG_AQ_DR_CS_separate"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--reference-space-key", default="annotation/ccf_2017")
    parser.add_argument("--acronyms", nargs="+", default=list(DEFAULT_ACRONYMS))
    parser.add_argument("--no-descendants", action="store_true")
    parser.add_argument("--allow-overlap", action="store_true", help="Do not make selected structures mutually exclusive.")
    parser.add_argument("--closing-iterations", type=int, default=0)
    parser.add_argument("--html", type=Path, default=None, help="Optional standalone K3D HTML output path.")
    parser.add_argument("--color", action="append", default=None, help="Override color, e.g. --color PAG=AA3377")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.manifest or (args.out_dir / f"allen_ccf_{args.resolution_um}um_manifest.json")
    colors = parse_color_arg(args.color)

    rspc = ReferenceSpaceCache(args.resolution_um, args.reference_space_key, manifest=manifest)
    tree = rspc.get_structure_tree(structure_graph_id=1)
    annotation, _meta = rspc.get_annotation_volume()

    raw_id_sets: dict[str, set[int]] = {}
    matched_nodes: dict[str, dict] = {}

    print("Matched structures:")
    for acronym in args.acronyms:
        ids_here, node = get_structure_id_set(
            tree,
            acronym,
            include_descendants=not args.no_descendants,
        )
        raw_id_sets[acronym] = ids_here
        matched_nodes[acronym] = node
        print(f"  {acronym:>6s}  id={node['id']:>5}  name={node['name']}  ids={len(ids_here)}")

    if args.allow_overlap:
        id_sets = raw_id_sets
    else:
        id_sets = make_exclusive_id_sets(raw_id_sets)
        for acronym in args.acronyms:
            removed = len(raw_id_sets[acronym]) - len(id_sets[acronym])
            if removed > 0:
                print(f"  {acronym:>6s}: removed {removed} overlapping descendant IDs assigned to a more specific selected structure")

    meshes_bregma: dict[str, trimesh.Trimesh] = {}
    outputs = {}

    for acronym in args.acronyms:
        ids = id_sets[acronym]
        if not ids:
            print(f"Skipping {acronym}: no exclusive IDs left.")
            continue

        mesh_ccf = make_mesh_from_ids(
            annotation=annotation,
            structure_ids=ids,
            resolution_um=float(args.resolution_um),
            closing_iterations=args.closing_iterations,
        )
        mesh_bregma = to_bregma_lps_mm(mesh_ccf)
        meshes_bregma[acronym] = mesh_bregma

        out_ccf = args.out_dir / f"{acronym}_ccf_ml_ap_dv_um.obj"
        out_bregma = args.out_dir / f"{acronym}_bregma_lps_mm.obj"
        mesh_ccf.export(out_ccf)
        mesh_bregma.export(out_bregma)
        outputs[acronym] = {
            "ccf_ml_ap_dv_um": str(out_ccf),
            "bregma_lps_mm": str(out_bregma),
            "color": f"0x{colors.get(acronym, 0x999999):06X}",
            "n_ids_used": len(ids),
            "n_vertices": int(len(mesh_bregma.vertices)),
            "n_faces": int(len(mesh_bregma.faces)),
        }
        print(f"Wrote {acronym}: {out_bregma}")

    metadata = {
        "acronyms": args.acronyms,
        "matched_structures": {
            ac: {"id": int(n["id"]), "name": n["name"], "acronym": n["acronym"]}
            for ac, n in matched_nodes.items()
        },
        "include_descendants": not args.no_descendants,
        "exclusive_selected_structures": not args.allow_overlap,
        "resolution_um": args.resolution_um,
        "reference_space_key": args.reference_space_key,
        "bregma_lps_mm": BREGMA_LPS_MM.tolist(),
        "outputs": outputs,
        "notes": [
            "Each structure is exported as a separate mesh so K3D can color it independently.",
            "bregma_lps_mm meshes can be loaded directly in the plotting coordinate frame used by ccf_generation.py.",
            "ccf_ml_ap_dv_um meshes use Allen corner-origin microns ordered [ML, AP, DV].",
        ],
    }
    out_meta = args.out_dir / "PAG_AQ_DR_CS_separate_metadata.json"
    out_meta.write_text(json.dumps(metadata, indent=2))
    print(f"Wrote metadata: {out_meta}")

    if args.html is not None:
        write_k3d_html(meshes_bregma=meshes_bregma, colors=colors, html_path=args.html)


if __name__ == "__main__":
    main()

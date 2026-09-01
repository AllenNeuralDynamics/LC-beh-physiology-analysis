import os
from pathlib import Path


def capsule_root():
    """Resolve repo root in a portable way.

    Resolution order:
      1. ``$CAPSULE_ROOT`` environment variable, if set.
      2. ``/root/capsule`` if it exists (the real Code Ocean mount), so
         behaviour on the capsule is unchanged.
      3. The repository root derived from this file's location
         (``<root>/code/beh_ephys_analysis/utils/capsule_migration.py``),
         so checkouts on any machine resolve ``code/...`` paths correctly.
    """
    env = os.environ.get("CAPSULE_ROOT")
    if env:
        return Path(env)
    if os.path.isdir("/root/capsule"):
        return Path("/root/capsule")
    return Path(__file__).resolve().parents[3]


# String form for cheap concatenation in path literals across the repo, e.g.
#   CAPSULE_ROOT + '/code/data_management/session_assets.csv'
CAPSULE_ROOT = str(capsule_root())


def capsule_directories():
    """
    Get standard capsule directory paths and ensure they exist.

    Creates a dictionary of commonly used paths within the capsule
    and creates any missing directories.

    Returns
    -------
    dict
        Dictionary with keys:
        - output_dir: Main output directory
        - manuscript_fig_dir: Manuscript figures directory
        - manuscript_fig_prep_dir: Manuscript prep directory
        - derived_dir: Derived data directory
        - data_dir: Raw data directory
    """
    root = capsule_root()
    output_dir = root / 'scratch' / 'results'
    dirs = {
        'output_dir': output_dir,
        'manuscript_fig_dir': output_dir / 'manuscript' / 'figures',
        'manuscript_fig_prep_dir': output_dir / 'manuscript' / 'prep',
        'derived_dir': root / 'data' / 'scratch_data',
        'data_dir': root / 'data',
    }
    for dir in dirs.values():
        if not dir.exists():
            dir.mkdir(parents=True, exist_ok=True)
    return dirs

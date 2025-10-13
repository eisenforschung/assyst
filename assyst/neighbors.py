"""Utility module to pick where to load neighbour_list from."""

from pyiron_snippets.import_alarm import ImportAlarm

with ImportAlarm(
    "matscipy not installed falling back to ASE implementation; "
    "install with 'conda install -c conda-forge matscipy' or "
    "'pip install matscipy'",
) as neighbor_alarm:
    try:
        from matscipy.neighbours import neighbour_list as neighbor_list
    except ImportError:
        from ase.neighborlist import neighbor_list

        raise

__all__ = ["neighbor_list"]

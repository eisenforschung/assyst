import uuid
from ase import Atoms

STAGE_KEY = "stage"
"""Key in :attr:`ase.Atoms.info` under which the workflow steps a structure went through are recorded."""


def record_stage(structure: Atoms, step: str) -> Atoms:
    """Append the name of a workflow step to the stage history of a structure.

    Steps are joined with ``+`` in the order they were applied, so a structure that was generated, volume relaxed
    and then rattled carries ``spg+volume_relax+rattle(0.05)``.

    Operates INPLACE.

    Args:
        structure (:class:`ase.Atoms`): the structure to tag
        step (:class:`str`): name of the step that just ran

    Returns:
        :class:`ase.Atoms`: the tagged structure
    """
    previous = structure.info.get(STAGE_KEY, "")
    structure.info[STAGE_KEY] = f"{previous}+{step}" if previous else step
    return structure


def stage_of(structure: Atoms, default: str = "unknown") -> str:
    """The workflow steps a structure went through, as recorded by :func:`.record_stage`.

    Args:
        structure (:class:`ase.Atoms`): the structure to inspect
        default (:class:`str`): returned for structures that carry no stage, i.e. those ASSYST did not make

    Returns:
        :class:`str`: the steps joined with ``+``, e.g. ``spg+volume_relax+full_relax``
    """
    return str(structure.info.get(STAGE_KEY, default))


def update_uuid(structure: Atoms) -> Atoms:
    """Updates the UUID of the structure and maintains a lineage.

    If the structure already has a UUID, it is appended to the 'lineage' list.
    A new UUID is then generated and stored in the 'uuid' key of the `info` dictionary.

    Args:
        structure (ase.Atoms): The structure to update.

    Returns:
        ase.Atoms: The updated structure.
    """
    if 'uuid' in structure.info:
        # Create a new list for lineage to avoid sharing it with parent structures
        lineage = list(structure.info.get('lineage', []))
        lineage.append(structure.info['uuid'])
        structure.info['lineage'] = lineage

    new_uuid = str(uuid.uuid4())
    structure.info['uuid'] = new_uuid

    if 'seed' not in structure.info:
        structure.info['seed'] = new_uuid

    return structure

import uuid
from ase import Atoms

STEP_KEY = "step"
"""Key in :attr:`ase.Atoms.info` under which the workflow step that produced a structure is recorded."""


def step_of(structure: Atoms, default: str = "unknown") -> str:
    """The workflow step that produced a structure, as recorded by :func:`.update_uuid`.

    Only the most recent step is kept, mirroring ``uuid``; the structures it came from are in ``lineage``.

    Args:
        structure (:class:`ase.Atoms`): the structure to inspect
        default (:class:`str`): returned for structures that carry no step, i.e. those ASSYST did not make

    Returns:
        :class:`str`: the name of the step, e.g. ``volume_relax``
    """
    return str(structure.info.get(STEP_KEY, default))


def update_uuid(structure: Atoms, step: str | None = None) -> Atoms:
    """Updates the UUID of the structure and maintains a lineage.

    If the structure already has a UUID, it is appended to the 'lineage' list.
    A new UUID is then generated and stored in the 'uuid' key of the `info` dictionary.

    Args:
        structure (ase.Atoms): The structure to update.
        step (str, optional): Name of the step that produced this new UUID, e.g. a relaxation or a
            perturbation. Stored in the 'step' key of `info` when given, replacing any previous value;
            read it back with :func:`.step_of`.

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

    if step is not None:
        structure.info[STEP_KEY] = step

    return structure

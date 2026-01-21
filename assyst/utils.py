import uuid
from ase import Atoms

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
        if 'lineage' not in structure.info:
            structure.info['lineage'] = []
        structure.info['lineage'].append(structure.info['uuid'])

    structure.info['uuid'] = str(uuid.uuid4())
    return structure

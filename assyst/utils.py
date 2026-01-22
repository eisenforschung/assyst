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
        # Create a new list for lineage to avoid sharing it with parent structures
        lineage = list(structure.info.get('lineage', []))
        lineage.append(structure.info['uuid'])
        structure.info['lineage'] = lineage

    new_uuid = str(uuid.uuid4())
    structure.info['uuid'] = new_uuid

    if 'seed' not in structure.info:
        structure.info['seed'] = new_uuid

    return structure

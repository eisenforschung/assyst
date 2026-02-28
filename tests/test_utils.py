import uuid
import pytest
from ase import Atoms
from assyst.utils import update_uuid

@pytest.fixture
def atoms():
    return Atoms('H2', positions=[[0, 0, 0], [0.74, 0, 0]])

def test_update_uuid_no_existing_uuid(atoms):
    """Test update_uuid when structure has no UUID."""
    # Ensure clean state
    if 'uuid' in atoms.info:
        del atoms.info['uuid']
    if 'lineage' in atoms.info:
        del atoms.info['lineage']
    if 'seed' in atoms.info:
        del atoms.info['seed']

    updated_atoms = update_uuid(atoms)

    assert 'uuid' in updated_atoms.info
    assert isinstance(updated_atoms.info['uuid'], str)
    # Check that it's a valid UUID
    uuid.UUID(updated_atoms.info['uuid'])

    # Should set seed if not present
    assert 'seed' in updated_atoms.info
    assert updated_atoms.info['seed'] == updated_atoms.info['uuid']

    # Lineage should not be created if it didn't exist and no previous UUID
    assert 'lineage' not in updated_atoms.info

def test_update_uuid_existing_uuid(atoms):
    """Test update_uuid when structure has an existing UUID."""
    old_uuid = str(uuid.uuid4())
    atoms.info['uuid'] = old_uuid

    updated_atoms = update_uuid(atoms)

    new_uuid = updated_atoms.info['uuid']
    assert old_uuid != new_uuid

    assert 'lineage' in updated_atoms.info
    assert old_uuid in updated_atoms.info['lineage']
    assert updated_atoms.info['lineage'][-1] == old_uuid

def test_update_uuid_existing_lineage(atoms):
    """Test update_uuid appends to existing lineage."""
    uuid1 = str(uuid.uuid4())
    uuid2 = str(uuid.uuid4())
    atoms.info['lineage'] = [uuid1]
    atoms.info['uuid'] = uuid2

    updated_atoms = update_uuid(atoms)

    assert len(updated_atoms.info['lineage']) == 2
    assert updated_atoms.info['lineage'] == [uuid1, uuid2]
    assert updated_atoms.info['uuid'] != uuid2

def test_update_uuid_preserves_seed(atoms):
    """Test that update_uuid preserves existing seed."""
    original_seed = "original_seed"
    atoms.info['seed'] = original_seed
    # It needs a uuid to trigger lineage update logic if any, but seed check is independent

    atoms.info['uuid'] = str(uuid.uuid4())
    updated_atoms = update_uuid(atoms)

    assert updated_atoms.info['seed'] == original_seed

def test_update_uuid_lineage_independence(atoms):
    """Test that lineage list is not shared between parent and child."""
    original_lineage = ["ancestor"]
    s3 = atoms.copy()
    s3.info['uuid'] = "uuid3"
    s3.info['lineage'] = original_lineage # s3.info['lineage'] is reference to original_lineage

    updated_s3 = update_uuid(s3)

    # updated_s3.info['lineage'] should have appended uuid3
    assert "uuid3" in updated_s3.info['lineage']

    # original_lineage should NOT have "uuid3"
    assert "uuid3" not in original_lineage
    assert id(updated_s3.info['lineage']) != id(original_lineage)

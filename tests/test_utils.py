import unittest
import uuid
from ase import Atoms
from assyst.utils import update_uuid

class TestUtils(unittest.TestCase):
    def setUp(self):
        self.atoms = Atoms('H2', positions=[[0, 0, 0], [0.74, 0, 0]])

    def test_update_uuid_no_existing_uuid(self):
        """Test update_uuid when structure has no UUID."""
        # Ensure clean state
        if 'uuid' in self.atoms.info:
            del self.atoms.info['uuid']
        if 'lineage' in self.atoms.info:
            del self.atoms.info['lineage']
        if 'seed' in self.atoms.info:
            del self.atoms.info['seed']

        updated_atoms = update_uuid(self.atoms)

        self.assertIn('uuid', updated_atoms.info)
        self.assertIsInstance(updated_atoms.info['uuid'], str)
        # Check that it's a valid UUID
        uuid.UUID(updated_atoms.info['uuid'])

        # Should set seed if not present
        self.assertIn('seed', updated_atoms.info)
        self.assertEqual(updated_atoms.info['seed'], updated_atoms.info['uuid'])

        # Lineage should not be created if it didn't exist and no previous UUID
        self.assertNotIn('lineage', updated_atoms.info)

    def test_update_uuid_existing_uuid(self):
        """Test update_uuid when structure has an existing UUID."""
        old_uuid = str(uuid.uuid4())
        self.atoms.info['uuid'] = old_uuid

        updated_atoms = update_uuid(self.atoms)

        new_uuid = updated_atoms.info['uuid']
        self.assertNotEqual(old_uuid, new_uuid)

        self.assertIn('lineage', updated_atoms.info)
        self.assertIn(old_uuid, updated_atoms.info['lineage'])
        self.assertEqual(updated_atoms.info['lineage'][-1], old_uuid)

    def test_update_uuid_existing_lineage(self):
        """Test update_uuid appends to existing lineage."""
        uuid1 = str(uuid.uuid4())
        uuid2 = str(uuid.uuid4())
        self.atoms.info['lineage'] = [uuid1]
        self.atoms.info['uuid'] = uuid2

        updated_atoms = update_uuid(self.atoms)

        self.assertEqual(len(updated_atoms.info['lineage']), 2)
        self.assertEqual(updated_atoms.info['lineage'], [uuid1, uuid2])
        self.assertNotEqual(updated_atoms.info['uuid'], uuid2)

    def test_update_uuid_preserves_seed(self):
        """Test that update_uuid preserves existing seed."""
        original_seed = "original_seed"
        self.atoms.info['seed'] = original_seed
        # It needs a uuid to trigger lineage update logic if any, but seed check is independent

        self.atoms.info['uuid'] = str(uuid.uuid4())
        updated_atoms = update_uuid(self.atoms)

        self.assertEqual(updated_atoms.info['seed'], original_seed)

    def test_update_uuid_lineage_independence(self):
        """Test that lineage list is not shared between parent and child."""
        original_lineage = ["ancestor"]
        s3 = self.atoms.copy()
        s3.info['uuid'] = "uuid3"
        s3.info['lineage'] = original_lineage # s3.info['lineage'] is reference to original_lineage

        updated_s3 = update_uuid(s3)

        # updated_s3.info['lineage'] should have appended uuid3
        self.assertIn("uuid3", updated_s3.info['lineage'])

        # original_lineage should NOT have "uuid3"
        self.assertNotIn("uuid3", original_lineage)
        self.assertNotEqual(id(updated_s3.info['lineage']), id(original_lineage))

if __name__ == '__main__':
    unittest.main()

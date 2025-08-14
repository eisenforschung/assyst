import unittest
from ase import Atoms
from assyst.filters import VolumeFilter

class TestVolumeFilter(unittest.TestCase):
    def test_volume_filter(self):
        filter = VolumeFilter(maximum_volume_per_atom=20.0)

        # Volume per atom = 8.0
        structure1 = Atoms('Cu', cell=[2, 2, 2], pbc=True)
        self.assertTrue(filter(structure1))

        # Volume per atom = 20.0
        structure2 = Atoms('Cu2', cell=[4, 5, 2], pbc=True)
        self.assertTrue(filter(structure2))

        # Volume per atom = 20.0001
        structure3 = Atoms('Cu', cell=[2, 2, 5.0001], pbc=True)
        self.assertFalse(filter(structure3))

import unittest
from ase import Atoms
from assyst.filters import AspectFilter

class TestAspectFilter(unittest.TestCase):
    def test_aspect_filter(self):
        filter = AspectFilter(maximum_aspect_ratio=2.0)

        # Aspect ratio = 1.0
        structure1 = Atoms('Cu', cell=[2, 2, 2], pbc=True)
        self.assertTrue(filter(structure1))

        # Aspect ratio = 2.0
        structure2 = Atoms('Cu', cell=[2, 2, 4], pbc=True)
        self.assertTrue(filter(structure2))

        # Aspect ratio = 2.5
        structure3 = Atoms('Cu', cell=[2, 2, 5], pbc=True)
        self.assertFalse(filter(structure3))

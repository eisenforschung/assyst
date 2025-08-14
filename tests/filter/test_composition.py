import unittest
from ase import Atoms
from assyst.filters import AndFilter, OrFilter, FilterBase

class MockFilter(FilterBase):
    def __init__(self, value):
        self.value = value
    def __call__(self, structure: Atoms) -> bool:
        return self.value

class TestComposition(unittest.TestCase):
    def test_and_filter(self):
        structure = Atoms()
        # True and True
        self.assertTrue(AndFilter(MockFilter(True), MockFilter(True))(structure))
        # True and False
        self.assertFalse(AndFilter(MockFilter(True), MockFilter(False))(structure))
        # False and True
        self.assertFalse(AndFilter(MockFilter(False), MockFilter(True))(structure))
        # False and False
        self.assertFalse(AndFilter(MockFilter(False), MockFilter(False))(structure))

    def test_or_filter(self):
        structure = Atoms()
        self.assertTrue(OrFilter(MockFilter(True), MockFilter(True))(structure))
        self.assertTrue(OrFilter(MockFilter(True), MockFilter(False))(structure))
        self.assertTrue(OrFilter(MockFilter(False), MockFilter(True))(structure))
        self.assertFalse(OrFilter(MockFilter(False), MockFilter(False))(structure))

    def test_and_operator(self):
        structure = Atoms()
        and_filter = MockFilter(True) & MockFilter(False)
        self.assertIsInstance(and_filter, AndFilter)
        self.assertFalse(and_filter(structure))

    def test_or_operator(self):
        structure = Atoms()
        or_filter = MockFilter(True) | MockFilter(False)
        self.assertIsInstance(or_filter, OrFilter)
        self.assertTrue(or_filter(structure))

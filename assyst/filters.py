'''Classes that filter structures according to some criteria.

The code in the other modules that uses them is set up such that simple
functions can always be passed as well and that the classes here are just for
convenience.'''

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations_with_replacement
from math import nan, inf

from ase import Atoms
from structuretoolkit import get_neighbors

@dataclass
class DistanceFilter:
    '''Filter structures that contain too close atoms.

    Setting a radius to NaN allows all bonds involving this atom.'''
    radii: dict[str, float]

    @staticmethod
    def _element_wise_dist(structure: Atoms):
        pair = defaultdict(lambda: inf)
        # on weird aspect ratios the neighbor searching code can allocate huge structures,
        # because it explicitly repeats the structure to create ghost atoms
        # since we only care about the presence of short distances between atoms and not the
        # real neighbor information, simply double the structure to make sure we see all bonds
        # and turn off PBC
        # this can miss neighbors in structures with highly acute lattice angles, but we'll live
        sr = structure.repeat(2)
        sr.pbc = [False, False, False]
        n = get_neighbors(sr, num_neighbors=len(structure), mode="ragged")
        for i, (I, D) in enumerate(zip(n.indices, n.distances)):
            for j, d in zip(I, D):
                ei, ej = sorted((sr.symbols[i], sr.symbols[j]))
                pair[ei, ej] = min(d, pair[ei, ej])
        return pair

    def __call__(self, structure: Atoms) -> bool:
        '''
        Return True if structure satifies minimum distance criteria.

        Args:
            structure (Atoms): structure to check

        Returns:
            `False`: at least on bond is shorter than the sum of given cutoff radii of the respective elements
            `True`: all bonds are than the sum of given cutoff radii of the respective elements
        '''
        pair = self._element_wise_dist(structure)
        for ei, ej in combinations_with_replacement(structure.symbols.species(), 2):
            ei, ej = sorted((ei, ej))
            if pair[ei, ej] < self.radii.get(ei, nan) + self.radii.get(ej, nan):
                return False
        return True

@dataclass
class AspectFilter:
    '''Filters structures with high aspect ratios.'''
    maximum_aspect_ratio: float = 6

    def __call__(self, structure: Atoms) -> bool:
        '''Return True if structure's cell has an agreeable aspect ratio.

        Args:
            structure (Atoms): structure to check

        Returns:
            `True`: lattice's aspect ratio is below or equal `:attr:`.maximum_aspect_ratio`.
            `False`: lattice's aspect ratio is above `:attr:`.maximum_aspect_ratio`.'''
        a, b, c = sorted(structure.cell.lengths())
        return c / a <= self.maximum_aspect_ratio

@dataclass
class VolumeFilter:
    '''Filters structures by volume.'''
    maximum_volume_per_atom: float

    def __call__(self, structure: Atoms) -> bool:
        '''Return True if structure's volume is within range.

        Args:
            structure (Atoms): structure to check

        Returns:
            `True`: volume per atom is smaller or equal than `:attr:.maximum_volume_per_atom`.
            `False`: otherwise'''
        return structure.cell.volume / len(structure) <= self.maximum_volume_per_atom

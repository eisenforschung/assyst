from dataclasses import dataclass
from itertools import combinations_with_replacement
from math import nan

from ase import Atoms
from structuretoolkit import get_neighbors

@dataclass
class DistanceFilter:
    """Filter structures that contain too close atoms."""
    radii: dict[str, float]

    @staticmethod
    def _element_wise_dist(structure: Atoms):
        pair = defaultdict(lambda: np.inf)
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
        """
        Return True if structure satifies minimum distance criteria.
        """
        pair = self._element_wise_dist(structure)
        for ei, ej in combinations_with_replacement(structure.symbols.species(), 2):
            ei, ej = sorted((ei, ej))
            if pair[ei, ej] < self.radii.get(ei, nan) + self.radii.get(ej, nan):
                return False
        return True

from dataclasses import dataclass
from collections.abc import Sequence
from itertools import product
from warnings import catch_warnings
from typing import Self

from ase import Atoms
from structuretoolkit.build.random import pyxtal
from tqdm.auto import tqdm
import math

@dataclass(frozen=True)
class Elements(Sequence):
    atoms: tuple[dict[str, int]]

    @property
    def elements(self) -> set[str]:
        """Set of elements present in elements."""
        e = set()
        for s in self.atoms:
            e = e.union(s.keys())
        return e

    def __add__(self, other: Self) -> Self:
        """Extend underlying list of stoichiometries."""
        return Elements(self.atoms + other.atoms)

    def __or__(self, other: Self) -> Self:
        """Inner product of underlying stoichiometries.

        Must not share elements with other.elements."""
        assert self.elements.isdisjoint(other.elements), "Can only or stoichiometries of different elements!"
        s = ()
        for me, you in zip(self.atoms, other.atoms):
            s += (me | you,)
        return Elements(s)

    def __mul__(self, other: Self) -> Self:
        """Outer product of underlying stoichiometries.

        Must not share elements with other.elements."""
        assert self.elements.isdisjoint(other.elements), "Can only multiply stoichiometries of different elements!"
        s = ()
        for me, you in product(self.atoms, other.atoms):
            s += (me | you,)
        return Elements(s)

    # Sequence Impl'
    def __getitem__(self, index: int) -> dict[str, int]:
        return self.atoms[index]

    def __len__(self) -> int:
        return len(self.atoms)

def sample_space_groups(
        elements: Elements,
        spacegroups: list[int] | tuple[int,...] | None = None,
        max_atoms: int = 10,
        max_structures: int | None = None,
) -> list[Atoms]:
    """
    Create symmetric random structures.

    Args:
        elements (Elements): list of compositions per structure
        spacegroups (list of int): which space groups to generate
        max_atoms (int): do not generate structures larger than this
        max_structures (int): generate at most this many structures
    Returns:
        list of Atoms: generated structures
    """

    if spacegroups is None:
        spacegroups = list(range(1,231))
    if max_structures is None:
        max_structures = math.inf

    structures = []
    with catch_warnings(category=UserWarning, action='ignore'):
        for stoich in (bar := tqdm(elements)):
            elements, num_atomss = zip(*stoich.items())
            stoich_str = "".join(f"{s}{n}" for s, n in zip(elements, num_atomss))
            bar.set_description(stoich_str)
            structures += [s['atoms'] for s in pyxtal(spacegroups, elements, num_atomss)]
            if len(structures) > max_structures:
                structures = structures[:max_structures]
                break
    return structures

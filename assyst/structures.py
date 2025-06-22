'''Crystal structure generation step of ASSYST.'''

from dataclasses import dataclass
from collections.abc import Sequence
from itertools import product
from warnings import catch_warnings
from typing import Self, Iterable, Iterator

from ase import Atoms
from structuretoolkit.build.random import pyxtal
from tqdm.auto import tqdm
import math

@dataclass(eq=True, frozen=True)
class Formulas(Sequence):
    '''Simple helper to generate lists of structure compositions.

    :func:`.sample_space_groups` is the intended consumer and expects an iterable of dictionaries, where each dictionary
    maps an element name to the number of atoms of this type in one structure.
    :class:`.Formulas` behaves as if it were such a tuple, but extends the basic python arithmetic operations to make
    building the list a bit simpler.

    The class can be initialized from any tuple of dictionaries.

    >>> el_manual = Formulas(({'Cu': 1}, {'Cu': 2}))

    :meth:`.unary_range` is a helper class method that initializes `Formulas` for a single element and takes the same
    arguments as the builtin `range`, except that it skips the zero.

    >>> el = Formulas.unary_range('Cu', 3)
    Formulas(atoms=({'Cu': 1}, {'Cu': 2}))
    >>> el == el_manual
    True

    Addition is overloaded to the addition of the underlying tuples.

    >>> Formulas.unary_range('Cu', 1, 5) == Formulas.unary_range('Cu', 1, 3) + Formulas.unary_range('Cu', 3, 5)

    The bitwise or operation is akin to the inner product

    >>> Formulas.unary_range('Cu', 3) | Formulas.unary_range('Ag', 3)
    Formulas(atoms=({'Cu': 1, 'Ag': 1}, {'Cu': 2, 'Ag': 2}))

    >>> Formulas.unary_range('Cu', 3) * Formulas.unary_range('Ag', 3)
    Formulas(atoms=({'Cu': 1, 'Ag': 1}, {'Cu': 2, 'Ag': 1}, {'Cu': 1, 'Ag': 2}, {'Cu': 2, 'Ag': 2}))
    '''
    atoms: tuple[dict[str, int], ...]

    @property
    def elements(self) -> set[str]:
        '''Set of elements present in elements.'''
        e = set()
        for s in self.atoms:
            e = e.union(s.keys())
        return e

    @classmethod
    def unary_range(cls, element: str, *range_args) -> Self:
        '''Creates '''
        return cls(tuple({element: i} for i in range(*range_args) if i > 0))

    def __add__(self, other: Self) -> Self:
        '''Extend underlying list of stoichiometries.'''
        return Formulas(self.atoms + other.atoms)

    def __or__(self, other: Self) -> Self:
        '''Inner product of underlying stoichiometries.

        Truncates to the length of the shortest of the two element sequences.
        Must not share elements with other.elements.'''
        assert self.elements.isdisjoint(other.elements), "Can only or stoichiometries of different elements!"
        s = ()
        for me, you in zip(self.atoms, other.atoms):
            s += (me | you,)
        return Formulas(s)

    def __mul__(self, other: Self) -> Self:
        '''Outer product of underlying stoichiometries.

        Must not share elements with other.elements.'''
        assert self.elements.isdisjoint(other.elements), "Can only multiply stoichiometries of different elements!"
        s = ()
        for me, you in product(self.atoms, other.atoms):
            s += (me | you,)
        return Formulas(s)

    # Sequence Impl'
    def __getitem__(self, index: int) -> dict[str, int]:
        return self.atoms[index]

    def __len__(self) -> int:
        return len(self.atoms)

def sample_space_groups(
        elements: Formulas | Iterable[dict[str, int]],
        spacegroups: list[int] | tuple[int,...] | None = None,
        max_atoms: int = 10,
        max_structures: int | None = None,
) -> Iterator[Atoms]:
    '''
    Create symmetric random structures.

    Args:
        elements (Formulas or iterable of dicts from str to int): list of chemical formulas
        spacegroups (list of int): which space groups to generate
        max_atoms (int): do not generate structures larger than this
        max_structures (int): generate at most this many structures

    Yields:
        `Atoms`: random symmetric crystal structures
    '''

    if spacegroups is None:
        spacegroups = list(range(1,231))
    if max_structures is None:
        max_structures = math.inf

    yielded = 0
    with catch_warnings(category=UserWarning, action='ignore'):
        for stoich in (bar := tqdm(elements)):
            elements, num_atoms = zip(*stoich.items())
            stoich_str = "".join(f"{s}{n}" for s, n in zip(elements, num_atoms))
            bar.set_description(stoich_str)
            for s in pyxtal(spacegroups, elements, num_atoms):
                yield s['atoms']
                yielded += 1
                if yielded >= max_structures:
                    return

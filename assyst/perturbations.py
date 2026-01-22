"""Classes to apply (random) perturbations to structures."""

from abc import ABC, abstractmethod
from ase import Atoms
from typing import Iterable, Callable, Self, Iterator, Union
from dataclasses import dataclass
import numpy as np

from .filters import Filter


def rattle(
    structure: Atoms, sigma: float, rng: Union[int, np.random.Generator, None] = None
) -> Atoms:
    """Randomly displace positions with gaussian noise.

    Operates INPLACE."""
    if len(structure) == 1:
        raise ValueError("Can only rattle structures larger than one atom.")

    if isinstance(rng, int):
        structure.rattle(stdev=sigma, seed=rng)
    else:
        if rng is None:
            rng = np.random
        structure.rattle(stdev=sigma, rng=rng)
    return structure


def element_scaled_rattle(
    structure: Atoms,
    sigma: float,
    reference: dict[str, float],
    rng: Union[int, np.random.Generator, None] = None,
) -> Atoms:
    """Randomly displace positions with gaussian noise relative to an elemental reference length.

    Operates like :func:`.rattle` but uses a standard deviation derived from the relative `sigma` and the `reference`,
    where this reference is given by element.

    Operates IN PLACE!

    Args:
        structure (:class:`ase.Atoms`): structure to perturb
        sigma (:class:`float`): relative standard deviation
        reference (:class:`dict` of :class:`str` to :class:`float`): reference length per element
        rng (int, numpy.random.Generator): seed or random number generator

    Raises:
        ValueError: if len(structure) == 1, create a super cell first before calling again
        ValueError: if reference values are not positive
        ValueError: if reference does not contain all elements in given structure
    """
    sigma = sigma * np.ones(len(structure))
    if not all(r > 0 for r in reference.values()):
        raise ValueError("Reference lengths must be strictly positive!")
    for i, sym in enumerate(structure.symbols):
        try:
            sigma[i] *= reference[sym]
        except KeyError:
            raise ValueError(f"No value for element {sym} provided in argument `reference`!") from None
    return rattle(structure, sigma.reshape(-1, 1), rng=rng)


def stretch(
    structure: Atoms,
    hydro: float,
    shear: float,
    minimum_strain=1e-3,
    rng: Union[int, np.random.Generator, None] = None,
) -> Atoms:
    """Randomly stretch cell with uniform noise.

    Ensures at least `minimum_strain` strain to avoid structures very close to their original structures.
    These don't offer a lot of new information and can also confuse VASP's symmetry analyzer.

    Operates INPLACE."""
    _rng = np.random.default_rng(rng)

    def get_strains(max_strain, size):
        signs = _rng.choice([-1, 1], size=size)
        magnitudes = _rng.uniform(minimum_strain, max_strain, size=size)
        return signs * magnitudes

    strain = np.zeros((3, 3))
    # Off-diagonal elements
    indices = np.triu_indices(3, k=1)
    strain[indices] = get_strains(shear, 3)
    strain += strain.T

    # Diagonal elements
    np.fill_diagonal(strain, 1 + get_strains(hydro, 3))

    structure.set_cell(structure.cell.array @ strain, scale_atoms=True)
    return structure


class PerturbationABC(ABC):
    """Apply some perturbation to a given structure."""

    def __call__(self, structure: Atoms) -> Atoms:
        if "perturbation" not in structure.info:
            structure.info["perturbation"] = str(self)
        else:
            structure.info["perturbation"] += "+" + str(self)
        return structure

    @abstractmethod
    def __str__(self) -> str:
        pass

    def __add__(self, other: Self) -> "Series":
        return Series((self, other))


Perturbation = Callable[[Atoms], Atoms] | PerturbationABC


def apply_perturbations(
    structures: Iterable[Atoms],
    perturbations: Iterable[Perturbation],
    filters: Iterable[Filter] | Filter | None = None,
    retries: int = 10,
) -> Iterator[Atoms]:
    """Apply a list of perturbations to each structure and yield the result of each perturbation separately.

    If a perturbation raises ValueError it is ignored.

    Args:
        structures: :class:`collections.abc.Iterable` of :class:`ase.Atoms` to perturb.
        perturbations: :class:`collections.abc.Iterable` of :class:`~.Perturbation` that modify structures.
        filters: :class:`collections.abc.Iterable` of :class:`~assyst.filters.Filter` to filter valid results (optional).
        retries: :class:`int`, max attempts per perturbation (default: 10).

    Yields:
        :class:`ase.Atoms`: perturbed structure that passes all filters.
    """
    if filters is None:
        filters = []
    if not isinstance(filters, Iterable):
        filters = [filters]
    perturbations = list(perturbations)

    for structure in structures:
        for mod in perturbations:
            try:
                for _ in range(retries):
                    m = mod(structure.copy())
                    if all(f(m) for f in filters):
                        yield m
                        break
            except ValueError:
                continue


@dataclass(frozen=True)
class Rattle(PerturbationABC):
    """Displace atoms by some absolute amount from a normal distribution."""

    sigma: float
    create_supercells: bool = False
    "Create minimal 2x2x2 super cells when applied to structures of only one atom."
    rng: Union[int, np.random.Generator, None] = None

    def __post_init__(self):
        object.__setattr__(self, "rng", np.random.default_rng(self.rng))

    def __getstate__(self):
        state = self.__dict__.copy()
        state["rng"] = self.rng.bit_generator.state
        return state

    def __setstate__(self, state):
        for key, value in state.items():
            if key == "rng":
                rng = np.random.default_rng()
                rng.bit_generator.state = value
                value = rng
            object.__setattr__(self, key, value)

    def __call__(self, structure: Atoms):
        if self.create_supercells and len(structure) == 1:
            structure = structure.repeat(2)
        structure = super().__call__(structure)
        return rattle(structure, self.sigma, rng=self.rng)

    def __str__(self):
        return f"rattle({self.sigma})"


@dataclass(frozen=True)
class ElementScaledRattle(PerturbationABC):
    """Displace atoms by some amount from a normal distribution.

    Operates like :class:`.Rattle` but uses a standard deviation derived from the relative `sigma` and the `reference`,
    where this reference is given by element.
    """

    sigma: float
    reference: dict[str, float]
    create_supercells: bool = False
    "Create minimal 2x2x2 super cells when applied to structures of only one atom."
    rng: Union[int, np.random.Generator, None] = None

    def __post_init__(self):
        object.__setattr__(self, "rng", np.random.default_rng(self.rng))

    def __getstate__(self):
        state = self.__dict__.copy()
        state["rng"] = self.rng.bit_generator.state
        return state

    def __setstate__(self, state):
        for key, value in state.items():
            if key == "rng":
                rng = np.random.default_rng()
                rng.bit_generator.state = value
                value = rng
            object.__setattr__(self, key, value)

    def __call__(self, structure: Atoms):
        if self.create_supercells and len(structure) == 1:
            structure = structure.repeat(2)
        structure = super().__call__(structure)
        return element_scaled_rattle(structure, self.sigma, self.reference, rng=self.rng)

    def __str__(self):
        return f"scaled_rattle({self.sigma})"


@dataclass(frozen=True)
class Stretch(PerturbationABC):
    """Apply random cell perturbation."""

    hydro: float
    shear: float
    minimum_strain: float = 1e-3
    rng: Union[int, np.random.Generator, None] = None

    def __post_init__(self):
        object.__setattr__(self, "rng", np.random.default_rng(self.rng))

    def __getstate__(self):
        state = self.__dict__.copy()
        state["rng"] = self.rng.bit_generator.state
        return state

    def __setstate__(self, state):
        for key, value in state.items():
            if key == "rng":
                rng = np.random.default_rng()
                rng.bit_generator.state = value
                value = rng
            object.__setattr__(self, key, value)

    def __call__(self, structure: Atoms):
        structure = super().__call__(structure)
        return stretch(structure, self.hydro, self.shear, self.minimum_strain, rng=self.rng)

    def __str__(self):
        return f"stretch(hydro={self.hydro}, shear={self.shear})"


@dataclass(frozen=True)
class Series(PerturbationABC):
    """Apply some perturbations in sequence."""

    perturbations: tuple[Perturbation, ...]

    def __call__(self, structure: Atoms) -> Atoms:
        for mod in self.perturbations:
            structure = mod(structure)
        return structure

    def __str__(self):
        return "+".join(str(mod) for mod in self.perturbations)


@dataclass(frozen=True)
class RandomChoice(PerturbationABC):
    """Apply either of two alternatives randomly."""

    choice_a: Perturbation
    choice_b: Perturbation
    chance: float
    "Probability to pick choice b"
    rng: Union[int, np.random.Generator, None] = None

    def __post_init__(self):
        object.__setattr__(self, "rng", np.random.default_rng(self.rng))

    def __getstate__(self):
        state = self.__dict__.copy()
        state["rng"] = self.rng.bit_generator.state
        return state

    def __setstate__(self, state):
        for key, value in state.items():
            if key == "rng":
                rng = np.random.default_rng()
                rng.bit_generator.state = value
                value = rng
            object.__setattr__(self, key, value)

    def __call__(self, structure: Atoms) -> Atoms:
        if self.rng.random() > self.chance:
            return self.choice_a(structure)
        else:
            return self.choice_b(structure)

    def __str__(self):
        return str(self.choice_a) + "|" + str(self.choice_b)


__all__ = [
        "rattle",
        "stretch",
        "PerturbationABC",
        "Perturbation",
        "apply_perturbations",
        "Rattle",
        "Stretch",
        "Series",
        "RandomChoice",
]

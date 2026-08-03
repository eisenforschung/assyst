"""Relaxation step of ASSYST."""

import warnings
from dataclasses import dataclass
from typing import Literal, Iterable, Iterator

from .calculators import AseCalculatorConfig
from .utils import update_uuid

from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms, FixSymmetry
from ase.filters import FrechetCellFilter
from ase.optimize import BFGS, FIRE, LBFGS, CellAwareBFGS

import numpy as np


@dataclass(frozen=True, eq=True)
class Relax:
    """Minimize energy with respect to internal positions.

    Also used as a base class for all other relaxation classes."""

    max_steps: int = 100
    force_tolerance: float = 1e-3
    algorithm: Literal["LBFGS", "BFGS", "FIRE"] = "LBFGS"
    calculator: AseCalculatorConfig | Calculator | None = None

    def apply_filter_and_constraints(self, structure: Atoms):
        """Hook to allow subclasses to add filters and constraints."""
        return structure

    def get_calculator(self, structure: Atoms) -> Calculator:
        """Resolve the calculator to use for *structure*.

        Prefers ``self.calculator`` (baked into the relaxer at construction time)
        over ``structure.calc``, so a custom engine does not have to be wrapped as
        an ASE calculator and glued onto ``Atoms`` by the caller.  Falls back to
        ``structure.calc`` for backwards compatibility with the old calling
        convention (:func:`relax` attaching a calculator to the structure).

        Args:
            structure (:class:`ase.Atoms`): structure about to be relaxed

        Returns:
            :class:`ase.calculators.calculator.Calculator`: the calculator to use

        Raises:
            ValueError: neither ``self.calculator`` nor ``structure.calc`` is set
        """
        calc = self.calculator
        if isinstance(calc, AseCalculatorConfig):
            calc = calc.get_calculator()
        if calc is None:
            calc = structure.calc
        if calc is None:
            raise ValueError(
                f"{type(self).__name__} has no calculator: set `calculator` on the "
                "Relax instance or attach one to `structure.calc`."
            )
        return calc

    def relax(self, structure: Atoms) -> Atoms:
        """Relax a structure and return result.

        Structure must have a calculator attached, unless this relaxer was
        constructed with `calculator` set.
        Returned structure will have a SinglePointCalculator with the final energy, forces and stresses attached.

        Args:
            structure (:class:`ase.Atoms`): structure to relax

        Returns:
            :class:`ase.Atoms`: relaxed structure with attached single point calculator.
        """
        calc = self.get_calculator(structure)
        structure = structure.copy()
        update_uuid(structure)
        structure.calc = calc
        optimizer_cls = {"LBFGS": LBFGS, "BFGS": BFGS, "FIRE": FIRE}[self.algorithm]
        optimizer = optimizer_cls(self.apply_filter_and_constraints(structure), logfile="/dev/null")
        with warnings.catch_warnings():
            # FrechetCellFilter occasionally emits a benign scipy.linalg.logm
            # accuracy warning during cell updates; the reported error is at
            # numerical noise level and does not affect the relaxation.
            warnings.filterwarnings(
                "ignore",
                message="logm result may be inaccurate",
                category=RuntimeWarning,
            )
            optimizer.run(fmax=self.force_tolerance, steps=self.max_steps)
        structure.calc = None
        structure.calc = SinglePointCalculator(
            structure,
            energy=calc.get_potential_energy(),
            forces=calc.get_forces(),
            stress=calc.get_stress(),
        )
        structure.constraints.clear()
        return structure


@dataclass(frozen=True, eq=True)
class CellRelax(Relax):
    """Minimize energy while keeping relative positions and volume constant."""

    def apply_filter_and_constraints(self, structure: Atoms):
        structure.set_constraint(FixAtoms(np.ones(len(structure), dtype=bool)))
        return FrechetCellFilter(structure, constant_volume=True)


@dataclass(frozen=True, eq=True)
class VolumeRelax(Relax):
    """Minimize energy while keeping relative positions and cell shape constant."""

    pressure: float = 0.0

    def apply_filter_and_constraints(self, structure: Atoms):
        structure.set_constraint(FixAtoms(np.ones(len(structure), dtype=bool)))
        return FrechetCellFilter(
            structure, hydrostatic_strain=True, scalar_pressure=self.pressure
        )


@dataclass(frozen=True, eq=True)
class SymmetryRelax(Relax):
    """Minimize energy with respect to internal positions and cell, while keeping space group fixed."""

    pressure: float = 0.0

    def apply_filter_and_constraints(self, structure: Atoms):
        structure.set_constraint(FixSymmetry(structure))
        return FrechetCellFilter(structure, scalar_pressure=self.pressure)


@dataclass(frozen=True, eq=True)
class FullRelax(Relax):
    """Minimize energy with respect to internal positions and cell without constraints."""

    pressure: float = 0.0

    def apply_filter_and_constraints(self, structure: Atoms):
        return FrechetCellFilter(structure, scalar_pressure=self.pressure)


def relax(
    structures: Iterable[Atoms],
    settings: Relax,
    calculator: AseCalculatorConfig | Calculator | None = None,
) -> Iterator[Atoms]:
    """Relax structures according the given relaxation settings.

    Output structures have the final energy and force attached as ase's SinglePointCalculator.

    Args:
        structures (:class:`collections.abc.Iterable` of :class:`ase.Atoms`): the structures to minimize
        settings (:class:`.Relax`): the kind of relaxation to perform (position, volume, etc.)
        calculator (:class:`.AseCalculatorConfig` or :class:`ase.calculators.calculator.Calculator`):
            the energy/force engine to use.  Optional when `settings.calculator` is
            already set, e.g. for a custom :class:`.Relax` subclass that bakes in its
            own (possibly non-ASE) engine.

    Yields:
        :class:`ase.Atoms`: the corresponding relaxed configuration to each input structure
    """
    for s in structures:
        s = s.copy()
        if calculator is not None:
            s.calc = calculator.get_calculator() if isinstance(calculator, AseCalculatorConfig) else calculator
        yield settings.relax(s)


__all__ = [
        "Relax",
        "CellRelax",
        "VolumeRelax",
        "SymmetryRelax",
        "FullRelax",
        "relax",
]

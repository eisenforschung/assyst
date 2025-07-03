'''Helper plotting functions.'''

from typing import Literal, Callable, Iterable

from ase import Atoms
from matscipy.neighbours import neighbour_list
import matplotlib.pyplot as plt
import numpy as np


def _volume(structures: list[Atoms]) -> Iterable[float]:
    return [s.cell.volume/len(s) for s in structures]


def _energy(structures: list[Atoms]) -> Iterable[float]:
    return [s.get_potential_energy()/len(s) for s in structures]


def volume_histogram(structures: list[Atoms], **kwargs):
    '''Plot histogram of per-atom volumes.

    Args:
        structures (list of :class:`ase.Atoms`):
            structures to plot
        **kwargs:
            passed through to `matplotlib.pyplot.plot`

    Returns:
        Return value of `matplotlib.pyplot.plot`'''
    return plt.hist(_volume(structures), **kwargs)


def distance_histogram(
        structures: list[Atoms],
        rmax: float = 6.0,
        reduce: Literal['min', 'mean'] | Callable[[Iterable[float]], float] = 'min',
        **kwargs
):
    '''Plot histogram of per-atom volumes.

    Args:
        structures (list of :class:`ase.Atoms`):
            structures to plot
        rmax (float):
            maximum cutoff to consider neighborhood
        reduce (callable from array of floats to float):
            applied to the neighbor distances per structure, and should reduce a single scalar that is binned
        **kwargs:
            passed through to `matplotlib.pyplot.plot`

    Returns:
        Return value of `matplotlib.pyplot.plot`'''
    kwargs.setdefault('bins', 100)
    _preset = {
            'min': np.min,
            'mean': np.mean,
    }
    reduce = _preset.get(reduce, reduce)
    return plt.hist([reduce(neighbour_list('d', s, float(rmax))) for s in structures], **kwargs)


def energy_volume(
        structures: list[Atoms],
        **kwargs
):
    '''Plot energy per atom versus volume per atom.

    Requires that :class:`ase.calculators.SinglePointCalculator` are attached to the atoms, either from a relaxation
    for final training set calculation.

    Args:
        structure: list[Atoms],
            structures to plot'''
    V = _volume(structures)
    E = _energy(structures)
    structures = list(structures)
    if len(structures) < 1000:
        if 's' not in kwargs and 'markersize' not in kwargs:
            kwargs['markersize'] = 5
        plt.scatter(V, E, **kwargs)
    else:
        plt.hexbin(V, E, **kwargs, bins='log')

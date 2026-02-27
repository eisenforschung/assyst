"""Helper plotting functions."""

from typing import Literal, Callable, Iterable
from collections import Counter, defaultdict

from ase import Atoms
import matplotlib.pyplot as plt
import numpy as np

from assyst.neighbors import neighbor_list


def _volume(structures: Iterable[Atoms]) -> list[float]:
    return [s.cell.volume / len(s) for s in structures]


def _energy(structures: Iterable[Atoms]) -> list[float]:
    return [s.get_potential_energy() / len(s) for s in structures]


def _concentration(
    structures: Iterable[Atoms], elements: Iterable[str] | None = None
) -> list[dict[str, float]]:
    structure_concentrations = [
        {k: v / len(s) for k, v in Counter(s.symbols).items()} for s in structures
    ]
    concentrations = defaultdict(lambda: np.zeros(len(structure_concentrations)))
    for i, d in enumerate(structure_concentrations):
        for e, c in d.items():
            concentrations[e][i] = c
    if elements is not None:
        concentrations = {e: concentrations[e] for e in elements}
    return concentrations


def _distance(
    structures: Iterable[Atoms], rmax: float
) -> list[Iterable[float]]:
    return [neighbor_list("d", s, float(rmax)) for s in structures]


def _plot_histogram(
    structures: Iterable[Atoms],
    extractor: Callable[[Iterable[Atoms]], Iterable[float]],
    xlabel: str,
    ylabel: str,
    **kwargs
):
    """Helper function to plot histograms.

    Args:
        structures (iterable of :class:`ase.Atoms`):
            structures to plot
        extractor (callable):
            function to extract data from structures
        xlabel (str):
            label for x-axis
        ylabel (str):
            label for y-axis
        **kwargs:
            passed through to :func:`matplotlib.pyplot.hist`

    Returns:
        Return value of :func:`matplotlib.pyplot.hist`
    """
    data = extractor(structures)
    res = plt.hist(data, **kwargs)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    return res


def volume_histogram(structures: list[Atoms], **kwargs):
    """Plot histogram of per-atom volumes.

    Args:
        structures (list of :class:`ase.Atoms`):
            structures to plot
        **kwargs:
            passed through to :func:`matplotlib.pyplot.hist`

    Returns:

        Return value of :func:`matplotlib.pyplot.hist`"""
    return _plot_histogram(
        structures,
        _volume,
        r"Volume [$\mathrm{\AA}^3/\mathrm{atom}$]",
        r"#$\,$Structures",
        **kwargs
    )


def size_histogram(structures: list[Atoms], **kwargs):
    """Plot histogram of number of atoms.

    Args:
        structures (list of :class:`ase.Atoms`):
            structures to plot
        **kwargs:
            passed through to :func:`matplotlib.pyplot.hist`

    Returns:
        Return value of :func:`matplotlib.pyplot.hist`"""
    return _plot_histogram(
        structures,
        lambda s: list(map(len, s)),
        "# Atoms",
        r"#$\,$Structures",
        **kwargs
    )


def concentration_histogram(
    structures: list[Atoms], elements: Iterable[str] | None = None, **kwargs
):
    """Plot histogram of concentrations.

    Args:
        structures (list of :class:`ase.Atoms`):
            structures to plot
        elements (iterable of str):
            which element concentrations to plot, by default all present
        **kwargs:
            passed through to :func:`matplotlib.pyplot.bar`"""
    conc = _concentration(structures, elements=elements)
    conc_step = np.diff(
        sorted(np.unique(np.concatenate([np.unique(c) for c in conc.values()])))
    ).min()
    kwargs.setdefault("width", conc_step)
    width = kwargs["width"]
    kwargs["width"] = width / len(conc)
    shifts = np.linspace(0, 1, len(conc), endpoint=False)
    for i, (e, c) in enumerate(conc.items()):
        x, h = np.unique(c, return_counts=True)
        plt.bar(x + shifts[i] * width - width / 2, h, label=e, align="edge", **kwargs)
    plt.legend()
    plt.xlabel("Concentration")
    plt.ylabel("#$\\,$Structures")


def distance_histogram(
    structures: list[Atoms],
    rmax: float = 6.0,
    reduce: Literal["min", "mean"] | Callable[[Iterable[float]], float] | None = "min",
    **kwargs,
):
    """Plot histogram of per-atom volumes.

    Args:
        structures (list of :class:`ase.Atoms`):
            structures to plot
        rmax (float):
            maximum cutoff to consider neighborhood
        reduce (callable from array of floats to float):
            applied to the neighbor distances per structure, and should reduce a single scalar that is binned
        **kwargs:
            passed through to :func:`matplotlib.pyplot.hist`

    Returns:
        Return value of :func:`matplotlib.pyplot.hist`"""
    kwargs.setdefault("bins", 100)
    labels = {
        "min": r"Minimum distance [$\mathrm{\AA}$]",
        "mean": r"Mean distance [$\mathrm{\AA}$]",
    }
    xlabel = labels.get(reduce, r"Distance [$\mathrm{\AA}$]")

    _preset = {
        "min": np.min,
        "mean": np.mean,
    }

    if reduce is None:
        def extractor(s):
            return np.concatenate(
                [neighbor_list("d", struct, float(rmax)) for struct in s]
            )
        ylabel = r"#$\,$Neighbours"
    else:
        reduce_func = _preset.get(reduce, reduce)
        def extractor(s):
            return [reduce_func(neighbor_list("d", struct, float(rmax))) for struct in s]
        ylabel = r"#$\,$Structures"

    return _plot_histogram(structures, extractor, xlabel, ylabel, **kwargs)


def radial_distribution(
    structures: list[Atoms],
    rmax: float = 6.0,
    **kwargs
):
    """Plot radial distribution of neighbors in training set.

    Calculates all neighbors in all structures and histograms them together.
    Bins are weighted by 1/(4 pi r^2), but because the density in each
    structure can be different, the plot does *not* yield something that can be
    directly compared to a Radial Distribution Function.
    It can be used to locate prefered bonding distances or sampling of the
    radial neighborhood in a training set given suitable data.

    Args:
        structures (list of :class:`ase.Atoms`):
            structures to plot
        rmax (float):
            maximum cutoff to consider neighborhood
        **kwargs: pass through to :func:`matplotlib.pyplot.hist`

    Returns:
        Return value of :func:`matplotlib.pyplot.hist`"""
    kwargs.setdefault("bins", 100)
    distances = np.concatenate([n for n in _distance(structures, rmax)])
    weights = 1 / (4 * np.pi * distances ** 2)
    res = plt.hist(distances, weights=weights, **kwargs)
    plt.xlabel(r"Distance [$\mathrm{\AA}$]")
    plt.ylabel("Radial distribution")
    return res


def energy_histogram(
        structures: list[Atoms],
        **kwargs
):
    """Plot energy per atom histogram.

    Requires that :class:`ase.calculators.singlepoint.SinglePointCalculator` are attached to the atoms, either from a
    relaxation for final training set calculation.

    Args:
        structures (list of :class:`ase.Atoms`): structures to plot
        **kwargs: pass through to :func:`matplotlib.pyplot.hist`

    Returns:
        Return value of :func:`matplotlib.pyplot.hist`"""
    kwargs.setdefault("bins", 100)
    return _plot_histogram(
        structures,
        _energy,
        r"Energy [eV/atom]",
        r"#$\,$Structures",
        **kwargs
    )


def energy_volume(structures: list[Atoms], **kwargs):
    """Plot energy per atom versus volume per atom.

    Requires that :class:`ase.calculators.singlepoint.SinglePointCalculator` are attached to the atoms, either from a
    relaxation for final training set calculation.

    Args:
        structure: list[Atoms],
            structures to plot"""
    V = _volume(structures)
    E = _energy(structures)
    structures = list(structures)
    if len(structures) < 1000:
        if "s" not in kwargs and "markersize" not in kwargs:
            kwargs["markersize"] = 5
        plt.scatter(V, E, **kwargs)
    else:
        plt.hexbin(V, E, **kwargs, bins="log")
    plt.xlabel(r"Volume [$\mathrm{\AA}^3/\mathrm{atom}$]")
    plt.ylabel(r"Energy [eV/atom]")


__all__ = [
        "volume_histogram",
        "size_histogram",
        "concentration_histogram",
        "distance_histogram",
        "radial_distribution",
        "energy_histogram",
        "energy_volume",
]

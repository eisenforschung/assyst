"""Helper plotting functions."""

from typing import Literal, Callable, Iterable
from collections import Counter, defaultdict

from ase import Atoms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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


_DISTANCE_LABELS = {
    "min": r"Minimum distance [$\mathrm{\AA}$]",
    "mean": r"Mean distance [$\mathrm{\AA}$]",
}


def _distance_xlabel(
    reduce: Literal["min", "mean"] | Callable[[Iterable[float]], float] | None,
) -> str:
    return _DISTANCE_LABELS.get(reduce, r"Distance [$\mathrm{\AA}$]")


def _reduce_distances(
    structures: Iterable[Atoms],
    rmax: float,
    reduce: Literal["min", "mean"] | Callable[[Iterable[float]], float] | None,
) -> list[float]:
    """Compute neighbor distances, optionally reduced per structure.

    Args:
        structures (iterable of :class:`ase.Atoms`):
            structures to process
        rmax (float):
            neighbor cutoff radius
        reduce (callable, "min", "mean", or None):
            if ``None``, return all neighbor distances concatenated; otherwise
            apply the reducer per structure and return one value per structure,
            skipping structures with no neighbors within *rmax*

    Returns:
        list of floats (or :class:`numpy.ndarray` when *reduce* is ``None``)
    """
    _preset = {"min": np.min, "mean": np.mean}
    if reduce is None:
        return np.concatenate(
            [neighbor_list("d", s, float(rmax)) for s in structures]
        )
    reduce_func = _preset.get(reduce, reduce)
    distances = []
    for s in structures:
        d = neighbor_list("d", s, float(rmax))
        if len(d) > 0:
            distances.append(reduce_func(d))
    return distances


def _plot_histogram(
    structures: Iterable[Atoms],
    extractor: Callable[[Iterable[Atoms]], Iterable[float] | dict[str, Iterable[float]]],
    xlabel: str,
    ylabel: str,
    **kwargs
):
    """Helper function to plot histograms.

    If the extractor returns a :class:`dict`, the values are assembled into a
    long-form :class:`pandas.DataFrame` and plotted with a single
    :func:`seaborn.histplot` call using the dict keys as ``hue``.  All series
    share a common bin grid (computed from all values combined) and default to
    ``element='step'``.  Otherwise :func:`matplotlib.pyplot.hist` is called
    with the returned data.

    Args:
        structures (iterable of :class:`ase.Atoms`):
            structures to plot
        extractor (callable):
            function to extract data from structures; may return a dict mapping
            labels to arrays of values, in which case multiple histograms are
            plotted
        xlabel (str):
            label for x-axis
        ylabel (str):
            label for y-axis
        **kwargs:
            passed through to :func:`matplotlib.pyplot.hist` or
            :func:`seaborn.histplot`; ``bins`` controls binning for both paths

    Returns:
        Return value of :func:`matplotlib.pyplot.hist`, or ``None`` when a dict
        is returned by the extractor.
    """
    data = extractor(structures)
    if isinstance(data, dict):
        df = pd.DataFrame({k: pd.Series(v) for k, v in data.items()})
        df_long = df.melt(var_name='variable', value_name='value')
        ax = plt.gca()
        all_values = df_long['value'].dropna().to_numpy()
        bins = kwargs.pop('bins', 'auto')
        bin_edges = np.histogram_bin_edges(all_values, bins=bins)
        kwargs.setdefault('element', 'step')
        sns.histplot(data=df_long, x='value', hue='variable', ax=ax, bins=bin_edges, **kwargs)
        res = None
    else:
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
            passed through to :func:`seaborn.histplot`
    """
    return _plot_histogram(
        structures,
        lambda s: _concentration(s, elements=elements),
        "Concentration",
        r"#$\,$Structures",
        **kwargs,
    )


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
            applied to the neighbor distances per structure, and should reduce a single scalar that is binned;
            if `None` plot all atomic distances concatenated
        **kwargs:
            passed through to :func:`matplotlib.pyplot.hist`

    Returns:
        Return value of :func:`matplotlib.pyplot.hist`"""
    kwargs.setdefault("bins", 100)
    xlabel = _distance_xlabel(reduce)
    ylabel = r"#$\,$Neighbours" if reduce is None else r"#$\,$Structures"
    return _plot_histogram(
        structures,
        lambda s: _reduce_distances(s, rmax, reduce),
        xlabel,
        ylabel,
        **kwargs,
    )


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


def energy_distance(
    structures: list[Atoms],
    rmax: float = 6.0,
    reduce: Literal["min", "mean"] | Callable[[Iterable[float]], float] = "min",
    **kwargs,
):
    """Plot energy per atom versus neighbor distance.

    Requires that :class:`ase.calculators.singlepoint.SinglePointCalculator` are attached to the atoms, either from a
    relaxation for final training set calculation.

    Args:
        structures (list of :class:`ase.Atoms`):
            structures to plot
        rmax (float):
            maximum cutoff to consider neighborhood
        reduce (callable from array of floats to float):
            applied to the neighbor distances per structure to reduce them to a
            single scalar; ``"min"`` and ``"mean"`` are recognized as shortcuts;
            structures with no neighbors within *rmax* are silently skipped
        **kwargs:
            passed through to :func:`matplotlib.pyplot.scatter` or
            :func:`matplotlib.pyplot.hexbin`"""
    xlabel = _distance_xlabel(reduce)
    structures = list(structures)
    D = _reduce_distances(structures, rmax, reduce)
    # Keep only structures that contributed to D (those with at least one neighbor)
    with_neighbors = [
        s for s in structures
        if len(neighbor_list("d", s, float(rmax))) > 0
    ]
    E = _energy(with_neighbors)
    if len(structures) < 1000:
        if "s" not in kwargs and "markersize" not in kwargs:
            kwargs["markersize"] = 5
        plt.scatter(D, E, **kwargs)
    else:
        plt.hexbin(D, E, **kwargs, bins="log")
    plt.xlabel(xlabel)
    plt.ylabel(r"Energy [eV/atom]")


def energy_volume(structures: list[Atoms], **kwargs):
    """Plot energy per atom versus volume per atom.

    Requires that :class:`ase.calculators.singlepoint.SinglePointCalculator` are attached to the atoms, either from a
    relaxation for final training set calculation.

    Args:
        structures (list of :class:`ase.Atoms`):
            structures to plot
        **kwargs:
            passed through to :func:`matplotlib.pyplot.scatter` or :func:`matplotlib.pyplot.hexbin`"""
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


def _lattice_parameters(structures: Iterable[Atoms]) -> dict[str, np.ndarray]:
    lengths = np.array([s.cell.lengths() for s in structures])
    return {"a": lengths[:, 0], "b": lengths[:, 1], "c": lengths[:, 2]}


def _lattice_angles(structures: Iterable[Atoms]) -> dict[str, np.ndarray]:
    angles = np.array([s.cell.angles() for s in structures])
    return {r"$\alpha$": angles[:, 0], r"$\beta$": angles[:, 1], r"$\gamma$": angles[:, 2]}


def _aspect_ratio(structures: Iterable[Atoms]) -> list[float]:
    return [
        max(s.cell.lengths()) / min(s.cell.lengths()) for s in structures
    ]


def lattice_parameter_histogram(structures: list[Atoms], **kwargs):
    """Plot histogram of lattice parameters a, b, and c.

    Args:
        structures (list of :class:`ase.Atoms`):
            structures to plot
        **kwargs:
            passed through to :func:`seaborn.histplot`
    """
    return _plot_histogram(
        structures,
        _lattice_parameters,
        r"Lattice parameter [$\mathrm{\AA}$]",
        r"#$\,$Structures",
        **kwargs,
    )


def lattice_angle_histogram(structures: list[Atoms], **kwargs):
    """Plot histogram of lattice angles alpha, beta, and gamma.

    Args:
        structures (list of :class:`ase.Atoms`):
            structures to plot
        **kwargs:
            passed through to :func:`seaborn.histplot`
    """
    return _plot_histogram(
        structures,
        _lattice_angles,
        r"Lattice angle [°]",
        r"#$\,$Structures",
        **kwargs,
    )


def aspect_ratio_histogram(structures: list[Atoms], **kwargs):
    """Plot histogram of cell aspect ratios (max / min lattice parameter).

    Args:
        structures (list of :class:`ase.Atoms`):
            structures to plot
        **kwargs:
            passed through to :func:`matplotlib.pyplot.hist`

    Returns:
        Return value of :func:`matplotlib.pyplot.hist`
    """
    return _plot_histogram(
        structures,
        _aspect_ratio,
        "Aspect ratio (max / min lattice parameter)",
        r"#$\,$Structures",
        **kwargs,
    )


__all__ = [
        "volume_histogram",
        "size_histogram",
        "concentration_histogram",
        "distance_histogram",
        "radial_distribution",
        "energy_histogram",
        "energy_distance",
        "energy_volume",
        "lattice_parameter_histogram",
        "lattice_angle_histogram",
        "aspect_ratio_histogram",
]

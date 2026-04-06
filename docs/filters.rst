Filters DSL
===========

ASSYST filters are callables that accept an :class:`ase.Atoms` structure and
return a ``bool``: ``True`` to keep the structure, ``False`` to drop it.

Because filters are plain callables, they work directly with Python's built-in
:func:`filter` function::

    from assyst.filters import DistanceFilter, AspectFilter

    structures = [...]   # list of ase.Atoms

    dist_filter = DistanceFilter({"Fe": 1.1, "N": 0.7})
    good = list(filter(dist_filter, structures))

Any callable with the signature ``(Atoms) -> bool`` is accepted wherever a
:class:`~assyst.filters.Filter` is expected — including plain functions and
lambdas::

    # keep only structures with at least 4 atoms
    good = list(filter(lambda s: len(s) >= 4, structures))

Composing filters
-----------------

:class:`~assyst.filters.FilterBase` subclasses support ``&`` (and) and ``|``
(or) to build compound filters without nesting explicit
:class:`~assyst.filters.AndFilter` / :class:`~assyst.filters.OrFilter`
instances::

    combined = DistanceFilter({"Fe": 1.1}) & AspectFilter(maximum_aspect_ratio=5)
    good = list(filter(combined, structures))

The ``&`` operator short-circuits: the right-hand filter is only evaluated if
the left-hand filter returns ``True``.

Available filter classes
------------------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Class
     - What it checks
   * - :class:`~assyst.filters.DistanceFilter`
     - Minimum interatomic distances (element-pair cutoffs)
   * - :class:`~assyst.filters.AspectFilter`
     - Maximum aspect ratio of the unit cell (longest / shortest lattice parameter)
   * - :class:`~assyst.filters.VolumeFilter`
     - Maximum volume per atom
   * - :class:`~assyst.filters.EnergyFilter`
     - Energy per atom within ``[min_energy, max_energy]`` (requires a single-point calculator)
   * - :class:`~assyst.filters.ForceFilter`
     - Maximum atomic force magnitude (requires a single-point calculator)

See :doc:`api/filters` for the full API reference of each class.

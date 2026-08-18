Metadata
====================

ASSYST tracks several metadata keys in the ``info`` attribute of the :class:`ase.Atoms` structures it generates and modifies.
These keys allow for identification, lineage tracking, and understanding the symmetry and perturbations applied to each structure.

Identification & Lineage
------------------------

These keys are managed by the :func:`assyst.utils.update_uuid` function and are used to track the derivation history of structures.

* ``uuid``: A Universally Unique Identifier (UUID) for the current structure.
* ``seed``: The UUID of the initial structure from which this structure was derived. It remains constant throughout a lineage.
* ``lineage``: A list of UUIDs of all parent structures, in the order they were generated.
* ``step``: The name of the workflow step that produced the current UUID.  Only set when the caller of :func:`assyst.utils.update_uuid` provides one; like ``uuid`` it reflects the most recent step only, the structures it came from are in ``lineage``.

The step names are

* ``pyxtal``: generation by :func:`assyst.crystals.pyxtal`, either directly or through :func:`assyst.crystals.sample`
* ``relax``, ``cell_relax``, ``volume_relax``, ``symmetry_relax``, ``full_relax``: the corresponding class in :mod:`assyst.relaxations`.  A relaxation against a non-zero pressure carries it, as in ``full_relax(pressure=3.0)``, since it minimizes to a different structure; the settings of the optimizer are not part of the name
* the string of the applied :class:`assyst.perturbations.PerturbationABC`, the same value that goes into ``perturbation``

This makes the sets of a plain ASSYST run tell themselves apart -- the generated structures carry ``pyxtal``, the
volume relaxed ones ``volume_relax``, the fully relaxed ones ``full_relax`` and the perturbed ones their perturbation
-- which the ``symmetry`` and ``perturbation`` keys alone cannot do.

Use :func:`assyst.utils.step_of` to read the key, and pass ``step=`` to :func:`assyst.utils.update_uuid` to record a
step of your own.

For more details on how lineage is tracked through the workflow, see :doc:`lineage`.

Symmetry
--------

When structures are generated using :func:`assyst.crystals.sample` or :func:`assyst.crystals.pyxtal`, the following symmetry-related keys are added:

* ``requested spacegroup``: The requested symmetry group number (e.g., space group 225).
* ``symmetry``: An alias for ``requested spacegroup``.
* ``spacegroup``: The actual symmetry group number of the generated structure, which may be higher than the requested one.
* ``repeat``: The iteration index when multiple structures are generated for the same symmetry group.

Perturbation
------------

When a perturbation is applied using a :class:`assyst.perturbations.PerturbationABC` subclass, information about the perturbation is recorded:

* ``perturbation``: A string description of the perturbation(s) applied (e.g., ``rattle(0.05)+stretch(hydro=0.05, shear=0.05)``). Multiple perturbations are concatenated with a ``+``.

.. toctree::
   :maxdepth: 1
   :hidden:

   lineage

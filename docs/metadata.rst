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

For more details on how lineage is tracked through the workflow, see :doc:`lineage`.

Symmetry
--------

When structures are generated using :func:`assyst.crystals.sample_space_groups` or :func:`assyst.crystals.pyxtal`, the following symmetry-related keys are added:

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

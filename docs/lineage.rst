Lineage Tracking
================

ASSYST automatically tracks the history of structures as they are generated and modified through the workflow.
This is achieved using UUIDs (Universally Unique Identifiers) stored in the structure's ``info`` dictionary.

For a quick reference of all metadata keys, see :doc:`metadata`.

Workflow Integration
--------------------

Initial Generation
~~~~~~~~~~~~~~~~~~

When a structure is first generated using :func:`.pyxtal` (or through :func:`.sample`), it is assigned a new UUID and ``step`` is set to ``"pyxtal"``.
At this stage, the ``seed`` is set to the same UUID, and the ``lineage`` is empty.

Perturbations
~~~~~~~~~~~~~

Whenever a :class:`.PerturbationABC` (like :class:`.Rattle` or :class:`.Stretch`) is applied to a structure, a new UUID is generated and ``step`` is set to that perturbation's own string form (e.g. ``rattle(0.1)``).
The previous UUID is appended to the ``lineage`` list. The ``seed`` remains unchanged.
The cumulative ``perturbation`` key still records every step of a :class:`.Series`, while ``step`` only reflects the most recent one.

Relaxations
~~~~~~~~~~~

Similarly, the :meth:`.Relax.relax` method generates a new UUID for the relaxed structure, updates the lineage, and sets ``step`` to the relaxation class's name (e.g. ``VolumeRelax``).

Example
-------

For a practical demonstration of how these fields are updated, please refer to the :doc:`Lineage Notebook <notebooks/Lineage>`.

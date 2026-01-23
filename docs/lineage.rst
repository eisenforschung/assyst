Lineage Tracking
================

ASSYST automatically tracks the history of structures as they are generated and modified through the workflow.
This is achieved using UUIDs (Universally Unique Identifiers) stored in the structure's ``info`` dictionary.

For a quick reference of all metadata keys, see :doc:`metadata`.

Workflow Integration
--------------------

Initial Generation
~~~~~~~~~~~~~~~~~~

When a structure is first generated using :func:`.pyxtal` (or through :func:`.sample_space_groups`), it is assigned a new UUID.
At this stage, the ``seed`` is set to the same UUID, and the ``lineage`` is empty.

Perturbations
~~~~~~~~~~~~~

Whenever a :class:`.PerturbationABC` (like :class:`.Rattle` or :class:`.Stretch`) is applied to a structure, a new UUID is generated.
The previous UUID is appended to the ``lineage`` list. The ``seed`` remains unchanged.

Relaxations
~~~~~~~~~~~

Similarly, the :meth:`.Relax.relax` method generates a new UUID for the relaxed structure and updates the lineage.

Example
-------

For a practical demonstration of how these fields are updated, please refer to the :doc:`Lineage Notebook <notebooks/Lineage>`.

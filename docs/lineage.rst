Lineage Tracking
================

ASSYST automatically tracks the history of structures as they are generated and modified through the workflow.
This is achieved using UUIDs (Universally Unique Identifiers) stored in the structure's ``info`` dictionary.

For a quick reference of all metadata keys, see :doc:`metadata`.

Workflow Integration
--------------------

Initial Generation
~~~~~~~~~~~~~~~~~~

When a structure is first generated using :func:`.pyxtal` (or through :func:`.sample`), it is assigned a new UUID.
At this stage, the ``seed`` is set to the same UUID, and the ``lineage`` is empty.

Perturbations
~~~~~~~~~~~~~

Whenever a :class:`.PerturbationABC` (like :class:`.Rattle` or :class:`.Stretch`) is applied to a structure, a new UUID is generated.
The previous UUID is appended to the ``lineage`` list. The ``seed`` remains unchanged.

Relaxations
~~~~~~~~~~~

Similarly, the :meth:`.Relax.relax` method generates a new UUID for the relaxed structure and updates the lineage.

Stages
~~~~~~

Where the UUIDs record *which* structure a structure came from, the ``stage`` key records *what was done to it*:
each of the steps above appends its name to it, so ``spg+volume_relax+full_relax`` is a generated structure that was
volume relaxed and then fully relaxed.  See :doc:`metadata` for the step names.

Example
-------

For a practical demonstration of how these fields are updated, please refer to the :doc:`Lineage Notebook <notebooks/Lineage>`.

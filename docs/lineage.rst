Lineage Tracking
================

ASSYST automatically tracks the history of structures as they are generated and modified through the workflow.
This is achieved using UUIDs (Universally Unique Identifiers) stored in the structure's ``info`` dictionary.

Keys in Atoms.info
------------------

Every structure managed by ASSYST will contain the following keys in its ``info`` attribute:

* ``uuid``: A unique identifier for the current structure.
* ``seed``: The UUID of the initial structure from which this structure was derived.
* ``lineage``: A list of UUIDs of all parent structures, in the order they were generated.

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

.. code-block:: python

    from assyst.crystals import pyxtal
    from assyst.perturbations import Rattle

    # 1. Generate initial structure
    atoms = pyxtal(1, species=['Cu'], num_ions=[1])
    print(f"Initial UUID: {atoms.info['uuid']}")
    print(f"Seed: {atoms.info['seed']}")
    print(f"Lineage: {atoms.info.get('lineage', [])}")

    # 2. Apply perturbation
    rattle = Rattle(sigma=0.1)
    perturbed = rattle(atoms.copy())
    print(f"Perturbed UUID: {perturbed.info['uuid']}")
    print(f"Seed: {perturbed.info['seed']}")
    print(f"Lineage: {perturbed.info['lineage']}")

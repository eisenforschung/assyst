Installation
============

Install via PyPI

.. code-block:: bash

   pip install assyst

or conda-forge

.. code-block:: bash

   conda install -c conda-forge assyst


Optional Dependencies
---------------------

ASSYST requires some ASE calculators to perform structure relaxations, but is agnostic to which specifically.
The example notebooks use either builtin ASE calculators or the
`Graph Atomic Cluster Expansion <https://gracemaker.readthedocs.io/en/latest/>`_.
When using pip, you can install the necessary packages with the ``grace`` optional dependency

.. code-block:: bash

   pip install assyst[grace]

When installing via conda follow the instructions on the GRACE home page or try the ``grace-tensorpotential`` package
from conda-forge.

The example notebooks also fit simple Atomic Cluster Expansion models, though not technically part of the ASSYST
workflow.
You will need to install the ``python-ace`` conda-forge package or follow the 
`instructions <https://pacemaker.readthedocs.io/en/latest/pacemaker/install/>`_.

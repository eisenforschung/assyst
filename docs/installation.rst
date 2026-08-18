Installation
============

`assyst` requires at least python 3.11 and has been tested on CPython.

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
These are only required to simulate acquiring reference data.
If you will use DFT or other simulation engines as training data, you won't need them.

When using pip, you can install the necessary packages with the ``grace`` optional dependency

.. code-block:: bash

   pip install assyst[grace]

When installing via conda follow the instructions on the GRACE home page or try the ``grace-tensorpotential`` package
from conda-forge.

The example notebooks also fit simple Atomic Cluster Expansion models, though not technically part of the ASSYST
workflow.
:class:`~assyst.leverage.AceFeaturizer`, which scores structures on the same linear ACE basis such a fit would use,
needs it as well; the rest of :mod:`assyst.leverage` does not.
You will need to install the ``python-ace`` conda-forge package or follow the
`instructions <https://pacemaker.readthedocs.io/en/latest/pacemaker/install/>`_.
The PyPI release of ``python-ace`` only ships a CPython 3.9 wheel, so it cannot be installed on the python versions
``assyst`` supports and there is no ``pip`` extra for it.

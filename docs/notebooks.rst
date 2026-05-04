Notebooks
=========

These are introduction workflows using `assyst`.
In :doc:`notebooks/SimpleUnary` we construct a full training set from start to finish and fit a small, linear MLIP to it.
This show cases all required steps in creating a running MLIP.

:doc:`notebooks/Quickstart/Crystals` is a more detailed explanation how to construct chemical sampling in your training
set.

:doc:`notebooks/PlotGallery` is a small gallery of available plots `assyst` offers for training data inspection.

:doc:`notebooks/Lineage` shows some technical details that trace structures as they flow through the workflow, if you
are interested in optimizing or debuging particular structures.

:doc:`notebooks/Relaxations` walks through the different relaxation modes available in :mod:`assyst.relaxations`
(positions only, volume only, cell shape, symmetry-preserving, full), including how to apply pressure and switch
optimizer algorithms.

:doc:`notebooks/Perturbations` show cases the perturbations available in :mod:`assyst.perturbations` for generating
off-equilibrium training structures: gaussian rattling, element-scaled rattling, random cell stretching, and how to
compose them in series or pick between alternatives at random.



.. toctree::
   :maxdepth: 1
   :hidden:
   :glob:

   notebooks/Quickstart/Crystals.ipynb
   notebooks/SimpleUnary.ipynb
   notebooks/Relaxations.ipynb
   notebooks/Perturbations.ipynb
   notebooks/PlotGallery.ipynb
   notebooks/Lineage.ipynb

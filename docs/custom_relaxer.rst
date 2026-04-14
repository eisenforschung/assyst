Custom Relaxers
===============

ASSYST's relaxation step is designed to be extensible.
If your preferred energy/force engine does not expose an ASE-compatible
:class:`ase.calculators.calculator.Calculator`, you can still plug it into
the workflow by subclassing :class:`assyst.relax.Relax` and overriding the
:meth:`~assyst.relax.Relax.relax` method.

When to subclass ``Relax``
--------------------------

The built-in :meth:`~assyst.relax.Relax.relax` implementation drives
minimization through ASE's LBFGS optimizer and therefore requires an ASE
calculator to be attached to the :class:`~ase.Atoms` object.
A custom subclass is the right tool when:

* the external code has its own minimizer (e.g. a force-field engine with
  native geometry optimisation), or
* the energy/force interface is not easily wrapped as an ASE calculator.

The contract your override must satisfy:

1. Accept a single :class:`~ase.Atoms` object (the structure to relax).
2. Call :func:`assyst.utils.update_uuid` on the returned structure so that
   provenance tracking continues to work.
3. Return a new :class:`~ase.Atoms` object with a
   :class:`~ase.calculators.singlepoint.SinglePointCalculator` carrying the
   final energy, forces, and stress.

Toy example
-----------

The snippet below shows a minimal custom relaxer that delegates geometry
optimisation to a hypothetical external library ``myengine``.

.. code-block:: python

    from dataclasses import dataclass

    import numpy as np
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator

    from assyst.relax import Relax
    from assyst.utils import update_uuid


    # ---------------------------------------------------------------------------
    # Toy stand-in for a non-ASE energy/force engine
    # ---------------------------------------------------------------------------

    class MyEngineError(RuntimeError):
        pass


    def myengine_run_relaxation(positions, cell, numbers, max_steps, ftol):
        """Pretend external relaxation routine.

        In practice this would call out to a C extension, a subprocess,
        or a REST API.  Here it simply returns the input unchanged with
        made-up energetics so the example runs without any real dependency.

        Returns
        -------
        relaxed_positions : np.ndarray, shape (N, 3)
        final_energy      : float   (eV)
        final_forces      : np.ndarray, shape (N, 3)  (eV/Å)
        final_stress      : np.ndarray, shape (6,)    (eV/Å³, Voigt order)
        """
        # --- replace this block with actual engine calls ---
        n_atoms = len(positions)
        relaxed_positions = positions.copy()
        final_energy = -float(n_atoms)          # 1 eV/atom binding
        final_forces = np.zeros((n_atoms, 3))   # converged → forces ≈ 0
        final_stress = np.zeros(6)
        # ---------------------------------------------------
        return relaxed_positions, final_energy, final_forces, final_stress


    # ---------------------------------------------------------------------------
    # Custom Relax subclass
    # ---------------------------------------------------------------------------

    @dataclass(frozen=True, eq=True)
    class MyEngineRelax(Relax):
        """Relax structures using ``myengine``'s native geometry optimiser.

        Inherits ``max_steps`` and ``force_tolerance`` from
        :class:`~assyst.relax.Relax`.
        """

        def relax(self, structure: Atoms) -> Atoms:
            # 1. Run the external relaxation
            relaxed_pos, energy, forces, stress = myengine_run_relaxation(
                positions=structure.get_positions(),
                cell=structure.get_cell(),
                numbers=structure.get_atomic_numbers(),
                max_steps=self.max_steps,
                ftol=self.force_tolerance,
            )

            # 2. Build the output Atoms object
            relaxed = structure.copy()
            relaxed.set_positions(relaxed_pos)

            # 3. Attach a SinglePointCalculator with the final energetics
            relaxed.calc = SinglePointCalculator(
                relaxed,
                energy=energy,
                forces=forces,
                stress=stress,
            )

            # 4. Update provenance (UUID / lineage) — do not skip this step
            update_uuid(relaxed)

            return relaxed


Using the custom relaxer in the workflow
----------------------------------------

Once defined, ``MyEngineRelax`` is a drop-in replacement anywhere
:class:`~assyst.relax.Relax` is accepted:

.. code-block:: python

    from assyst.relax import relax as assyst_relax

    settings = MyEngineRelax(max_steps=200, force_tolerance=5e-4)

    relaxed_structures = list(
        assyst_relax(
            structures=my_structures,
            settings=settings,
            calculator=None,   # calculator is unused by MyEngineRelax.relax
        )
    )

.. note::

    The top-level :func:`assyst.relax.relax` function attaches an ASE
    calculator to each structure before calling ``settings.relax``.  If your
    custom ``relax`` method does not need an ASE calculator you can pass
    ``calculator=None`` *and* iterate over the structures directly, bypassing
    the helper function altogether:

    .. code-block:: python

        relaxer = MyEngineRelax()
        relaxed_structures = [relaxer.relax(s) for s in my_structures]

Specifying Compositions
=======================

The :class:`~assyst.crystals.Formulas` class is how you tell ASSYST which chemical compositions to generate
structures for. It holds a list of formula units — each formula unit is a dictionary that maps an element
symbol to the number of atoms of that element per unit cell.

.. code-block:: python

    from assyst.crystals import Formulas

    # Manually: two formula units for pure copper — 1-atom and 2-atom cells
    cu = Formulas(({'Cu': 1}, {'Cu': 2}))

For a range of stoichiometries the :meth:`~assyst.crystals.Formulas.range` helper is usually more
convenient. It works like Python's built-in :func:`range`, but skips zero and the stop value is exclusive:

.. code-block:: python

    # Cu₁, Cu₂, Cu₃, Cu₄
    cu = Formulas.range('Cu', 1, 5)

Combining compositions
----------------------

The real power of :class:`~assyst.crystals.Formulas` comes from combining single-element lists into
multi-element lists using three operators.

``+`` — concatenate
~~~~~~~~~~~~~~~~~~~~

Two :class:`~assyst.crystals.Formulas` objects are joined end-to-end. This is useful for combining
independent lists, e.g. unary and binary compositions for a mixed training set:

.. code-block:: python

    cu    = Formulas.range('Cu', 1, 5)   # Cu₁ … Cu₄
    ag    = Formulas.range('Ag', 1, 5)   # Ag₁ … Ag₄
    unary = cu + ag
    # Formulas(atoms=({'Cu': 1}, {'Cu': 2}, {'Cu': 3}, {'Cu': 4},
    #                 {'Ag': 1}, {'Ag': 2}, {'Ag': 3}, {'Ag': 4}))

``|`` — pair element-wise
~~~~~~~~~~~~~~~~~~~~~~~~~

Elements from two lists are merged at matching positions (like Python's :func:`zip`). The result has
the same length as the shorter of the two inputs. Use this when you want to keep the Cu:Ag ratio
*fixed* across all formula units:

.. code-block:: python

    cu    = Formulas.range('Cu', 1, 5)   # Cu₁ … Cu₄
    ag    = Formulas.range('Ag', 1, 3)   # Ag₁, Ag₂
    paired = cu | ag
    # Formulas(atoms=({'Cu': 1, 'Ag': 1}, {'Cu': 2, 'Ag': 2}))

The two operands must cover **different** elements.

``*`` — all combinations (Cartesian product)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every formula from the left list is combined with every formula from the right list. Use this
to sample *all* Cu:Ag ratios:

.. code-block:: python

    cu  = Formulas.range('Cu', 1, 3)   # Cu₁, Cu₂
    ag  = Formulas.range('Ag', 1, 3)   # Ag₁, Ag₂
    all_combos = cu * ag
    # Formulas(atoms=({'Cu': 1, 'Ag': 1}, {'Cu': 1, 'Ag': 2},
    #                 {'Cu': 2, 'Ag': 1}, {'Cu': 2, 'Ag': 2}))

The two operands must cover **different** elements.

Multi-element systems
---------------------

Both ``|`` and ``*`` extend naturally to ternary (and higher) systems by chaining:

.. code-block:: python

    cu  = Formulas.range('Cu', 1, 4)
    ag  = Formulas.range('Ag', 1, 4)
    au  = Formulas.range('Au', 1, 4)
    ternary = cu * ag * au   # all Cu–Ag–Au combinations

Trimming by atom count
----------------------

Generated formula units may exceed the number of atoms that a structure generator can
handle efficiently. :meth:`~assyst.crystals.Formulas.trim` removes formula units outside a given
atom-count range:

.. code-block:: python

    large = Formulas.range('Cu', 1, 10) * Formulas.range('Ag', 1, 10)
    small = large.trim(min_atoms=2, max_atoms=8)

Passing to :func:`~assyst.crystals.sample_space_groups`
---------------------------------------------------------

:class:`~assyst.crystals.Formulas` is the expected input to :func:`~assyst.crystals.sample_space_groups`:

.. code-block:: python

    from assyst.crystals import Formulas, sample_space_groups

    formulas = Formulas.range('Cu', 1, 5) * Formulas.range('Ag', 1, 5)
    structures = sample_space_groups(formulas)

Any plain iterable of dicts is also accepted, so you can supply your own list directly without
using :class:`~assyst.crystals.Formulas` at all:

.. code-block:: python

    structures = sample_space_groups([{'Cu': 2, 'Ag': 1}, {'Cu': 3, 'Ag': 1}])

See :doc:`api/crystals` for the full API reference.

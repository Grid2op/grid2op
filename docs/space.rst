.. currentmodule:: grid2op.Space

Space
===================================

Objectives
-----------
This module exposes the action space definition, the observation space definition (both depend on the underlying
power grid and on the type of Action / Observation chosen).

It also define a dedicated representation of the powergrid, that is "powerflow agnostic" (does not depends on the
implementation of the :class:`grid2op.Backend`) and from which inherit most of grid2op objects: the :class:`GridObjects`

Detailed Documentation by class
--------------------------------
.. automodule:: grid2op.Space
    :members:
    :show-inheritance:
    :special-members:
    :autosummary:

.. include:: final.rst


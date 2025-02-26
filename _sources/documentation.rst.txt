.. _documentation:

Documentation
=============

Solver class
------------

.. py:module:: edipack2triqs.solver

.. autoclass:: EDIpackSolver
   :special-members: __init__
   :members:
   :exclude-members: chi2_fit_bath

Objects representing bath parameters
------------------------------------

.. py:module:: edipack2triqs.bath

.. autoclass:: Bath
   :members:

These classes represent collections of bath parameters used by EDIpack.
Each of the classes corresponds to one bath geometry (*normal*, *hybrid* or
*general*).

.. autoclass:: BathNormal
   :members:
.. autoclass:: BathHybrid
   :members:
.. autoclass:: BathGeneral
   :members:

Bath fitting
------------

.. py:module:: edipack2triqs.fit

:py:class:`BathFittingParams` is a collection of parameters used
in EDIpack's bath fitting procedure. The procedure can be invoked by calling
a method of :py:class:`edipack2triqs.solver.EDIpackSolver`.

.. autoclass:: BathFittingParams
.. automethod:: edipack2triqs.solver.EDIpackSolver.chi2_fit_bath

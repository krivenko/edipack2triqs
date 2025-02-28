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

Objects representing sets of bath parameters
--------------------------------------------

.. py:module:: edipack2triqs.bath

.. autoclass:: Bath
   :members:

The following classes derived from :py:class:`Bath` represent sets of bath
parameters used by **EDIpack**. Each of the classes corresponds to one bath
geometry (*normal*, *hybrid* or *general*). They support addition, subtraction
and multiplication by a scalar, which translate into the respective operations
with the stored parameters.

.. autoclass:: BathNormal
   :members:
   :show-inheritance:
.. autoclass:: BathHybrid
   :members:
   :show-inheritance:
.. autoclass:: BathGeneral
   :members:
   :show-inheritance:

Bath fitting
------------

.. py:module:: edipack2triqs.fit

:py:class:`BathFittingParams` is a collection of parameters used
in **EDIpack**'s bath fitting procedure. The procedure can be invoked by
calling a method of :py:class:`edipack2triqs.solver.EDIpackSolver`.

.. autoclass:: BathFittingParams
   :members:
.. automethod:: edipack2triqs.solver.EDIpackSolver.chi2_fit_bath

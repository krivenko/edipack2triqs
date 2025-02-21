.. _documentation:

Documentation
=============

Solver class
------------

.. py:module:: edipack2triqs.solver

The central class of **edipack2triqs** is
:py:class:`EDIpackSolver`. It represents the underlying
EDIpack solver library and its internal state. Only one instance of this class
can exist at a time.

.. autoclass:: EDIpackSolver
   :members:
   :exclude-members: chi2_fit_bath

Objects representing bath parameters
------------------------------------

.. py:module:: edipack2triqs.bath

These classes represent collections of bath parameters used by EDIpack.
Each of the classes corresponds to one bath geometry (*normal*, *hybrid* or
*general*).

.. autoclass:: BathNormal
   :members:
   :inherited-members:
.. autoclass:: BathHybrid
   :members:
   :inherited-members:
.. autoclass:: BathGeneral
   :members:
   :inherited-members:

Bath fitting
------------

.. py:module:: edipack2triqs.fit

:py:class:`BathFittingParams` is a collection of parameters used
in EDIpack's bath fitting procedure. The procedure can be invoked by calling
a method of :py:class:`edipack2triqs.solver.EDIpackSolver`.

.. autoclass:: BathFittingParams
.. automethod:: edipack2triqs.solver.EDIpackSolver.chi2_fit_bath

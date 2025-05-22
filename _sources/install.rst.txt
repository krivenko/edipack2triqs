.. highlight:: bash

.. _install:

Installation
************

Release version from Anaconda
-----------------------------

The latest released version of **edipack2triqs** along with its dependencies
is available from Anaconda.

.. code::

   conda install -c conda-forge -c edipack edipack2triqs

Current development version from GitHub
---------------------------------------

#. Install `EDIpack <https://edipack.github.io/EDIpack>`_ following the
   :ref:`installation instructions <edipack:edipack_install>`.

#. Install `the Python API for EDIpack <https://edipack.github.io/EDIpack2py>`_
   using one of the :ref:`supported methods <edipack2py:edipack2py_install>`.

#. Install :ref:`TRIQS <triqslibs:welcome>` library,
   see :ref:`TRIQS installation instruction <triqslibs:triqs_install>`.

#. Clone the source code of **edipack2triqs** from GitHub and install
   it using ``pip``,

.. code::

     $ git clone https://github.com/krivenko/edipack2triqs
     $ cd edipack2triqs
     $ pip install .

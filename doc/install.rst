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

#. Install `EDIpack2 <https://edipack.github.io/EDIpack2.0>`_ following the
   :ref:`installation instructions <edipack2:edipack_install>`.

#. Install `the Python API for EDIpack2 <https://edipack.github.io/EDIpy2.0>`_
   using one of the :ref:`supported methods <edipy2:edipy_install>`.

#. Install :ref:`TRIQS <triqslibs:welcome>` library,
   see :ref:`TRIQS installation instruction <triqslibs:triqs_install>`.

#. Clone the source code of **edipack2triqs** from GitHub and install
   it using ``pip``,

.. code::

     $ git clone https://github.com/krivenko/edipack2triqs
     $ cd edipack2triqs
     $ pip install .

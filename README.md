[![Test (Ubuntu)](https://github.com/krivenko/edipack2triqs/actions/workflows/test-ubuntu.yml/badge.svg)](
https://github.com/krivenko/edipack2triqs/actions/workflows/test-ubuntu.yml)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-red)](
https://krivenko.github.io/edipack2triqs)

edipack2triqs: Compatibility layer between EDIpack2 and TRIQS
=============================================================

**edipack2triqs** is a thin compatibility layer between
**EDIpack2** (A Massively Parallel Exact Diagonalization solver for generic
Quantum Impurity problems) and **TRIQS** (Toolbox for Research on Interacting
Quantum Systems).

Copyright (c) 2024-2025, Igor Krivenko, Lorenzo Crippa

Dependencies
------------

* NumPy
* NetworkX
* mpi4py
* [edipy2 >= 2.0.8 (EDIpack2.0)](https://github.com/edipack/EDIpack2.0)
* [TRIQS 3.x.y](https://github.com/TRIQS/triqs), tested with TRIQS 3.2.1

Installation
------------

```bash
git clone https://github.com/krivenko/edipack2triqs
cd edipack2triqs
pip install .
```

Usage examples
--------------

See https://krivenko.github.io/edipack2triqs/examples.html.

Missing features
----------------

* Phonons
* Imaginary time susceptibilities
* Sector selection
* Conversion of bath objects into operator expressions

License
-------

edipack2triqs is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

edipack2triqs is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
edipack2triqs (in the file LICENSE.txt in this directory).
If not, see <http://www.gnu.org/licenses/>.

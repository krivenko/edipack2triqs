[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-red)](
https://EDIpack.github.io/edipack2triqs)
[![Anaconda-Server Badge](https://anaconda.org/edipack/edipack2triqs/badges/version.svg)](
https://anaconda.org/edipack/edipack2triqs)
[![Test](https://github.com/EDIpack/edipack2triqs/actions/workflows/test.yml/badge.svg)](
https://github.com/EDIpack/edipack2triqs/actions/workflows/test.yml)

edipack2triqs: Compatibility layer between EDIpack and TRIQS
============================================================

**edipack2triqs** is a thin compatibility layer between
**EDIpack** (A Massively Parallel Exact Diagonalization solver for generic
Quantum Impurity problems) and **TRIQS** (Toolbox for Research on Interacting
Quantum Systems).

Copyright (c) 2024-2026, Igor Krivenko, Lorenzo Crippa

Dependencies
------------

* NumPy
* NetworkX
* mpi4py
* [edipack2py >= 6.0.0](https://github.com/EDIpack/EDIpack2py)
* [TRIQS >= 4.0.0](https://github.com/TRIQS/triqs)

Installation
------------

**Via [Conda](https://anaconda.org/anaconda/conda)**

```bash
conda install -c conda-forge -c edipack edipack2triqs
```

**From the source code [repository on GitHub](https://github.com/EDIpack/edipack2triqs)**

```bash
git clone https://github.com/EDIpack/edipack2triqs
cd edipack2triqs
pip install .
```

Usage examples
--------------

See https://EDIpack.github.io/edipack2triqs/examples.html.

Citing
------

Please, consider citing the [accompanying SciPost Phys. Codebases paper](
https://scipost.org/10.21468/SciPostPhysCodeb.58)
[[arXiv:2506.01363](https://arxiv.org/abs/2506.01363)], if you find this package
useful for your research.

```BibTeX
@Article{10.21468/SciPostPhysCodeb.58,
  title= {{Next-generation EDIpack: A Lanczos-based package for quantum
           impurity models featuring general broken-symmetry phases,
           flexible bath topologies and multi-platform interoperability}},
  author = {Lorenzo Crippa and Igor Krivenko and Samuele Giuli and
            Gabriele Bellomia and Alexander Kowalski and Francesco Petocchi and
            Alberto Scazzola and Markus Wallerberger and Giacomo Mazza and
            Luca de Medici and Giorgio Sangiovanni and Massimo Capone and
            Adriano Amaricci},
  journal = {SciPost Phys. Codebases},
  pages = {58},
  year = {2025},
  publisher = {SciPost},
  doi = {10.21468/SciPostPhysCodeb.58},
  url = {https://scipost.org/10.21468/SciPostPhysCodeb.58}
}
```

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

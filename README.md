# edipack2triqs: Compatibility layer between EDIpack and TRIQS

Copyright (c) 2024-2025, Igor Krivenko, Lorenzo Crippa

## Dependencies

* NumPy
* mpi4py
* [edipy2 >= 0.1.0 (EDIpack2.0)](https://github.com/aamaricci/EDIpack2.0)
* [TRIQS 3.x.y](https://triqs.github.io/triqs/latest), tested with TRIQS 3.2.1

## Installation

```bash
git clone https://github.com/krivenko/edipack2triqs
cd edipack2triqs
pip install .
```

## Missing features

* Phonons
* Imaginary time susceptibilities
* Sector selection
* Bath fitting

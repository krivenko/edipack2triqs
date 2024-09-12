# edipack2triqs: Compatibility layer between EDIpack and TRIQS

Copyright (c) 2024, Igor Krivenko

## Dependencies

* NumPy
* [edipy (EDIpack) >= 4.0.0](https://github.com/aamaricci/EDIpack)
* [TRIQS 3.x.y](https://triqs.github.io/triqs/latest), tested with TRIQS 3.2.1

## Installation

```bash
git clone https://github.com/krivenko/edipack2triqs
cd edipack2triqs
pip install .
```

## Missing features

* `replica` bath type
* Phonons
* Imaginary time susceptibilities
* Bath fitting
* Sector selection

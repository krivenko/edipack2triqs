from functools import reduce
from operator import mul
from typing import Union
from contextlib import contextmanager
import os

import numpy as np

import triqs.operators as op

IndicesType = tuple[Union[int, str], Union[int, str]]
CanonicalType = tuple[bool, IndicesType]


def is_diagonal(a: np.ndarray):
    """
    Check if matrix (rank-2 array) diagonal.
    """
    return np.array_equal(a, np.diag(np.diag(a)))


def is_spin_diagonal(h: np.ndarray):
    "Check if array is diagonal in its first two indices"
    return np.all(h[0, 1, ...] == 0) and np.all(h[1, 0, ...] == 0)


def is_spin_degenerate(h: np.ndarray):
    """
    Check if array is proportional to an identity matrix in its first two
    indices
    """
    return is_spin_diagonal(h) and \
        np.allclose(h[0, 0, ...], h[1, 1, ...], atol=1e-10)


def make_nambu(h: np.ndarray, Delta: np.ndarray):
    """
    Reinterpret first two indices of h as Nambu indices and fill the
    off-diagonal elements from Delta.
    """
    h_nambu = h.copy()
    h_nambu[1, 1, :, :] *= -1
    h_nambu[0, 1, :, :] = Delta
    h_nambu[1, 0, :, :] = np.conj(Delta.T)
    return h_nambu


def canonical2op(dag: bool, ind: IndicesType):
    """
    Return a many-body operator made out of one canonical operator
    c_dag(*ind) or c(*ind).
    """
    return op.c_dag(*ind) if dag else op.c(*ind)


def monomial2op(mon: list[CanonicalType]):
    "Return a many-body operator made out of one monomial."
    return reduce(mul, map(lambda c: canonical2op(*c), mon), op.Operator(1))


def validate_fops_up_dn(fops_up: list[IndicesType],
                        fops_dn: list[IndicesType],
                        name_fops_up: str,
                        name_fops_dn: str):
    """
    Check that two fundamental sets fops_up and fops_dn
    - do not contain repeated elements
    - have the same size
    - are disjoint
    """
    fops_up_s = set(fops_up)
    fops_dn_s = set(fops_dn)
    assert len(fops_up) == len(fops_up_s), \
        f"No repeated entries are allowed in {name_fops_up}"
    assert len(fops_dn) == len(fops_dn_s), \
        f"No repeated entries are allowed in {name_fops_dn}"
    assert len(fops_up) == len(fops_dn), \
        f"Fundamental sets {name_fops_up} and {name_fops_dn} " \
        "must be of equal size"
    assert fops_up_s.isdisjoint(fops_dn_s), \
        f"Fundamental sets {name_fops_up} and {name_fops_dn} " \
        "must be disjoint"


def spin_conjugate(OP: op.Operator,
                   fops_up: list[IndicesType],
                   fops_dn: list[IndicesType]):
    """
    Return a spin conjugate of a many-body operator OP.
    fops_up and fops_dn are fundamental sets of spin-up and spin-down operators
    respectively.
    """
    validate_fops_up_dn(fops_up, fops_dn, "fops_up", "fops_dn")

    spin_conj_map = {u: d for u, d in zip(fops_up, fops_dn)}
    spin_conj_map.update({d: u for d, u in zip(fops_dn, fops_up)})

    res = op.Operator()
    for mon, coeff in OP:
        new_mon = [(dag, spin_conj_map[tuple(ind)]) for dag, ind in mon]
        res += coeff * monomial2op(new_mon)
    return res


def normal_part(OP: op.Operator):
    """
    Return the particle number conversing part of a many-body operator OP.
    """
    res = op.Operator()
    for mon, coeff in OP:
        if sum((1 if dag else -1) for dag, ind in mon) == 0:
            res += coeff * monomial2op(mon)
    return res


def non_int_part(OP: op.Operator):
    """
    Return the non-interacting part of a many-body operator OP.
    """
    res = op.Operator()
    for mon, coeff in OP:
        if len(mon) < 3:
            res += coeff * monomial2op(mon)
    return res


@contextmanager
def chdircontext(path):
    """
    Emulates contextlib.chdir(path) from Python 3.11.
    """
    oldpwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(oldpwd)


def write_config(f, config):
    """
    Write a name-value configuration file recognized by EDIpack.
    """
    for name, value in config.items():
        if isinstance(value, bool):
            v = 'T' if value else 'F'
        elif isinstance(value, np.ndarray):
            v = ','.join(map(str, value))
        else:
            v = value
        f.write(f"{name}={v}    !\n")

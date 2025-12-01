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


def basis_op_to_matrix(OP: op.Operator,
                       nspin: int,
                       fops_bath_up: list[IndicesType],
                       fops_bath_dn: list[IndicesType]):
    """
    Transform a quadratic operator into a basis matrix.

    The two matrices returned by this function are the normal and anomalous
    components of the basis matrix.
    """
    nbath_total = len(fops_bath_up)

    fops_bath = fops_bath_up + fops_bath_dn

    # Resulting matrices
    mat_n = np.zeros((2, 2, nbath_total, nbath_total), dtype=complex, order='F')
    mat_a = np.zeros((nbath_total, nbath_total), dtype=complex, order='F')

    for mon, coeff in OP:
        if len(mon) != 2:
            raise RuntimeError(
                "Only quadratic operators are allowed "
                f"in the bath basis specification, got {OP}"
            )
        daggers = tuple(dag for dag, ind in mon)
        indices = tuple(tuple(ind) for dag, ind in mon)

        if (indices[0] not in fops_bath) or (indices[1] not in fops_bath):
            raise RuntimeError(
                f"Unexpected term {coeff * monomial2op(mon)} "
                f"in {OP}"
            )

        spin1, b1 = divmod(fops_bath.index(indices[0]), nbath_total)
        spin2, b2 = divmod(fops_bath.index(indices[1]), nbath_total)

        # Normal matrix element
        if daggers == (True, False):
            mat_n[spin1, spin2, b1, b2] = coeff

        # Anomalous matrix element
        elif daggers[0] == daggers[1]:
            if spin1 == spin2:  # Not representable in Nambu notation
                raise RuntimeError(
                    "Unexpected same-spin anomalous term "
                    f"{coeff * monomial2op(mon)}"
                )
            # Fill mat_a from creation-creation terms,
            # ignore annihilation-annihilation terms
            if daggers == (True, True):
                if spin1 == 0:
                    mat_a[b1, b2] = coeff
                else:
                    mat_a[b2, b1] = -coeff

    return mat_n, mat_a

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

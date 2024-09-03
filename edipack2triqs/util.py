from functools import reduce
from operator import mul
from typing import Union

import triqs.operators as op

IndicesType = list[Union[int, str]]
CanonicalType = tuple[bool, IndicesType]


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
        f"Lists {name_fops_up} and {name_fops_dn} must be of equal size"
    assert fops_up_s.isdisjoint(fops_dn_s), \
        f"Lists {name_fops_up} and {name_fops_dn} must be disjoint"


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

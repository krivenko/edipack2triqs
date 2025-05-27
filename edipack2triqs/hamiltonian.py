"""
Hamiltonian and its parameters
"""

from itertools import product
from dataclasses import dataclass
from typing import Union

import numpy as np

import triqs.operators as op

from .util import (is_spin_diagonal,
                   is_spin_degenerate,
                   IndicesType,
                   monomial2op,
                   normal_part,
                   spin_conjugate)
from .bath import BathNormal, BathHybrid, BathGeneral


@dataclass
class HamiltonianParams:
    """Parameters of the Hamiltonian"""

    # EDIpack exact diagonalization mode (normal, superc, nonsu2)
    ed_mode: str
    # Non-interacting part of the impurity Hamiltonian
    Hloc: np.ndarray
    # Bath parameters
    bath: Union[BathNormal, BathHybrid, BathGeneral]
    # Interaction matrix U_{ijkl}
    U: np.ndarray


def _is_density(hloc: np.ndarray):
    "Check if a given local Hamiltonian is diagonal in both spin and orbital"
    nspin = hloc.shape[0]
    norb = hloc.shape[2]
    assert hloc.shape == (nspin, nspin, norb, norb)
    for s1, s2, o1, o2 in np.ndindex(hloc.shape):
        # Skip the density terms
        if s1 == s2 and o1 == o2:
            continue
        if hloc[s1, s2, o1, o2] != 0:
            return False
    return True


def _is_density_density(U: np.ndarray):
    "Check if a given interaction matrix is of density-density type"
    assert U.ndim == 8
    norb = U.shape[0]
    assert U.shape == (norb, 2) * 4
    for o1, s1, o2, s2, o3, s3, o4, s4 in np.ndindex(U.shape):
        i1, i2, i3, i4 = (o1, s1), (o2, s2), (o3, s3), (o4, s4)
        # Skip the density-density terms
        if (i1 == i3 and i2 == i4) or (i1 == i4 and i2 == i3):
            continue
        if U[o1, s1, o2, s2, o3, s3, o4, s4] != 0:
            return False
    return True


def _make_bath(ed_mode: str,
               nspin: int,
               Hloc: np.ndarray,
               h: np.ndarray,
               V: np.ndarray,
               Delta: np.ndarray):
    """
    Make a bath parameters object.
    """

    # Can we use bath_type = 'normal'?
    if BathNormal.is_usable(Hloc, h, V, Delta):
        return BathNormal.from_hamiltonian(ed_mode, nspin, Hloc, h, V, Delta)
    # Can we use bath_type = 'hybrid'?
    elif BathHybrid.is_usable(h, Delta):
        return BathHybrid.from_hamiltonian(ed_mode, nspin, Hloc, h, V, Delta)
    # Can we use bath_type = 'general'?
    else:
        try:
            return BathGeneral.from_hamiltonian(ed_mode,
                                                nspin,
                                                Hloc,
                                                h,
                                                V,
                                                Delta)
        except RuntimeError:
            raise RuntimeError(
                "Cannot find a suitable bath mode for the given Hamiltonian"
            )


def parse_hamiltonian(hamiltonian: op.Operator,  # noqa: C901
                      fops_imp_up: list[IndicesType],
                      fops_imp_dn: list[IndicesType],
                      fops_bath_up: list[IndicesType],
                      fops_bath_dn: list[IndicesType]) -> HamiltonianParams:
    """
    Parse a given Hamiltonian and extract parameters from it.
    """

    if not (hamiltonian - op.dagger(hamiltonian)).is_zero():
        raise RuntimeError("Hamiltonian is not Hermitian")

    fops_imp = fops_imp_up + fops_imp_dn
    fops_bath = fops_bath_up + fops_bath_dn

    assert set(fops_imp).isdisjoint(set(fops_bath)), \
        "All fundamental sets must be disjoint"

    norb = len(fops_imp_up)
    nbath_total = len(fops_bath_up)

    # Coefficients Hloc[spin1, spin2, orb1, orb2] in front of
    # d^+(spin1, orb1) d(spin2, orb2)
    Hloc = np.zeros((2, 2, norb, norb), dtype=complex)
    # Coefficients h[spin1, spin2, b1, b2] in front of
    # a^+(spin1, b1) a(spin2, b2)
    h = np.zeros((2, 2, nbath_total, nbath_total), dtype=complex)
    # Coefficients V[spin1, spin2, orb, b] in front of
    # d^+(spin1, orb) a(spin2, b)
    V = np.zeros((2, 2, norb, nbath_total))
    # Coefficients \Delta[b1, b2] in front of c^+(up, b1) c^+(dn, b2)
    Delta = np.zeros((nbath_total, nbath_total), dtype=complex)
    # Coefficients U[orb1, spin1, orb2, spin2, orb3, spin3, orb4, spin4]
    # in front of
    # (1/2) c^+(spin1, orb1) c^+(spin2, orb2) c(spin4, orb4) c(spin3, orb3)
    U = np.zeros((norb, 2) * 4, dtype=float)

    for mon, coeff in hamiltonian:
        # Skipping an irrelevant constant term
        if len(mon) == 0:
            continue

        daggers = [dag for dag, ind in mon]
        indices = [tuple(ind) for dag, ind in mon]

        # U(1)-symmetric quadratic term
        if daggers == [True, False]:
            # d^+ d
            if (indices[0] in fops_imp) and (indices[1] in fops_imp):
                spin1, orb1 = divmod(fops_imp.index(indices[0]), norb)
                spin2, orb2 = divmod(fops_imp.index(indices[1]), norb)
                Hloc[spin1, spin2, orb1, orb2] = coeff
            # d^+ a
            elif (indices[0] in fops_imp) and (indices[1] in fops_bath):
                spin1, orb = divmod(fops_imp.index(indices[0]), norb)
                spin2, b = divmod(fops_bath.index(indices[1]), nbath_total)
                V[spin1, spin2, orb, b] = coeff
            # a^+ d
            elif (indices[0] in fops_bath) and (indices[1] in fops_imp):
                continue
            # a^+ a
            elif (indices[0] in fops_bath) and (indices[1] in fops_bath):
                spin1, b1 = divmod(fops_bath.index(indices[0]), nbath_total)
                spin2, b2 = divmod(fops_bath.index(indices[1]), nbath_total)
                h[spin1, spin2, b1, b2] = coeff
            else:
                raise RuntimeError(
                    f"Unexpected quadratic term {coeff * monomial2op(mon)}"
                )

        # U(1)-symmetric quartic term
        elif daggers == [True, True, False, False]:
            try:
                spin1, orb1 = divmod(fops_imp.index(indices[0]), norb)
                spin2, orb2 = divmod(fops_imp.index(indices[1]), norb)
                spin3, orb3 = divmod(fops_imp.index(indices[2]), norb)
                spin4, orb4 = divmod(fops_imp.index(indices[3]), norb)
            except ValueError:
                raise RuntimeError(
                    f"Unexpected interaction term {coeff * monomial2op(mon)}"
                )

            if coeff.imag != 0:
                raise RuntimeError(
                    "Unsupported complex interaction term "
                    f"{coeff * monomial2op(mon)}"
                )

            U[orb1, spin1, orb2, spin2, orb4, spin4, orb3, spin3] = 0.5 * coeff
            U[orb1, spin1, orb2, spin2, orb3, spin3, orb4, spin4] = -0.5 * coeff
            U[orb2, spin2, orb1, spin1, orb4, spin4, orb3, spin3] = -0.5 * coeff
            U[orb2, spin2, orb1, spin1, orb3, spin3, orb4, spin4] = 0.5 * coeff

        # Anomalous term creation-creation
        elif daggers == [True, True]:
            if (indices[0] in fops_bath) and (indices[1] in fops_bath):
                spin1, b1 = divmod(fops_bath.index(indices[0]), nbath_total)
                spin2, b2 = divmod(fops_bath.index(indices[1]), nbath_total)
                if spin1 == spin2:  # Not representable in Nambu notation
                    term = coeff * monomial2op(mon)
                    raise RuntimeError(
                        f"Unexpected same-spin anomalous term {term}"
                    )
                if spin1 == 0:
                    Delta[b1, b2] = coeff
                else:
                    Delta[b2, b1] = -coeff
            else:
                term = coeff * monomial2op(mon)
                raise RuntimeError(f"Unexpected anomalous term {term}")

        # Anomalous term annihilation-annihilation
        elif daggers == [False, False]:
            if (indices[0] in fops_bath) and (indices[1] in fops_bath):
                spin1, b1 = divmod(fops_bath.index(indices[0]), nbath_total)
                spin2, b2 = divmod(fops_bath.index(indices[1]), nbath_total)
                if spin1 == spin2:  # Not representable in Nambu notation
                    raise RuntimeError(
                        f"Unexpected same-spin anomalous term {term}"
                    )
                continue
            else:
                raise RuntimeError(f"Unexpected anomalous term {term}")

        else:
            raise RuntimeError(
                f"Unsupported Hamiltonian term {coeff * monomial2op(mon)}"
            )

    hamiltonian_n = normal_part(hamiltonian)
    hamiltonian_n_conj = spin_conjugate(
        hamiltonian_n, fops_imp_up + fops_bath_up, fops_imp_dn + fops_bath_dn
    )
    nspin = 1 if (hamiltonian_n_conj - hamiltonian_n).is_zero() else 2

    if nspin == 1:
        # Internal consistency check: Hloc, h and V must be spin-degenerate
        assert is_spin_degenerate(Hloc)
        assert is_spin_degenerate(h)
        assert is_spin_degenerate(V)
        if (Delta == 0).all():
            ed_mode = "normal"
        else:
            ed_mode = "superc"
    else:  # nspin == 2
        if not (Delta == 0).all():
            raise RuntimeError(
                "Magnetism in presence of a superconducting bath "
                "is not supported"
            )
        if is_spin_diagonal(Hloc) and \
           is_spin_diagonal(h) and is_spin_diagonal(V):
            ed_mode = "normal"
        else:
            ed_mode = "nonsu2"

    params = HamiltonianParams(
        ed_mode,
        Hloc=np.zeros((nspin, nspin, norb, norb), dtype=complex, order='F'),
        bath=_make_bath(ed_mode, nspin, Hloc, h, V, Delta),
        U=U
    )

    for spin1, spin2 in product(range(nspin), range(nspin)):
        params.Hloc[spin1, spin2, ...] = Hloc[spin1, spin2, ...]

    return params

"""
Hamiltonian and its parameters
"""

from itertools import product
from dataclasses import dataclass, field
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


def default_Uloc():
    return np.array([2.0, 0, 0, 0, 0])


@dataclass
class HamiltonianParams:
    """Parameters of the Hamiltonian"""

    # EDIpack exact diagonalization mode (normal, superc, nonsu2)
    ed_mode: str
    # Non-interacting part of the impurity Hamiltonian
    Hloc: np.ndarray
    # Bath parameters
    bath: Union[BathNormal, BathHybrid, BathGeneral]
    # Local intra-orbital interactions U (one value per orbital)
    Uloc: np.ndarray = field(default_factory=default_Uloc)
    # Local inter-orbital interaction U'
    Ust: float = 0
    # Hund's coupling
    Jh: float = 0
    # Spin-exchange coupling constant
    Jx: float = 0
    # Pair-hopping coupling constant
    Jp: float = 0


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
    if BathNormal.is_usable(Hloc, h, V):
        return BathNormal(ed_mode, nspin, Hloc, h, V, Delta)
    # Can we use bath_type = 'hybrid'?
    elif BathHybrid.is_usable(h):
        return BathHybrid(ed_mode, nspin, Hloc, h, V, Delta)
    # Can we use bath_type = 'general'?
    else:
        try:
            return BathGeneral(ed_mode, nspin, Hloc, h, V, Delta)
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
    # TODO
    Delta = np.zeros((2, nbath_total))

    Uloc = np.zeros(5, dtype=float)
    Ust, UstmJ = [], []
    Jx, Jp = [], []

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

            # A density-density interaction
            if (spin1, orb1) == (spin4, orb4) and \
               (spin2, orb2) == (spin3, orb3):
                # Interaction with different spins
                if spin1 != spin2:
                    # Intra-orbital
                    if orb1 == orb2:
                        Uloc[orb1] = coeff
                    # Inter-orbital
                    else:
                        Ust.append(coeff)
                # Interaction with the same spin
                else:
                    UstmJ.append(coeff)

            # A non-density-density interaction
            else:
                # Pair-hopping
                if (orb1 == orb2) and (orb3 == orb4):
                    Jp.append(coeff if spin2 == spin3 else -coeff)
                # Spin-exchange
                elif (spin1 == spin4) and (spin2 == spin3) and \
                     (orb1 == orb3) and (orb2 == orb4):
                    Jx.append(coeff)
                elif (spin1 == spin3) and (spin2 == spin4) and \
                     (orb1 == orb4) and (orb2 == orb3):
                    Jx.append(-coeff)
                else:
                    term = coeff * monomial2op(mon)
                    raise RuntimeError(f"Unexpected interaction term {term}")

        # Anomalous term creation-creation
        elif daggers == [True, True]:
            if (indices[0] in fops_bath) and (indices[1] in fops_bath):
                spin1, b1 = divmod(fops_bath.index(indices[0]), nbath_total)
                spin2, b2 = divmod(fops_bath.index(indices[1]), nbath_total)
                if b1 == b2:
                    # \Delta c^+_dn c^+_up
                    Delta[0, b1] = (1 if spin1 == 1 else -1) * coeff
                else:
                    raise RuntimeError(
                        f"Unexpected off-diagonal anomalous bath term {term}"
                    )
            else:
                raise RuntimeError(f"Unexpected anomalous term {term}")

        # Anomalous term annihilation-annihilation
        elif daggers == [False, False]:
            if (indices[0] in fops_bath) and (indices[1] in fops_bath):
                spin1, b1 = divmod(fops_bath.index(indices[0]), nbath_total)
                spin2, b2 = divmod(fops_bath.index(indices[1]), nbath_total)
                if b1 == b2:
                    # \Delta^* c_up c_dn
                    Delta[1, b1] = (1 if spin1 == 0 else -1) * coeff
                else:
                    raise RuntimeError(
                        f"Unexpected off-diagonal anomalous bath term {term}"
                    )
            else:
                raise RuntimeError(f"Unexpected anomalous term {term}")

        else:
            raise RuntimeError(
                f"Unsupported Hamiltonian term {coeff * monomial2op(mon)}"
            )

    def all_close(vals):
        return all(np.isclose(v, vals[0], atol=1e-10) for v in vals)

    assert all_close(Ust), \
        "Inconsistent values of U' for different pairs of orbitals"
    assert all_close(UstmJ), \
        "Inconsistent values of U' - J for different pairs of orbitals"
    assert all_close(Jx), \
        "Inconsistent values of J_X for different pairs of orbitals"
    assert all_close(Jp), \
        "Inconsistent values of J_P for different pairs of orbitals"

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
        Uloc=Uloc,
        Ust=Ust[0] if len(Ust) > 0 else .0,
        Jx=Jx[0] if len(Jx) > 0 else .0,
        Jp=Jp[0] if len(Jp) > 0 else .0
    )
    params.Jh = -(UstmJ[0] if len(UstmJ) > 0 else .0) + params.Ust

    for spin1, spin2 in product(range(nspin), range(nspin)):
        params.Hloc[spin1, spin2, ...] = Hloc[spin1, spin2, ...]

    return params

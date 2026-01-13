"""
Hamiltonian and its parameters
"""

from itertools import product
from dataclasses import dataclass
from types import NoneType
from typing import Union, Optional

import numpy as np

import triqs.operators as op

from . import EDMode
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

    # Exact diagonalization mode
    ed_mode: EDMode
    # Non-interacting part of the impurity Hamiltonian
    Hloc: np.ndarray
    # Anomalous part of the impurity Hamiltonian
    Hloc_an: np.ndarray
    # Bath object (None if no bath is present)
    bath: Union[BathNormal, BathHybrid, BathGeneral, NoneType]
    # Interaction matrix U_{ijkl}
    U: np.ndarray

    def Hloc_op(self,
                fops_imp_up: list[IndicesType],
                fops_imp_dn: list[IndicesType]) -> op.Operator:
        "Hloc as a TRIQS many-body operator object"
        fops = (fops_imp_up, fops_imp_dn)
        h = op.Operator()
        hloc_it = np.nditer(self.Hloc, flags=['multi_index'])
        if self.Hloc.shape[0] == 1:  # nspin == 1
            for coeff in hloc_it:
                orb1, orb2 = hloc_it.multi_index[2:]
                h += coeff * op.c_dag(*fops[0][orb1]) * op.c(*fops[0][orb2])
                h += coeff * op.c_dag(*fops[1][orb1]) * op.c(*fops[1][orb2])
        else:  # nspin == 2
            for coeff in hloc_it:
                s1, s2, orb1, orb2 = hloc_it.multi_index
                h += coeff * op.c_dag(*fops[s1][orb1]) * op.c(*fops[s2][orb2])
        return h

    def Hloc_an_op(self,
                   fops_imp_up: list[IndicesType],
                   fops_imp_dn: list[IndicesType]) -> op.Operator:
        "Hloc_an as a TRIQS many-body operator object"
        h = op.Operator()
        hloc_an_it = np.nditer(self.Hloc_an, flags=['multi_index'])
        for coeff in hloc_an_it:
            orb1, orb2 = hloc_an_it.multi_index[2:]
            h += coeff * \
                op.c_dag(*fops_imp_up[orb1]) * op.c_dag(*fops_imp_dn[orb2])
            h += np.conj(coeff) * \
                op.c(*fops_imp_dn[orb2]) * op.c(*fops_imp_up[orb1])
        return h


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


def _make_bath(ed_mode: EDMode,
               nspin: int,
               Hloc: np.ndarray,
               Hloc_an: np.ndarray,
               h: np.ndarray,
               V: np.ndarray,
               Delta: np.ndarray):
    """
    Make a bath parameters object.
    """

    # Can we use bath_type = 'normal'?
    if BathNormal.is_usable(Hloc, Hloc_an, h, V, Delta):
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


def _raise_term_error(msg, mon, coeff):
    "Raise a runtime error about an unexpected Hamiltonian term"
    term = coeff * monomial2op(mon)
    raise RuntimeError(msg.format(term))


def extract_quadratic(h: op.Operator,
                      fops_up: list[IndicesType],
                      fops_dn: list[IndicesType],
                      ignore_unexpected: bool = True) \
        -> tuple[np.ndarray, np.ndarray]:
    """
    Extract matrices of the normal and anomalous quadratic contributions to a
    given Hamiltonian 'h'. An unexpected term in 'h' results in an exception if
    `ignore_unexpected=False`.
    """
    fops = fops_up + fops_dn
    dim = len(fops_up)

    M = np.zeros((2, 2, dim, dim), dtype=complex)
    M_an = np.zeros((1, 1, dim, dim), dtype=complex)

    for mon, coeff in h:
        if len(mon) != 2:
            if ignore_unexpected:
                continue
            else:
                _raise_term_error("Unexpected non-quadratic term {}",
                                  mon,
                                  coeff)

        daggers = [dag for dag, ind in mon]
        indices = [tuple(ind) for dag, ind in mon]
        if not ((indices[0] in fops) and (indices[1] in fops)):
            if ignore_unexpected:
                continue
            else:
                _raise_term_error("Unexpected quadratic term {}", mon, coeff)

        spin1, orb1 = divmod(fops.index(indices[0]), dim)
        spin2, orb2 = divmod(fops.index(indices[1]), dim)

        # Normal quadratic term
        if daggers == [True, False]:
            M[spin1, spin2, orb1, orb2] = coeff
        # Anomalous term
        elif daggers[0] == daggers[1]:
            if spin1 == spin2:  # Not representable in Nambu notation
                if ignore_unexpected:
                    continue
                else:
                    _raise_term_error("Unexpected same-spin anomalous term {}",
                                      mon,
                                      coeff)
            # Creation-creation term
            if daggers[0]:
                if spin1 == 0:
                    M_an[0, 0, orb1, orb2] = coeff
                else:
                    M_an[0, 0, orb2, orb1] = -coeff

    return M, M_an


def parse_hamiltonian(hamiltonian: op.Operator,  # noqa: C901
                      fops_imp_up: list[IndicesType],
                      fops_imp_dn: list[IndicesType],
                      fops_bath_up: list[IndicesType],
                      fops_bath_dn: list[IndicesType],
                      f_ed_mode: Optional[EDMode] = None) -> HamiltonianParams:
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

    def is_imp(index):
        return index in fops_imp

    def is_bath(index):
        return index in fops_bath

    def get_spin_orb(index):
        return divmod(fops_imp.index(index), norb)

    def get_spin_b(index):
        return divmod(fops_bath.index(index), nbath_total)

    # Coefficients Hloc[spin1, spin2, orb1, orb2] in front of
    # d^+(spin1, orb1) d(spin2, orb2)
    Hloc = np.zeros((2, 2, norb, norb), dtype=complex)
    # Coefficients Hloc_anomalous[orb1, orb2]
    # in front of c^+(up, orb1) c^+(dn, orb2)
    # Hloc_anomalous[orb1, orb2]* is the coefficient in front of
    # c(dn, orb2) c(up, orb1)
    Hloc_an = np.zeros((1, 1, norb, norb), dtype=complex)
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
            if is_imp(indices[0]) and is_imp(indices[1]):
                spin1, orb1 = get_spin_orb(indices[0])
                spin2, orb2 = get_spin_orb(indices[1])
                Hloc[spin1, spin2, orb1, orb2] = coeff
            # d^+ a
            elif is_imp(indices[0]) and is_bath(indices[1]):
                spin1, orb = get_spin_orb(indices[0])
                spin2, b = get_spin_b(indices[1])
                V[spin1, spin2, orb, b] = coeff
            # a^+ d
            elif is_bath(indices[0]) and is_imp(indices[1]):
                continue
            # a^+ a
            elif is_bath(indices[0]) and is_bath(indices[1]):
                spin1, b1 = get_spin_b(indices[0])
                spin2, b2 = get_spin_b(indices[1])
                h[spin1, spin2, b1, b2] = coeff
            else:
                _raise_term_error("Unexpected quadratic term {}", mon, coeff)

        # U(1)-symmetric quartic term
        elif daggers == [True, True, False, False]:
            try:
                spin1, orb1 = get_spin_orb(indices[0])
                spin2, orb2 = get_spin_orb(indices[1])
                spin3, orb3 = get_spin_orb(indices[2])
                spin4, orb4 = get_spin_orb(indices[3])
            except ValueError:
                _raise_term_error("Unexpected interaction term {}", mon, coeff)

            if coeff.imag != 0:
                _raise_term_error("Unsupported complex interaction term {}",
                                  mon,
                                  coeff)

            U[orb1, spin1, orb2, spin2, orb4, spin4, orb3, spin3] = 0.5 * coeff
            U[orb1, spin1, orb2, spin2, orb3, spin3, orb4, spin4] = -0.5 * coeff
            U[orb2, spin2, orb1, spin1, orb4, spin4, orb3, spin3] = -0.5 * coeff
            U[orb2, spin2, orb1, spin1, orb3, spin3, orb4, spin4] = 0.5 * coeff

        # Anomalous term creation-creation
        elif daggers == [True, True]:
            if is_bath(indices[0]) and is_bath(indices[1]):
                spin1, b1 = get_spin_b(indices[0])
                spin2, b2 = get_spin_b(indices[1])
                if spin1 == spin2:  # Not representable in Nambu notation
                    _raise_term_error("Unexpected same-spin anomalous term {}",
                                      mon,
                                      coeff)
                if spin1 == 0:
                    Delta[b1, b2] = coeff
                else:
                    Delta[b2, b1] = -coeff
            elif is_imp(indices[0]) and is_imp(indices[1]):
                spin1, orb1 = get_spin_orb(indices[0])
                spin2, orb2 = get_spin_orb(indices[1])
                if spin1 == spin2:  # Not representable in Nambu notation
                    _raise_term_error("Unexpected same-spin anomalous term {}",
                                      mon,
                                      coeff)
                if spin1 == 0:
                    Hloc_an[0, 0, orb1, orb2] = coeff
                else:
                    Hloc_an[0, 0, orb2, orb1] = -coeff
            else:
                _raise_term_error("Unexpected anomalous term {}", mon, coeff)

        # Anomalous term annihilation-annihilation
        elif daggers == [False, False]:
            if is_bath(indices[0]) and is_bath(indices[1]):
                spin1, b1 = get_spin_b(indices[0])
                spin2, b2 = get_spin_b(indices[1])
                if spin1 == spin2:  # Not representable in Nambu notation
                    _raise_term_error("Unexpected same-spin anomalous term {}",
                                      mon,
                                      coeff)
                continue
            elif is_imp(indices[0]) and is_imp(indices[1]):
                spin1, orb1 = get_spin_orb(indices[0])
                spin2, orb2 = get_spin_orb(indices[1])
                if spin1 == spin2:  # Not representable in Nambu notation
                    _raise_term_error("Unexpected same-spin anomalous term {}",
                                      mon,
                                      coeff)
                continue
            else:
                _raise_term_error("Unexpected anomalous term {}", mon, coeff)

        else:
            _raise_term_error("Unsupported Hamiltonian term {}", mon, coeff)

    hamiltonian_n = normal_part(hamiltonian)
    hamiltonian_n_conj = spin_conjugate(
        hamiltonian_n, fops_imp_up + fops_bath_up, fops_imp_dn + fops_bath_dn
    )
    nspin = 1 if (hamiltonian_n_conj - hamiltonian_n).is_zero() else 2

    # EDIpack does not seem to reliably work when \Delta is asymmetric
    # See https://github.com/EDIpack/EDIpack/issues/35#issuecomment-3637501715
    if (Delta != Delta.T).any():
        raise RuntimeError(
            "All pairing terms between bath states b and b' must be symmetric "
            "under the index swap b <-> b'"
        )
    # Same for the anomalous local Hamiltonian
    if (Hloc_an[0, 0] != Hloc_an[0, 0].T).any():
        raise RuntimeError(
            "All pairing terms between impurity orbitals o and o' must be "
            "symmetric under the index swap o <-> o'"
        )

    # ed_mode selection
    superc = (Delta != 0).any() or (Hloc_an != 0).any()
    if nspin == 1:
        # Internal consistency check: Hloc, h and V must be spin-degenerate
        assert is_spin_degenerate(Hloc)
        assert is_spin_degenerate(h)
        assert is_spin_degenerate(V)
        ed_mode = EDMode.SUPERC if superc else EDMode.NORMAL
    else:  # nspin == 2
        if superc:
            raise RuntimeError(
                "Magnetism in presence of pairing terms is not supported"
            )
        if is_spin_diagonal(Hloc) and \
           is_spin_diagonal(h) and is_spin_diagonal(V):
            ed_mode = EDMode.NORMAL
        else:
            ed_mode = EDMode.NONSU2

    # Check if the forced ED mode is compatible with the deduced one
    if (f_ed_mode is not None) and (ed_mode != f_ed_mode):
        if ed_mode == EDMode.NORMAL:
            ed_mode = f_ed_mode
            if f_ed_mode == EDMode.NONSU2:
                nspin = 2
            elif (f_ed_mode == EDMode.SUPERC and nspin != 1):
                raise RuntimeError(
                    "Requested exact diagonalization mode EDMode.SUPERC "
                    "requires a spin-degenerate Hamiltonian"
                )

        else:
            raise RuntimeError(
                f"Requested exact diagonalization mode {f_ed_mode} "
                f"is incompatible with the Hamiltonian (must be {ed_mode})"
            )

    bath = _make_bath(ed_mode, nspin, Hloc, Hloc_an, h, V, Delta) \
        if nbath_total > 0 else None
    params = HamiltonianParams(
        ed_mode,
        Hloc=np.zeros((nspin, nspin, norb, norb), dtype=complex, order='F'),
        Hloc_an=Hloc_an,
        bath=bath,
        U=U
    )

    for spin1, spin2 in product(range(nspin), range(nspin)):
        params.Hloc[spin1, spin2, ...] = Hloc[spin1, spin2, ...]

    return params

"""
Hamiltonian and its parameters
"""

from itertools import product
from dataclasses import dataclass, field
from typing import Union

import numpy as np

import triqs.operators as op

from .util import (is_diagonal,
                   IndicesType,
                   monomial2op,
                   spin_conjugate)


class BathNormal:
    """Parameters of a bath with normal topology"""

    # EDIpack bath type
    name: str = 'normal'

    def __init__(self,
                 ed_mode: str,
                 nspin: int,
                 Hloc: np.ndarray,
                 h: np.ndarray,
                 V: np.ndarray):
        norb = Hloc.shape[2]
        nbath_total = h.shape[1]
        # Number of bath sites
        self.nbath = nbath_total // norb

        size = nspin * norb * self.nbath

        # EDIpack-compatible bath parameter array
        self.data = np.zeros(size * (2 if ed_mode == "normal" else 3),
                             dtype=float)

        # Energy levels view
        self.eps = self.data[:size].reshape((nspin, norb, self.nbath))
        assert not self.eps.flags['OWNDATA']
        # Same-spin hopping amplitudes view
        self.V = self.data[size:2 * size].reshape((nspin, norb, self.nbath))
        assert not self.V.flags['OWNDATA']

        if ed_mode == "nonsu2":
            # Spin-flip hopping amplitudes view
            self.U = self.data[2 * size:].reshape((nspin, norb, self.nbath))
            assert not self.U.flags['OWNDATA']
        else:
            pass  # TODO: superc

        for spin1, spin2 in product(range(nspin), range(nspin)):
            # Lists of bath states coupled to each impurity orbital
            bs = [[] for orb in range(norb)]
            # List of bath states decoupled from the impurity
            dec_bs = []
            for b in range(nbath_total):
                orbs = np.nonzero(V[spin1, spin2, :, b])[0]
                (bs[orbs[0]] if (len(orbs) != 0) else dec_bs).append(b)
            for orb in range(norb):
                # Assign the decoupled bath states to some orbitals
                n_missing_states = self.nbath - len(bs[orb])
                for _ in range(n_missing_states):
                    bs[orb].append(dec_bs.pop(0))
                # Fill the parameters
                for nu, b in enumerate(bs[orb]):
                    if spin1 == spin2:
                        self.eps[spin1, orb, nu] = h[spin1, b, b]
                        self.V[spin1, orb, nu] = V[spin1, spin2, orb, b]
                    elif ed_mode == "nonsu2":
                        self.U[spin1, orb, nu] = V[spin1, spin2, orb, b]


class BathHybrid:
    """Parameters of a bath with hybrid topology"""

    # EDIpack bath type
    name: str = 'hybrid'

    def __init__(self,
                 ed_mode: str,
                 nspin: int,
                 Hloc: np.ndarray,
                 h: np.ndarray,
                 V: np.ndarray):
        norb = Hloc.shape[2]
        self.nbath = h.shape[1]

        eps_size = nspin * self.nbath
        size = eps_size * norb

        # EDIpack-compatible bath parameter array
        self.data = np.zeros(
            eps_size + size * (1 if ed_mode == "normal" else 2),
            dtype=float)

        # Energy levels view
        self.eps = self.data[:eps_size].reshape(nspin, self.nbath)
        assert not self.eps.flags['OWNDATA']
        # Same-spin hopping amplitudes view
        self.V = self.data[eps_size:eps_size + size].reshape(
            (nspin, norb, self.nbath)
        )
        assert not self.V.flags['OWNDATA']

        if ed_mode == "nonsu2":
            # Spin-flip hopping amplitudes view
            self.U = self.data[eps_size + size:].reshape(
                (nspin, norb, self.nbath)
            )
            assert not self.U.flags['OWNDATA']
        else:
            pass  # TODO: superc

        for spin1, spin2, nu in product(range(nspin),
                                        range(nspin),
                                        range(self.nbath)):
            if spin1 == spin2:
                self.eps[spin1, nu] = h[spin1, nu, nu]
                self.V[spin1, :, nu] = V[spin1, spin2, :, nu]
            elif ed_mode == "nonsu2":
                self.U[spin1, :, nu] = V[spin1, spin2, :, nu]


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
    bath: Union[BathNormal, BathHybrid]  # TODO: BathReplica
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
               V: np.ndarray):
    """
    Make a bath parameters object.
    """

    norb = Hloc.shape[2]
    nbath_total = h.shape[1]  # Total number of bath states

    # Can we use bath_type = 'normal'?
    # - The total number of bath states must be a multiple of norb
    # - All spin components of Hloc must be diagonal
    # - All spin components of h must be diagonal
    # - Each bath state is coupled to one impurity orbital
    # - Each impurity orbital is coupled to at most nbath_total/norb bath states
    if (nbath_total % norb == 0) and \
       all(is_diagonal(Hloc[spin1, spin2, ...])
           for spin1, spin2 in product(range(2), range(2))) and \
       all(is_diagonal(h[spin, ...]) for spin in range(2)) and \
       (np.count_nonzero(V, axis=2) <= 1).all() and \
       (np.count_nonzero(V, axis=3) <= (nbath_total // norb)).all():
        return BathNormal(ed_mode, nspin, Hloc, h, V)

    # Can we use bath_type = 'hybrid'?
    # - All spin components of h must be diagonal
    elif all(is_diagonal(h[spin, ...]) for spin in range(2)):
        return BathHybrid(ed_mode, nspin, Hloc, h, V)

    # Can we use bath_type = 'replica'
    # - The total number of bath states must be a multiple of norb
    # elif False: #(bath_size % norb == 0):
    #    params.Nbath = bath_size // norb
    #    params.bath_type = "replica"
    #    # TODO: Set params.bath
    else:
        raise RuntimeError(
            "Cannot find a suitable bath mode for the given Hamiltonian"
        )


def _is_spin_diagonal(h: np.ndarray):
    "Check if array is diagonal in its first two indices"
    return np.all(h[0, 1, ...] == 0) and np.all(h[1, 0, ...] == 0)


def _is_spin_degenerate(h: np.ndarray):
    """
    Check if array is proportional to an identity matrix in its first two
    indices
    """
    return _is_spin_diagonal(h) and \
        np.allclose(h[0, 0, ...], h[1, 1, ...], atol=1e-10)


def parse_hamiltonian(hamiltonian: op.Operator,
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

    hamiltonian_conj = spin_conjugate(hamiltonian,
                                      fops_imp_up + fops_bath_up,
                                      fops_imp_dn + fops_bath_dn)
    nspin = 1 if (hamiltonian_conj - hamiltonian).is_zero() else 2

    Hloc = np.zeros((2, 2, norb, norb), dtype=complex)
    h = np.zeros((2, nbath_total, nbath_total))
    V = np.zeros((2, 2, norb, nbath_total))

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
                if spin1 != spin2:
                    raise RuntimeError("Spin non-diagonal h is not supported")
                h[spin1, b1, b2] = coeff
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

    if nspin == 1:
        # Internal consistency check: Hloc, h and V must be spin-degenerate
        assert _is_spin_degenerate(Hloc)
        assert np.allclose(h[0, ...], h[1, ...], atol=1e-10)
        assert _is_spin_degenerate(V)
        ed_mode = "normal"
    else:  # nspin == 2
        ed_mode = "normal" if \
            (_is_spin_diagonal(Hloc) and _is_spin_diagonal(V)) \
            else "nonsu2"
        # TODO: superc

    params = HamiltonianParams(
        ed_mode,
        Hloc=np.zeros((nspin, nspin, norb, norb), dtype=complex, order='F'),
        bath=_make_bath(ed_mode, nspin, Hloc, h, V),
        Uloc=Uloc,
        Ust=Ust[0] if len(Ust) > 0 else .0,
        Jx=Jx[0] if len(Jx) > 0 else .0,
        Jp=Jp[0] if len(Jp) > 0 else .0
    )
    params.Jh = -(UstmJ[0] if len(UstmJ) > 0 else .0) + params.Ust

    for spin1, spin2 in product(range(nspin), range(nspin)):
        params.Hloc[spin1, spin2, ...] = Hloc[spin1, spin2, ...]

    return params

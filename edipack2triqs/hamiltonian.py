"""
Hamiltonian and its parameters
"""

from itertools import product
from dataclasses import dataclass, field
from typing import Union

import numpy as np
import networkx as nx

import triqs.operators as op

from .util import (is_diagonal,
                   IndicesType,
                   monomial2op,
                   normal_part,
                   spin_conjugate)


def _bath_states_to_orbs(V: np.ndarray):
    """
    For each bath state, find all impurity orbitals it is connected to by
    a hopping amplitude matrix 'V'.
    """
    # np.unique() removes repeated orbitals possibly introduced by multiple
    # combinations of spin indices
    nbath_total = V.shape[3]
    return [list(np.unique(np.nonzero(V[:, :, :, b])[2]))
            for b in range(nbath_total)]


def _orbs_to_bath_states(V: np.ndarray):
    """
    For each impurity orbital, find all bath states it is connected to by
    a hopping amplitude matrix 'V'.
    """
    # np.unique() removes repeated bath states possibly introduced by multiple
    # combinations of spin indices
    norb = V.shape[2]
    return [list(np.unique(np.nonzero(V[:, :, orb, :])[2]))
            for orb in range(norb)]


class BathNormal:
    """Parameters of a bath with normal topology"""

    # EDIpack bath type
    name: str = 'normal'

    @classmethod
    def is_usable(cls, Hloc: np.ndarray, h: np.ndarray, V: np.ndarray):
        norb = Hloc.shape[2]
        nbath_total = h.shape[2]  # Total number of bath states

        # - The total number of bath states must be a multiple of norb
        # - All spin components of Hloc must be diagonal
        # - h must be spin-diagonal
        # - All spin components of h must be diagonal
        # - Each bath state is coupled to at most one impurity orbital
        # - Each impurity orbital is coupled to at most nbath_total/norb
        #   bath states
        return (nbath_total % norb == 0) and \
            all(is_diagonal(Hloc[spin1, spin2, ...])
                for spin1, spin2 in product(range(2), repeat=2)) and \
            _is_spin_diagonal(h) and \
            all(is_diagonal(h[spin, spin, ...]) for spin in range(2)) and \
            all(len(orbs) <= 1 for orbs in _bath_states_to_orbs(V)) and \
            all(len(bs) <= (nbath_total // norb)
                for bs in _orbs_to_bath_states(V))

    def __init__(self,
                 ed_mode: str,
                 nspin: int,
                 Hloc: np.ndarray,
                 h: np.ndarray,
                 V: np.ndarray,
                 Delta: np.ndarray):
        norb = Hloc.shape[2]
        nbath_total = h.shape[2]
        # Number of bath sites
        self.nbath = nbath_total // norb

        size = nspin * norb * self.nbath

        # EDIpack-compatible bath parameter array
        self.data = np.zeros(size * (2 if ed_mode == "normal" else 3),
                             dtype=float)

        params_shape = (nspin, norb, self.nbath)

        # View: Energy levels
        self.eps = self.data[:size].reshape(params_shape)
        assert not self.eps.flags['OWNDATA']

        if ed_mode == "nonsu2":
            # View: Same-spin hopping amplitudes
            self.V = self.data[size:2 * size].reshape(params_shape)
            assert not self.V.flags['OWNDATA']
            # View: Spin-flip hopping amplitudes
            self.U = self.data[2 * size:].reshape(params_shape)
            assert not self.U.flags['OWNDATA']
        elif ed_mode == "superc":
            # View: Local SC order parameters of the bath
            self.Delta = self.data[size:2 * size].reshape(params_shape)
            assert not self.Delta.flags['OWNDATA']
            # View: Same-spin hopping amplitudes
            self.V = self.data[2 * size:].reshape(params_shape)
            assert not self.V.flags['OWNDATA']
        else:  # ed_mode == "normal"
            # View: Same-spin hopping amplitudes
            self.V = self.data[size:2 * size].reshape(params_shape)
            assert not self.V.flags['OWNDATA']

        for spin1, spin2 in product(range(nspin), repeat=2):
            # Lists of bath states coupled to each impurity orbital
            bs = [[] for orb in range(norb)]
            # List of bath states decoupled from the impurity
            dec_bs = []
            for b in range(nbath_total):
                orbs = np.flatnonzero(V[spin1, spin2, :, b])
                (bs[orbs[0]] if (len(orbs) != 0) else dec_bs).append(b)
            for orb in range(norb):
                # Assign the decoupled bath states to some orbitals
                n_missing_states = self.nbath - len(bs[orb])
                for _ in range(n_missing_states):
                    bs[orb].append(dec_bs.pop(0))
                # Fill the parameters
                for nu, b in enumerate(bs[orb]):
                    if spin1 == spin2:
                        self.eps[spin1, orb, nu] = np.real_if_close(
                            h[spin1, spin2, b, b]
                        )
                        if ed_mode == "superc":
                            self.Delta[spin1, orb, nu] = Delta[spin1, b]
                        self.V[spin1, orb, nu] = V[spin1, spin2, orb, b]
                    elif ed_mode == "nonsu2":
                        self.U[spin1, orb, nu] = V[spin1, spin2, orb, b]


class BathHybrid:
    """Parameters of a bath with hybrid topology"""

    # EDIpack bath type
    name: str = 'hybrid'

    @classmethod
    def is_usable(cls, h: np.ndarray):
        # - h must be spin-diagonal
        # - All spin components of h must be diagonal
        return _is_spin_diagonal(h) and \
            all(is_diagonal(h[spin, spin, ...]) for spin in range(2))

    def __init__(self,
                 ed_mode: str,
                 nspin: int,
                 Hloc: np.ndarray,
                 h: np.ndarray,
                 V: np.ndarray,
                 Delta: np.ndarray):
        norb = Hloc.shape[2]
        self.nbath = h.shape[2]

        eps_size = nspin * self.nbath
        size = eps_size * norb

        # EDIpack-compatible bath parameter array
        self.data = np.zeros(
            {"normal": eps_size + size,
             "superc": 2 * eps_size + size,
             "nonsu2": eps_size + 2 * size}[ed_mode],
            dtype=float)

        eps_shape = (nspin, self.nbath)
        shape = (nspin, norb, self.nbath)

        # View: Energy levels
        self.eps = self.data[:eps_size].reshape(eps_shape)
        assert not self.eps.flags['OWNDATA']

        if ed_mode == "nonsu2":
            # View: Same-spin hopping amplitudes
            self.V = self.data[eps_size:eps_size + size].reshape(shape)
            assert not self.V.flags['OWNDATA']
            # View: Spin-flip hopping amplitudes
            self.U = self.data[eps_size + size:].reshape(shape)
            assert not self.U.flags['OWNDATA']
        elif ed_mode == "superc":
            # View: Local SC order parameters of the bath
            self.Delta = self.data[eps_size:2 * eps_size].reshape(eps_shape)
            assert not self.Delta.flags['OWNDATA']
            # View: Same-spin hopping amplitudes
            self.V = self.data[2 * eps_size:].reshape(shape)
            assert not self.V.flags['OWNDATA']
        else:  # ed_mode == "normal"
            # View: Same-spin hopping amplitudes
            self.V = self.data[eps_size:].reshape(shape)
            assert not self.V.flags['OWNDATA']

        for spin1, spin2, nu in product(range(nspin),
                                        range(nspin),
                                        range(self.nbath)):
            if spin1 == spin2:
                self.eps[spin1, nu] = np.real_if_close(h[spin1, spin2, nu, nu])
                if ed_mode == "superc":
                    self.Delta[spin1, nu] = Delta[spin1, nu]
                self.V[spin1, :, nu] = V[spin1, spin2, :, nu]
            elif ed_mode == "nonsu2":
                self.U[spin1, :, nu] = V[spin1, spin2, :, nu]


class BathGeneral:
    """Parameters of a bath with general topology"""

    # EDIpack bath type
    name: str = 'general'

    @classmethod
    def is_replica_valid(cls, replica: set[int], bs2orbs: list[list[int]]):
        """
        Check that all bath states of a given replica are connected to different
        impurity orbitals (if any).
        """
        orbs = [bs2orbs[b][0] for b in replica if len(bs2orbs[b]) != 0]
        return len(set(orbs)) == len(orbs)

    @classmethod
    def merge_inc_replicas(cls,
                           inc_replicas: list[set[int]],
                           norb: int,
                           bs2orbs: list[list[int]]):
        """
        Merge incomplete replicas to form a few complete replicas of size norb.
        """
        # Number of complete replicas to form
        nreps = sum(map(len, inc_replicas)) // norb
        # Select which complete replica each incomplete replica will be part of
        irep2rep = [0] * len(inc_replicas)
        # Current size of each replica
        repsizes = [0] * nreps

        def check_replicas():
            for rep in range(nreps):
                selected_ireps = [irep for i, irep in enumerate(inc_replicas)
                                  if irep2rep[i] == rep]
                if not cls.is_replica_valid(set().union(*selected_ireps),
                                            bs2orbs):
                    return False
            return True

        def assign_irep2rep(irep):
            if irep == len(inc_replicas):
                assert repsizes == [norb] * nreps
                return check_replicas()

            for rep in range(nreps):
                irep2rep[irep] = rep
                irep_size = len(inc_replicas[irep])
                if repsizes[rep] + irep_size <= norb:
                    repsizes[rep] += irep_size
                    if assign_irep2rep(irep + 1):
                        return True
                    repsizes[rep] -= irep_size

            return False

        if assign_irep2rep(0) is None:
            raise RuntimeError("Could not form replica bases")
        else:
            replicas = []
            for rep in range(nreps):
                selected_ireps = [irep for i, irep in enumerate(inc_replicas)
                                  if irep2rep[i] == rep]
                replicas.append(set().union(*selected_ireps))
            return replicas

    @classmethod
    def build_replica_bases(cls,
                            norb: int,
                            h: np.ndarray,
                            V: np.ndarray):
        """
        Distribute nbath_total bath basis states between a few replicas, each
        of size norb. The replica bases being built are subject to three
        conditions.

        - Basis states connected by a nonzero matrix element of h must belong
          to the same replica.
        - Each bath basis state is connected to at most one impurity orbital.
        - If two bath states are connected to the same impurity orbital,
          then they cannot belong to the same replica.
        """
        nbath_total = h.shape[2]

        if nbath_total % norb != 0:
            raise RuntimeError(
                "Total number of bath states is not a multiple of norb"
            )

        if not _is_spin_diagonal(V):
            raise RuntimeError("Bath hybridization matrix is not spin-diagonal")

        bath_states = range(nbath_total)
        bs2orbs = _bath_states_to_orbs(V)

        if any(len(orbs) > 1 for orbs in bs2orbs):
            raise RuntimeError(
                "A bath state is connected to more than one impurity orbital"
            )

        # Graph representation of the bath Hamiltonian
        # Basis states are vertices and nonzero matrix elements are edges
        h_graph = nx.Graph()
        h_graph.add_nodes_from(bath_states)
        for spin1, spin2, b1, b2 in zip(*np.nonzero(h)):
            h_graph.add_edge(int(b1), int(b2))

        # Replica bases
        replicas = []
        # Incomplete replicas of sizes < norb. These will have to be merged to
        # form proper replicas.
        inc_replicas = []

        # Connected components of the graph are candidates for the replica bases
        for replica in list(nx.connected_components(h_graph)):
            if len(replica) > norb:
                raise RuntimeError(
                    f"One of replicas has more than norb = {norb} states"
                )
            elif len(replica) == norb:
                if not cls.is_replica_valid(replica, bs2orbs):
                    raise RuntimeError(
                        "An impurity orbital is connected to a replica "
                        "more than once"
                    )
                replicas.append(replica)
            else:
                inc_replicas.append(replica)
        replicas += cls.merge_inc_replicas(inc_replicas, norb, bs2orbs)

        # Order replica basis according to the orbital
        def order_replica(replica):
            res = []
            # Bath states in replica that are decoupled from the impurity
            dec_bs = list(filter(lambda b: len(bs2orbs[b]) == 0, replica))
            for orb in range(norb):
                b = [b for b in replica if bs2orbs[b] == [orb]]
                res.append(b[0] if len(b) != 0 else dec_bs.pop())
            return res

        # Consistency check
        ordered_replicas = list(map(order_replica, replicas))
        for replica in ordered_replicas:
            assert all(bs2orbs[replica[orb]] in ([orb], [])
                       for orb in range(norb))

        return ordered_replicas

    @classmethod
    def build_linear_combination(cls,
                                 replicas: list[list[int]],
                                 nspin: int,
                                 h: np.ndarray):
        """
        Analyse a given bath Hamiltonian h and build its representation as a
        linear combination of basis matrices for a single replica. The basis
        matrices are chosen to (1) be Hermitian and (2) have at most 2 non-zero
        elements.
        """

        nbath = len(replicas)
        norb = len(replicas[0])

        # For each replica, collect all non-zero matrix elements of h
        h_elements = [dict() for nu in range(nbath)]
        for nu in range(nbath):
            replica = replicas[nu]
            for (orb1, b1), (orb2, b2) in product(enumerate(replica), repeat=2):
                for spin1, spin2 in product(range(nspin), repeat=2):
                    idx1 = (spin1, orb1)
                    idx2 = (spin2, orb2)
                    val = h[spin1, spin2, b1, b2]
                    if val != 0:
                        h_elements[nu][(idx1, idx2)] = val

        # Collect indices of all nonzero matrix elements of h
        h_elements_real_idx = set()
        h_elements_imag_idx = set()
        for h_elements_nu in h_elements:
            for (idx1, idx2), val in h_elements_nu.items():
                if idx1 > idx2:
                    continue
                if val.real != 0:
                    h_elements_real_idx.add((idx1, idx2))
                if val.imag != 0:
                    h_elements_imag_idx.add((idx1, idx2))

        h_elements_real_idx = list(h_elements_real_idx)
        h_elements_imag_idx = list(h_elements_imag_idx)
        nsym = len(h_elements_real_idx) + len(h_elements_imag_idx)

        # Build basis matrices
        hvec = np.zeros((nspin, nspin, norb, norb, nsym),
                        dtype=complex, order='F')

        isym = 0
        for idx1, idx2 in h_elements_real_idx:
            spin1, orb1 = idx1
            spin2, orb2 = idx2
            hvec[spin1, spin2, orb1, orb2, isym] = 1.0
            hvec[spin2, spin1, orb2, orb1, isym] = 1.0
            isym += 1
        for idx1, idx2 in h_elements_imag_idx:
            spin1, orb1 = idx1
            spin2, orb2 = idx2
            hvec[spin1, spin2, orb1, orb2, isym] = -1.0j
            hvec[spin2, spin1, orb2, orb1, isym] = 1.0j
            isym += 1

        # Extract lambda parameters
        lambdavec = np.zeros((nbath, nsym), order='F')
        for nu in range(nbath):
            isym = 0
            for idx1, idx2 in h_elements_real_idx:
                lambdavec[nu, isym] = h_elements[nu].get((idx1, idx2), 0).real
                isym += 1
            for idx1, idx2 in h_elements_imag_idx:
                lambdavec[nu, isym] = -h_elements[nu].get((idx1, idx2), 0).imag
                isym += 1

        return hvec, lambdavec

    def __init__(self,
                 ed_mode: str,
                 nspin: int,
                 Hloc: np.ndarray,
                 h: np.ndarray,
                 V: np.ndarray,
                 Delta: np.ndarray):
        norb = Hloc.shape[2]
        nbath_total = h.shape[2]
        # Number of replicas
        self.nbath = nbath_total // norb

        replicas = self.build_replica_bases(norb, h, V)

        self.hvec, self.lambdavec = \
            self.build_linear_combination(replicas, nspin, h)
        self.nsym = self.hvec.shape[-1]

        V_size = nspin * norb
        replica_params_size = V_size + self.nsym

        def replica_offset(nu):
            return 1 + nu * replica_params_size

        self.data = np.zeros(1 + self.nbath * replica_params_size, dtype=float)
        self.data[0] = self.nsym

        # View: Hopping amplitudes
        self.V = [self.data[replica_offset(nu):replica_offset(nu) + V_size].
                  reshape(nspin, norb)
                  for nu in range(self.nbath)]
        assert all(not V_nu.flags['OWNDATA'] for V_nu in self.V)

        # View: Linear coefficients of the replica matrix linear combination
        self.l = [self.data[replica_offset(nu) + V_size:  # noqa: E741
                            replica_offset(nu) + V_size + self.nsym].
                  reshape(self.nsym)
                  for nu in range(self.nbath)]
        assert all(not l_nu.flags['OWNDATA'] for l_nu in self.lambdavec)

        # Fill V and lambda
        for nu in range(self.nbath):
            replica = replicas[nu]
            for spin in range(nspin):
                for orb, b in enumerate(replica):
                    self.V[nu][spin, orb] = V[spin, spin, orb, b]
            for isym in range(self.nsym):
                self.l[nu][isym] = self.lambdavec[nu, isym]


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
        assert _is_spin_degenerate(Hloc)
        assert _is_spin_degenerate(h)
        assert _is_spin_degenerate(V)
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
        if _is_spin_diagonal(Hloc) and \
           _is_spin_diagonal(h) and _is_spin_diagonal(V):
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

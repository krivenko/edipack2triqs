"""
Objects representing various EDIpack bath topologies.
"""

from itertools import product
from copy import deepcopy
from typing import Optional

import numpy as np
import networkx as nx

from h5.formats import register_class

from . import EDMode
from .util import is_diagonal, is_spin_diagonal


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
    For each impurity orbital, find all bath levels it is connected to by
    a hopping amplitude matrix 'V'.
    """
    # np.unique() removes repeated bath levels possibly introduced by multiple
    # combinations of spin indices
    norb = V.shape[2]
    return [list(np.unique(np.nonzero(V[:, :, orb, :])[2]))
            for orb in range(norb)]


class Bath:
    """
    Base class for all bath classes.
    """

    def assert_compatible(self, other):
        assert type(self) is type(other), "Incompatible bath object types"
        assert self.data.shape == other.data.shape, \
            "Incompatible bath topologies"

    # Multiply all bath parameters by a constant
    def __imul__(self, x):
        self.data *= x
        return self

    def __mul__(self, x):
        res = deepcopy(self)
        res *= x
        return res

    def __rmul__(self, x):
        res = deepcopy(self)
        res *= x
        return res

    def __neg__(self):
        res = deepcopy(self)
        res *= -1
        return res

    # Addition and subtraction of bath parameters
    def __iadd__(self, other):
        self.assert_compatible(other)
        self.data += other.data
        return self

    def __isub__(self, other):
        self.assert_compatible(other)
        self.data -= other.data
        return self

    def __add__(self, other):
        res = deepcopy(self)
        res += other
        return res

    def __sub__(self, other):
        res = deepcopy(self)
        res -= other
        return res


class BathNormal(Bath):
    """
    Parameters of a bath with normal topology.
    The normal topology means that each of ``norb`` impurity orbitals is
    connected to ``nbath`` independent bath levels. There are, therefore,
    ``norb * nbath`` bath levels in total.

    Instances of this class are compatible with TRIQS'
    :ref:`HDF5 </documentation/manual/hdf5/ref.rst>` interface.
    """

    # EDIpack bath type
    name: str = 'normal'

    nbath: int
    "Number of bath levels per impurity orbital."

    eps: np.ndarray
    """
    Bath energy levels as an array of the shape ``(nspin, norb, nbath)``.
    """
    Delta: Optional[np.ndarray]
    """
    Local super-conducting order parameters of the bath as an array
    of the shape ``(nspin, norb, nbath)``. Only present when ``ed_mode ==
    EDMode.SUPERC``.
    """
    V: np.ndarray
    """
    Spin-diagonal hopping amplitudes between the impurity and the bath
    as an array of the shape ``(nspin, norb, nbath)``.
    """
    U: Optional[np.ndarray]
    """
    Spin off-diagonal hopping amplitudes between the impurity and the bath
    as an array of the shape ``(nspin, norb, nbath)``. Only present when
    ``ed_mode == EDMode.NONSU2``.
    """

    def __init__(self,
                 ed_mode: EDMode,
                 nspin: int,
                 norb: int,
                 nbath: int,
                 data: Optional[np.ndarray] = None):
        self.nbath = nbath

        size = nspin * norb * nbath

        # EDIpack-compatible bath parameter array
        data_size = size * (2 if ed_mode == EDMode.NORMAL else 3)
        if data is None:
            self.data = np.zeros(data_size, dtype=float)
        else:
            assert data.dtype == float
            assert data.shape == (data_size,)
            self.data = data

        params_shape = (nspin, norb, nbath)

        # View: Energy levels
        self.eps = self.data[:size].reshape(params_shape)
        assert self.eps.base is self.data

        if ed_mode == EDMode.NONSU2:
            # View: Same-spin hopping amplitudes
            self.V = self.data[size:2 * size].reshape(params_shape)
            assert self.V.base is self.data
            # View: Spin-flip hopping amplitudes
            self.U = self.data[2 * size:].reshape(params_shape)
            assert self.U.base is self.data
        elif ed_mode == EDMode.SUPERC:
            # View: Local SC order parameters of the bath
            self.Delta = self.data[size:2 * size].reshape(params_shape)
            assert self.Delta.base is self.data
            # View: Same-spin hopping amplitudes
            self.V = self.data[2 * size:].reshape(params_shape)
            assert self.V.base is self.data
        elif ed_mode == EDMode.NORMAL:
            # View: Same-spin hopping amplitudes
            self.V = self.data[size:2 * size].reshape(params_shape)
            assert self.V.base is self.data
        else:
            raise RuntimeError("Unknown ED mode")

    @property
    def ed_mode(self) -> EDMode:
        """
        ED mode this bath object is usable with.
        """
        if hasattr(self, "U"):
            return EDMode.NONSU2
        elif hasattr(self, "Delta"):
            return EDMode.SUPERC
        else:
            return EDMode.NORMAL

    def __deepcopy__(self, memo):
        nspin, norb, nbath = self.V.shape
        return BathNormal(self.ed_mode, nspin, norb, nbath, self.data.copy())

    def __reduce_to_dict__(self):
        "HDFArchive serialization"
        return {"ed_mode": self.ed_mode.value,
                "nspin": self.V.shape[0],
                "norb": self.V.shape[1],
                "nbath": self.V.shape[2],
                "data": self.data}

    @classmethod
    def __factory_from_dict__(cls, name, d):
        "HDFArchive deserialization"
        return BathNormal(
            EDMode(d["ed_mode"]), d["nspin"], d["norb"], d["nbath"], d["data"]
        )

    @classmethod
    def is_usable(cls,
                  Hloc: np.ndarray,
                  Hloc_an: np.ndarray,
                  h: np.ndarray,
                  V: np.ndarray,
                  Delta: np.ndarray):
        norb = Hloc.shape[2]
        nbath_total = h.shape[2]  # Total number of bath states

        # - The total number of bath states must be a multiple of norb
        # - All spin components of Hloc must be diagonal
        # - Hloc_an must be diagonal
        # - h must be spin-diagonal
        # - Delta must be diagonal
        # - All spin components of h must be diagonal
        # - Each bath state is coupled to at most one impurity orbital
        # - Each impurity orbital is coupled to at most nbath_total/norb
        #   bath states
        return (nbath_total % norb == 0) and \
            all(is_diagonal(Hloc[spin1, spin2, ...])
                for spin1, spin2 in product(range(2), repeat=2)) and \
            is_diagonal(Hloc_an[0, 0, ...]) and \
            is_spin_diagonal(h) and \
            is_diagonal(Delta) and \
            all(is_diagonal(h[spin, spin, ...]) for spin in range(2)) and \
            all(len(orbs) <= 1 for orbs in _bath_states_to_orbs(V)) and \
            all(len(bs) <= (nbath_total // norb)
                for bs in _orbs_to_bath_states(V))

    @classmethod
    def from_hamiltonian(cls,
                         ed_mode: EDMode,
                         nspin: int,
                         Hloc: np.ndarray,
                         h: np.ndarray,
                         V: np.ndarray,
                         Delta: np.ndarray):
        norb = Hloc.shape[2]
        nbath_total = h.shape[2]
        # Number of bath sites
        nbath = nbath_total // norb

        bath = cls(ed_mode, nspin, norb, nbath)

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
                n_missing_states = nbath - len(bs[orb])
                for _ in range(n_missing_states):
                    bs[orb].append(dec_bs.pop(0))
                # Fill the parameters
                for nu, b in enumerate(bs[orb]):
                    if spin1 == spin2:
                        bath.eps[spin1, orb, nu] = np.real_if_close(
                            h[spin1, spin2, b, b]
                        )
                        if ed_mode == EDMode.SUPERC:
                            bath.Delta[spin1, orb, nu] = np.real_if_close(
                                Delta[b, b]
                            )
                        bath.V[spin1, orb, nu] = V[spin1, spin2, orb, b]
                    elif ed_mode == EDMode.NONSU2:
                        bath.U[spin1, orb, nu] = V[spin1, spin2, orb, b]

        return bath


register_class(BathNormal)


class BathHybrid(Bath):
    """
    Parameters of a bath with hybrid topology.
    In the hybrid topology there are ``nbath`` independent bath levels. Each of
    these levels is connected to each impurity orbital via hopping amplitudes
    ``V``.

    Instances of this class are compatible with TRIQS'
    :ref:`HDF5 </documentation/manual/hdf5/ref.rst>` interface.
    """

    # EDIpack bath type
    name: str = 'hybrid'

    nbath: int
    "Total number of bath levels."

    eps: np.ndarray
    """
    Bath energy levels as an array of the shape ``(nspin, nbath)``.
    """
    Delta: Optional[np.ndarray]
    """
    Local super-conducting order parameters of the bath as an array
    of the shape ``(nspin, nbath)``. Only present when
    ``ed_mode == EDMode.SUPERC``.
    """
    V: np.ndarray
    """
    Spin-diagonal hopping amplitudes between the impurity and the bath
    as an array of the shape ``(nspin, norb, nbath)``.
    """
    U: Optional[np.ndarray]
    """
    Spin off-diagonal hopping amplitudes between the impurity and the bath
    as an array of the shape ``(nspin, norb, nbath)``. Only present when
    ``ed_mode == EDMode.NONSU2``.
    """

    def __init__(self,
                 ed_mode: EDMode,
                 nspin: int,
                 norb: int,
                 nbath: int,
                 data: Optional[np.ndarray] = None):
        self.nbath = nbath

        eps_size = nspin * nbath
        size = eps_size * norb

        # EDIpack-compatible bath parameter array
        data_size = {EDMode.NORMAL: eps_size + size,
                     EDMode.SUPERC: 2 * eps_size + size,
                     EDMode.NONSU2: eps_size + 2 * size}[ed_mode]
        if data is None:
            self.data = np.zeros(data_size, dtype=float)
        else:
            assert data.dtype == float
            assert data.shape == (data_size,)
            self.data = data

        eps_shape = (nspin, nbath)
        shape = (nspin, norb, nbath)

        # View: Energy levels
        self.eps = self.data[:eps_size].reshape(eps_shape)
        assert self.eps.base is self.data

        if ed_mode == EDMode.NONSU2:
            # View: Same-spin hopping amplitudes
            self.V = self.data[eps_size:eps_size + size].reshape(shape)
            assert self.V.base is self.data
            # View: Spin-flip hopping amplitudes
            self.U = self.data[eps_size + size:].reshape(shape)
            assert self.U.base is self.data
        elif ed_mode == EDMode.SUPERC:
            # View: Local SC order parameters of the bath
            self.Delta = self.data[eps_size:2 * eps_size].reshape(eps_shape)
            assert self.Delta.base is self.data
            # View: Same-spin hopping amplitudes
            self.V = self.data[2 * eps_size:].reshape(shape)
            assert self.V.base is self.data
        elif ed_mode == EDMode.NORMAL:
            # View: Same-spin hopping amplitudes
            self.V = self.data[eps_size:].reshape(shape)
            assert self.V.base is self.data
        else:
            raise RuntimeError("Unknown ED mode")

    @property
    def ed_mode(self) -> EDMode:
        """
        ED mode this bath object is usable with.
        """
        if hasattr(self, "U"):
            return EDMode.NONSU2
        elif hasattr(self, "Delta"):
            return EDMode.SUPERC
        else:
            return EDMode.NORMAL

    def __deepcopy__(self, memo):
        nspin, norb, nbath = self.V.shape
        return BathHybrid(self.ed_mode, nspin, norb, nbath, self.data.copy())

    def __reduce_to_dict__(self):
        "HDFArchive serialization"
        return {"ed_mode": self.ed_mode.value,
                "nspin": self.V.shape[0],
                "norb": self.V.shape[1],
                "nbath": self.V.shape[2],
                "data": self.data}

    @classmethod
    def __factory_from_dict__(cls, name, d):
        "HDFArchive deserialization"
        return BathHybrid(
            EDMode(d["ed_mode"]), d["nspin"], d["norb"], d["nbath"], d["data"]
        )

    @classmethod
    def is_usable(cls, h: np.ndarray, Delta: np.ndarray):
        # - h must be spin-diagonal
        # - All spin components of h must be diagonal
        # - Delta must be diagonal
        return is_spin_diagonal(h) and \
            all(is_diagonal(h[spin, spin, ...]) for spin in range(2)) and \
            is_diagonal(Delta)

    @classmethod
    def from_hamiltonian(cls,
                         ed_mode: EDMode,
                         nspin: int,
                         Hloc: np.ndarray,
                         h: np.ndarray,
                         V: np.ndarray,
                         Delta: np.ndarray):
        norb = Hloc.shape[2]
        nbath = h.shape[2]

        bath = cls(ed_mode, nspin, norb, nbath)

        for spin1, spin2, nu in product(range(nspin),
                                        range(nspin),
                                        range(nbath)):
            if spin1 == spin2:
                bath.eps[spin1, nu] = np.real_if_close(h[spin1, spin2, nu, nu])
                if ed_mode == EDMode.SUPERC:
                    bath.Delta[spin1, nu] = np.real_if_close(Delta[nu, nu])
                bath.V[spin1, :, nu] = V[spin1, spin2, :, nu]
            elif ed_mode == EDMode.NONSU2:
                bath.U[spin1, :, nu] = V[spin1, spin2, :, nu]

        return bath


register_class(BathHybrid)


class BathGeneral(Bath):
    r"""
    Parameters of a bath with general topology. General bath is a set of
    ``nbath`` independent replicas of the impurity. Hamiltonian of each replica
    is constructed as a linear combination of ``nsym`` basis matrices
    :math:`\hat O_i` with coefficients :math:`\lambda^\nu_i`. Each impurity
    orbital is coupled to the corresponding orbital of each replica via hopping
    amplitudes ``V``.

    Instances of this class are compatible with TRIQS'
    :ref:`HDF5 </documentation/manual/hdf5/ref.rst>` interface.
    """

    # EDIpack bath type
    name: str = 'general'

    nbath: int
    "Number of replicas."
    nsym: int
    r"Number of bath basis matrices :math:`\hat O_i`."
    hvec: np.ndarray
    r"""
    Basis matrices :math:`\hat O_i` as an array of the shape
    (nspin, nspin, norb, norb, nsym).
    """
    l: list[np.ndarray]
    r"""
    Coefficients of linear combinations :math:`\lambda^\nu_i`. Each of the
    ``nbath`` elements is an array of length ``nsym`` corresponding to the
    :math:`\nu`-th replica.
    """
    V: list[np.ndarray]
    r"""
    Hopping amplitudes :math:`V^\nu_{\sigma,\alpha}`. Each of the ``nbath``
    elements is an array of the shape ``(nspin, norb)`` corresponding to the
    :math:`\nu`-th replica.
    """

    def __init__(self,
                 nspin: int,
                 norb: int,
                 nbath: int,
                 hvec: np.ndarray,
                 data: Optional[np.ndarray] = None):
        self.nbath = nbath
        self.hvec = hvec
        self.nsym = hvec.shape[-1]

        V_size = nspin * norb
        replica_params_size = V_size + self.nsym

        def replica_offset(nu):
            return 1 + nu * replica_params_size

        data_size = 1 + nbath * replica_params_size
        if data is None:
            self.data = np.zeros(data_size, dtype=float)
            self.data[0] = self.nsym
        else:
            assert data.dtype == float
            assert data.shape == (data_size,)
            assert data[0] == self.nsym
            self.data = data

        # View: Hopping amplitudes
        self.V = [self.data[replica_offset(nu):replica_offset(nu) + V_size].
                  reshape(nspin, norb)
                  for nu in range(self.nbath)]
        assert all(V_nu.base is self.data for V_nu in self.V)

        # View: Linear coefficients of the replica matrix linear combination
        self.l = [self.data[replica_offset(nu) + V_size:  # noqa: E741
                            replica_offset(nu) + V_size + self.nsym].
                  reshape(self.nsym)
                  for nu in range(self.nbath)]
        assert all(l_nu.base is self.data for l_nu in self.l)

    def __deepcopy__(self, memo):
        nspin, norb = self.V[0].shape
        return BathGeneral(nspin, norb, self.nbath,
                           deepcopy(self.hvec, memo),
                           self.data.copy())

    def __reduce_to_dict__(self):
        "HDFArchive serialization"
        return {"nspin": self.V[0].shape[0],
                "norb": self.V[0].shape[1],
                "nbath": self.nbath,
                "hvec": self.hvec,
                "data": self.data}

    @classmethod
    def __factory_from_dict__(cls, name, d):
        "HDFArchive deserialization"
        return BathGeneral(
            d["nspin"], d["norb"], d["nbath"],
            d["hvec"], d["data"]
        )

    def assert_compatible(self, other):
        assert type(self) is type(other), "Incompatible bath object types"
        assert self.data.shape == other.data.shape, \
            "Incompatible bath topologies"
        assert (self.hvec == other.hvec).all(), \
            "Incompatible general bath objects (different basis matrices)"

    # Multiply all bath parameters by a constant
    def __imul__(self, x):
        # Skipping the first element that stores nsym
        self.data[1:] *= x
        return self

    # Addition of bath parameters
    def __iadd__(self, other):
        self.assert_compatible(other)
        # Skipping the first element that stores nsym
        self.data[1:] += other.data[1:]
        return self

    # Subtraction of bath parameters
    def __isub__(self, other):
        self.assert_compatible(other)
        # Skipping the first element that stores nsym
        self.data[1:] -= other.data[1:]
        return self

    @classmethod
    def _is_replica_valid(cls, replica: set[int], bs2orbs: list[list[int]]):
        """
        Check that all bath states of a given replica are connected to different
        impurity orbitals (if any).
        """
        orbs = [bs2orbs[b][0] for b in replica if len(bs2orbs[b]) != 0]
        return len(set(orbs)) == len(orbs)

    @classmethod
    def _merge_inc_replicas(cls,
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
                if not cls._is_replica_valid(set().union(*selected_ireps),
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
    def _build_replica_bases(cls,
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
                "Total number of bath levels is not a multiple of norb"
            )

        if not is_spin_diagonal(V):
            raise RuntimeError("Bath hybridization matrix is not spin-diagonal")

        bath_states = range(nbath_total)
        bs2orbs = _bath_states_to_orbs(V)

        if any(len(orbs) > 1 for orbs in bs2orbs):
            raise RuntimeError(
                "A bath level is connected to more than one impurity orbital"
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
                if not cls._is_replica_valid(replica, bs2orbs):
                    raise RuntimeError(
                        "An impurity orbital is connected to a replica "
                        "more than once"
                    )
                replicas.append(replica)
            else:
                inc_replicas.append(replica)
        replicas += cls._merge_inc_replicas(inc_replicas, norb, bs2orbs)

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
    def _build_linear_combination(cls,  # noqa: C901
                                  replicas: list[list[int]],
                                  nspin: int,
                                  h: np.ndarray,
                                  is_nambu: bool):
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
                # In the superconducting case, check that all elements from the
                # inambu1 = inambu2 = 1 block have negated counterparts in the
                # inambu1 = inambu2 = 0 block. Disregard the former.
                if is_nambu:
                    inambu1, orb1 = idx1
                    inambu2, orb2 = idx2
                    if inambu1 == 1 and inambu2 == 1:
                        val00 = h_elements_nu.get(((0, orb1), (0, orb2)), 0)
                        if abs(val00 + val) > 1e-10:
                            raise RuntimeError(
                                "Inconsistent matrix elements in the diagonal "
                                "Nambu blocks of h"
                            )
                        else:
                            continue

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
            if is_nambu and spin1 == 0 and spin2 == 0:
                hvec[1, 1, orb1, orb2, isym] = -1.0
                hvec[1, 1, orb2, orb1, isym] = -1.0
            isym += 1
        for idx1, idx2 in h_elements_imag_idx:
            spin1, orb1 = idx1
            spin2, orb2 = idx2
            hvec[spin1, spin2, orb1, orb2, isym] = -1.0j
            hvec[spin2, spin1, orb2, orb1, isym] = 1.0j
            if is_nambu and spin1 == 0 and spin2 == 0:
                hvec[1, 1, orb1, orb2, isym] = 1.0j
                hvec[1, 1, orb2, orb1, isym] = -1.0j
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

    @classmethod
    def from_hamiltonian(cls,
                         ed_mode: EDMode,
                         nspin: int,
                         Hloc: np.ndarray,
                         h: np.ndarray,
                         V: np.ndarray,
                         Delta: np.ndarray):

        norb = Hloc.shape[2]
        nbath_total = h.shape[2]
        # Number of replicas
        nbath = nbath_total // norb

        # In the superconducting case, reinterpret first two indices of h as
        # Nambu indices and fill the off-diagonal elements from Delta
        is_nambu = (ed_mode == EDMode.SUPERC)
        nnambu = 1
        if is_nambu:
            nnambu = 2
            h = h.copy()
            h[1, 1, :, :] *= -1
            h[0, 1, :, :] = Delta
            h[1, 0, :, :] = np.conj(Delta.T)

        replicas = cls._build_replica_bases(norb, h, V)

        hvec, lambdavec = \
            cls._build_linear_combination(replicas, nnambu * nspin, h, is_nambu)

        bath = cls(nspin, norb, nbath, hvec)

        # Fill l
        for nu in range(bath.nbath):
            for isym in range(bath.nsym):
                bath.l[nu][isym] = lambdavec[nu, isym]

        # Fill V
        for nu in range(bath.nbath):
            replica = replicas[nu]
            for spin in range(nspin):
                for orb, b in enumerate(replica):
                    bath.V[nu][spin, orb] = V[spin, spin, orb, b]

        return bath

    @property
    def lambdavec(self):
        return np.asarray(self.l, order='F')


register_class(BathGeneral)

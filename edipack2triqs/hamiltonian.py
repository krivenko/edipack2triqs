"""
Hamiltonian and its parameters
"""

from dataclasses import dataclass, field
import numpy as np

import triqs.operators as op

from .util import (is_diagonal,
                   IndicesType,
                   monomial2op,
                   validate_fops_up_dn,
                   spin_conjugate)


def default_Uloc():
    return np.array([2.0, 0, 0, 0, 0])


@dataclass
class HamiltonianParams:
    """Parameters of the Hamiltonian"""

    # Non-interacting part of the impurity Hamiltonian
    Hloc: np.ndarray
    # Bath parameters
    bath: np.ndarray
    # Number of bath sites
    Nbath: int = 6
    # Bath type, one of 'normal', 'hybrid' and 'replica'
    bath_type: str = 'normal'
    # Local intra-orbital interactions U (one value per orbital)"
    Uloc: np.ndarray = field(default_factory=default_Uloc)
    # Local inter-orbital interaction U'"
    Ust: float = 0
    # Hund's coupling"
    Jh: float = 0
    # Spin-exchange coupling constant"
    Jx: float = 0
    # Pair-hopping coupling constant
    Jp: float = 0


def parse_hamiltonian(hamiltonian: op.Operator,
                      fops_imp_up: list[IndicesType],
                      fops_imp_dn: list[IndicesType],
                      fops_bath_up: list[IndicesType],
                      fops_bath_dn: list[IndicesType]) -> HamiltonianParams:
    """
    Parse a given Hamiltonian and extract parameters from it.
    """

    validate_fops_up_dn(fops_imp_up,
                        fops_imp_dn,
                        "fops_imp_up",
                        "fops_imp_dn")
    validate_fops_up_dn(fops_bath_up,
                        fops_bath_dn,
                        "fops_bath_up",
                        "fops_bath_dn")

    if not (hamiltonian - op.dagger(hamiltonian)).is_zero():
        raise RuntimeError("Hamiltonian is not Hermitian")

    fops_imp = fops_imp_up + fops_imp_dn
    fops_bath = fops_bath_up + fops_bath_dn

    assert set(fops_imp).isdisjoint(set(fops_bath)), \
        "All fundamental sets must be disjoint"

    norb = len(fops_imp_up)
    bath_size = len(fops_bath_up)

    hamiltonian_conj = spin_conjugate(hamiltonian,
                                      fops_imp_up + fops_bath_up,
                                      fops_imp_dn + fops_bath_dn)
    nspin = 1 if (hamiltonian_conj - hamiltonian).is_zero() else 2

    Hloc = np.zeros((2, norb, norb))
    h = np.zeros((2, bath_size, bath_size))
    V = np.zeros((2, bath_size, norb))

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
                if spin1 != spin2:
                    raise RuntimeError(
                        "Spin non-diagonal H^{loc} is not supported"
                    )
                Hloc[spin1, orb1, orb2] = coeff
            # d^+ a
            elif (indices[0] in fops_imp) and (indices[1] in fops_bath):
                spin1, orb = divmod(fops_imp.index(indices[0]), norb)
                spin2, b = divmod(fops_bath.index(indices[1]), bath_size)
                if spin1 != spin2:
                    raise RuntimeError("Spin non-diagonal V is not supported")
                V[spin1, b, orb] = coeff
            # a^+ d
            elif (indices[0] in fops_bath) and (indices[1] in fops_imp):
                continue
            # a^+ a
            elif (indices[0] in fops_bath) and (indices[1] in fops_bath):
                spin1, b1 = divmod(fops_bath.index(indices[0]), bath_size)
                spin2, b2 = divmod(fops_bath.index(indices[1]), bath_size)
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

    params = HamiltonianParams(
        Hloc=np.zeros((nspin, nspin, norb, norb), dtype=float, order='F'),
        bath=None,
        bath_type=None,
        Uloc=Uloc,
        Ust=Ust[0] if len(Ust) > 0 else .0,
        Jx=Jx[0] if len(Jx) > 0 else .0,
        Jp=Jp[0] if len(Jp) > 0 else .0
    )
    params.Jh = -(UstmJ[0] if len(UstmJ) > 0 else .0) + params.Ust

    if nspin == 1:
        # Internal consistency check: Hloc must be spin-degenerate
        assert np.allclose(Hloc[0, ...], Hloc[1, ...], atol=1e-10)

    for spin in range(nspin):
        params.Hloc[spin, spin, ...] = Hloc[spin, ...]

    # Can we use bath_type = 'normal'?
    # - The total number of bath states must be a multiple of norb
    # - All spin components of Hloc must be diagonal
    # - All spin components of h must be diagonal
    # - Each bath state is coupled to at most one orbital
    if (bath_size % norb == 0) and \
       all(is_diagonal(Hloc[spin, ...]) for spin in range(2)) and \
       all(is_diagonal(h[spin, ...]) for spin in range(2)) and \
       (np.count_nonzero(V, axis=2) <= 1).all():
        params.Nbath = bath_size // norb
        params.bath_type = "normal"
        # TODO: Set params.bath

    # Can we use bath_type = 'hybrid'?
    # - All spin components of h must be diagonal
    elif all(is_diagonal(h[spin, ...]) for spin in range(2)):
        params.Nbath = bath_size
        params.bath_type = "hybrid"
        # TODO: Set params.bath

    # TODO: bath_type = 'replica'
    elif False:
        pass
    else:
        raise RuntimeError(
            "Cannot find a suitable bath mode for the given Hamiltonian"
        )

    return params

# TODO: detect correct ed_total_ud()

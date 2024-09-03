"""
Hamiltonian and its parameters
"""

from dataclasses import dataclass, field
import numpy as np

import triqs.operators as op

from .util import IndicesType, monomial2op


def default_Uloc():
    return np.array([2.0, 0, 0, 0, 0])


@dataclass
class HamiltonianParams:
    """Parameters of the Hamiltonian"""

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


def parse_hamiltonian(h: op.Operator,
                      fops_imp_up: list[IndicesType],
                      fops_imp_dn: list[IndicesType]) -> HamiltonianParams:
    """
    Parse a given Hamiltonian h and extract parameters from it.
    """

    norb = len(fops_imp_up)
    fops_imp = fops_imp_up + fops_imp_dn

    Uloc = np.zeros(5, dtype=float)
    Ust, UstmJ = [], []
    Jx, Jp = [], []

    for mon, coeff in h:
        # Skipping an irrelevant constant term
        if len(mon) == 0:
            continue

        daggers, indices = zip(*mon)

        # U(1)-symmetric quadratic term
        if daggers == (True, False):
            # TODO
            pass

        # U(1)-symmetric quartic term
        elif daggers == (True, True, False, False):
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

    if not all_close(Ust):
        raise RuntimeError(
            "Inconsistent values of U' for different orbital pairs"
        )
    if not all_close(UstmJ):
        raise RuntimeError(
            "Inconsistent values of U' - J for different orbital pairs"
        )
    if not all_close(Jx):
        raise RuntimeError(
            "Inconsistent values of J_X for different orbital pairs"
        )
    if not all_close(Jp):
        raise RuntimeError(
            "Inconsistent values of J_P for different orbital pairs"
        )

    params = HamiltonianParams(
        Uloc=Uloc,
        Ust=Ust[0] if len(Ust) > 0 else .0,
        Jx=Jx[0] if len(Jx) > 0 else .0,
        Jp=Jp[0] if len(Jp) > 0 else .0
    )
    params.Jh = -(UstmJ[0] if len(UstmJ) > 0 else .0) + params.Ust

    return params

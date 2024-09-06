import numpy as np

import triqs.operators as op

from edipy import global_env as ed

from .util import IndicesType
from .hamiltonian import parse_hamiltonian


class EDIpackSolver:

    # edipy stores simulation parameters as attributes of the module itself.
    # Therefore, state of a simulation must be controlled by at most one object
    # at any time -> singleton.
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(
                EDIpackSolver, cls
            ).__new__(cls)
        else:
            raise RuntimeError(
                "Only one instance of EDIpackSolver can be created"
            )
        return cls._instance

    def __init__(self,
                 h: op.Operator,
                 fops_imp_up: list[IndicesType],
                 fops_imp_dn: list[IndicesType],
                 fops_bath_up: list[IndicesType],
                 fops_bath_dn: list[IndicesType]):
        self.params = parse_hamiltonian(h,
                                        fops_imp_up,
                                        fops_imp_dn,
                                        fops_bath_up,
                                        fops_bath_dn)
        # Pass general parameters to EDIpack
        ed.Nspin = self.params.Hloc.shape[0]
        ed.Norb = self.params.Hloc.shape[2]
        assert ed.Norb <= 5, f"At most 5 orbitals are allowed, got {ed.Norb}"

        bath_params = self.params.bath

        # Initialize EDIpack solver
        ed.Nbath = bath_params.nbath
        ed.bath_type = bath_params.name
        self.bath = np.zeros(ed.get_bath_dimension(), dtype=float)
        ed.init_solver(self.bath)

        # Pass bath parameters to EDIpack
        if bath_params.name in ('normal', 'hybrid'):
            assert self.bath.size == bath_params.eps.size + bath_params.V.size
            self.bath[:bath_params.eps.size] = bath_params.eps.flatten()
            self.bath[bath_params.eps.size:] = bath_params.V.flatten()
        else:
            # TODO
            raise RuntimeError("'replica' bath topology is not supported yet")

        # Pass interaction parameters to EDIpack
        ed.Uloc = self.params.Uloc
        ed.Ust = self.params.Ust
        ed.Jh = self.params.Jh
        ed.Jx = self.params.Jx
        ed.Jp = self.params.Jp

    def solve(self):
        ed.solve(self.bath, self.params.Hloc)

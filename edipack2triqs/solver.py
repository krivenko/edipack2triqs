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
            ).__new__(cls, *args, **kwargs)
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
        ed.Nspin = self.params.Hloc.shape[0]
        ed.Norb = self.params.Hloc.shape[2]
        assert ed.Norb <= 5, f"At most 5 orbitals are allowed, got {ed.Norb}"

        ed.Nbath = self.params.Nbath
        ed.bath_type = self.params.bath_type
        ed.init_solver(self.params.bath)

    def solve(self):
        # TODO
        ed.solve(self.params.bath, self.params.Hloc)

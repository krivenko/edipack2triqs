import triqs.operators as op

from edipy import global_env as ed

from .util import IndicesType, validate_fops_up_dn


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

        validate_fops_up_dn(fops_imp_up,
                            fops_imp_dn,
                            "fops_imp_up",
                            "fops_imp_dn")

        self.norb = len(fops_imp_up)
        ed.Norb = self.norb

        # TODO: Extract Uloc and Ust from h
        # TODO: Extract Jh, Jx, Jp
        # TODO: Deduce bath fops
        # TODO: Call spin_conjugate() to decide edipy.ed_input_vars.norb=1/2

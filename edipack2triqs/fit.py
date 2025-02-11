"""
Bath fitting tools.
"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

import numpy as np

from triqs.gf import BlockGf, MeshImFreq

from edipy2 import global_env as ed

from .util import chdircontext


@dataclass(frozen=True, kw_only=True)
class BathFittingParams:
    """Parameters of bath fitting"""

    # Fitting scheme: delta, weiss
    scheme: str = "delta"
    # Minimization routine type: CGnr, minimize
    method: str = "CGnr"
    # Gradient evaluation: analytic, numeric
    grad: str = "analytic"
    # Conjugate-gradient tolerance
    tol: float = 0.00001
    # Conjugate-gradient stopping condition:
    # target (|F_{n-1} - F_n| < tol * (1+F_n)),
    # vars (||x_{n-1} - x_n|| < tol * (1+||x_n||)),
    # or both
    stop: str = "both"
    # Max number of iterations
    niter: int = 500
    # Conjugate-gradient weight form: 1, 1/n, 1/w_n
    weight: str = "1"
    # Conjugate-gradient norm definition: elemental, frobenius
    norm: str = "elemental"
    # Fit power for the calculation of the generalized distance as
    # |G0 - G0and| ** pow
    pow: int = 2
    # Flag to pick old/False (Krauth) or new/True (Lichtenstein) version of
    # the minimize CG routine
    minimize_ver: bool = False
    # Unknown parameter used in the CG minimize procedure
    minimize_hh: float = 1e-4

    def __dict__(self):
        assert self.scheme in ("delta", "weiss"), "Invalid value of 'scheme'"
        assert self.method in ("CGnr", "minimize"), "Invalid value of 'method'"
        assert self.grad in ("analytic", "numeric"), "Invalid value of 'grad'"
        assert self.tol >= 0, "'tol' cannot be negative"
        assert self.stop in ("target", "vars", "both"), \
            "Invalid value of 'stop'"
        assert self.niter > 0, "'niter' must be positive"
        assert self.weight in ("1", "1/n", "1/w_n"), "Invalid value of 'weight'"
        assert self.norm in ("elemental", "frobenius"), \
            "Invalid value of 'norm'"
        assert self.pow > 0, "'pow' must be positive"

        return {
            "CG_SCHEME": self.scheme,
            "CG_METHOD": {"CGnr": 0, "minimize": 1}[self.method],
            "CG_GRAD": {"analytic": 0, "minimize": 1}[self.grad],
            "CG_FTOL": self.tol,
            "CG_STOP": {"target": 1, "vars": 2, "both": 0}[self.stop],
            "CG_NITER": self.niter,
            "CG_WEIGHT": {"1": 1, "1/n": 2, "1/w_n": 3}[self.weight],
            "CG_NORM": self.norm,
            "CG_POW": self.pow,
            "CG_MINIMIZE_VER": self.minimize_ver,
            "CG_MINIMIZE_HH": self.minimize_hh
        }


def _chi2_fit_bath(self, g: BlockGf, f: Optional[BlockGf] = None):
    """
    Perform bath parameter fit of a given Green's function.
    """

    if (ed.get_ed_mode() == 2) != (f is not None):
        raise RuntimeError(
            "The anomalous GF is required iff the bath is superconducting"
        )

    fitted_bath = deepcopy(self.h_params.bath)

    def extract_triqs_data(d):
        return np.transpose(d[ed.Lmats:, ...], (1, 2, 0))

    with chdircontext(self.wdname):
        if ed.get_ed_mode() == 1:  # Normal, here nspin is important
            assert set(g.indices) == set(self.gf_block_names), \
                "Unexpected block structure of g"

            func_up = extract_triqs_data(g[self.gf_block_names[0]].data)
            fitted_bath.data[:] = ed.chi2_fitgf(func_up,
                                                fitted_bath.data,
                                                ispin=0)
            if ed.Nspin == 1:
                fitted_bath.data[:] = \
                    ed.spin_symmetrize_bath(fitted_bath.data[:])
            else:
                func_dn = extract_triqs_data(g[self.gf_block_names[1]].data)
                fitted_bath.data[:] = ed.chi2_fitgf(func_dn,
                                                    fitted_bath.data,
                                                    ispin=1)

        elif ed.get_ed_mode() == 2:  # superc, here nspin is 1
            func_up = extract_triqs_data(g[self.gf_block_names[0]].data)
            func_an = extract_triqs_data(f[self.gf_an_block_names].data)
            fitted_bath.data[:] = ed.chi2_fitgf(func_up,
                                                func_an,
                                                fitted_bath.data)

        elif ed.get_ed_mode() == 3:  # nonsu2, here nspin is 2
            func = extract_triqs_data(g[self.gf_block_names[0]].data)
            fitted_bath.data[:] = ed.chi2_fitgf(func, fitted_bath.data)

        else:
            raise RuntimeError("Unrecognized ED mode")

    #
    # Create fitted G0 or \Delta
    #

    mesh = MeshImFreq(beta=ed.beta, S="Fermion", n_iw=ed.Lmats)
    z_vals = np.array([complex(z) for z in mesh])
    get_method = ed.get_g0and if (self.config["CG_SCHEME"] == "weiss") \
        else ed.get_delta

    g_out = g.copy()

    def pack_triqs_data(d):
        return np.transpose(d, (2, 0, 1))

    with chdircontext(self.wdname):
        if ed.get_ed_mode() == 1:  # normal
            out = get_method(z_vals, fitted_bath.data, ishape=5, typ='n')
            g_out[self.gf_block_names[0]].data[:] = \
                pack_triqs_data(out[0, 0, ...])
            g_out[self.gf_block_names[1]].data[:] = \
                pack_triqs_data(out[0, 0, ...] if (self.nspin == 1)
                                else out[1, 1, ...])
            return fitted_bath, g_out

        elif ed.get_ed_mode() == 2:  # superc
            out = get_method(z_vals, fitted_bath.data, ishape=5, typ='n')
            for bn in self.gf_block_names:
                g_out[bn] = pack_triqs_data(out[0, 0, ...])
            out_an = get_method(z_vals, fitted_bath.data, ishape=5, typ='a')
            f_out = f.copy()
            f_out[self.gf_an_block_names].data[:] = \
                pack_triqs_data(out_an[0, 0, ...])
            return fitted_bath, g_out, f_out

        else:  # nonsu2
            out = get_method(z_vals, fitted_bath.data, ishape=3)
            g_out[self.gf_block_names[0]].data[:] = pack_triqs_data(out)
            return fitted_bath, g_out

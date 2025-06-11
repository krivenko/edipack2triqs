"""
Generate reference results for TestEDIpackSolver* using AtomDiag.
"""

import inspect
import numpy as np

import triqs.operators as op
from triqs.atom_diag import (AtomDiag,
                             atomic_density_matrix,
                             atomic_g_iw,
                             atomic_g_w,
                             trace_rho_op)
from triqs.gf.tools import dyson
from h5 import HDFArchive

from edipack2triqs.util import non_int_part


# Write computed reference results into HDF5 archive
write_h5 = False


def make_mkind(spins, orbs, spin_blocks):
    "Returns a mapping (spin, orb) -> (block name, inner index)"
    if spin_blocks:
        return lambda spin, o: (spin, o)
    else:
        up, dn = spins
        return lambda spin, o: (f"{up}_{dn}", len(orbs) * spins.index(spin) + o)


def ref_results(h5_group_name, **params):
    "Either generate reference results or load them from an HDF5 archive"

    filename = inspect.stack()[1].filename[:-2] + "h5"
    if write_h5:
        results = make_reference_results(**params)
        with HDFArchive(filename, 'a') as ar:
            ar.create_group(h5_group_name)
            ar[h5_group_name] = results
            return results
    else:
        with HDFArchive(filename, 'r') as ar:
            return ar[h5_group_name]


def make_reference_results(*,
                           h,
                           spins, orbs, fops,
                           beta, n_iw, energy_window, n_w, broadening,
                           spin_blocks=True,
                           superc=False,
                           zerotemp=False):
    "Generate reference results"

    up, dn = spins
    mki = make_mkind(spins, orbs, spin_blocks)

    c_up = [op.c(*mki(up, o)) for o in orbs]
    c_dn = [op.c(*mki(dn, o)) for o in orbs]
    c_dag_up = [op.c_dag(*mki(up, o)) for o in orbs]
    c_dag_dn = [op.c_dag(*mki(dn, o)) for o in orbs]
    n_up = [op.n(*mki(up, o)) for o in orbs]
    n_dn = [op.n(*mki(dn, o)) for o in orbs]

    N = [n_up[o] + n_dn[o] for o in orbs]
    D = [n_up[o] * n_dn[o] for o in orbs]
    S_x = [c_dag_up[o] * c_dn[o] + c_dag_dn[o] * c_up[o] for o in orbs]
    S_y = [c_dag_dn[o] * c_up[o] - c_dag_up[o] * c_dn[o] for o in orbs]
    S_z = [n_up[o] - n_dn[o] for o in orbs]

    if spin_blocks:
        gf_struct = [(up, len(orbs)), (dn, len(orbs))]
    else:
        gf_struct = [(f'{up}_{dn}', 2 * len(orbs))]

    ad = AtomDiag(h, fops)
    rho = atomic_density_matrix(ad, beta)
    g_w = atomic_g_w(ad, beta, gf_struct, energy_window, n_w, broadening)

    def avg(ops):
        return np.array([trace_rho_op(rho, ops[o], ad) for o in orbs])

    results = {'densities': avg(N),
               'double_occ': avg(D),
               'magn_x': avg(S_x),
               'magn_y': 1j * avg(S_y),
               'magn_z': avg(S_z),
               'g_w': g_w}

    if not zerotemp:
        g_iw = atomic_g_iw(ad, beta, gf_struct, n_iw)
        results['g_iw'] = g_iw

    if not superc:
        h0 = non_int_part(h)
        ad0 = AtomDiag(h0, fops)
        g0_w = atomic_g_w(ad0, beta, gf_struct, energy_window, n_w, broadening)
        results['Sigma_w'] = dyson(G0_iw=g0_w, G_iw=g_w)
        if not zerotemp:
            g0_iw = atomic_g_iw(ad0, beta, gf_struct, n_iw)
            results['Sigma_iw'] = dyson(G0_iw=g0_iw, G_iw=g_iw)

    return results

"""
Generate reference results for TestEDIpackSolver* using pomerol2triqs.
"""

import inspect
import numpy as np
from itertools import product

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
    "Generate reference results using pomerol2triqs"

    from pomerol2triqs import PomerolED

    up, dn = spins
    mki = make_mkind(spins, orbs, spin_blocks)
    norb = len(orbs)

    # Conversion from TRIQS to Pomerol notation for operator indices
    # TRIQS: block_name, inner_index
    # Pomerol: site_label, orbital_index, spin_name
    index_converter = {}
    for bn, inner in fops:
        # Bath
        if bn[0] == "B":  # Bath
            site = "B"
            spin = "down" if bn[-2:] == dn else up
            orb = inner
        else:  # local
            site = "L"
            if spin_blocks:
                spin = "down" if bn[-2:] == dn else up
            else:
                spin = "down" if inner >= norb else up
            orb = inner % norb
        index_converter[(bn, inner)] = (site, orb, spin)

    def make_ed(h_):
        ed = PomerolED(index_converter, verbose=False)
        ed.ops_melem_tol = 0
        ed.rho_threshold = 0
        ed.diagonalize(h_)
        return ed

    ed = make_ed(h)

    def make_avg(spins):
        res = [ed.ensemble_average(*[mki(s, o) for s in spins], beta)
               for o in orbs]
        return np.array(res)

    # Occupation and magnetization
    n_up_avg = make_avg((up, up))
    n_dn_avg = make_avg((dn, dn))
    S_p_avg = make_avg((up, dn))
    S_m_avg = make_avg((dn, up))
    # Double occupancy
    D_avg = make_avg((up, dn, dn, up))

    # Green's functions
    if spin_blocks:
        gf_struct = [(up, norb), (dn, norb)]
    else:
        gf_struct = [(f"{up}_{dn}", 2 * norb)]

    tols = {"pole_res": 1e-12, "coeff_tol": 1e-12}
    g_w = ed.G_w(gf_struct, beta, energy_window, n_w, broadening, **tols)

    results = {'densities': n_up_avg + n_dn_avg,
               'double_occ': D_avg,
               'magn_x': S_p_avg + S_m_avg,
               'magn_y': -1j * (S_p_avg - S_m_avg),
               'magn_z': n_up_avg - n_dn_avg,
               'g_w': g_w}

    # Superconductive \phi
    if superc:
        phi = np.empty((norb, norb), dtype=complex)
        for o1, o2 in product(orbs, repeat=2):
            phi[o1, o2] = ed.ensemble_average(mki(up, o1), mki(dn, o2),
                                              beta, (False, False))
        results['phi'] = phi

    if not zerotemp:
        g_iw = ed.G_iw(gf_struct, beta, n_iw, **tols)
        results['g_iw'] = g_iw

    if not superc:
        ed0 = make_ed(non_int_part(h))
        g0_w = ed0.G_w(gf_struct, beta, energy_window, n_w, broadening, **tols)
        results['Sigma_w'] = dyson(G0_iw=g0_w, G_iw=g_w)
        if not zerotemp:
            g0_iw = ed0.G_iw(gf_struct, beta, n_iw, **tols)
            results['Sigma_iw'] = dyson(G0_iw=g0_iw, G_iw=g_iw)

    return results

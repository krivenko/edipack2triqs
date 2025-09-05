"""
Generate reference results for TestEDIpackSolver* using pomerol2triqs.
"""

import inspect
import numpy as np
from itertools import product

import triqs.operators as op
from triqs.operators.util.extractors import extract_h_dict
from triqs.gf import (BlockGf,
                      Gf,
                      MeshReFreq,
                      MeshImFreq,
                      MeshImTime,
                      conjugate,
                      transpose)
from triqs.gf.descriptors import Omega, iOmega_n
from triqs.gf.tools import inverse, dyson
from h5 import HDFArchive

from edipack2triqs.util import monomial2op, non_int_part


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
                           n_tau=None,
                           spin_blocks=True,
                           superc=False,
                           zerotemp=False,
                           chi_spin=False,
                           chi_dens=False,
                           chi_pair=False,
                           chi_exct=False):
    "Generate reference results using pomerol2triqs"

    up, dn = spins
    mki = make_mkind(spins, orbs, spin_blocks)
    norb = len(orbs)

    index_converter = _make_index_converter(fops, spins, norb, spin_blocks)
    ed = _make_pomerol_ed(index_converter, h)

    #
    # Static observables
    #

    def make_avg(spins):
        res = [
            ed.ensemble_average(*[mki(s, o) for s in spins], beta) for o in orbs
        ]
        return np.array(res)

    # Occupation and magnetization
    n_up_avg = make_avg((up, up))
    n_dn_avg = make_avg((dn, dn))
    S_p_avg = make_avg((up, dn))
    S_m_avg = make_avg((dn, up))
    # Double occupancy
    D_avg = make_avg((up, dn, dn, up))

    results = {'densities': n_up_avg + n_dn_avg,
               'double_occ': D_avg,
               'magn_x': S_p_avg + S_m_avg,
               'magn_y': -1j * (S_p_avg - S_m_avg),
               'magn_z': n_up_avg - n_dn_avg}

    # Superconductive \phi
    if superc:
        phi = np.empty((norb, norb), dtype=complex)
        for o1, o2 in product(orbs, repeat=2):
            phi[o1, o2] = ed.ensemble_average(
                mki(up, o1), mki(dn, o2), beta, (False, False)
            )
        results['phi'] = phi

    # Tolerances for construction of Lehmann representation
    tols = {"pole_res": 1e-12, "coeff_tol": 1e-12}

    #
    # Green's functions and self-energies
    #

    # Structure of normal GF
    gf_struct = [(up, norb), (dn, norb)] if spin_blocks else \
                [(f"{up}_{dn}", 2 * norb)]

    # Normal GF calculations
    if not superc:
        ed0 = _make_pomerol_ed(index_converter, non_int_part(h))

        hloc = _extract_hloc(h, gf_struct)

        # Real frequency
        g_w = ed.G_w(gf_struct, beta, energy_window, n_w, broadening, **tols)
        g0_w = ed0.G_w(gf_struct, beta, energy_window, n_w, broadening, **tols)
        results['g_w'] = g_w
        results['g0_w'] = g0_w
        results['Sigma_w'] = dyson(G0_iw=g0_w, G_iw=g_w)
        Delta_w = g0_w.copy()
        for b, sn in gf_struct:
            Delta_w[b] << Omega + 1j * broadening - hloc[b] - inverse(g0_w[b])
        results['Delta_w'] = Delta_w

        # Matsubara frequency
        if not zerotemp:
            g_iw = ed.G_iw(gf_struct, beta, n_iw, **tols)
            g0_iw = ed0.G_iw(gf_struct, beta, n_iw, **tols)
            results['g_iw'] = g_iw
            results['g0_iw'] = g0_iw
            results['Sigma_iw'] = dyson(G0_iw=g0_iw, G_iw=g_iw)
            Delta_iw = g0_iw.copy()
            for b, sn in gf_struct:
                Delta_iw[b] << iOmega_n - hloc[b] - inverse(g0_iw[b])
            results['Delta_iw'] = Delta_iw

    # Nambu GF calculations
    else:
        # Calculations using the single block
        fops = [_merge_spin_blocks(bl, i, spins, norb) for bl, i in fops]
        index_converter = _make_index_converter(fops, spins, norb, False)
        up_dn = f"{up}_{dn}"
        gf_struct = [(up_dn, 2 * norb)]

        h = _merge_spin_blocks_in_expr(h, spins, norb)
        ed = _make_pomerol_ed(index_converter, h)
        ed0 = _make_pomerol_ed(index_converter, non_int_part(h))

        hloc = _extract_hloc(h, gf_struct)[up_dn]
        hloc[norb:, norb:] *= -1

        # Real frequency
        gf_args = [gf_struct, beta, energy_window, n_w]

        g_w = ed.G_w(*gf_args, broadening, **tols)[up_dn]
        f_w = ed.F_w(*gf_args, broadening, **tols)[up_dn][:norb, norb:]
        fbar_w = transpose(conjugate(
            ed.F_w(*gf_args, -broadening, **tols)[up_dn][:norb, norb:]
        ))

        g0_w = ed0.G_w(*gf_args, broadening, **tols)[up_dn]
        f0_w = ed0.F_w(*gf_args, broadening, **tols)[up_dn][:norb, norb:]
        f0bar_w = transpose(conjugate(
            ed0.F_w(*gf_args, -broadening, **tols)[up_dn][:norb, norb:]
        ))

        # Green's functions
        results['g_w'] = BlockGf(
            block_list=[g_w[:norb, :norb], g_w[norb:, norb:]],
            name_list=[up, dn]
        )
        results['g_an_w'] = BlockGf(block_list=[f_w], name_list=[up_dn])

        # Non-interacting Green's functions
        results['g0_w'] = BlockGf(
            block_list=[g0_w[:norb, :norb], g0_w[norb:, norb:]],
            name_list=[up, dn]
        )
        results['g0_an_w'] = BlockGf(block_list=[f0_w], name_list=[up_dn])

        # Self-energies
        G_nambu_w = _make_nambu_gf(g_w, f_w, fbar_w)
        G0_nambu_w = _make_nambu_gf(g0_w, f0_w, f0bar_w)
        Sigma_nambu_w = dyson(G0_iw=G0_nambu_w, G_iw=G_nambu_w)
        Sigma_w, Sigma_an_w = _unpack_nambu_gf(Sigma_nambu_w, spins)
        results['Sigma_w'] = Sigma_w
        results['Sigma_an_w'] = Sigma_an_w

        # Hybridization functions
        Delta_nambu_w = G0_nambu_w.copy()
        Delta_nambu_w << Omega + 1j * broadening - hloc - inverse(G0_nambu_w)
        Delta_w, Delta_an_w = _unpack_nambu_gf(Delta_nambu_w, spins)
        results['Delta_w'] = Delta_w
        results['Delta_an_w'] = Delta_an_w

        # Matsubara frequency
        if not zerotemp:
            g_iw = ed.G_iw(gf_struct, beta, n_iw, **tols)[up_dn]
            f_iw = ed.F_iw(gf_struct, beta, n_iw, **tols)[up_dn][:norb, norb:]
            fbar_iw = _reflect_freq(transpose(conjugate(f_iw)))

            g0_iw = ed0.G_iw(gf_struct, beta, n_iw, **tols)[up_dn]
            f0_iw = ed0.F_iw(gf_struct, beta, n_iw, **tols)[up_dn][:norb, norb:]
            f0bar_iw = _reflect_freq(transpose(conjugate(f0_iw)))

            # Green's functions
            results['g_iw'] = BlockGf(
                block_list=[g_iw[:norb, :norb], g_iw[norb:, norb:]],
                name_list=[up, dn]
            )
            results['g_an_iw'] = BlockGf(block_list=[f_iw], name_list=[up_dn])

            # Non-interacting Green's functions
            results['g0_iw'] = BlockGf(
                block_list=[g0_iw[:norb, :norb], g0_iw[norb:, norb:]],
                name_list=[up, dn]
            )
            results['g0_an_iw'] = BlockGf(block_list=[f0_iw], name_list=[up_dn])

            # Self-energies
            G_nambu_iw = _make_nambu_gf(g_iw, f_iw, fbar_iw)
            G0_nambu_iw = _make_nambu_gf(g0_iw, f0_iw, f0bar_iw)
            Sigma_nambu_iw = dyson(G0_iw=G0_nambu_iw, G_iw=G_nambu_iw)
            Sigma_iw, Sigma_an_iw = _unpack_nambu_gf(Sigma_nambu_iw, spins)
            results['Sigma_iw'] = Sigma_iw
            results['Sigma_an_iw'] = Sigma_an_iw

            # Hybridization functions
            Delta_nambu_iw = G0_nambu_iw.copy()
            Delta_nambu_iw << iOmega_n - hloc - inverse(G0_nambu_iw)
            Delta_iw, Delta_an_iw = _unpack_nambu_gf(Delta_nambu_iw, spins)
            results['Delta_iw'] = Delta_iw
            results['Delta_an_iw'] = Delta_an_iw

    #
    # Response functions
    #

    if True in (chi_spin, chi_dens, chi_pair, chi_exct):
        make_reference_chi_results(results, ed, mki,
                                   spins=spins, orbs=orbs,
                                   beta=beta,
                                   n_iw=n_iw,
                                   energy_window=energy_window,
                                   n_w=n_w,
                                   broadening=broadening,
                                   n_tau=n_tau,
                                   tols=tols,
                                   zerotemp=zerotemp,
                                   chi_spin=chi_spin,
                                   chi_dens=chi_dens,
                                   chi_pair=chi_pair,
                                   chi_exct=chi_exct)

    return results


def make_reference_chi_results(results, ed, mki, *,
                               spins, orbs, beta,
                               n_iw,
                               energy_window, n_w, broadening,
                               n_tau,
                               tols,
                               zerotemp,
                               chi_spin, chi_dens, chi_pair, chi_exct):
    "Generate reference results for susceptibilities using pomerol2triqs"

    up, dn = spins
    norb = len(orbs)

    def chi_w(ind1, ind2, ind3, ind4, channel='PH'):
        return ed.chi_w(ind1, ind2, ind3, ind4, beta,
                        energy_window, n_w, broadening, channel=channel, **tols)

    def chi_iw(ind1, ind2, ind3, ind4, channel='PH'):
        return ed.chi_iw(ind1, ind2, ind3, ind4, beta, n_iw,
                         channel=channel, **tols)

    def chi_tau(ind1, ind2, ind3, ind4, channel='PH'):
        return ed.chi_tau(ind1, ind2, ind3, ind4, beta, n_tau,
                          channel=channel, **tols)

    axes = [('w', MeshReFreq(energy_window, n_w), chi_w)]
    if not zerotemp:
        axes.append(('iw', MeshImFreq(beta, "Boson", n_iw), chi_iw))
        axes.append(('tau', MeshImTime(beta, "Boson", n_tau), chi_tau))

    for axis, mesh, func in axes:
        # Spin and density
        if chi_spin or chi_dens:
            chi_nn = np.asarray([
                Gf(mesh=mesh, target_shape=(norb, norb)) for _ in range(4)
            ]).reshape((2, 2))

            for (s1, spin1), (s2, spin2) in product(enumerate(spins), repeat=2):
                for o1, o2 in product(orbs, orbs):
                    ind1, ind2 = mki(spin1, o1), mki(spin2, o2)
                    chi_nn[s1, s2][o1, o2] = func(ind1, ind1, ind2, ind2)
        if chi_spin:
            results[f"chi_spin_{axis}"] = 0.25 * (
                chi_nn[0, 0] - chi_nn[0, 1] - chi_nn[1, 0] + chi_nn[1, 1])
        if chi_dens:
            results[f"chi_dens_{axis}"] = np.sum(chi_nn)

        # Exciton
        if chi_exct:
            chi_exct = Gf(mesh=mesh, target_shape=(3, norb, norb))
            for o1, o2 in product(orbs, orbs):
                for (s1, spin1), (s2, spin2) in product(enumerate(spins),
                                                        repeat=2):
                    chi_ss = func(mki(spin1, o2), mki(spin1, o1),
                                  mki(spin2, o1), mki(spin2, o2))
                    # Singlet
                    chi_exct[0, o1, o2] += chi_ss
                    # Triplet z
                    chi_exct[2, o1, o2] += (-1) ** int(s1 != s2) * chi_ss
                    # Triplet x
                    chi_exct[1, o1, o2] += func(
                        mki(spin1, o2), mki(spins[1 - s1], o1),
                        mki(spin2, o1), mki(spins[1 - s2], o2))
            results[f"chi_exct_{axis}"] = chi_exct

        # Pair
        if chi_pair:
            chi_pair = Gf(mesh=mesh, target_shape=(norb, norb))
            for o1, o2 in product(orbs, orbs):
                chi_pair[o1, o2] = func(
                    mki(dn, o1), mki(dn, o2), mki(up, o1), mki(up, o2),
                    channel='PP')
            results[f"chi_pair_{axis}"] = chi_pair


def _make_index_converter(fops, spins, norb, spin_blocks):
    """
    Conversion from TRIQS to Pomerol notation for operator indices
    TRIQS: block_name, inner_index
    Pomerol: site_label, orbital_index, spin_name
    """
    up, dn = spins
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
    return index_converter


def _merge_spin_blocks(bl, i, spins, norb):
    "Map from the spin block notation to the notation with a single block"
    up, dn = spins
    if bl == up:
        return (f"{up}_{dn}", i)
    elif bl == dn:
        return (f"{up}_{dn}", norb + i)
    else:
        return (bl, i)


def _merge_spin_blocks_in_expr(h, spins, norb):
    """
    Translate a many-body operator expression from the spin block notation to
    the notation with a single block.
    """
    res = op.Operator()
    for mon, coeff in h:
        new_mon = [
            (dag, _merge_spin_blocks(*ind, spins, norb)) for dag, ind in mon
        ]
        res += coeff * monomial2op(new_mon)
    return res


def _extract_hloc(h, gf_struct):
    """
    Extract local quadratic part from a Hamiltonian expression.
    """
    h_dict = extract_h_dict(h, True)
    hloc = {}
    for bn, bs in gf_struct:
        hloc_b = np.zeros((bs, bs), dtype=complex)
        for i, j in product(range(bs), repeat=2):
            hloc_b[i, j] = h_dict.get(((bn, i), (bn, j)), .0)
        hloc[bn] = hloc_b
    return hloc


def _make_pomerol_ed(index_converter, h):
    "Construct a PomerolED object"
    from pomerol2triqs import PomerolED

    ed = PomerolED(index_converter, verbose=False)
    ed.ops_melem_tol = 0
    ed.rho_threshold = 0
    ed.diagonalize(h)
    return ed


def _reflect_freq(g):
    "Apply (complex) frequency reflection z -> -z to a Green's function"
    res = g.copy()
    res.data[:] = np.flip(res.data, axis=0)
    return res


def _make_nambu_gf(g, f, fbar):
    "Make a Nambu Green's function from spin-diagonal and anomalous components"
    mesh = g.mesh
    norb = g.target_shape[0] // 2

    g_nam = Gf(mesh=mesh, target_shape=(2 * norb, 2 * norb))
    g_nam[:norb, :norb] = g[:norb, :norb]
    g_nam[:norb, norb:] = f
    g_nam[norb:, :norb] = fbar
    # Fill remaining Nambu blocks using symmetry relations
    if isinstance(mesh, MeshImFreq):
        g_nam[norb:, norb:] = -conjugate(g[norb:, norb:])
    else:  # MeshReFreq
        assert mesh.w_min == -mesh.w_max
        g_nam[norb:, norb:] = -_reflect_freq(conjugate(g[norb:, norb:]))

    return g_nam


def _unpack_nambu_gf(g_nambu, spins):
    """
    Extract spin-diagonal and anomalous components from a Nambu Green's
    function.
    """
    up, dn = spins
    up_dn = f"{up}_{dn}"
    mesh = g_nambu.mesh
    norb = g_nambu.target_shape[0] // 2

    G = BlockGf(mesh=mesh, gf_struct=[(up, norb), (dn, norb)])
    G[up] = g_nambu[:norb, :norb]
    # Use symmetry relation to fill g[dn]
    if isinstance(mesh, MeshImFreq):
        G[dn] = -conjugate(g_nambu[norb:, norb:])
    else:
        G[dn] = -_reflect_freq(conjugate(g_nambu[norb:, norb:]))
    f = BlockGf(block_list=[g_nambu[:norb, norb:]], name_list=[up_dn])

    return G, f

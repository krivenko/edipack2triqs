import unittest
import gc
import inspect
from itertools import product

import numpy as np
from numpy.testing import assert_allclose

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
from triqs.utility.comparison_tests import (assert_gfs_are_close,
                                            assert_block_gfs_are_close)

from h5 import HDFArchive

from edipack2triqs.util import monomial2op, non_int_part


def make_pomerol_ed(index_converter, h):
    "Construct a PomerolED object"
    from pomerol2triqs import PomerolED

    ed = PomerolED(index_converter, verbose=False)
    ed.ops_melem_tol = 0
    ed.rho_threshold = 0
    ed.diagonalize(h)
    return ed


def extract_hloc(h, gf_struct):
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


def extract_hloc_an(h, norb, bn):
    """
    Extract anomalous impurity terms from a Hamiltonian expression.
    """
    h_loc_an = np.zeros((norb, norb), dtype=complex)
    for mon, coeff in h:
        if len(mon) != 2:
            continue
        dag1, ind1 = mon[0]
        dag2, ind2 = mon[1]
        if ((dag1, dag2) != (True, True)
           or (ind1[0] != bn) or (ind2[0] != bn)):
            continue
        spin1, orb1 = divmod(ind1[1], norb)
        spin2, orb2 = divmod(ind2[1], norb)
        if (spin1 == 0) and (spin2 == 1):
            h_loc_an[orb1, orb2] = coeff
        elif (spin1 == 1) and (spin2 == 0):
            h_loc_an[orb2, orb1] = -coeff
    return h_loc_an


def make_op(o: op.Operator, mkind):
    "Returns a function f(*args) -> o(*mkind(*args))"
    return lambda *args: o(*mkind(*args))


def herm(g):
    "Hermitian conjugate"
    return transpose(conjugate(g))


def reflect_freq(g):
    "Apply (complex) frequency reflection z -> -z to a Green's function"
    res = g.copy()
    res.data[:] = np.flip(res.data, axis=0)
    return res


class TestSolver(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Set cls.fops_bath_* if cls.mkind_bath is present
        if not hasattr(cls, 'mkind_bath'):
            return
        mkind = cls.mkind_bath
        args = list(mkind.index_ranges)
        cls.fops_bath_up = [mkind(cls.spins[0], *arg) for arg in args]
        cls.fops_bath_dn = [mkind(cls.spins[1], *arg) for arg in args]

    def tearDown(self):
        "Make sure EDIpackSolver.__del__() is called"
        gc.collect()

    #
    # Common parameters of the impurity
    #

    spins = ('up', 'dn')
    up_dn = spins[0] + "_" + spins[1]

    norb = 2
    orbs = range(norb)

    @classmethod
    def make_mkind_imp(cls, spin_blocks):
        "Returns a mapping (spin, orb) -> (block name, inner index)"
        if spin_blocks:
            return lambda spin, o: (spin, o)
        else:
            return lambda spin, o: (cls.up_dn,
                                    cls.norb * cls.spins.index(spin) + o)

    @classmethod
    def make_fops_imp(cls, spin_blocks=True):
        "Returns the up-down pair of impurity fundamental operator sets"
        mkind = cls.make_mkind_imp(spin_blocks)
        return tuple([mkind(spin, o) for o in cls.orbs] for spin in cls.spins)

    @classmethod
    def make_op_imp(cls, o, spin_blocks):
        "Returns a function f(spin, orb) -> o(*mkind_imp(spin, orb))"
        return make_op(o, cls.make_mkind_imp(spin_blocks))

    @classmethod
    def _merge_spin_blocks(cls, bl, i):
        "Map from the spin block notation to the notation with a single block"
        if bl == cls.spins[0]:
            return (cls.up_dn, i)
        elif bl == cls.spins[1]:
            return (cls.up_dn, cls.norb + i)
        else:
            return (bl, i)

    @classmethod
    def _merge_spin_blocks_in_expr(cls, h):
        """
        Translate a many-body operator expression from the spin block notation
        to the notation with a single block.
        """
        res = op.Operator()
        for mon, coeff in h:
            new_mon = [
                (dag, cls._merge_spin_blocks(*ind)) for dag, ind in mon
            ]
            res += coeff * monomial2op(new_mon)
        return res

    #
    # Common bath-related routines
    #

    @classmethod
    def bath_index_ranges(cls, *ranges):
        "Decorator: Attach ranges of bath indices to a mkind_bath function"
        def wrapper(f):
            f.index_ranges = product(*ranges)
            return f
        return wrapper

    @classmethod
    def make_bath_op(cls, o):
        "Returns a function f(spin, ...) -> o(*mkind_bath(spin, ...))"
        return make_op(o, cls.mkind_bath)

    #
    # Class methods returning various contributions to the Hamiltonian operator
    # object.
    #

    @classmethod
    def make_h_loc(cls, mat, spin_blocks=True):
        d_dag, d = [cls.make_op_imp(o, spin_blocks) for o in (op.c_dag, op.c)]
        return sum(mat[s1, s2, o1, o2] * d_dag(spin1, o1) * d(spin2, o2)
                   for (s1, spin1), (s2, spin2), o1, o2
                   in product(enumerate(cls.spins), enumerate(cls.spins),
                              cls.orbs, cls.orbs))

    @classmethod
    def make_h_loc_an(cls, mat):
        d_dag, d = [cls.make_op_imp(o, True) for o in (op.c_dag, op.c)]
        return sum(mat[o1, o2] * d_dag('up', o1) * d_dag('dn', o2)
                   + np.conj(mat[o1, o2]) * d('dn', o2) * d('up', o1)
                   for o1, o2 in product(cls.orbs, cls.orbs))

    @classmethod
    def make_h_int(cls, *, Uloc, Ust, Jh, Jx, Jp, spin_blocks=True):
        d_dag, d, n = [cls.make_op_imp(o, spin_blocks)
                       for o in (op.c_dag, op.c, op.n)]
        h_int = sum(Uloc[o] * n('up', o) * n('dn', o) for o in cls.orbs)
        h_int += Ust * sum(int(o1 != o2) * n('up', o1) * n('dn', o2)
                           for o1, o2 in product(cls.orbs, cls.orbs))
        h_int += (Ust - Jh) * \
            sum(int(o1 < o2) * n(spin, o1) * n(spin, o2)
                for spin, o1, o2 in product(cls.spins, cls.orbs, cls.orbs))
        h_int -= Jx * sum(int(o1 != o2)
                          * d_dag('up', o1) * d('dn', o1)
                          * d_dag('dn', o2) * d('up', o2)
                          for o1, o2 in product(cls.orbs, cls.orbs))
        h_int += Jp * sum(int(o1 != o2)
                          * d_dag('up', o1) * d_dag('dn', o1)
                          * d('dn', o2) * d('up', o2)
                          for o1, o2 in product(cls.orbs, cls.orbs))
        return h_int

    @classmethod
    def change_int_params(cls, U, *, Uloc, Ust, Jh, Jx, Jp):
        # Uloc
        for s, o in product(range(2), cls.orbs):
            U[o, s, o, 1 - s, o, s, o, 1 - s] = 0.5 * Uloc[o]
            U[o, s, o, 1 - s, o, 1 - s, o, s] = -0.5 * Uloc[o]
        for s, o1, o2 in product(range(2), cls.orbs, cls.orbs):
            if o1 == o2:
                continue
            # Ust
            U[o1, s, o2, 1 - s, o1, s, o2, 1 - s] = 0.5 * Ust
            U[o1, s, o2, 1 - s, o2, 1 - s, o1, s] = -0.5 * Ust
            # Ust - Jh
            U[o1, s, o2, s, o1, s, o2, s] = 0.5 * (Ust - Jh)
            U[o1, s, o2, s, o2, s, o1, s] = -0.5 * (Ust - Jh)
            # Jx
            U[o1, s, o2, 1 - s, o2, s, o1, 1 - s] = 0.5 * Jx
            U[o1, s, o2, 1 - s, o1, 1 - s, o2, s] = -0.5 * Jx
            # Jp
            U[o1, s, o1, 1 - s, o2, s, o2, 1 - s] = 0.5 * Jp
            U[o1, s, o1, 1 - s, o2, 1 - s, o2, s] = -0.5 * Jp

    #
    # Assertion methods
    #

    @classmethod
    def assert_static_obs(cls, solver, atol, **refs):
        "Assert correctness of computed static observables."
        assert_allclose(solver.densities, refs['densities'], atol=atol)
        assert_allclose(solver.double_occ, refs['double_occ'], atol=atol)
        assert_allclose(solver.magnetization[:, 0], refs['magn_x'], atol=atol)
        assert_allclose(solver.magnetization[:, 1], refs['magn_y'], atol=atol)
        assert_allclose(solver.magnetization[:, 2], refs['magn_z'], atol=atol)
        if 'phi' in refs:
            assert_allclose(
                solver.superconductive_phi
                * np.exp(1j * solver.superconductive_phi_arg),
                refs['phi'],
                atol=atol
            )

    @classmethod
    def assert_gfs(cls, solver, atol=1e-6, has_bath=True, **refs):
        "Assert correctness of computed GFs and related quantities."
        if has_bath:
            gf_list = ['g_iw', 'g_an_iw', 'g_w', 'g_an_w',
                       'g0_iw', 'g0_an_iw', 'g0_w', 'g0_an_w',
                       'Sigma_iw', 'Sigma_an_iw', 'Sigma_w', 'Sigma_an_w',
                       'Delta_iw', 'Delta_an_iw', 'Delta_w', 'Delta_an_w']
        else:
            gf_list = ['g_iw', 'Sigma_iw', 'g_w', 'Sigma_w']
        for gf in gf_list:
            if gf in refs:
                try:
                    assert_block_gfs_are_close(getattr(solver, gf), refs[gf],
                                               precision=atol)
                except AssertionError as error:
                    print(f"Failed check for {gf}:")
                    raise error

    @classmethod
    def assert_chi(cls, solver, atol=1e-6, **refs):
        "Assert correctness of computed susceptibilities."
        for axis, chan in product(['iw', 'w', 'tau'],
                                  ['spin', 'dens', 'pair', 'exct']):
            chi = f"chi_{chan}_{axis}"
            if chi in refs:
                try:
                    assert_gfs_are_close(getattr(solver, chi), refs[chi],
                                         precision=atol)
                except AssertionError as error:
                    print(f"Failed check for {chi}:")
                    raise error

    #
    # Miscellaneous
    #

    @classmethod
    def _make_index_converter(cls, fops, spin_blocks):
        """
        Conversion from TRIQS to Pomerol notation for operator indices
        TRIQS: block_name, inner_index
        Pomerol: site_label, orbital_index, spin_name
        """
        up, dn = cls.spins
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
                    spin = "down" if inner >= cls.norb else up
                orb = inner % cls.norb
            index_converter[(bn, inner)] = (site, orb, spin)
        return index_converter

    # solve() parameters that enable computation of susceptibilities
    chi_params = {f"chi_{ch}": True for ch in ["spin", "dens", "pair", "exct"]}

    #
    # Generate reference results
    #

    # Call pomerol2triqs to obtain reference data for unit tests and
    # write computed reference results into HDF5 archive.
    generate_ref_data = False

    @classmethod
    def ref_results(cls, h5_group_name, **params):
        "Either generate reference results or load them from an HDF5 archive"

        filename = inspect.stack()[1].filename[:-2] + "h5"
        if cls.generate_ref_data:
            results = cls.make_reference_results(**params)
            with HDFArchive(filename, 'a') as ar:
                ar.create_group(h5_group_name)
                ar[h5_group_name] = results
                return results
        else:
            with HDFArchive(filename, 'r') as ar:
                return ar[h5_group_name]

    @classmethod
    def make_reference_results(cls, *,
                               h,
                               fops,
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

        mki = cls.make_mkind_imp(spin_blocks)

        index_converter = cls._make_index_converter(fops, spin_blocks)
        ed = make_pomerol_ed(index_converter, h)

        results = {}

        # Static observables
        cls.make_reference_static_obs_results(results, ed, mki, beta=beta,
                                              superc=superc)

        # Tolerances for construction of Lehmann representation
        tols = {"pole_res": 1e-12, "coeff_tol": 1e-12}

        # Green's functions and self-energies
        if superc:
            # Calculations using single block up_dn
            index_converter_sb = cls._make_index_converter(
                [cls._merge_spin_blocks(bl, i) for bl, i in fops],
                False
            )
            h_sb = cls._merge_spin_blocks_in_expr(h)
            ed_sb = make_pomerol_ed(index_converter_sb, h_sb)
            ed0_sb = make_pomerol_ed(index_converter_sb, non_int_part(h_sb))
            cls.make_reference_gfs_superc_results(results, h_sb, ed_sb, ed0_sb,
                                                  beta=beta,
                                                  n_iw=n_iw,
                                                  energy_window=energy_window,
                                                  n_w=n_w,
                                                  broadening=broadening,
                                                  tols=tols,
                                                  zerotemp=zerotemp)
        else:
            ed0 = make_pomerol_ed(index_converter, non_int_part(h))
            cls.make_reference_gfs_normal_results(results, h, ed, ed0,
                                                  beta=beta,
                                                  n_iw=n_iw,
                                                  energy_window=energy_window,
                                                  n_w=n_w,
                                                  broadening=broadening,
                                                  spin_blocks=spin_blocks,
                                                  tols=tols,
                                                  zerotemp=zerotemp)

        # Response functions
        if any((chi_spin, chi_dens, chi_pair, chi_exct)):
            cls.make_reference_chi_results(results, ed, mki,
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

    @classmethod
    def make_reference_static_obs_results(cls, results, ed, mki, *,
                                          beta, superc):
        "Generate reference results for static observables using pomerol2triqs"
        def make_avg(spins):
            res = [
                ed.ensemble_average(*[mki(spin, o) for spin in spins], beta)
                for o in cls.orbs
            ]
            return np.array(res)

        # Occupation and magnetization
        n_up_avg = make_avg(('up', 'up'))
        n_dn_avg = make_avg(('dn', 'dn'))
        S_p_avg = make_avg(('up', 'dn'))
        S_m_avg = make_avg(('dn', 'up'))
        # Double occupancy
        D_avg = make_avg(('up', 'dn', 'dn', 'up'))

        results['densities'] = n_up_avg + n_dn_avg
        results['double_occ'] = D_avg
        results['magn_x'] = S_p_avg + S_m_avg
        results['magn_y'] = -1j * (S_p_avg - S_m_avg)
        results['magn_z'] = n_up_avg - n_dn_avg

        # Superconductive \phi
        if superc:
            phi = np.empty((cls.norb, cls.norb), dtype=complex)
            for o1, o2 in product(cls.orbs, repeat=2):
                phi[o1, o2] = ed.ensemble_average(
                    mki('up', o1), mki('dn', o2), beta, (False, False)
                )
            results['phi'] = phi

    @classmethod
    def make_reference_gfs_normal_results(cls, results, h, ed, ed0, *,
                                          beta,
                                          n_iw,
                                          energy_window, n_w, broadening,
                                          spin_blocks,
                                          tols,
                                          zerotemp):
        """
        Generate reference results for GFs and related quantities in the normal
        (non-superconducting) case using pomerol2triqs.
        """
        if spin_blocks:
            gf_struct = [(spin, cls.norb) for spin in cls.spins]
        else:
            gf_struct = [(cls.up_dn, 2 * cls.norb)]

        hloc = extract_hloc(h, gf_struct)

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
        if zerotemp:
            return
        g_iw = ed.G_iw(gf_struct, beta, n_iw, **tols)
        g0_iw = ed0.G_iw(gf_struct, beta, n_iw, **tols)
        results['g_iw'] = g_iw
        results['g0_iw'] = g0_iw
        results['Sigma_iw'] = dyson(G0_iw=g0_iw, G_iw=g_iw)
        Delta_iw = g0_iw.copy()
        for b, sn in gf_struct:
            Delta_iw[b] << iOmega_n - hloc[b] - inverse(g0_iw[b])
        results['Delta_iw'] = Delta_iw

    @classmethod
    def make_reference_gfs_superc_results(cls, results, h, ed, ed0, *,
                                          beta,
                                          n_iw,
                                          energy_window, n_w, broadening,
                                          tols,
                                          zerotemp):
        """
        Generate reference results for GFs and related quantities in the
        superconducting case using pomerol2triqs.
        """
        gf_struct = [(cls.up_dn, 2 * cls.norb)]

        # Slices for extraction of Nambu blocks
        nam11 = (slice(None, cls.norb), slice(None, cls.norb))
        nam12 = (slice(None, cls.norb), slice(cls.norb, None))
        nam21 = (slice(cls.norb, None), slice(None, cls.norb))
        nam22 = (slice(cls.norb, None), slice(cls.norb, None))

        def make_nambu_gf(g, f, fbar):
            """
            Make a Nambu Green's function from spin-diagonal and anomalous
            components.
            """
            mesh = g.mesh
            g_nam = Gf(mesh=mesh, target_shape=(2 * cls.norb, 2 * cls.norb))
            g_nam[nam11] = g[nam11]
            g_nam[nam12] = f
            g_nam[nam21] = fbar
            # Fill remaining Nambu blocks using symmetry relations
            if isinstance(mesh, MeshImFreq):
                g_nam[nam22] = -conjugate(g[nam22])
            else:  # MeshReFreq
                assert mesh.w_min == -mesh.w_max
                g_nam[nam22] = -reflect_freq(conjugate(g[nam22]))
            return g_nam

        def unpack_nambu_gf(g_nambu):
            """
            Extract spin-diagonal and anomalous components from a Nambu Green's
            function.
            """
            mesh = g_nambu.mesh
            G = BlockGf(mesh=mesh,
                        gf_struct=[('up', cls.norb), ('dn', cls.norb)])
            G['up'] = g_nambu[nam11]
            # Use symmetry relation to fill g[dn]
            if isinstance(mesh, MeshImFreq):
                G['dn'] = -conjugate(g_nambu[nam22])
            else:
                G['dn'] = -reflect_freq(conjugate(g_nambu[nam22]))
            f = BlockGf(block_list=[g_nambu[nam12]], name_list=[cls.up_dn])
            return G, f

        hloc = extract_hloc(h, gf_struct)[cls.up_dn]
        hloc[nam22] *= -1
        hloc_an = extract_hloc_an(h, cls.norb, cls.up_dn)
        hloc[nam12] = hloc_an
        hloc[nam21] = np.conj(hloc_an)

        # Real frequency
        gf_args = [gf_struct, beta, energy_window, n_w]

        g_w = ed.G_w(*gf_args, broadening, **tols)[cls.up_dn]
        f_w = ed.F_w(*gf_args, broadening, **tols)[cls.up_dn][nam12]
        fbar_w = herm(ed.F_w(*gf_args, -broadening, **tols)[cls.up_dn][nam12])

        g0_w = ed0.G_w(*gf_args, broadening, **tols)[cls.up_dn]
        f0_w = ed0.F_w(*gf_args, broadening, **tols)[cls.up_dn][nam12]
        f0bar_w = herm(ed0.F_w(*gf_args, -broadening, **tols)[cls.up_dn][nam12])

        # Green's functions
        results['g_w'] = BlockGf(
            block_list=[g_w[nam11], g_w[nam22]], name_list=cls.spins
        )
        results['g_an_w'] = BlockGf(block_list=[f_w], name_list=[cls.up_dn])

        # Non-interacting Green's functions
        results['g0_w'] = BlockGf(
            block_list=[g0_w[nam11], g0_w[nam22]], name_list=cls.spins
        )
        results['g0_an_w'] = BlockGf(block_list=[f0_w], name_list=[cls.up_dn])

        # Self-energies
        G_nambu_w = make_nambu_gf(g_w, f_w, fbar_w)
        G0_nambu_w = make_nambu_gf(g0_w, f0_w, f0bar_w)
        Sigma_nambu_w = dyson(G0_iw=G0_nambu_w, G_iw=G_nambu_w)
        Sigma_w, Sigma_an_w = unpack_nambu_gf(Sigma_nambu_w)
        results['Sigma_w'] = Sigma_w
        results['Sigma_an_w'] = Sigma_an_w

        # Hybridization functions
        Delta_nambu_w = G0_nambu_w.copy()
        Delta_nambu_w << Omega + 1j * broadening - hloc - inverse(G0_nambu_w)
        Delta_w, Delta_an_w = unpack_nambu_gf(Delta_nambu_w)
        results['Delta_w'] = Delta_w
        results['Delta_an_w'] = Delta_an_w

        # Matsubara frequency
        if zerotemp:
            return

        g_iw = ed.G_iw(gf_struct, beta, n_iw, **tols)[cls.up_dn]
        f_iw = ed.F_iw(gf_struct, beta, n_iw, **tols)[cls.up_dn][nam12]
        fbar_iw = reflect_freq(herm(f_iw))

        g0_iw = ed0.G_iw(gf_struct, beta, n_iw, **tols)[cls.up_dn]
        f0_iw = ed0.F_iw(gf_struct, beta, n_iw, **tols)[cls.up_dn][nam12]
        f0bar_iw = reflect_freq(herm(f0_iw))

        # Green's functions
        results['g_iw'] = BlockGf(
            block_list=[g_iw[nam11], g_iw[nam22]], name_list=cls.spins
        )
        results['g_an_iw'] = BlockGf(block_list=[f_iw], name_list=[cls.up_dn])

        # Non-interacting Green's functions
        results['g0_iw'] = BlockGf(
            block_list=[g0_iw[nam11], g0_iw[nam22]], name_list=cls.spins
        )
        results['g0_an_iw'] = BlockGf(block_list=[f0_iw], name_list=[cls.up_dn])

        # Self-energies
        G_nambu_iw = make_nambu_gf(g_iw, f_iw, fbar_iw)
        G0_nambu_iw = make_nambu_gf(g0_iw, f0_iw, f0bar_iw)
        Sigma_nambu_iw = dyson(G0_iw=G0_nambu_iw, G_iw=G_nambu_iw)
        Sigma_iw, Sigma_an_iw = unpack_nambu_gf(Sigma_nambu_iw)
        results['Sigma_iw'] = Sigma_iw
        results['Sigma_an_iw'] = Sigma_an_iw

        # Hybridization functions
        Delta_nambu_iw = G0_nambu_iw.copy()
        Delta_nambu_iw << iOmega_n - hloc - inverse(G0_nambu_iw)
        Delta_iw, Delta_an_iw = unpack_nambu_gf(Delta_nambu_iw)
        results['Delta_iw'] = Delta_iw
        results['Delta_an_iw'] = Delta_an_iw

    @classmethod
    def make_reference_chi_results(cls, results, ed, mki, *,
                                   beta,
                                   n_iw,
                                   energy_window, n_w, broadening,
                                   n_tau,
                                   tols,
                                   zerotemp,
                                   chi_spin, chi_dens, chi_pair, chi_exct):
        "Generate reference results for susceptibilities using pomerol2triqs"

        def chi_w(ind1, ind2, ind3, ind4, channel='PH'):
            return ed.chi_w(ind1, ind2, ind3, ind4, beta,
                            energy_window, n_w, broadening,
                            channel=channel, **tols)

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
                    Gf(mesh=mesh, target_shape=(cls.norb, cls.norb))
                    for _ in range(4)
                ]).reshape((2, 2))

                for (s1, spin1), (s2, spin2) in product(enumerate(cls.spins),
                                                        repeat=2):
                    for o1, o2 in product(cls.orbs, cls.orbs):
                        ind1, ind2 = mki(spin1, o1), mki(spin2, o2)
                        chi_nn[s1, s2][o1, o2] = func(ind1, ind1, ind2, ind2)
            if chi_spin:
                results[f"chi_spin_{axis}"] = 0.25 * (
                    chi_nn[0, 0] - chi_nn[0, 1] - chi_nn[1, 0] + chi_nn[1, 1])
            if chi_dens:
                results[f"chi_dens_{axis}"] = np.sum(chi_nn)

            # Exciton
            if chi_exct:
                chi_exct = Gf(mesh=mesh, target_shape=(3, cls.norb, cls.norb))
                for o1, o2 in product(cls.orbs, cls.orbs):
                    for (s1, spin1), (s2, spin2) in \
                            product(enumerate(cls.spins), repeat=2):
                        chi_ss = func(mki(spin1, o2), mki(spin1, o1),
                                      mki(spin2, o1), mki(spin2, o2))
                        # Singlet
                        chi_exct[0, o1, o2] += chi_ss
                        # Triplet z
                        chi_exct[2, o1, o2] += (-1) ** int(s1 != s2) * chi_ss
                        # Triplet x
                        chi_exct[1, o1, o2] += func(
                            mki(spin1, o2), mki(cls.spins[1 - s1], o1),
                            mki(spin2, o1), mki(cls.spins[1 - s2], o2))
                results[f"chi_exct_{axis}"] = chi_exct

            # Pair
            if chi_pair:
                chi_pair = Gf(mesh=mesh, target_shape=(cls.norb, cls.norb))
                for o1, o2 in product(cls.orbs, cls.orbs):
                    chi_pair[o1, o2] = func(
                        mki(cls.spins[1], o1), mki(cls.spins[1], o2),
                        mki(cls.spins[0], o1), mki(cls.spins[0], o2),
                        channel='PP')
                results[f"chi_pair_{axis}"] = chi_pair

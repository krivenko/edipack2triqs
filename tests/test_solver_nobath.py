import unittest
import gc
from itertools import product

import numpy as np
from numpy.testing import assert_allclose
from numpy import multiply as mul

import triqs.operators as op
from triqs.atom_diag import (AtomDiag,
                             atomic_density_matrix,
                             atomic_g_iw,
                             atomic_g_w,
                             trace_rho_op)
from triqs.gf.tools import dyson
from triqs.utility.comparison_tests import assert_block_gfs_are_close
from h5 import HDFArchive

from edipack2triqs.solver import EDIpackSolver
from edipack2triqs.util import non_int_part


generate_packaged_ref_results = False
packaged_ref_results_name = __file__[:-2] + "h5"

s0 = np.eye(2)
sx = np.array([[0, 1], [1, 0]])
sz = np.array([[1, 0], [0, -1]])


class TestEDIpackSolverNoBath(unittest.TestCase):

    spins = ('up', 'dn')
    orbs = range(2)

    @classmethod
    def make_fops_imp(cls, spin_blocks=True):
        if spin_blocks:
            return ([('up', o) for o in cls.orbs],
                    [('dn', o) for o in cls.orbs])
        else:
            return ([('up_dn', so) for so in cls.orbs],
                    [('up_dn', so + len(cls.orbs)) for so in cls.orbs])

    @classmethod
    def make_mkind(cls, spin_blocks):
        if spin_blocks:
            return lambda spin, o: (spin, o)
        else:
            return lambda spin, o: ('up_dn',
                                    len(cls.orbs) * cls.spins.index(spin) + o)

    @classmethod
    def make_h_loc(cls, h_loc, spin_blocks=True):
        mki = cls.make_mkind(spin_blocks)
        return sum(h_loc[s1, s2, o1, o2]
                   * op.c_dag(*mki(spin1, o1)) * op.c(*mki(spin2, o2))
                   for (s1, spin1), (s2, spin2), o1, o2
                   in product(enumerate(cls.spins), enumerate(cls.spins),
                              cls.orbs, cls.orbs))

    @classmethod
    def make_h_int(cls, *, Uloc, Ust, Jh, Jx, Jp, spin_blocks=True):
        mki = cls.make_mkind(spin_blocks)
        h_int = sum(Uloc[o] * op.n(*mki('up', o)) * op.n(*mki('dn', o))
                    for o in cls.orbs)
        h_int += Ust * sum(int(o1 != o2)
                           * op.n(*mki('up', o1)) * op.n(*mki('dn', o2))
                           for o1, o2 in product(cls.orbs, cls.orbs))
        h_int += (Ust - Jh) * \
            sum(int(o1 < o2) * op.n(*mki(s, o1)) * op.n(*mki(s, o2))
                for s, o1, o2 in product(cls.spins, cls.orbs, cls.orbs))
        h_int -= Jx * sum(int(o1 != o2)
                          * op.c_dag(*mki('up', o1)) * op.c(*mki('dn', o1))
                          * op.c_dag(*mki('dn', o2)) * op.c(*mki('up', o2))
                          for o1, o2 in product(cls.orbs, cls.orbs))
        h_int += Jp * sum(int(o1 != o2)
                          * op.c_dag(*mki('up', o1)) * op.c_dag(*mki('dn', o1))
                          * op.c(*mki('dn', o2)) * op.c(*mki('up', o2))
                          for o1, o2 in product(cls.orbs, cls.orbs))
        return h_int

    @classmethod
    def make_h_sc(cls, Delta):
        return sum(Delta[o, nu] * (
            op.c_dag('B_up', nu * 2 + o) * op.c_dag('B_dn', nu * 2 + o)
            + op.c('B_dn', nu * 2 + o) * op.c('B_up', nu * 2 + o))
            for o, nu in product(cls.orbs, range(2))
        )

    @classmethod
    def make_ref_results(
        cls, h, fops, beta, n_iw, energy_window, n_w, eta,
        h5_name,
        spin_blocks=True, zerotemp=False
    ):
        if not generate_packaged_ref_results:
            with HDFArchive(packaged_ref_results_name, 'r') as ar:
                return ar[h5_name]

        mki = cls.make_mkind(spin_blocks)

        c_up = [op.c(*mki('up', o)) for o in cls.orbs]
        c_dn = [op.c(*mki('dn', o)) for o in cls.orbs]
        c_dag_up = [op.c_dag(*mki('up', o)) for o in cls.orbs]
        c_dag_dn = [op.c_dag(*mki('dn', o)) for o in cls.orbs]
        n_up = [op.n(*mki('up', o)) for o in cls.orbs]
        n_dn = [op.n(*mki('dn', o)) for o in cls.orbs]

        N = [n_up[o] + n_dn[o] for o in cls.orbs]
        D = [n_up[o] * n_dn[o] for o in cls.orbs]
        S_x = [c_dag_up[o] * c_dn[o] + c_dag_dn[o] * c_up[o] for o in cls.orbs]
        S_y = [c_dag_dn[o] * c_up[o] - c_dag_up[o] * c_dn[o] for o in cls.orbs]
        S_z = [n_up[o] - n_dn[o] for o in cls.orbs]

        if spin_blocks:
            gf_struct = [('up', len(cls.orbs)), ('dn', len(cls.orbs))]
        else:
            gf_struct = [('up_dn', 2 * len(cls.orbs))]

        ad = AtomDiag(h, fops)
        rho = atomic_density_matrix(ad, beta)
        g_w = atomic_g_w(ad, beta, gf_struct, energy_window, n_w, eta)

        def avg(ops):
            return np.array([trace_rho_op(rho, ops[o], ad) for o in cls.orbs])

        results = {'densities': avg(N),
                   'double_occ': avg(D),
                   'magn_x': avg(S_x),
                   'magn_y': 1j * avg(S_y),
                   'magn_z': avg(S_z),
                   'g_w': g_w}

        if not zerotemp:
            g_iw = atomic_g_iw(ad, beta, gf_struct, n_iw)
            results['g_iw'] = g_iw

        h0 = non_int_part(h)
        ad0 = AtomDiag(h0, fops)
        g0_w = atomic_g_w(ad0, beta, gf_struct, energy_window, n_w, eta)
        results['Sigma_w'] = dyson(G0_iw=g0_w, G_iw=g_w)
        if not zerotemp:
            g0_iw = atomic_g_iw(ad0, beta, gf_struct, n_iw)
            results['Sigma_iw'] = dyson(G0_iw=g0_iw, G_iw=g_iw)

        if generate_packaged_ref_results:
            with HDFArchive(packaged_ref_results_name, 'a') as ar:
                ar.create_group(h5_name)
                ar[h5_name] = results

        return results

    @classmethod
    def change_int_params(cls, U, new_int_params):
        Uloc = new_int_params["Uloc"]
        Ust = new_int_params["Ust"]
        Jh = new_int_params["Jh"]
        Jx = new_int_params["Jx"]
        Jp = new_int_params["Jp"]
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

    @classmethod
    def assert_all(cls, s, **refs):
        assert_allclose(s.densities, refs['densities'], atol=1e-8)
        assert_allclose(s.double_occ, refs['double_occ'], atol=1e-8)
        assert_allclose(s.magnetization[:, 0], refs['magn_x'], atol=1e-8)
        assert_allclose(s.magnetization[:, 1], refs['magn_y'], atol=1e-8)
        assert_allclose(s.magnetization[:, 2], refs['magn_z'], atol=1e-8)
        assert_block_gfs_are_close(s.g_w, refs['g_w'])
        if 'g_iw' in refs:
            assert_block_gfs_are_close(s.g_iw, refs['g_iw'])
        if 'Sigma_w' in refs:
            assert_block_gfs_are_close(s.Sigma_w, refs['Sigma_w'])
        if 'Sigma_iw' in refs:
            assert_block_gfs_are_close(s.Sigma_iw, refs['Sigma_iw'])

    def test_zerotemp(self):
        h_loc = self.make_h_loc(mul.outer(s0, np.diag([-0.5, -0.6])))
        h_int = self.make_h_int(Uloc=np.array([1.0, 2.0]),
                                Ust=0.8,
                                Jh=0.2,
                                Jx=0.1,
                                Jp=0.15)

        fops_imp_up, fops_imp_dn = self.make_fops_imp()
        fops = fops_imp_up + fops_imp_dn

        h = h_loc + h_int
        solver = EDIpackSolver(
            h,
            fops_imp_up, fops_imp_dn,
            zerotemp=True,
            verbose=0,
            lanc_nstates_total=17,
            lanc_nstates_sector=17
        )

        self.assertEqual(solver.nspin, 1)
        self.assertEqual(solver.norb, 2)
        self.assertIsNone(solver.bath)

        n_iw = 100
        energy_window = (-2.0, 2.0)
        n_w = 600
        broadening = 0.05
        solver.solve(n_iw=n_iw,
                     energy_window=energy_window,
                     n_w=n_w,
                     broadening=broadening)

        ## Reference solution
        refs = self.make_ref_results(h, fops, 10000,
                                     n_iw,
                                     energy_window, n_w, broadening,
                                     "zerotemp",
                                     zerotemp=True)
        self.assert_all(solver, **refs)

    def test_nspin1(self):
        h_loc = self.make_h_loc(mul.outer(s0, np.diag([-0.5, -0.6])))
        h_int = self.make_h_int(Uloc=np.array([1.0, 2.0]),
                                Ust=0.8,
                                Jh=0.2,
                                Jx=0.1,
                                Jp=0.15)

        fops_imp_up, fops_imp_dn = self.make_fops_imp()
        fops = fops_imp_up + fops_imp_dn

        h = h_loc + h_int
        solver = EDIpackSolver(
            h,
            fops_imp_up, fops_imp_dn,
            verbose=0,
            lanc_nstates_total=17,
            lanc_nstates_sector=17
        )

        self.assertEqual(solver.nspin, 1)
        self.assertEqual(solver.norb, 2)
        self.assertIsNone(solver.bath)

        # Part I: Initial solve()
        beta = 100.0
        n_iw = 100
        energy_window = (-2.0, 2.0)
        n_w = 600
        broadening = 0.05
        solver.solve(beta=beta,
                     n_iw=n_iw,
                     energy_window=energy_window,
                     n_w=n_w,
                     broadening=broadening)

        ## Reference solution
        refs = self.make_ref_results(h, fops, beta,
                                     n_iw,
                                     energy_window, n_w, broadening,
                                     "nspin1_1")
        self.assert_all(solver, **refs)

        # Part II: update interaction parameters
        new_int_params = {'Uloc': np.array([2.0, 3.0]),
                          'Ust': 0.6,
                          'Jh': 0.1,
                          'Jx': 0.2,
                          'Jp': 0.0}
        self.change_int_params(solver.U, new_int_params)
        solver.comm.barrier()

        beta = 120.0
        n_iw = 200
        energy_window = (-1.5, 1.5)
        n_w = 400
        broadening = 0.03
        solver.solve(beta=beta,
                     n_iw=n_iw,
                     energy_window=energy_window,
                     n_w=n_w,
                     broadening=broadening)

        ## Reference solution
        h_int = self.make_h_int(**new_int_params)
        h = h_loc + h_int
        refs = self.make_ref_results(h, fops, beta,
                                     n_iw,
                                     energy_window, n_w, broadening,
                                     "nspin1_2")
        self.assert_all(solver, **refs)

    def test_nspin2(self):
        h_loc = self.make_h_loc(mul.outer(np.diag([0.8, 1.2]),
                                          np.diag([-0.5, -0.6])))
        h_int = self.make_h_int(Uloc=np.array([1.0, 2.0]),
                                Ust=0.8,
                                Jh=0.2,
                                Jx=0.1,
                                Jp=0.15)

        fops_imp_up, fops_imp_dn = self.make_fops_imp()
        fops = fops_imp_up + fops_imp_dn

        h = h_loc + h_int
        solver = EDIpackSolver(
            h,
            fops_imp_up, fops_imp_dn,
            verbose=0,
            lanc_nstates_total=17,
            lanc_nstates_sector=17
        )

        self.assertEqual(solver.nspin, 2)
        self.assertEqual(solver.norb, 2)
        self.assertIsNone(solver.bath)

        # Part I: Initial solve()
        beta = 100.0
        n_iw = 100
        energy_window = (-2.0, 2.0)
        n_w = 600
        broadening = 0.05
        solver.solve(beta=beta,
                     n_iw=n_iw,
                     energy_window=energy_window,
                     n_w=n_w,
                     broadening=broadening)

        ## Reference solution
        refs = self.make_ref_results(h, fops, beta,
                                     n_iw,
                                     energy_window, n_w, broadening,
                                     "nspin2_1")
        self.assert_all(solver, **refs)

        # Part II: update interaction parameters
        new_int_params = {'Uloc': np.array([2.0, 3.0]),
                          'Ust': 0.6,
                          'Jh': 0.1,
                          'Jx': 0.2,
                          'Jp': 0.0}
        self.change_int_params(solver.U, new_int_params)
        solver.comm.barrier()

        beta = 120.0
        n_iw = 200
        energy_window = (-1.5, 1.5)
        n_w = 400
        broadening = 0.03
        solver.solve(beta=beta,
                     n_iw=n_iw,
                     energy_window=energy_window,
                     n_w=n_w,
                     broadening=broadening)

        ## Reference solution
        h_int = self.make_h_int(**new_int_params)
        h = h_loc + h_int
        refs = self.make_ref_results(h, fops, beta,
                                     n_iw,
                                     energy_window, n_w, broadening,
                                     "nspin2_2")
        self.assert_all(solver, **refs)

    def test_nonsu2(self):
        h_loc = self.make_h_loc(mul.outer(np.array([[0.8, 0.2],
                                                    [0.2, 1.2]]),
                                          np.diag([-0.5, -0.6])),
                                spin_blocks=False)
        h_int = self.make_h_int(Uloc=np.array([1.0, 2.0]),
                                Ust=0.8,
                                Jh=0.2,
                                Jx=0.1,
                                Jp=0.15,
                                spin_blocks=False)

        fops_imp_up, fops_imp_dn = self.make_fops_imp(False)
        fops = fops_imp_up + fops_imp_dn

        h = h_loc + h_int
        solver = EDIpackSolver(
            h,
            fops_imp_up, fops_imp_dn,
            verbose=0,
            lanc_nstates_total=17,
            lanc_nstates_sector=17
        )

        self.assertEqual(solver.nspin, 2)
        self.assertEqual(solver.norb, 2)
        self.assertIsNone(solver.bath)

        # Part I: Initial solve()
        beta = 100.0
        n_iw = 100
        energy_window = (-1.5, 1.5)
        n_w = 600
        broadening = 0.05
        solver.solve(beta=beta,
                     n_iw=n_iw,
                     energy_window=energy_window,
                     n_w=n_w,
                     broadening=broadening)

        ## Reference solution
        refs = self.make_ref_results(h, fops, beta,
                                     n_iw,
                                     energy_window, n_w, broadening,
                                     "nonsu2_hloc_1",
                                     spin_blocks=False)
        self.assert_all(solver, **refs)

        # Part II: update interaction parameters
        new_int_params = {'Uloc': np.array([2.0, 3.0]),
                          'Ust': 0.6,
                          'Jh': 0.1,
                          'Jx': 0.2,
                          'Jp': 0.0}
        self.change_int_params(solver.U, new_int_params)
        solver.comm.barrier()

        beta = 120.0
        n_iw = 200
        energy_window = (-1.5, 1.5)
        n_w = 400
        broadening = 0.03
        solver.solve(beta=beta,
                     n_iw=n_iw,
                     energy_window=energy_window,
                     n_w=n_w,
                     broadening=broadening)

        ## Reference solution
        h_int = self.make_h_int(**new_int_params, spin_blocks=False)
        h = h_loc + h_int
        refs = self.make_ref_results(h, fops, beta,
                                     n_iw,
                                     energy_window, n_w, broadening,
                                     "nonsu2_hloc_2",
                                     spin_blocks=False)
        self.assert_all(solver, **refs)

    def tearDown(self):
        # Make sure EDIpackSolver.__del__() is called
        gc.collect()


if __name__ == '__main__':
    unittest.main()

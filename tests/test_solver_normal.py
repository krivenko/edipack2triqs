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
                             trace_rho_op)
from triqs.utility.comparison_tests import assert_block_gfs_are_close

from edipack2triqs.solver import EDIpackSolver


class TestEDIpackSolverBathNormal(unittest.TestCase):

    # Interaction parameters for make_H_int()
    spins = ('up', 'dn')
    orbs = range(2)
    fops_imp_up = [('up', o) for o in orbs]
    fops_imp_dn = [('dn', o) for o in orbs]

    def make_h_loc(self, h_loc):
        orbs = range(h_loc.shape[1])
        return sum(h_loc[s, o1, o2] * op.c_dag(spin, o1) * op.c(spin, o2)
                   for (s, spin), o1, o2
                   in product(enumerate(self.spins), orbs, orbs))

    def make_h_int(self, *, Uloc, Ust, Jh, Jx, Jp):
        h_int = sum(Uloc[o] * op.n('up', o) * op.n('dn', o) for o in self.orbs)
        h_int += Ust * sum(int(o1 != o2) * op.n('up', o1) * op.n('dn', o2)
                           for o1, o2 in product(self.orbs, self.orbs))
        h_int += (Ust - Jh) * \
            sum(int(o1 < o2) * op.n(s, o1) * op.n(s, o2)
                for s, o1, o2 in product(self.spins, self.orbs, self.orbs))
        h_int -= Jx * sum(int(o1 != o2)
                          * op.c_dag('up', o1) * op.c('dn', o1)
                          * op.c_dag('dn', o2) * op.c('up', o2)
                          for o1, o2 in product(self.orbs, self.orbs))
        h_int += Jp * sum(int(o1 != o2)
                          * op.c_dag('up', o1) * op.c_dag('dn', o1)
                          * op.c('dn', o2) * op.c('up', o2)
                          for o1, o2 in product(self.orbs, self.orbs))
        return h_int

    def make_h_bath(self, eps, V):
        h_bath = sum(eps[o, nu]
                     * op.c_dag("B_" + s, nu * 2 + o)
                     * op.c("B_" + s, nu * 2 + o)
                     for s, o, nu in product(self.spins, self.orbs, range(2)))
        h_bath += sum(V[o, nu] * (
                      op.c_dag(s, o) * op.c("B_" + s, nu * 2 + o)
                      + op.c_dag("B_" + s, nu * 2 + o) * op.c(s, o))
                      for s, o, nu in product(self.spins, self.orbs, range(2)))
        return h_bath

    def make_ref_results(self, h, fops, beta, n_iw):
        ad = AtomDiag(h, fops)
        rho = atomic_density_matrix(ad, beta)

        densities = [
            trace_rho_op(rho, op.n('up', o) + op.n('dn', o), ad)
            for o in self.orbs]
        double_occ = [
            trace_rho_op(rho, op.n('up', o) * op.n('dn', o), ad)
            for o in self.orbs]
        magnetization = [
            trace_rho_op(rho, op.n('up', o) - op.n('dn', o), ad)
            for o in self.orbs]

        gf_struct = [('up', len(self.orbs)), ('dn', len(self.orbs))]
        g_iw = atomic_g_iw(ad, beta, gf_struct, n_iw)

        return densities, double_occ, magnetization, g_iw

    def test_nspin1(self):
        h_loc = self.make_h_loc(mul.outer([1, 1], np.diag([0.5, 0.6])))
        h_int = self.make_h_int(Uloc=np.array([1.0, 2.0]),
                                Ust=0.8,
                                Jh=0.2,
                                Jx=0.1,
                                Jp=0.15)

        fops_bath_up = [('B_up', nu * 2 + o)
                        for nu, o in product(range(2), self.orbs)]
        fops_bath_dn = [('B_dn', nu * 2 + o)
                        for nu, o in product(range(2), self.orbs)]

        fops = self.fops_imp_up + self.fops_imp_dn + fops_bath_up + fops_bath_dn

        eps = np.array([[-0.5, 0.5],
                        [-0.7, 0.7]])
        V = np.array([[0.1, 0.2],
                      [0.3, 0.4]])
        h_bath = self.make_h_bath(eps, V)

        h = h_loc + h_int + h_bath
        solver = EDIpackSolver(h,
                               self.fops_imp_up,
                               self.fops_imp_dn,
                               fops_bath_up,
                               fops_bath_dn,
                               verbose=0)

        self.assertEqual(solver.nspin, 1)
        self.assertEqual(solver.norb, 2)
        self.assertEqual(solver.bath().name, "normal")
        self.assertEqual(solver.bath().nbath, 2)

        # Part I: Initial solve()
        beta = 100.0
        n_iw = 100
        solver.solve(beta=beta, n_iw=n_iw)

        ## Reference solution
        densities_ref, double_occ_ref, magnetization_ref, g_iw_ref = \
            self.make_ref_results(h, fops, beta, n_iw)

        assert_allclose(solver.densities(), densities_ref, atol=1e-8)
        assert_allclose(solver.double_occ(), double_occ_ref, atol=1e-8)
        assert_allclose(solver.magnetization(), magnetization_ref, atol=1e-8)
        assert_block_gfs_are_close(solver.g_iw(), g_iw_ref)

        # Part II: update_int_params()
        new_int_params = {'Uloc': np.array([2.0, 3.0]),
                          'Ust': 0.6,
                          'Jh': 0.1,
                          'Jx': 0.2,
                          'Jp': 0.0}
        solver.update_int_params(**new_int_params)

        beta = 120.0
        n_iw = 200
        solver.solve(beta=beta, n_iw=n_iw)

        ## Reference solution
        h_int = self.make_h_int(**new_int_params)
        h = h_loc + h_int + h_bath
        densities_ref, double_occ_ref, magnetization_ref, g_iw_ref = \
            self.make_ref_results(h, fops, beta, n_iw)

        assert_allclose(solver.densities(), densities_ref, atol=1e-8)
        assert_allclose(solver.double_occ(), double_occ_ref, atol=1e-8)
        assert_allclose(solver.magnetization(), magnetization_ref, atol=1e-8)
        assert_block_gfs_are_close(solver.g_iw(), g_iw_ref)

        # Part III: Updated bath parameters
        eps = np.array([[-0.5, 0.5],
                        [-0.7, 0.8]])
        V = np.array([[0.1, 0.2],
                      [0.5, 0.4]])

        solver.bath().eps[:] = eps
        solver.bath().V[:] = V

        beta = 100.0
        n_iw = 50
        solver.solve(beta=beta, n_iw=n_iw)

        ## Reference solution
        h_bath = self.make_h_bath(eps, V)
        h = h_loc + h_int + h_bath
        densities_ref, double_occ_ref, magnetization_ref, g_iw_ref = \
            self.make_ref_results(h, fops, beta, n_iw)

        assert_allclose(solver.densities(), densities_ref, atol=1e-8)
        assert_allclose(solver.double_occ(), double_occ_ref, atol=1e-8)
        assert_allclose(solver.magnetization(), magnetization_ref, atol=1e-8)
        assert_block_gfs_are_close(solver.g_iw(), g_iw_ref)

    def test_nspin2(self):
        h_loc = self.make_h_loc(mul.outer([0.8, 1.2], np.diag([0.5, 0.6])))
        h_int = self.make_h_int(Uloc=np.array([1.0, 2.0]),
                                Ust=0.8,
                                Jh=0.2,
                                Jx=0.1,
                                Jp=0.15)

        fops_bath_up = [('B_up', nu * 2 + o)
                        for nu, o in product(range(2), self.orbs)]
        fops_bath_dn = [('B_dn', nu * 2 + o)
                        for nu, o in product(range(2), self.orbs)]

        fops = self.fops_imp_up + self.fops_imp_dn + fops_bath_up + fops_bath_dn

        eps = np.array([[-0.5, 0.5],
                        [-0.7, 0.7]])
        V = np.array([[0.1, 0.2],
                      [0.3, 0.4]])
        h_bath = self.make_h_bath(eps, V)

        h = h_loc + h_int + h_bath
        solver = EDIpackSolver(h,
                               self.fops_imp_up,
                               self.fops_imp_dn,
                               fops_bath_up,
                               fops_bath_dn,
                               verbose=0)

        self.assertEqual(solver.nspin, 2)
        self.assertEqual(solver.norb, 2)
        self.assertEqual(solver.bath().name, "normal")
        self.assertEqual(solver.bath().nbath, 2)

        # Part I: Initial solve()
        beta = 100.0
        n_iw = 100
        solver.solve(beta=beta, n_iw=n_iw)

        ## Reference solution
        densities_ref, double_occ_ref, magnetization_ref, g_iw_ref = \
            self.make_ref_results(h, fops, beta, n_iw)

        assert_allclose(solver.densities(), densities_ref, atol=1e-8)
        assert_allclose(solver.double_occ(), double_occ_ref, atol=1e-8)
        assert_allclose(solver.magnetization(), magnetization_ref, atol=1e-8)
        assert_block_gfs_are_close(solver.g_iw(), g_iw_ref)

        # Part II: update_int_params()
        new_int_params = {'Uloc': np.array([2.0, 3.0]),
                          'Ust': 0.6,
                          'Jh': 0.1,
                          'Jx': 0.2,
                          'Jp': 0.0}
        solver.update_int_params(**new_int_params)

        beta = 120.0
        n_iw = 200
        solver.solve(beta=beta, n_iw=n_iw)

        ## Reference solution
        h_int = self.make_h_int(**new_int_params)
        h = h_loc + h_int + h_bath
        densities_ref, double_occ_ref, magnetization_ref, g_iw_ref = \
            self.make_ref_results(h, fops, beta, n_iw)

        assert_allclose(solver.densities(), densities_ref, atol=1e-8)
        assert_allclose(solver.double_occ(), double_occ_ref, atol=1e-8)
        assert_allclose(solver.magnetization(), magnetization_ref, atol=1e-8)
        assert_block_gfs_are_close(solver.g_iw(), g_iw_ref)

        # Part III: Updated bath parameters
        eps = np.array([[-0.5, 0.5],
                        [-0.7, 0.8]])
        V = np.array([[0.1, 0.2],
                      [0.5, 0.4]])

        solver.bath().eps[:] = eps
        solver.bath().V[:] = V

        beta = 100.0
        n_iw = 50
        solver.solve(beta=beta, n_iw=n_iw)

        ## Reference solution
        h_bath = self.make_h_bath(eps, V)
        h = h_loc + h_int + h_bath
        densities_ref, double_occ_ref, magnetization_ref, g_iw_ref = \
            self.make_ref_results(h, fops, beta, n_iw)

        assert_allclose(solver.densities(), densities_ref, atol=1e-8)
        assert_allclose(solver.double_occ(), double_occ_ref, atol=1e-8)
        assert_allclose(solver.magnetization(), magnetization_ref, atol=1e-8)
        assert_block_gfs_are_close(solver.g_iw(), g_iw_ref)

    def tearDown(self):
        # Make sure EDIpackSolver.__del__() is called
        gc.collect()


if __name__ == '__main__':
    unittest.main()

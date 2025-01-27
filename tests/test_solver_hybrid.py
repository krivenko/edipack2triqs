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
from triqs.utility.comparison_tests import assert_block_gfs_are_close

from edipack2triqs.solver import EDIpackSolver


s0 = np.eye(2)
sx = np.array([[0, 1], [1, 0]])
sz = np.array([[1, 0], [0, -1]])


class TestEDIpackSolverBathHybrid(unittest.TestCase):

    spins = ('up', 'dn')
    orbs = range(2)

    fops_bath_up = [('B_up', nu) for nu in range(3)]
    fops_bath_dn = [('B_dn', nu) for nu in range(3)]

    @classmethod
    def make_fops_imp(cls, spin_blocks):
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
    def make_h_loc(cls, h_loc, spin_blocks):
        mki = cls.make_mkind(spin_blocks)
        return sum(h_loc[s1, s2, o1, o2]
                   * op.c_dag(*mki(spin1, o1)) * op.c(*mki(spin2, o2))
                   for (s1, spin1), (s2, spin2), o1, o2
                   in product(enumerate(cls.spins), enumerate(cls.spins),
                              cls.orbs, cls.orbs))

    @classmethod
    def make_h_int(cls, spin_blocks, *, Uloc, Ust, Jh, Jx, Jp):
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
    def make_h_bath(cls, eps, V, spin_blocks):
        mki = cls.make_mkind(spin_blocks)
        h_bath = sum(eps[s, nu]
                     * op.c_dag("B_" + spin, nu) * op.c("B_" + spin, nu)
                     for nu, (s, spin)
                     in product(range(3), enumerate(cls.spins)))
        h_bath += sum(V[s1, s2, o, nu] * (
                      op.c_dag(*mki(spin1, o)) * op.c("B_" + spin2, nu)
                      + op.c_dag("B_" + spin2, nu) * op.c(*mki(spin1, o)))
                      for (s1, spin1), (s2, spin2), o, nu
                      in product(enumerate(cls.spins), enumerate(cls.spins),
                                 cls.orbs, range(3)))
        return h_bath

    @classmethod
    def make_h_sc(cls, Delta):
        return sum(Delta[nu] * (
            op.c_dag('B_up', nu) * op.c_dag('B_dn', nu)
            + op.c('B_dn', nu) * op.c('B_up', nu))
            for nu in range(3)
        )

    @classmethod
    def make_ref_results(
        cls, h, fops, beta, n_iw, energy_window, n_w, eta, spin_blocks
    ):
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

        def avg(ops):
            return np.array([trace_rho_op(rho, ops[o], ad) for o in cls.orbs])

        return {'densities': avg(N),
                'double_occ': avg(D),
                'magn_x': avg(S_x),
                'magn_y': 1j * avg(S_y),
                'magn_z': avg(S_z),
                'g_iw': atomic_g_iw(ad, beta, gf_struct, n_iw),
                'g_w': atomic_g_w(ad, beta, gf_struct, energy_window, n_w, eta)}

    @classmethod
    def assert_all(cls, s, **refs):
        assert_allclose(s.densities(), refs['densities'], atol=1e-8)
        assert_allclose(s.double_occ(), refs['double_occ'], atol=1e-8)
        assert_allclose(s.magnetization(comp='x'), refs['magn_x'], atol=1e-8)
        assert_allclose(s.magnetization(comp='y'), refs['magn_y'], atol=1e-8)
        assert_allclose(s.magnetization(comp='z'), refs['magn_z'], atol=1e-8)
        assert_block_gfs_are_close(s.g_iw(), refs['g_iw'])
        assert_block_gfs_are_close(s.g_w(), refs['g_w'])

    def test_nspin1(self):
        h_loc = self.make_h_loc(mul.outer(s0, np.diag([0.5, 0.6])), True)
        h_int = self.make_h_int(True,
                                Uloc=np.array([1.0, 2.0]),
                                Ust=0.8,
                                Jh=0.2,
                                Jx=0.1,
                                Jp=0.15)

        fops_imp_up, fops_imp_dn = self.make_fops_imp(True)
        fops = fops_imp_up + fops_imp_dn + self.fops_bath_up + self.fops_bath_dn

        eps = np.array([-0.1, -0.2, -0.3])
        V = np.array([[0.4, 0.7, 0.1],
                      [0.3, 0.5, 0.2]])
        h_bath = self.make_h_bath(mul.outer([1, 1], eps),
                                  mul.outer(s0, V),
                                  True)

        h = h_loc + h_int + h_bath
        solver = EDIpackSolver(h,
                               fops_imp_up,
                               fops_imp_dn,
                               self.fops_bath_up,
                               self.fops_bath_dn,
                               lanc_nstates_sector=4,
                               lanc_nstates_total=4,
                               verbose=0)

        self.assertEqual(solver.nspin, 1)
        self.assertEqual(solver.norb, 2)
        self.assertEqual(solver.bath().name, "hybrid")
        self.assertEqual(solver.bath().nbath, 3)

        # Part I: Initial solve()
        beta = 100.0
        n_iw = 100
        energy_window = (-2.0, 2.0)
        n_w = 600
        broadening = 0.005
        solver.solve(beta=beta,
                     n_iw=n_iw,
                     energy_window=energy_window,
                     n_w=n_w,
                     broadening=broadening)

        ## Reference solution
        refs = self.make_ref_results(h, fops, beta,
                                     n_iw,
                                     energy_window, n_w, broadening,
                                     True)
        self.assert_all(solver, **refs)

        # Part II: update_int_params()
        new_int_params = {'Uloc': np.array([2.0, 3.0]),
                          'Ust': 0.6,
                          'Jh': 0.1,
                          'Jx': 0.2,
                          'Jp': 0.0}
        solver.update_int_params(**new_int_params)

        beta = 120.0
        n_iw = 200
        energy_window = (-1.5, 1.5)
        n_w = 400
        broadening = 0.003
        solver.solve(beta=beta,
                     n_iw=n_iw,
                     energy_window=energy_window,
                     n_w=n_w,
                     broadening=broadening)

        ## Reference solution
        h_int = self.make_h_int(True, **new_int_params)
        h = h_loc + h_int + h_bath
        refs = self.make_ref_results(h, fops, beta,
                                     n_iw,
                                     energy_window, n_w, broadening,
                                     True)
        self.assert_all(solver, **refs)

        # Part III: Updated bath parameters
        eps = np.array([-0.1, -0.2, -0.4])
        V = np.array([[0.4, 0.7, 0.1],
                      [0.3, 0.8, 0.2]])

        solver.bath().eps[0, ...] = eps
        solver.bath().V[0, ...] = V

        beta = 100.0
        n_iw = 50
        energy_window = (-2.5, 2.5)
        n_w = 800
        broadening = 0.002
        solver.solve(beta=beta,
                     n_iw=n_iw,
                     energy_window=energy_window,
                     n_w=n_w,
                     broadening=broadening)

        ## Reference solution
        h_bath = self.make_h_bath(mul.outer([1, 1], eps),
                                  mul.outer(s0, V),
                                  True)
        h = h_loc + h_int + h_bath
        refs = self.make_ref_results(h, fops, beta,
                                     n_iw,
                                     energy_window, n_w, broadening,
                                     True)
        self.assert_all(solver, **refs)

    def test_nspin2(self):
        h_loc = self.make_h_loc(mul.outer(np.diag([0.8, 1.2]),
                                          np.diag([0.5, 0.6])),
                                True)
        h_int = self.make_h_int(True,
                                Uloc=np.array([1.0, 2.0]),
                                Ust=0.8,
                                Jh=0.2,
                                Jx=0.1,
                                Jp=0.15)

        fops_imp_up, fops_imp_dn = self.make_fops_imp(True)
        fops = fops_imp_up + fops_imp_dn + self.fops_bath_up + self.fops_bath_dn

        eps = np.array([-0.1, -0.2, -0.3])
        V = np.array([[0.4, 0.7, 0.1],
                      [0.3, 0.5, 0.2]])
        h_bath = self.make_h_bath(mul.outer([1, -1], eps),
                                  mul.outer(np.diag([1, 0.9]), V),
                                  True)

        h = h_loc + h_int + h_bath
        solver = EDIpackSolver(h,
                               fops_imp_up,
                               fops_imp_dn,
                               self.fops_bath_up,
                               self.fops_bath_dn,
                               lanc_nstates_sector=4,
                               lanc_nstates_total=4,
                               verbose=0)

        self.assertEqual(solver.nspin, 2)
        self.assertEqual(solver.norb, 2)
        self.assertEqual(solver.bath().name, "hybrid")
        self.assertEqual(solver.bath().nbath, 3)

        # Part I: Initial solve()
        beta = 100.0
        n_iw = 100
        energy_window = (-1.5, 1.5)
        n_w = 600
        broadening = 0.005
        solver.solve(beta=beta,
                     n_iw=n_iw,
                     energy_window=energy_window,
                     n_w=n_w,
                     broadening=broadening)

        ## Reference solution
        refs = self.make_ref_results(h, fops, beta,
                                     n_iw,
                                     energy_window, n_w, broadening,
                                     True)
        self.assert_all(solver, **refs)

        # Part II: update_int_params()
        new_int_params = {'Uloc': np.array([2.0, 3.0]),
                          'Ust': 0.6,
                          'Jh': 0.1,
                          'Jx': 0.2,
                          'Jp': 0.0}
        solver.update_int_params(**new_int_params)

        beta = 120.0
        n_iw = 200
        energy_window = (-1.0, 1.0)
        n_w = 400
        broadening = 0.003
        solver.solve(beta=beta,
                     n_iw=n_iw,
                     energy_window=energy_window,
                     n_w=n_w,
                     broadening=broadening)

        ## Reference solution
        h_int = self.make_h_int(True, **new_int_params)
        h = h_loc + h_int + h_bath
        refs = self.make_ref_results(h, fops, beta,
                                     n_iw,
                                     energy_window, n_w, broadening,
                                     True)
        self.assert_all(solver, **refs)

        # Part III: Updated bath parameters
        eps = np.array([-0.1, -0.2, -0.4])
        V = np.array([[0.4, 0.7, 0.1],
                      [0.3, 0.8, 0.2]])

        solver.bath().eps[:] = mul.outer([1, -1], eps)
        solver.bath().V[:] = mul.outer([1, 0.9], V)

        beta = 100.0
        n_iw = 50
        energy_window = (-1.5, 1.5)
        n_w = 800
        broadening = 0.002
        solver.solve(beta=beta,
                     n_iw=n_iw,
                     energy_window=energy_window,
                     n_w=n_w,
                     broadening=broadening)

        ## Reference solution
        h_bath = self.make_h_bath(mul.outer([1, -1], eps),
                                  mul.outer(np.diag([1, 0.9]), V),
                                  True)
        h = h_loc + h_int + h_bath
        refs = self.make_ref_results(h, fops, beta,
                                     n_iw,
                                     energy_window, n_w, broadening,
                                     True)
        self.assert_all(solver, **refs)

    def test_nonsu2_hloc(self):
        h_loc = self.make_h_loc(mul.outer(np.array([[0.8, 0.2],
                                                    [0.2, 1.2]]),
                                          np.diag([0.5, 0.6])),
                                False)
        h_int = self.make_h_int(False,
                                Uloc=np.array([1.0, 2.0]),
                                Ust=0.8,
                                Jh=0.2,
                                Jx=0.1,
                                Jp=0.15)

        fops_imp_up, fops_imp_dn = self.make_fops_imp(False)
        fops = fops_imp_up + fops_imp_dn + self.fops_bath_up + self.fops_bath_dn

        eps = np.array([-0.1, -0.2, -0.3])
        V = np.array([[0.4, 0.7, 0.1],
                      [0.3, 0.5, 0.2]])
        h_bath = self.make_h_bath(mul.outer([1, -1], eps),
                                  mul.outer(np.diag([1, 0.9]), V),
                                  False)

        h = h_loc + h_int + h_bath
        solver = EDIpackSolver(h,
                               fops_imp_up,
                               fops_imp_dn,
                               self.fops_bath_up,
                               self.fops_bath_dn,
                               lanc_nstates_sector=4,
                               lanc_nstates_total=4,
                               verbose=0)

        self.assertEqual(solver.nspin, 2)
        self.assertEqual(solver.norb, 2)
        self.assertEqual(solver.bath().name, "hybrid")
        self.assertEqual(solver.bath().nbath, 3)

        # Part I: Initial solve()
        beta = 100.0
        n_iw = 100
        energy_window = (-1.5, 1.5)
        n_w = 600
        broadening = 0.005
        solver.solve(beta=beta,
                     n_iw=n_iw,
                     energy_window=energy_window,
                     n_w=n_w,
                     broadening=broadening)

        ## Reference solution
        refs = self.make_ref_results(h, fops, beta,
                                     n_iw,
                                     energy_window, n_w, broadening,
                                     False)
        self.assert_all(solver, **refs)

        # Part II: update_int_params()
        new_int_params = {'Uloc': np.array([2.0, 3.0]),
                          'Ust': 0.6,
                          'Jh': 0.1,
                          'Jx': 0.2,
                          'Jp': 0.0}
        solver.update_int_params(**new_int_params)

        beta = 120.0
        n_iw = 200
        energy_window = (-1.0, 1.0)
        n_w = 400
        broadening = 0.003
        solver.solve(beta=beta,
                     n_iw=n_iw,
                     energy_window=energy_window,
                     n_w=n_w,
                     broadening=broadening)

        ## Reference solution
        h_int = self.make_h_int(False, **new_int_params)
        h = h_loc + h_int + h_bath
        refs = self.make_ref_results(h, fops, beta,
                                     n_iw,
                                     energy_window, n_w, broadening,
                                     False)
        self.assert_all(solver, **refs)

        # Part III: Updated bath parameters
        eps = np.array([-0.1, -0.2, -0.4])
        V = np.array([[0.4, 0.7, 0.1],
                      [0.3, 0.8, 0.2]])

        solver.bath().eps[:] = mul.outer([1, -1], eps)
        solver.bath().V[:] = mul.outer([1, 0.9], V)

        beta = 100.0
        n_iw = 50
        energy_window = (-1.5, 1.5)
        n_w = 800
        broadening = 0.002
        solver.solve(beta=beta,
                     n_iw=n_iw,
                     energy_window=energy_window,
                     n_w=n_w,
                     broadening=broadening)

        ## Reference solution
        h_bath = self.make_h_bath(mul.outer([1, -1], eps),
                                  mul.outer(np.diag([1, 0.9]), V),
                                  False)
        h = h_loc + h_int + h_bath
        refs = self.make_ref_results(h, fops, beta,
                                     n_iw,
                                     energy_window, n_w, broadening,
                                     False)
        self.assert_all(solver, **refs)

    def test_nonsu2_bath(self):
        h_loc = self.make_h_loc(mul.outer(np.diag([0.8, 1.2]),
                                          np.diag([0.5, 0.6])),
                                False)
        h_int = self.make_h_int(False,
                                Uloc=np.array([1.0, 2.0]),
                                Ust=0.8,
                                Jh=0.2,
                                Jx=0.1,
                                Jp=0.15)

        fops_imp_up, fops_imp_dn = self.make_fops_imp(False)
        fops = fops_imp_up + fops_imp_dn + self.fops_bath_up + self.fops_bath_dn

        eps = np.array([-0.1, -0.2, -0.3])
        V = np.array([[0.4, 0.7, 0.1],
                      [0.3, 0.5, 0.2]])
        h_bath = self.make_h_bath(mul.outer([1, -1], eps),
                                  mul.outer(sz + 0.2 * sx, V),
                                  False)

        h = h_loc + h_int + h_bath
        solver = EDIpackSolver(h,
                               fops_imp_up,
                               fops_imp_dn,
                               self.fops_bath_up,
                               self.fops_bath_dn,
                               lanc_nstates_sector=4,
                               lanc_nstates_total=4,
                               verbose=0)

        self.assertEqual(solver.nspin, 2)
        self.assertEqual(solver.norb, 2)
        self.assertEqual(solver.bath().name, "hybrid")
        self.assertEqual(solver.bath().nbath, 3)

        # Part I: Initial solve()
        beta = 100.0
        n_iw = 100
        energy_window = (-1.5, 1.5)
        n_w = 600
        broadening = 0.005
        solver.solve(beta=beta,
                     n_iw=n_iw,
                     energy_window=energy_window,
                     n_w=n_w,
                     broadening=broadening)

        ## Reference solution
        refs = self.make_ref_results(h, fops, beta,
                                     n_iw,
                                     energy_window, n_w, broadening,
                                     False)
        self.assert_all(solver, **refs)

        # Part II: update_int_params()
        new_int_params = {'Uloc': np.array([2.0, 3.0]),
                          'Ust': 0.6,
                          'Jh': 0.1,
                          'Jx': 0.2,
                          'Jp': 0.0}
        solver.update_int_params(**new_int_params)

        beta = 120.0
        n_iw = 200
        energy_window = (-1.0, 1.0)
        n_w = 400
        broadening = 0.003
        solver.solve(beta=beta,
                     n_iw=n_iw,
                     energy_window=energy_window,
                     n_w=n_w,
                     broadening=broadening)

        ## Reference solution
        h_int = self.make_h_int(False, **new_int_params)
        h = h_loc + h_int + h_bath
        refs = self.make_ref_results(h, fops, beta,
                                     n_iw,
                                     energy_window, n_w, broadening,
                                     False)
        self.assert_all(solver, **refs)

        # Part III: Updated bath parameters
        eps = np.array([-0.1, -0.2, -0.4])
        V = np.array([[0.4, 0.7, 0.1],
                      [0.3, 0.8, 0.2]])

        solver.bath().eps[:] = mul.outer([1, -1], eps)
        solver.bath().V[:] = mul.outer([1, 0.9], V)
        solver.bath().U[:] = mul.outer([0.2, 0.2], V)

        beta = 100.0
        n_iw = 50
        energy_window = (-1.5, 1.5)
        n_w = 800
        broadening = 0.002
        solver.solve(beta=beta,
                     n_iw=n_iw,
                     energy_window=energy_window,
                     n_w=n_w,
                     broadening=broadening)

        ## Reference solution
        h_bath = self.make_h_bath(mul.outer([1, -1], eps),
                                  mul.outer(np.diag([1, 0.9]) + 0.2 * sx, V),
                                  False)
        h = h_loc + h_int + h_bath
        refs = self.make_ref_results(h, fops, beta,
                                     n_iw,
                                     energy_window, n_w, broadening,
                                     False)
        self.assert_all(solver, **refs)

    def test_superc(self):
        h_loc = self.make_h_loc(mul.outer(s0, np.diag([0.5, 0.6])), True)
        h_int = self.make_h_int(True,
                                Uloc=np.array([1.0, 2.0]),
                                Ust=0.8,
                                Jh=0.2,
                                Jx=0.1,
                                Jp=0.15)

        fops_imp_up, fops_imp_dn = self.make_fops_imp(True)
        fops = fops_imp_up + fops_imp_dn + self.fops_bath_up + self.fops_bath_dn

        eps = np.array([-0.1, -0.2, -0.3])
        V = np.array([[0.4, 0.7, 0.1],
                      [0.3, 0.5, 0.2]])
        h_bath = self.make_h_bath(mul.outer([1, 1], eps),
                                  mul.outer(s0, V),
                                  True)

        Delta = np.array([0.6, 0.7, 0.8])
        h_sc = self.make_h_sc(Delta)

        h = h_loc + h_int + h_bath + h_sc
        solver = EDIpackSolver(h,
                               fops_imp_up,
                               fops_imp_dn,
                               self.fops_bath_up,
                               self.fops_bath_dn,
                               lanc_nstates_sector=4,
                               lanc_nstates_total=4,
                               verbose=0)

        self.assertEqual(solver.nspin, 1)
        self.assertEqual(solver.norb, 2)
        self.assertEqual(solver.bath().name, "hybrid")
        self.assertEqual(solver.bath().nbath, 3)

        # Part I: Initial solve()
        beta = 100.0
        n_iw = 100
        energy_window = (-2.0, 2.0)
        n_w = 600
        broadening = 0.005
        solver.solve(beta=beta,
                     n_iw=n_iw,
                     energy_window=energy_window,
                     n_w=n_w,
                     broadening=broadening)

        ## Reference solution
        refs = self.make_ref_results(h, fops, beta,
                                     n_iw,
                                     energy_window, n_w, broadening,
                                     True)
        self.assert_all(solver, **refs)

        # Part II: update_int_params()
        new_int_params = {'Uloc': np.array([2.0, 3.0]),
                          'Ust': 0.6,
                          'Jh': 0.1,
                          'Jx': 0.2,
                          'Jp': 0.0}
        solver.update_int_params(**new_int_params)

        beta = 120.0
        n_iw = 200
        energy_window = (-1.5, 1.5)
        n_w = 400
        broadening = 0.003
        solver.solve(beta=beta,
                     n_iw=n_iw,
                     energy_window=energy_window,
                     n_w=n_w,
                     broadening=broadening)

        ## Reference solution
        h_int = self.make_h_int(True, **new_int_params)
        h = h_loc + h_int + h_bath + h_sc
        refs = self.make_ref_results(h, fops, beta,
                                     n_iw,
                                     energy_window, n_w, broadening,
                                     True)
        self.assert_all(solver, **refs)

        # Part III: Updated bath parameters
        eps = np.array([-0.1, -0.2, -0.4])
        V = np.array([[0.4, 0.7, 0.1],
                      [0.3, 0.8, 0.2]])
        Delta = np.array([0.5, 0.7, 0.6])

        solver.bath().eps[0, ...] = eps
        solver.bath().V[0, ...] = V
        solver.bath().Delta[0, ...] = Delta

        beta = 100.0
        n_iw = 50
        energy_window = (-2.5, 2.5)
        n_w = 800
        broadening = 0.002
        solver.solve(beta=beta,
                     n_iw=n_iw,
                     energy_window=energy_window,
                     n_w=n_w,
                     broadening=broadening)

        ## Reference solution
        h_bath = self.make_h_bath(mul.outer([1, 1], eps),
                                  mul.outer(s0, V),
                                  True)
        h_sc = self.make_h_sc(Delta)
        h = h_loc + h_int + h_bath + h_sc
        refs = self.make_ref_results(h, fops, beta,
                                     n_iw,
                                     energy_window, n_w, broadening,
                                     True)
        self.assert_all(solver, **refs)

        solver.g_iw(anomalous=True)
        solver.Sigma_iw(anomalous=True)
        solver.g_w(anomalous=True)
        solver.Sigma_w(anomalous=True)

    def tearDown(self):
        # Make sure EDIpackSolver.__del__() is called
        gc.collect()


if __name__ == '__main__':
    unittest.main()

import unittest
import gc
from itertools import product

import numpy as np
from numpy.testing import assert_allclose
from numpy import multiply as mul

import triqs.operators as op
from triqs.utility.comparison_tests import assert_block_gfs_are_close

from edipack2triqs.solver import EDIpackSolver

from . import reference as ref
ref.write_h5 = False


s0 = np.eye(2)
sx = np.array([[0, 1], [1, 0]])
sz = np.array([[1, 0], [0, -1]])


class TestEDIpackSolverBathHybrid(unittest.TestCase):

    spins = ('up', 'dn')
    orbs = range(2)

    fops_bath_up = [('B_up', nu) for nu in range(3)]
    fops_bath_dn = [('B_dn', nu) for nu in range(3)]

    @classmethod
    def make_fops_imp(cls, spin_blocks=True):
        if spin_blocks:
            return ([('up', o) for o in cls.orbs],
                    [('dn', o) for o in cls.orbs])
        else:
            return ([('up_dn', so) for so in cls.orbs],
                    [('up_dn', so + len(cls.orbs)) for so in cls.orbs])

    @classmethod
    def make_h_loc(cls, h_loc, spin_blocks=True):
        mki = ref.make_mkind(cls.spins, cls.orbs, spin_blocks)
        return sum(h_loc[s1, s2, o1, o2]
                   * op.c_dag(*mki(spin1, o1)) * op.c(*mki(spin2, o2))
                   for (s1, spin1), (s2, spin2), o1, o2
                   in product(enumerate(cls.spins), enumerate(cls.spins),
                              cls.orbs, cls.orbs))

    @classmethod
    def make_h_int(cls, *, Uloc, Ust, Jh, Jx, Jp, spin_blocks=True):
        mki = ref.make_mkind(cls.spins, cls.orbs, spin_blocks)
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
    def make_h_bath(cls, eps, V, spin_blocks=True):
        mki = ref.make_mkind(cls.spins, cls.orbs, spin_blocks)
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
        h_loc = self.make_h_loc(mul.outer(s0, np.diag([0.5, 0.6])))
        h_int = self.make_h_int(Uloc=np.array([1.0, 2.0]),
                                Ust=0.8,
                                Jh=0.2,
                                Jx=0.1,
                                Jp=0.15)

        fops_imp_up, fops_imp_dn = self.make_fops_imp()
        fops = fops_imp_up + fops_imp_dn + self.fops_bath_up + self.fops_bath_dn
        struct_params = {"spins": self.spins, "orbs": self.orbs, "fops": fops}

        eps = np.array([-0.1, -0.2, -0.3])
        V = np.array([[0.4, 0.7, 0.1],
                      [0.3, 0.5, 0.2]])
        h_bath = self.make_h_bath(mul.outer([1, 1], eps), mul.outer(s0, V))

        h = h_loc + h_int + h_bath
        solver = EDIpackSolver(h,
                               fops_imp_up,
                               fops_imp_dn,
                               self.fops_bath_up,
                               self.fops_bath_dn,
                               zerotemp=True,
                               verbose=0)

        self.assertEqual(solver.nspin, 1)
        self.assertEqual(solver.norb, 2)
        self.assertEqual(solver.bath.name, "hybrid")
        self.assertEqual(solver.bath.nbath, 3)

        solve_params = {
            "n_iw": 100,
            "energy_window": (-2.0, 2.0),
            "n_w": 600,
            "broadening": 0.05
        }
        solver.solve(**solve_params)

        ## Reference solution
        refs = ref.ref_results("zerotemp", h=h, beta=10000, zerotemp=True,
                               **struct_params, **solve_params)
        self.assert_all(solver, **refs)

    def test_nspin1(self):
        h_loc = self.make_h_loc(mul.outer(s0, np.diag([0.5, 0.6])))
        h_int = self.make_h_int(Uloc=np.array([1.0, 2.0]),
                                Ust=0.8,
                                Jh=0.2,
                                Jx=0.1,
                                Jp=0.15)

        fops_imp_up, fops_imp_dn = self.make_fops_imp()
        fops = fops_imp_up + fops_imp_dn + self.fops_bath_up + self.fops_bath_dn
        struct_params = {"spins": self.spins, "orbs": self.orbs, "fops": fops}

        eps = np.array([-0.1, -0.2, -0.3])
        V = np.array([[0.4, 0.7, 0.1],
                      [0.3, 0.5, 0.2]])
        h_bath = self.make_h_bath(mul.outer([1, 1], eps), mul.outer(s0, V))

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
        self.assertEqual(solver.bath.name, "hybrid")
        self.assertEqual(solver.bath.nbath, 3)

        # Part I: Initial solve()
        solve_params = {
            "beta": 60.0,
            "n_iw": 100,
            "energy_window": (-2.0, 2.0),
            "n_w": 600,
            "broadening": 0.05
        }
        solver.solve(**solve_params)

        ## Reference solution
        refs = ref.ref_results("nspin1_1", h=h, **struct_params, **solve_params)
        self.assert_all(solver, **refs)

        # Part II: update interaction parameters
        new_int_params = {'Uloc': np.array([2.0, 3.0]),
                          'Ust': 0.6,
                          'Jh': 0.1,
                          'Jx': 0.2,
                          'Jp': 0.0}
        self.change_int_params(solver.U, new_int_params)
        solver.comm.barrier()

        solve_params = {
            "beta": 70.0,
            "n_iw": 200,
            "energy_window": (-1.5, 1.5),
            "n_w": 400,
            "broadening": 0.03
        }
        solver.solve(**solve_params)

        ## Reference solution
        h_int = self.make_h_int(**new_int_params)
        h = h_loc + h_int + h_bath
        refs = ref.ref_results("nspin1_2", h=h, **struct_params, **solve_params)
        self.assert_all(solver, **refs)

        # Part III: Updated bath parameters
        eps = np.array([-0.1, -0.2, -0.4])
        V = np.array([[0.4, 0.7, 0.1],
                      [0.3, 0.8, 0.2]])

        solver.bath.eps[0, ...] = eps
        solver.bath.V[0, ...] = V

        solve_params = {
            "beta": 60.0,
            "n_iw": 50,
            "energy_window": (-2.5, 2.5),
            "n_w": 800,
            "broadening": 0.02
        }
        solver.solve(**solve_params)

        ## Reference solution
        h_bath = self.make_h_bath(mul.outer([1, 1], eps), mul.outer(s0, V))
        h = h_loc + h_int + h_bath
        refs = ref.ref_results("nspin1_3", h=h, **struct_params, **solve_params)
        self.assert_all(solver, **refs)

    def test_nspin2(self):
        h_loc = self.make_h_loc(mul.outer(np.diag([0.8, 1.2]),
                                          np.diag([0.5, 0.6])))
        h_int = self.make_h_int(Uloc=np.array([1.0, 2.0]),
                                Ust=0.8,
                                Jh=0.2,
                                Jx=0.1,
                                Jp=0.15)

        fops_imp_up, fops_imp_dn = self.make_fops_imp()
        fops = fops_imp_up + fops_imp_dn + self.fops_bath_up + self.fops_bath_dn
        struct_params = {"spins": self.spins, "orbs": self.orbs, "fops": fops}

        eps = np.array([-0.1, -0.2, -0.3])
        V = np.array([[0.4, 0.7, 0.1],
                      [0.3, 0.5, 0.2]])
        h_bath = self.make_h_bath(mul.outer([1, -1], eps),
                                  mul.outer(np.diag([1, 0.9]), V))

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
        self.assertEqual(solver.bath.name, "hybrid")
        self.assertEqual(solver.bath.nbath, 3)

        # Part I: Initial solve()
        solve_params = {
            "beta": 60.0,
            "n_iw": 100,
            "energy_window": (-1.5, 1.5),
            "n_w": 600,
            "broadening": 0.05
        }
        solver.solve(**solve_params)

        ## Reference solution
        refs = ref.ref_results("nspin2_1", h=h, **struct_params, **solve_params)
        self.assert_all(solver, **refs)

        # Part II: update interaction parameters
        new_int_params = {'Uloc': np.array([2.0, 3.0]),
                          'Ust': 0.6,
                          'Jh': 0.1,
                          'Jx': 0.2,
                          'Jp': 0.0}
        self.change_int_params(solver.U, new_int_params)
        solver.comm.barrier()

        solve_params = {
            "beta": 70.0,
            "n_iw": 200,
            "energy_window": (-1.0, 1.0),
            "n_w": 400,
            "broadening": 0.03
        }
        solver.solve(**solve_params)

        ## Reference solution
        h_int = self.make_h_int(**new_int_params)
        h = h_loc + h_int + h_bath
        refs = ref.ref_results("nspin2_2", h=h, **struct_params, **solve_params)
        self.assert_all(solver, **refs)

        # Part III: Updated bath parameters
        eps = np.array([-0.1, -0.2, -0.4])
        V = np.array([[0.4, 0.7, 0.1],
                      [0.3, 0.8, 0.2]])

        solver.bath.eps[:] = mul.outer([1, -1], eps)
        solver.bath.V[:] = mul.outer([1, 0.9], V)

        solve_params = {
            "beta": 60.0,
            "n_iw": 50,
            "energy_window": (-1.5, 1.5),
            "n_w": 800,
            "broadening": 0.02
        }
        solver.solve(**solve_params)

        ## Reference solution
        h_bath = self.make_h_bath(mul.outer([1, -1], eps),
                                  mul.outer(np.diag([1, 0.9]), V))
        h = h_loc + h_int + h_bath
        refs = ref.ref_results("nspin2_3", h=h, **struct_params, **solve_params)
        self.assert_all(solver, **refs)

    def test_nonsu2_hloc(self):
        h_loc = self.make_h_loc(mul.outer(np.array([[0.8, 0.2],
                                                    [0.2, 1.2]]),
                                          np.diag([0.5, 0.6])),
                                spin_blocks=False)
        h_int = self.make_h_int(Uloc=np.array([1.0, 2.0]),
                                Ust=0.8,
                                Jh=0.2,
                                Jx=0.1,
                                Jp=0.15,
                                spin_blocks=False)

        fops_imp_up, fops_imp_dn = self.make_fops_imp(spin_blocks=False)
        fops = fops_imp_up + fops_imp_dn + self.fops_bath_up + self.fops_bath_dn
        struct_params = {"spins": self.spins, "orbs": self.orbs, "fops": fops}

        eps = np.array([-0.1, -0.2, -0.3])
        V = np.array([[0.4, 0.7, 0.1],
                      [0.3, 0.5, 0.2]])
        h_bath = self.make_h_bath(mul.outer([1, -1], eps),
                                  mul.outer(np.diag([1, 0.9]), V),
                                  spin_blocks=False)

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
        self.assertEqual(solver.bath.name, "hybrid")
        self.assertEqual(solver.bath.nbath, 3)

        # Part I: Initial solve()
        solve_params = {
            "beta": 60.0,
            "n_iw": 100,
            "energy_window": (-1.5, 1.5),
            "n_w": 600,
            "broadening": 0.05
        }
        solver.solve(**solve_params)

        ## Reference solution
        refs = ref.ref_results("nonsu2_hloc_1", h=h, spin_blocks=False,
                               **struct_params, **solve_params)
        self.assert_all(solver, **refs)

        # Part II: update interaction parameters
        new_int_params = {'Uloc': np.array([2.0, 3.0]),
                          'Ust': 0.6,
                          'Jh': 0.1,
                          'Jx': 0.2,
                          'Jp': 0.0}
        self.change_int_params(solver.U, new_int_params)
        solver.comm.barrier()

        solve_params = {
            "beta": 70.0,
            "n_iw": 200,
            "energy_window": (-1.0, 1.0),
            "n_w": 400,
            "broadening": 0.03
        }
        solver.solve(**solve_params)

        ## Reference solution
        h_int = self.make_h_int(**new_int_params, spin_blocks=False)
        h = h_loc + h_int + h_bath
        refs = ref.ref_results("nonsu2_hloc_2", h=h, spin_blocks=False,
                               **struct_params, **solve_params)
        self.assert_all(solver, **refs)

        # Part III: Updated bath parameters
        eps = np.array([-0.1, -0.2, -0.4])
        V = np.array([[0.4, 0.7, 0.1],
                      [0.3, 0.8, 0.2]])

        solver.bath.eps[:] = mul.outer([1, -1], eps)
        solver.bath.V[:] = mul.outer([1, 0.9], V)

        solve_params = {
            "beta": 60.0,
            "n_iw": 50,
            "energy_window": (-1.5, 1.5),
            "n_w": 800,
            "broadening": 0.02
        }
        solver.solve(**solve_params)

        ## Reference solution
        h_bath = self.make_h_bath(mul.outer([1, -1], eps),
                                  mul.outer(np.diag([1, 0.9]), V),
                                  spin_blocks=False)
        h = h_loc + h_int + h_bath
        refs = ref.ref_results("nonsu2_hloc_3", h=h, spin_blocks=False,
                               **struct_params, **solve_params)
        self.assert_all(solver, **refs)

    def test_nonsu2_bath(self):
        h_loc = self.make_h_loc(mul.outer(np.diag([0.8, 1.2]),
                                          np.diag([0.5, 0.6])),
                                spin_blocks=False)
        h_int = self.make_h_int(Uloc=np.array([1.0, 2.0]),
                                Ust=0.8,
                                Jh=0.2,
                                Jx=0.1,
                                Jp=0.15,
                                spin_blocks=False)

        fops_imp_up, fops_imp_dn = self.make_fops_imp(spin_blocks=False)
        fops = fops_imp_up + fops_imp_dn + self.fops_bath_up + self.fops_bath_dn
        struct_params = {"spins": self.spins, "orbs": self.orbs, "fops": fops}

        eps = np.array([-0.1, -0.2, -0.3])
        V = np.array([[0.4, 0.7, 0.1],
                      [0.3, 0.5, 0.2]])
        h_bath = self.make_h_bath(mul.outer([1, -1], eps),
                                  mul.outer(sz + 0.2 * sx, V),
                                  spin_blocks=False)

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
        self.assertEqual(solver.bath.name, "hybrid")
        self.assertEqual(solver.bath.nbath, 3)

        # Part I: Initial solve()
        solve_params = {
            "beta": 60.0,
            "n_iw": 100,
            "energy_window": (-1.5, 1.5),
            "n_w": 600,
            "broadening": 0.05
        }
        solver.solve(**solve_params)

        ## Reference solution
        refs = ref.ref_results("nonsu2_bath_1", h=h, spin_blocks=False,
                               **struct_params, **solve_params)
        self.assert_all(solver, **refs)

        # Part II: update interaction parameters
        new_int_params = {'Uloc': np.array([2.0, 3.0]),
                          'Ust': 0.6,
                          'Jh': 0.1,
                          'Jx': 0.2,
                          'Jp': 0.0}
        self.change_int_params(solver.U, new_int_params)
        solver.comm.barrier()

        solve_params = {
            "beta": 70.0,
            "n_iw": 200,
            "energy_window": (-1.0, 1.0),
            "n_w": 400,
            "broadening": 0.03
        }
        solver.solve(**solve_params)

        ## Reference solution
        h_int = self.make_h_int(**new_int_params, spin_blocks=False)
        h = h_loc + h_int + h_bath
        refs = ref.ref_results("nonsu2_bath_2", h=h, spin_blocks=False,
                               **struct_params, **solve_params)
        self.assert_all(solver, **refs)

        # Part III: Updated bath parameters
        eps = np.array([-0.1, -0.2, -0.4])
        V = np.array([[0.4, 0.7, 0.1],
                      [0.3, 0.8, 0.2]])

        solver.bath.eps[:] = mul.outer([1, -1], eps)
        solver.bath.V[:] = mul.outer([1, 0.9], V)
        solver.bath.U[:] = mul.outer([0.2, 0.2], V)

        solve_params = {
            "beta": 60.0,
            "n_iw": 50,
            "energy_window": (-1.5, 1.5),
            "n_w": 800,
            "broadening": 0.02
        }
        solver.solve(**solve_params)

        ## Reference solution
        h_bath = self.make_h_bath(mul.outer([1, -1], eps),
                                  mul.outer(np.diag([1, 0.9]) + 0.2 * sx, V),
                                  spin_blocks=False)
        h = h_loc + h_int + h_bath
        refs = ref.ref_results("nonsu2_bath_3", h=h, spin_blocks=False,
                               **struct_params, **solve_params)
        self.assert_all(solver, **refs)

    def test_superc(self):
        h_loc = self.make_h_loc(mul.outer(s0, np.diag([0.5, 0.6])))
        h_int = self.make_h_int(Uloc=np.array([1.0, 2.0]),
                                Ust=0.8,
                                Jh=0.2,
                                Jx=0.1,
                                Jp=0.15)

        fops_imp_up, fops_imp_dn = self.make_fops_imp()
        fops = fops_imp_up + fops_imp_dn + self.fops_bath_up + self.fops_bath_dn
        struct_params = {"spins": self.spins, "orbs": self.orbs, "fops": fops}

        eps = np.array([-0.1, -0.2, -0.3])
        V = np.array([[0.4, 0.7, 0.1],
                      [0.3, 0.5, 0.2]])
        h_bath = self.make_h_bath(mul.outer([1, 1], eps), mul.outer(s0, V))

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
        self.assertEqual(solver.bath.name, "hybrid")
        self.assertEqual(solver.bath.nbath, 3)

        # Part I: Initial solve()
        solve_params = {
            "beta": 60.0,
            "n_iw": 100,
            "energy_window": (-2.0, 2.0),
            "n_w": 600,
            "broadening": 0.05
        }
        solver.solve(**solve_params)

        ## Reference solution
        refs = ref.ref_results("superc_1", superc=True, h=h,
                               **struct_params, **solve_params)
        self.assert_all(solver, **refs)

        # Part II: update interaction parameters
        new_int_params = {'Uloc': np.array([2.0, 3.0]),
                          'Ust': 0.6,
                          'Jh': 0.1,
                          'Jx': 0.2,
                          'Jp': 0.0}
        self.change_int_params(solver.U, new_int_params)
        solver.comm.barrier()

        solve_params = {
            "beta": 70.0,
            "n_iw": 200,
            "energy_window": (-1.5, 1.5),
            "n_w": 400,
            "broadening": 0.03
        }
        solver.solve(**solve_params)

        ## Reference solution
        h_int = self.make_h_int(**new_int_params)
        h = h_loc + h_int + h_bath + h_sc
        refs = ref.ref_results("superc_2", h=h, superc=True,
                               **struct_params, **solve_params)
        self.assert_all(solver, **refs)

        # Part III: Updated bath parameters
        eps = np.array([-0.1, -0.2, -0.4])
        V = np.array([[0.4, 0.7, 0.1],
                      [0.3, 0.8, 0.2]])
        Delta = np.array([0.5, 0.7, 0.6])

        solver.bath.eps[0, ...] = eps
        solver.bath.V[0, ...] = V
        solver.bath.Delta[0, ...] = Delta

        solve_params = {
            "beta": 60.0,
            "n_iw": 50,
            "energy_window": (-2.5, 2.5),
            "n_w": 800,
            "broadening": 0.02
        }
        solver.solve(**solve_params)

        ## Reference solution
        h_bath = self.make_h_bath(mul.outer([1, 1], eps), mul.outer(s0, V))
        h_sc = self.make_h_sc(Delta)
        h = h_loc + h_int + h_bath + h_sc
        refs = ref.ref_results("superc_3", h=h, superc=True,
                               **struct_params, **solve_params)
        self.assert_all(solver, **refs)

        # TODO
        solver.g_an_iw
        solver.Sigma_an_iw
        solver.g_an_w
        solver.Sigma_an_w

    def tearDown(self):
        # Make sure EDIpackSolver.__del__() is called
        gc.collect()


if __name__ == '__main__':
    unittest.main()

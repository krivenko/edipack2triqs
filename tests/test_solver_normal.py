from itertools import product

import numpy as np
from numpy import multiply as mul
from numpy.testing import assert_equal

import triqs.operators as op

from edipack2triqs import EDMode
from edipack2triqs.solver import EDIpackSolver, LanczosParams

from .test_solver import TestSolver


s0 = np.eye(2)
sx = np.array([[0, 1], [1, 0]])
sz = np.array([[1, 0], [0, -1]])


class TestEDIpackSolverBathNormal(TestSolver):

    nbath = 2
    bsites = list(range(nbath))

    @classmethod
    @TestSolver.bath_index_ranges(bsites, TestSolver.orbs)
    def mkind_bath(cls, spin, nu, orb):
        "Map (spin, nu, orb) -> (block name, inner index)"
        return (f"B_{spin}", cls.norb * nu + orb)

    @classmethod
    def make_h_bath(cls, eps, V, spin_blocks=True):
        d_dag, d = [cls.make_op_imp(o, spin_blocks) for o in (op.c_dag, op.c)]
        a_dag, a = [cls.make_bath_op(o) for o in (op.c_dag, op.c)]
        h_bath = sum(eps[s, o, nu] * a_dag(spin, nu, o) * a(spin, nu, o)
                     for (s, spin), o, nu
                     in product(enumerate(cls.spins), cls.orbs, cls.bsites))
        h_bath += sum(V[s1, s2, o, nu] * (
                      d_dag(spin1, o) * a(spin2, nu, o)
                      + a_dag(spin2, nu, o) * d(spin1, o))
                      for (s1, spin1), (s2, spin2), o, nu
                      in product(enumerate(cls.spins), enumerate(cls.spins),
                                 cls.orbs, cls.bsites))
        return h_bath

    @classmethod
    def make_h_sc(cls, Delta):
        a_dag, a = [cls.make_bath_op(o) for o in (op.c_dag, op.c)]
        return sum(Delta[o, nu] * (
            a_dag('up', nu, o) * a_dag('dn', nu, o)
            + a('dn', nu, o) * a('up', nu, o))
            for o, nu in product(cls.orbs, cls.bsites)
        )

    @classmethod
    def assert_all(cls, s, **refs):
        cls.assert_static_obs(s, 1e-8, **refs)
        cls.assert_gfs(s, **refs)
        cls.assert_chi(s, atol=5e-5, **refs)

    def test_zerotemp(self):
        h_loc_mat = mul.outer(s0, np.diag([-0.5, -0.6]))
        h_loc = self.make_h_loc(h_loc_mat)
        h_int = self.make_h_int(
            Uloc=[0.1, 0.2], Ust=0.4, Jh=0.2, Jx=0.1, Jp=0.15
        )

        fops_imp_up, fops_imp_dn = self.make_fops_imp()
        fops = fops_imp_up + fops_imp_dn + self.fops_bath_up + self.fops_bath_dn

        eps = np.array([[-0.5, 0.5],
                        [-0.7, 0.7]])
        V = np.array([[0.1, 0.2],
                      [0.3, 0.4]])
        h_bath = self.make_h_bath(mul.outer([1, 1], eps), mul.outer(s0, V))

        h = h_loc + h_int + h_bath
        solver = EDIpackSolver(
            h,
            fops_imp_up, fops_imp_dn,
            self.fops_bath_up, self.fops_bath_dn,
            zerotemp=True,
            verbose=0
        )

        self.assertEqual(solver.h_params.ed_mode, EDMode.NORMAL)
        self.assertEqual(solver.nspin, 1)
        self.assertEqual(solver.norb, self.norb)
        self.assertEqual(solver.hloc, h_loc)
        assert_equal(solver.hloc_mat, h_loc_mat[:1, :1, ...])
        assert_equal(solver.hloc_an_mat, np.zeros((1, 1, self.norb, self.norb)))
        self.assertEqual(solver.bath.name, "normal")
        self.assertEqual(solver.bath.nbath, self.nbath)

        solve_params = {
            "n_iw": 10,
            "energy_window": (-2.0, 2.0),
            "n_w": 60,
            "broadening": 0.05
        }
        solver.solve(**solve_params)

        ## Reference solution
        refs = self.ref_results("zerotemp", h=h, fops=fops,
                                beta=10000, zerotemp=True, **solve_params)
        self.assert_all(solver, **refs)

    def test_nspin1(self):
        h_loc_mat = mul.outer(s0, np.diag([-0.5, -0.6]))
        h_loc = self.make_h_loc(h_loc_mat)
        h_int = self.make_h_int(
            Uloc=[0.1, 0.2], Ust=0.4, Jh=0.2, Jx=0.1, Jp=0.15
        )

        fops_imp_up, fops_imp_dn = self.make_fops_imp()
        fops = fops_imp_up + fops_imp_dn + self.fops_bath_up + self.fops_bath_dn

        eps = np.array([[-0.5, 0.5],
                        [-0.7, 0.7]])
        V = np.array([[0.1, 0.2],
                      [0.3, 0.4]])
        h_bath = self.make_h_bath(mul.outer([1, 1], eps), mul.outer(s0, V))

        h = h_loc + h_int + h_bath
        solver = EDIpackSolver(
            h,
            fops_imp_up, fops_imp_dn,
            self.fops_bath_up, self.fops_bath_dn,
            lanczos_params=LanczosParams(nstates_sector=4, nstates_total=10),
            verbose=0
        )

        self.assertEqual(solver.h_params.ed_mode, EDMode.NORMAL)
        self.assertEqual(solver.nspin, 1)
        self.assertEqual(solver.norb, self.norb)
        self.assertEqual(solver.hloc, h_loc)
        assert_equal(solver.hloc_mat, h_loc_mat[:1, :1, ...])
        assert_equal(solver.hloc_an_mat, np.zeros((1, 1, self.norb, self.norb)))
        self.assertEqual(solver.bath.name, "normal")
        self.assertEqual(solver.bath.nbath, self.nbath)

        # Part I: Initial solve()
        solve_params = {
            "beta": 60.0,
            "n_iw": 10,
            "energy_window": (-2.0, 2.0),
            "n_w": 60,
            "broadening": 0.05,
            "n_tau": 10,
            **self.chi_params
        }
        solver.solve(**solve_params)

        ## Reference solution
        refs = self.ref_results("nspin1_1", h=h, fops=fops, **solve_params)
        self.assert_all(solver, **refs)

        # Part II: update interaction parameters
        new_int_params = {
            'Uloc': [0.15, 0.25], 'Ust': 0.35, 'Jh': 0.1, 'Jx': 0.2, 'Jp': 0.0
        }
        self.change_int_params(solver.U, **new_int_params)
        solver.hloc = h_loc
        solver.comm.barrier()

        solve_params = {
            "beta": 70.0,
            "n_iw": 20,
            "energy_window": (-1.5, 1.5),
            "n_w": 40,
            "broadening": 0.03,
            "n_tau": 11,
            **self.chi_params
        }
        solver.solve(**solve_params)

        ## Reference solution
        h_int = self.make_h_int(**new_int_params)
        h = h_loc + h_int + h_bath
        refs = self.ref_results("nspin1_2", h=h, fops=fops, **solve_params)
        self.assert_all(solver, **refs)

        # Part III: Updated bath parameters
        eps = np.array([[-0.5, 0.5],
                        [-0.7, 0.8]])
        V = np.array([[0.1, 0.2],
                      [0.5, 0.4]])

        solver.bath.eps[0, ...] = eps
        solver.bath.V[0, ...] = V

        solve_params = {
            "beta": 60.0,
            "n_iw": 30,
            "energy_window": (-2.5, 2.5),
            "n_w": 80,
            "broadening": 0.02,
            "n_tau": 12,
            **self.chi_params
        }
        solver.solve(**solve_params)

        # Reference solution
        h_bath = self.make_h_bath(mul.outer([1, 1], eps), mul.outer(s0, V))
        h = h_loc + h_int + h_bath
        refs = self.ref_results("nspin1_3", h=h, fops=fops, **solve_params)
        self.assert_all(solver, **refs)

    def test_nspin2(self):
        h_loc_mat = mul.outer(np.diag([0.8, 1.2]),
                              np.diag([-0.5, -0.6]))
        h_loc = self.make_h_loc(h_loc_mat)
        h_int = self.make_h_int(
            Uloc=[0.1, 0.2], Ust=0.4, Jh=0.2, Jx=0.1, Jp=0.15
        )

        fops_imp_up, fops_imp_dn = self.make_fops_imp()
        fops = fops_imp_up + fops_imp_dn + self.fops_bath_up + self.fops_bath_dn

        eps = np.array([[-0.5, 0.5],
                        [-0.7, 0.7]])
        V = np.array([[0.1, 0.2],
                      [0.3, 0.4]])
        h_bath = self.make_h_bath(mul.outer([1, -1], eps),
                                  mul.outer(np.diag([1, 0.9]), V))

        h = h_loc + h_int + h_bath
        solver = EDIpackSolver(
            h,
            fops_imp_up, fops_imp_dn,
            self.fops_bath_up, self.fops_bath_dn,
            lanczos_params=LanczosParams(nstates_sector=4, nstates_total=10),
            verbose=0
        )

        self.assertEqual(solver.h_params.ed_mode, EDMode.NORMAL)
        self.assertEqual(solver.nspin, 2)
        self.assertEqual(solver.norb, self.norb)
        self.assertEqual(solver.hloc, h_loc)
        assert_equal(solver.hloc_mat, h_loc_mat)
        assert_equal(solver.hloc_an_mat, np.zeros((1, 1, self.norb, self.norb)))
        self.assertEqual(solver.bath.name, "normal")
        self.assertEqual(solver.bath.nbath, self.nbath)

        # Part I: Initial solve()
        solve_params = {
            "beta": 70.0,
            "n_iw": 10,
            "energy_window": (-2.0, 2.0),
            "n_w": 60,
            "broadening": 0.05,
            "n_tau": 10,
            **self.chi_params
        }
        solver.solve(**solve_params)

        ## Reference solution
        refs = self.ref_results("nspin2_1", h=h, fops=fops, **solve_params)
        self.assert_all(solver, **refs)

        # Part II: update interaction parameters
        new_int_params = {
            'Uloc': [0.15, 0.25], 'Ust': 0.35, 'Jh': 0.1, 'Jx': 0.2, 'Jp': 0.0
        }
        self.change_int_params(solver.U, **new_int_params)
        solver.hloc = h_loc
        solver.comm.barrier()

        solve_params = {
            "beta": 80.0,
            "n_iw": 20,
            "energy_window": (-1.5, 1.5),
            "n_w": 40,
            "broadening": 0.03,
            "n_tau": 11,
            **self.chi_params
        }
        solver.solve(**solve_params)

        ## Reference solution
        h_int = self.make_h_int(**new_int_params)
        h = h_loc + h_int + h_bath
        refs = self.ref_results("nspin2_2", h=h, fops=fops, **solve_params)
        self.assert_all(solver, **refs)

        # Part III: Updated bath parameters
        eps = np.array([[-0.5, 0.5],
                        [-0.7, 0.8]])
        V = np.array([[0.1, 0.2],
                      [0.5, 0.4]])

        solver.bath.eps[:] = mul.outer([1, -1], eps)
        solver.bath.V[:] = mul.outer([1, 0.9], V)

        solve_params = {
            "beta": 70.0,
            "n_iw": 30,
            "energy_window": (-2.5, 2.5),
            "n_w": 80,
            "broadening": 0.02,
            "n_tau": 12,
            **self.chi_params
        }
        solver.solve(**solve_params)

        ## Reference solution
        h_bath = self.make_h_bath(mul.outer([1, -1], eps),
                                  mul.outer(np.diag([1, 0.9]), V))
        h = h_loc + h_int + h_bath
        refs = self.ref_results("nspin2_3", h=h, fops=fops, **solve_params)
        self.assert_all(solver, **refs)

    def test_nonsu2_hloc(self):
        h_loc_mat = mul.outer(np.array([[0.8, 0.2],
                                        [0.2, 1.2]]),
                              np.diag([-0.5, -0.6]))
        h_loc = self.make_h_loc(h_loc_mat, spin_blocks=False)
        h_int = self.make_h_int(
            Uloc=[0.1, 0.2], Ust=0.4, Jh=0.2, Jx=0.1, Jp=0.15, spin_blocks=False
        )

        fops_imp_up, fops_imp_dn = self.make_fops_imp(False)
        fops = fops_imp_up + fops_imp_dn + self.fops_bath_up + self.fops_bath_dn

        eps = np.array([[-0.5, 0.5],
                        [-0.7, 0.7]])
        V = np.array([[0.1, 0.2],
                      [0.3, 0.4]])
        h_bath = self.make_h_bath(mul.outer([1, -1], eps),
                                  mul.outer(np.diag([1, 0.9]), V),
                                  spin_blocks=False)

        h = h_loc + h_int + h_bath
        solver = EDIpackSolver(
            h,
            fops_imp_up, fops_imp_dn,
            self.fops_bath_up, self.fops_bath_dn,
            lanczos_params=LanczosParams(nstates_sector=4, nstates_total=10),
            verbose=0
        )

        self.assertEqual(solver.h_params.ed_mode, EDMode.NONSU2)
        self.assertEqual(solver.nspin, 2)
        self.assertEqual(solver.norb, self.norb)
        self.assertEqual(solver.hloc, h_loc)
        assert_equal(solver.hloc_mat, h_loc_mat)
        assert_equal(solver.hloc_an_mat, np.zeros((1, 1, self.norb, self.norb)))
        self.assertEqual(solver.bath.name, "normal")
        self.assertEqual(solver.bath.nbath, self.nbath)

        # Part I: Initial solve()
        solve_params = {
            "beta": 70.0,
            "n_iw": 10,
            "energy_window": (-1.5, 1.5),
            "n_w": 60,
            "broadening": 0.05
        }
        solver.solve(**solve_params)

        ## Reference solution
        refs = self.ref_results("nonsu2_hloc_1", h=h, fops=fops,
                                spin_blocks=False, **solve_params)
        self.assert_all(solver, **refs)

        # Part II: update interaction parameters
        new_int_params = {
            'Uloc': [0.15, 0.25], 'Ust': 0.35, 'Jh': 0.1, 'Jx': 0.2, 'Jp': 0.0
        }
        self.change_int_params(solver.U, **new_int_params)
        solver.hloc = h_loc
        solver.comm.barrier()

        solve_params = {
            "beta": 80.0,
            "n_iw": 20,
            "energy_window": (-1.5, 1.5),
            "n_w": 40,
            "broadening": 0.03
        }
        solver.solve(**solve_params)

        ## Reference solution
        h_int = self.make_h_int(**new_int_params, spin_blocks=False)
        h = h_loc + h_int + h_bath
        refs = self.ref_results("nonsu2_hloc_2", h=h, fops=fops,
                                spin_blocks=False, **solve_params)
        self.assert_all(solver, **refs)

        # Part III: Updated bath parameters
        eps = np.array([[-0.5, 0.5],
                        [-0.7, 0.8]])
        V = np.array([[0.1, 0.2],
                      [0.5, 0.4]])

        solver.bath.eps[:] = mul.outer([1, -1], eps)
        solver.bath.V[:] = mul.outer([1, 0.9], V)

        solve_params = {
            "beta": 70.0,
            "n_iw": 30,
            "energy_window": (-1.5, 1.5),
            "n_w": 50,
            "broadening": 0.02
        }
        solver.solve(**solve_params)

        ## Reference solution
        h_bath = self.make_h_bath(mul.outer([1, -1], eps),
                                  mul.outer(np.diag([1, 0.9]), V),
                                  spin_blocks=False)
        h = h_loc + h_int + h_bath
        refs = self.ref_results("nonsu2_hloc_3", h=h, fops=fops,
                                spin_blocks=False, **solve_params)
        self.assert_all(solver, **refs)

    def test_nonsu2_bath(self):
        h_loc_mat = mul.outer(np.diag([0.8, 1.2]),
                              np.diag([-0.5, -0.6]))
        h_loc = self.make_h_loc(h_loc_mat, spin_blocks=False)
        h_int = self.make_h_int(
            Uloc=[0.1, 0.2], Ust=0.4, Jh=0.2, Jx=0.1, Jp=0.15, spin_blocks=False
        )

        fops_imp_up, fops_imp_dn = self.make_fops_imp(False)
        fops = fops_imp_up + fops_imp_dn + self.fops_bath_up + self.fops_bath_dn

        eps = np.array([[-0.5, 0.5],
                        [-0.7, 0.7]])
        V = np.array([[0.1, 0.2],
                      [0.3, 0.4]])
        h_bath = self.make_h_bath(mul.outer([1, -1], eps),
                                  mul.outer(sz + 0.2 * sx, V),
                                  spin_blocks=False)

        h = h_loc + h_int + h_bath
        solver = EDIpackSolver(
            h,
            fops_imp_up, fops_imp_dn,
            self.fops_bath_up, self.fops_bath_dn,
            lanczos_params=LanczosParams(nstates_sector=6, nstates_total=12),
            verbose=0
        )

        self.assertEqual(solver.h_params.ed_mode, EDMode.NONSU2)
        self.assertEqual(solver.nspin, 2)
        self.assertEqual(solver.norb, self.norb)
        self.assertEqual(solver.hloc, h_loc)
        assert_equal(solver.hloc_mat, h_loc_mat)
        assert_equal(solver.hloc_an_mat, np.zeros((1, 1, self.norb, self.norb)))
        self.assertEqual(solver.bath.name, "normal")
        self.assertEqual(solver.bath.nbath, self.nbath)

        # Part I: Initial solve()
        solve_params = {
            "beta": 60.0,
            "n_iw": 10,
            "energy_window": (-1.5, 1.5),
            "n_w": 60,
            "broadening": 0.05
        }
        solver.solve(**solve_params)

        ## Reference solution
        refs = self.ref_results("nonsu2_bath_1", h=h, fops=fops,
                                spin_blocks=False, **solve_params)
        self.assert_all(solver, **refs)

        # Part II: update interaction parameters
        new_int_params = {
            'Uloc': [0.15, 0.25], 'Ust': 0.35, 'Jh': 0.1, 'Jx': 0.2, 'Jp': 0.0
        }
        self.change_int_params(solver.U, **new_int_params)
        solver.hloc = h_loc
        solver.comm.barrier()

        solve_params = {
            "beta": 70.0,
            "n_iw": 20,
            "energy_window": (-1.5, 1.5),
            "n_w": 40,
            "broadening": 0.03
        }
        solver.solve(**solve_params)

        ## Reference solution
        h_int = self.make_h_int(**new_int_params, spin_blocks=False)
        h = h_loc + h_int + h_bath
        refs = self.ref_results("nonsu2_bath_2", h=h, fops=fops,
                                spin_blocks=False, **solve_params)
        self.assert_all(solver, **refs)

        # Part III: Updated bath parameters
        eps = np.array([[-0.5, 0.5],
                        [-0.7, 0.8]])
        V = np.array([[0.1, 0.2],
                      [0.5, 0.4]])

        solver.bath.eps[:] = mul.outer([1, -1], eps)
        solver.bath.V[:] = mul.outer([1, 0.9], V)
        solver.bath.U[:] = mul.outer([0.2, 0.2], V)

        solve_params = {
            "beta": 60.0,
            "n_iw": 30,
            "energy_window": (-1.5, 1.5),
            "n_w": 50,
            "broadening": 0.02
        }
        solver.solve(**solve_params)

        ## Reference solution
        h_bath = self.make_h_bath(mul.outer([1, -1], eps),
                                  mul.outer(np.diag([1, 0.9]) + 0.2 * sx, V),
                                  spin_blocks=False)
        h = h_loc + h_int + h_bath
        refs = self.ref_results("nonsu2_bath_3", h=h, fops=fops,
                                spin_blocks=False, **solve_params)
        self.assert_all(solver, **refs)

    def test_h_loc_an(self):
        h_loc_mat = mul.outer(s0, np.diag([-0.5, -0.6]))
        h_loc = self.make_h_loc(h_loc_mat)
        h_int = self.make_h_int(
            Uloc=[0.1, 0.2], Ust=0.4, Jh=0.2, Jx=0.1, Jp=0.15
        )

        fops_imp_up, fops_imp_dn = self.make_fops_imp()
        fops = fops_imp_up + fops_imp_dn + self.fops_bath_up + self.fops_bath_dn

        eps = np.array([[-0.5, 0.5],
                        [-0.7, 0.7]])
        V = np.array([[0.1, 0.2],
                      [0.3, 0.4]])
        h_bath = self.make_h_bath(mul.outer([1, 1], eps), mul.outer(s0, V))

        h_loc_an_mat = np.diag([0.1, 0.2])
        h_loc_an = self.make_h_loc_an(h_loc_an_mat)

        h = h_loc + h_loc_an + h_int + h_bath
        solver = EDIpackSolver(
            h,
            fops_imp_up, fops_imp_dn,
            self.fops_bath_up, self.fops_bath_dn,
            lanczos_params=LanczosParams(nstates_sector=4, nstates_total=10),
            verbose=0
        )

        self.assertEqual(solver.h_params.ed_mode, EDMode.SUPERC)
        self.assertEqual(solver.nspin, 1)
        self.assertEqual(solver.norb, self.norb)
        self.assertEqual(solver.hloc, h_loc + h_loc_an)
        assert_equal(solver.hloc_mat, h_loc_mat[:1, :1, ...])
        assert_equal(solver.hloc_an_mat,
                     h_loc_an_mat.reshape((1, 1, self.norb, self.norb)))
        self.assertEqual(solver.bath.name, "normal")
        self.assertEqual(solver.bath.nbath, self.nbath)

        # Part I: Initial solve()
        solve_params = {
            "beta": 60.0,
            "n_iw": 10,
            "energy_window": (-2.0, 2.0),
            "n_w": 60,
            "broadening": 0.05
        }
        solver.solve(**solve_params)

        ## Reference solution
        refs = self.ref_results("h_loc_an_1", h=h, fops=fops,
                                superc=True, **solve_params)
        self.assert_all(solver, **refs)

        # Part II: update interaction parameters and pairing field
        new_int_params = {
            'Uloc': [0.15, 0.25], 'Ust': 0.35, 'Jh': 0.1, 'Jx': 0.2, 'Jp': 0.0
        }
        self.change_int_params(solver.U, **new_int_params)
        h_loc_an_mat = np.diag([0.2, 0.3])
        h_loc_an = self.make_h_loc_an(h_loc_an_mat)
        solver.hloc = h_loc + h_loc_an
        solver.comm.barrier()

        solve_params = {
            "beta": 70.0,
            "n_iw": 20,
            "energy_window": (-1.5, 1.5),
            "n_w": 40,
            "broadening": 0.03
        }
        solver.solve(**solve_params)

        # Reference solution
        h_int = self.make_h_int(**new_int_params)
        h_loc_an = self.make_h_loc_an(h_loc_an_mat)
        h = h_loc + h_loc_an + h_int + h_bath
        refs = self.ref_results("h_loc_an_2", h=h, fops=fops,
                                superc=True, **solve_params)
        self.assert_all(solver, **refs)

        # Part III: Updated bath parameters
        eps = np.array([[-0.5, 0.5],
                        [-0.7, 0.8]])
        V = np.array([[0.1, 0.2],
                      [0.5, 0.4]])

        solver.bath.eps[0, ...] = eps
        solver.bath.V[0, ...] = V

        solve_params = {
            "beta": 60.0,
            "n_iw": 30,
            "energy_window": (-1.5, 1.5),
            "n_w": 80,
            "broadening": 0.02
        }
        solver.solve(**solve_params)

        # Reference solution
        h_bath = self.make_h_bath(mul.outer([1, 1], eps), mul.outer(s0, V))
        h = h_loc + h_loc_an + h_int + h_bath
        refs = self.ref_results("h_loc_an_3", h=h, fops=fops,
                                superc=True, **solve_params)
        self.assert_all(solver, **refs)

    def test_superc(self):
        h_loc_mat = mul.outer(s0, np.diag([-0.5, -0.6]))
        h_loc = self.make_h_loc(h_loc_mat)
        h_int = self.make_h_int(
            Uloc=[0.1, 0.2], Ust=0.4, Jh=0.2, Jx=0.1, Jp=0.15
        )

        fops_imp_up, fops_imp_dn = self.make_fops_imp()
        fops = fops_imp_up + fops_imp_dn + self.fops_bath_up + self.fops_bath_dn

        eps = np.array([[-0.5, 0.5],
                        [-0.7, 0.7]])
        V = np.array([[0.1, 0.2],
                      [0.3, 0.4]])
        h_bath = self.make_h_bath(mul.outer([1, 1], eps), mul.outer(s0, V))

        Delta = np.array([[0.6, 0.7],
                          [0.8, 0.6]])
        h_sc = self.make_h_sc(Delta)

        h = h_loc + h_int + h_bath + h_sc
        solver = EDIpackSolver(
            h,
            fops_imp_up, fops_imp_dn,
            self.fops_bath_up, self.fops_bath_dn,
            lanczos_params=LanczosParams(nstates_sector=4, nstates_total=10),
            verbose=0
        )

        self.assertEqual(solver.h_params.ed_mode, EDMode.SUPERC)
        self.assertEqual(solver.nspin, 1)
        self.assertEqual(solver.norb, self.norb)
        self.assertEqual(solver.hloc, h_loc)
        assert_equal(solver.hloc_mat, h_loc_mat[:1, :1, ...])
        assert_equal(solver.hloc_an_mat, np.zeros((1, 1, self.norb, self.norb)))
        self.assertEqual(solver.bath.name, "normal")
        self.assertEqual(solver.bath.nbath, self.nbath)

        # Part I: Initial solve()
        solve_params = {
            "beta": 60.0,
            "n_iw": 10,
            "energy_window": (-2.0, 2.0),
            "n_w": 60,
            "broadening": 0.05
        }
        solver.solve(**solve_params)

        ## Reference solution
        refs = self.ref_results("superc_1", h=h, fops=fops,
                                superc=True, **solve_params)
        self.assert_all(solver, **refs)

        # Part II: update interaction parameters
        new_int_params = {
            'Uloc': [0.15, 0.25], 'Ust': 0.35, 'Jh': 0.1, 'Jx': 0.2, 'Jp': 0.0
        }
        self.change_int_params(solver.U, **new_int_params)
        solver.hloc = h_loc
        solver.comm.barrier()

        solve_params = {
            "beta": 70.0,
            "n_iw": 20,
            "energy_window": (-1.5, 1.5),
            "n_w": 40,
            "broadening": 0.03
        }
        solver.solve(**solve_params)

        # Reference solution
        h_int = self.make_h_int(**new_int_params)
        h = h_loc + h_int + h_bath + h_sc
        refs = self.ref_results("superc_2", h=h, fops=fops,
                                superc=True, **solve_params)
        self.assert_all(solver, **refs)

        # Part III: Updated bath parameters
        eps = np.array([[-0.5, 0.5],
                        [-0.7, 0.8]])
        V = np.array([[0.1, 0.2],
                      [0.5, 0.4]])
        Delta = np.array([[0.5, 0.7],
                          [0.7, 0.6]])

        solver.bath.eps[0, ...] = eps
        solver.bath.Delta[0, ...] = Delta
        solver.bath.V[0, ...] = V

        solve_params = {
            "beta": 60.0,
            "n_iw": 30,
            "energy_window": (-1.5, 1.5),
            "n_w": 80,
            "broadening": 0.02
        }
        solver.solve(**solve_params)

        # Reference solution
        h_bath = self.make_h_bath(mul.outer([1, 1], eps), mul.outer(s0, V))
        h_sc = self.make_h_sc(Delta)
        h = h_loc + h_int + h_bath + h_sc
        refs = self.ref_results("superc_3", h=h, fops=fops,
                                superc=True, **solve_params)
        self.assert_all(solver, **refs)

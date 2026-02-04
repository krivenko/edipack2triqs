from itertools import product

import numpy as np
from numpy import multiply as mul

import triqs.operators as op

from edipack2triqs.solver import EDIpackSolver, LanczosParams

from .test_solver import TestSolver


s0 = np.eye(2)
sx = np.array([[0, 1], [1, 0]])
sz = np.array([[1, 0], [0, -1]])


class TestEDIpackSolverSectors(TestSolver):

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
            ed_sectors=True,
            ed_sectors_shift=0,
            lanczos_params=LanczosParams(nstates_sector=4, nstates_total=10),
            verbose=2
        )

        self.assertIsNone(solver.sectors)
        # (N_{up}, N_{dn})
        sectors = [(3, 3), (2, 4), (4, 2), (3, 4), (4, 3), (4, 4)]
        solver.sectors = sectors
        self.assertEqual(solver.sectors, sectors)

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
        refs = self.ref_results("nspin1", h=h, fops=fops, **solve_params)
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
            ed_sectors=True,
            ed_sectors_shift=0,
            lanczos_params=LanczosParams(nstates_sector=4, nstates_total=10),
            verbose=2
        )

        self.assertIsNone(solver.sectors)
        # (N_{up}, N_{dn})
        sectors = [(3, 3), (2, 4), (4, 2), (3, 4), (4, 3), (4, 4)]
        solver.sectors = sectors
        self.assertEqual(solver.sectors, sectors)

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
        refs = self.ref_results("nspin2", h=h, fops=fops, **solve_params)
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
            ed_sectors=True,
            ed_sectors_shift=0,
            lanczos_params=LanczosParams(nstates_sector=4, nstates_total=10),
            verbose=2
        )

        self.assertIsNone(solver.sectors)
        sectors = [5, 6, 7, 8]  # N_{tot}
        solver.sectors = sectors
        self.assertEqual(solver.sectors, sectors)

        solve_params = {
            "beta": 70.0,
            "n_iw": 10,
            "energy_window": (-1.5, 1.5),
            "n_w": 60,
            "broadening": 0.05
        }
        solver.solve(**solve_params)

        ## Reference solution
        refs = self.ref_results("nonsu2_hloc", h=h, fops=fops,
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
            ed_sectors=True,
            ed_sectors_shift=0,
            lanczos_params=LanczosParams(nstates_sector=6, nstates_total=12),
            verbose=2
        )

        self.assertIsNone(solver.sectors)
        sectors = [5, 6, 7, 8]  # N_{tot}
        solver.sectors = sectors
        self.assertEqual(solver.sectors, sectors)

        solve_params = {
            "beta": 60.0,
            "n_iw": 10,
            "energy_window": (-1.5, 1.5),
            "n_w": 60,
            "broadening": 0.05
        }
        solver.solve(**solve_params)

        ## Reference solution
        refs = self.ref_results("nonsu2_bath", h=h, fops=fops,
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
            ed_sectors=True,
            ed_sectors_shift=0,
            lanczos_params=LanczosParams(nstates_sector=4, nstates_total=10),
            verbose=2
        )

        self.assertIsNone(solver.sectors)
        sectors = [-2, -1, 0, 1, 2]  # S_z
        solver.sectors = sectors
        self.assertEqual(solver.sectors, sectors)

        solve_params = {
            "beta": 60.0,
            "n_iw": 10,
            "energy_window": (-2.0, 2.0),
            "n_w": 60,
            "broadening": 0.05
        }
        solver.solve(**solve_params)

        ## Reference solution
        refs = self.ref_results("h_loc_an", h=h, fops=fops,
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
            ed_sectors=True,
            ed_sectors_shift=0,
            lanczos_params=LanczosParams(nstates_sector=4, nstates_total=10),
            verbose=2
        )

        self.assertIsNone(solver.sectors)
        sectors = [-2, -1, 0, 1, 2]  # S_z
        solver.sectors = sectors
        self.assertEqual(solver.sectors, sectors)

        solve_params = {
            "beta": 60.0,
            "n_iw": 10,
            "energy_window": (-2.0, 2.0),
            "n_w": 60,
            "broadening": 0.05
        }
        solver.solve(**solve_params)

        ## Reference solution
        refs = self.ref_results("superc", h=h, fops=fops,
                                superc=True, **solve_params)
        self.assert_all(solver, **refs)

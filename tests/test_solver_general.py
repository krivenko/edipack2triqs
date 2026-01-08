from itertools import product

import numpy as np
from numpy import multiply as mul
from numpy.testing import assert_equal

import triqs.operators as op

from edipack2triqs import EDMode
from edipack2triqs.solver import EDIpackSolver

from .test_solver import TestSolver


s0 = np.eye(2)
sx = np.array([[0, 1], [1, 0]])
sz = np.array([[1, 0], [0, -1]])


class TestEDIpackSolverBathGeneral(TestSolver):

    nbath = 2
    bsites = range(nbath)

    @classmethod
    @TestSolver.bath_index_ranges(bsites, TestSolver.orbs)
    def mkind_bath(cls, spin, nu, orb):
        "Map (spin, nu, orb) -> (block name, inner index)"
        return (f"B_{spin}", cls.norb * nu + orb)

    @classmethod
    def make_h_bath(cls, h, V, spin_blocks=True):
        d_dag, d = [cls.make_op_imp(o, spin_blocks) for o in (op.c_dag, op.c)]
        a_dag, a = [cls.make_bath_op(o) for o in (op.c_dag, op.c)]
        h_bath = sum(h[s1, s2, o1, o2, nu]
                     * a_dag(spin1, nu, o1) * a(spin2, nu, o2)
                     for (s1, spin1), (s2, spin2), o1, o2, nu
                     in product(enumerate(cls.spins), enumerate(cls.spins),
                                cls.orbs, cls.orbs, range(2)))
        h_bath += sum(V[s, o, nu] * (
                      d_dag(spin, o) * a(spin, nu, o)
                      + a_dag(spin, nu, o) * d(spin, o))
                      for (s, spin), o, nu
                      in product(enumerate(cls.spins), cls.orbs, range(2)))
        return h_bath

    @classmethod
    def make_h_sc(cls, Delta):
        a_dag, a = [cls.make_bath_op(o) for o in (op.c_dag, op.c)]
        h_sc = sum(Delta[o1, o2, nu] * a_dag('up', nu, o1) * a_dag('dn', nu, o2)
                   for o1, o2, nu in product(cls.orbs, cls.orbs, range(2)))
        return h_sc + op.dagger(h_sc)

    @classmethod
    def assert_all(cls, s, **refs):
        cls.assert_static_obs(s, 1e-8, **refs)
        cls.assert_gfs(s, **refs)
        cls.assert_chi(s, **refs)

    @classmethod
    def find_basis_mat(cls, hvec, mat):
        for isym in range(hvec.shape[-1]):
            if (hvec[:, :, :, :, isym] == mat).all():
                return isym
        raise ValueError(f"Basis matrix {mat} not found")

    def test_zerotemp(self):
        h_loc_mat = mul.outer(s0, np.diag([-0.5, -0.6]))
        h_loc = self.make_h_loc(h_loc_mat)
        h_int = self.make_h_int(
            Uloc=[0.1, 0.2], Ust=0.4, Jh=0.2, Jx=0.1, Jp=0.15
        )

        fops_imp_up, fops_imp_dn = self.make_fops_imp()
        fops = fops_imp_up + fops_imp_dn + self.fops_bath_up + self.fops_bath_dn

        h = np.moveaxis(np.array([[[0.5, 0.1],
                                   [0.1, 0.6]],
                                  [[-0.5, 0.0],
                                   [0.0, -0.6]]]), 0, 2)
        V = np.array([[0.2, -0.2],
                      [0.4, -0.4]])
        h_bath = self.make_h_bath(mul.outer(s0, h), mul.outer([1, 1], V))

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
        self.assertEqual(solver.norb, 2)
        self.assertEqual(solver.hloc, h_loc)
        assert_equal(solver.hloc_mat, h_loc_mat[:1, :1, ...])
        assert_equal(solver.hloc_an_mat, np.zeros((1, 1, 2, 2)))
        self.assertEqual(solver.bath.name, "general")
        self.assertEqual(solver.bath.nbath, 2)

        solve_params = {
            "n_iw": 10,
            "energy_window": (-1.0, 1.0),
            "n_w": 40,
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

        h = np.moveaxis(np.array([[[0.5, 0.1],
                                   [0.1, 0.6]],
                                  [[-0.5, 0.0],
                                   [0.0, -0.6]]]), 0, 2)
        V = np.array([[0.2, -0.2],
                      [0.4, -0.4]])
        h_bath = self.make_h_bath(mul.outer(s0, h), mul.outer([1, 1], V))

        h = h_loc + h_int + h_bath
        solver = EDIpackSolver(
            h,
            fops_imp_up, fops_imp_dn,
            self.fops_bath_up, self.fops_bath_dn,
            lanc_nstates_sector=4,
            lanc_nstates_total=14,
            verbose=0
        )

        self.assertEqual(solver.h_params.ed_mode, EDMode.NORMAL)
        self.assertEqual(solver.nspin, 1)
        self.assertEqual(solver.norb, 2)
        self.assertEqual(solver.hloc, h_loc)
        assert_equal(solver.hloc_mat, h_loc_mat[:1, :1, ...])
        assert_equal(solver.hloc_an_mat, np.zeros((1, 1, 2, 2)))
        self.assertEqual(solver.bath.name, "general")
        self.assertEqual(solver.bath.nbath, 2)

        # Part I: Initial solve()
        solve_params = {
            "beta": 70.0,
            "n_iw": 10,
            "energy_window": (-1.0, 1.0),
            "n_w": 40,
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
            "beta": 80.0,
            "n_iw": 10,
            "energy_window": (-1.0, 1.0),
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
        h = np.moveaxis(np.array([[[0.5, 0.2],
                                   [0.2, 0.6]],
                                  [[-0.5, 0.0],
                                   [0.0, -0.6]]]), 0, 2)
        V = np.array([[0.1, 0.2],
                      [0.5, 0.4]])

        mat = np.zeros((1, 1, 2, 2), dtype=complex)
        mat[0, 0, 0, 1] = mat[0, 0, 1, 0] = 1

        bath = solver.bath
        bath.l[0][self.find_basis_mat(bath.hvec, mat)] = 0.2
        bath.V[0][:] = V[:, 0]
        bath.V[1][:] = V[:, 1]

        solve_params = {
            "beta": 70.0,
            "n_iw": 10,
            "energy_window": (-1.0, 1.0),
            "n_w": 40,
            "broadening": 0.02,
            "n_tau": 12,
            **self.chi_params
        }
        solver.solve(**solve_params)

        ## Reference solution
        h_bath = self.make_h_bath(mul.outer(s0, h), mul.outer([1, 1], V))
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

        h = np.moveaxis(np.array([[[0.5, 0.1],
                                   [0.1, 0.6]],
                                  [[-0.5, 0.0],
                                   [0.0, -0.6]]]), 0, 2)
        V = np.array([[0.2, -0.2],
                      [0.4, -0.4]])
        h_bath = self.make_h_bath(mul.outer(sz, h),
                                  mul.outer([1, 0.9], V))

        h = h_loc + h_int + h_bath
        solver = EDIpackSolver(
            h,
            fops_imp_up, fops_imp_dn,
            self.fops_bath_up, self.fops_bath_dn,
            lanc_nstates_sector=4,
            lanc_nstates_total=14,
            verbose=0
        )

        self.assertEqual(solver.h_params.ed_mode, EDMode.NORMAL)
        self.assertEqual(solver.nspin, 2)
        self.assertEqual(solver.norb, 2)
        self.assertEqual(solver.hloc, h_loc)
        assert_equal(solver.hloc_mat, h_loc_mat)
        assert_equal(solver.hloc_an_mat, np.zeros((1, 1, 2, 2)))
        self.assertEqual(solver.bath.name, "general")
        self.assertEqual(solver.bath.nbath, 2)

        # Part I: Initial solve()
        solve_params = {
            "beta": 60.0,
            "n_iw": 10,
            "energy_window": (-1.0, 1.0),
            "n_w": 40,
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
            "beta": 70.0,
            "n_iw": 10,
            "energy_window": (-1.0, 1.0),
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
        h = np.moveaxis(np.array([[[0.5, 0.2],
                                   [0.2, 0.6]],
                                  [[-0.5, 0.0],
                                   [0.0, -0.6]]]), 0, 2)
        V = np.array([[0.1, 0.2],
                      [0.5, 0.4]])

        mat_up = np.zeros((2, 2, 2, 2), dtype=complex)
        mat_dn = np.zeros((2, 2, 2, 2), dtype=complex)
        mat_up[0, 0, 0, 1] = mat_up[0, 0, 1, 0] = 1
        mat_dn[1, 1, 0, 1] = mat_dn[1, 1, 1, 0] = 1

        bath = solver.bath
        bath.l[0][self.find_basis_mat(bath.hvec, mat_up)] = 0.2
        bath.l[0][self.find_basis_mat(bath.hvec, mat_dn)] = -0.2
        bath.V[0][:] = mul.outer([1, 0.9], V[:, 0])
        bath.V[1][:] = mul.outer([1, 0.9], V[:, 1])

        solve_params = {
            "beta": 60.0,
            "n_iw": 10,
            "energy_window": (-1.0, 1.0),
            "n_w": 40,
            "n_tau": 12,
            "broadening": 0.02,
            **self.chi_params
        }
        solver.solve(**solve_params)

        ## Reference solution
        h_bath = self.make_h_bath(mul.outer(sz, h), mul.outer([1, 0.9], V))
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

        fops_imp_up, fops_imp_dn = self.make_fops_imp(spin_blocks=False)
        fops = fops_imp_up + fops_imp_dn + self.fops_bath_up + self.fops_bath_dn

        h = np.moveaxis(np.array([[[0.5, 0.1 - 0.1j],
                                   [0.1 + 0.1j, 0.6]],
                                  [[-0.5, 0.0],
                                   [0.0, -0.6]]]), 0, 2)
        V = np.array([[0.1, 0.2],
                      [0.3, 0.4]])
        h_bath = self.make_h_bath(mul.outer(sz, h),
                                  mul.outer([1, 0.9], V),
                                  spin_blocks=False)

        h = h_loc + h_int + h_bath
        solver = EDIpackSolver(
            h,
            fops_imp_up, fops_imp_dn,
            self.fops_bath_up, self.fops_bath_dn,
            lanc_nstates_sector=4,
            lanc_nstates_total=14,
            verbose=0
        )

        self.assertEqual(solver.h_params.ed_mode, EDMode.NONSU2)
        self.assertEqual(solver.nspin, 2)
        self.assertEqual(solver.norb, 2)
        self.assertEqual(solver.hloc, h_loc)
        assert_equal(solver.hloc_mat, h_loc_mat)
        assert_equal(solver.hloc_an_mat, np.zeros((1, 1, 2, 2)))
        self.assertEqual(solver.bath.name, "general")
        self.assertEqual(solver.bath.nbath, 2)

        # Part I: Initial solve()
        solve_params = {
            "beta": 60.0,
            "n_iw": 10,
            "energy_window": (-1.0, 1.0),
            "n_w": 40,
            "broadening": 0.05
        }
        solver.solve(**solve_params)

        # Reference solution
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
            "beta": 70.0,
            "n_iw": 20,
            "energy_window": (-1.0, 1.0),
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
        h = np.moveaxis(np.array([[[0.5, 0.2 - 0.1j],
                                   [0.2 + 0.1j, 0.6]],
                                  [[-0.5, 0.0],
                                   [0.0, -0.6]]]), 0, 2)
        V = np.array([[0.1, 0.2],
                      [0.35, 0.4]])

        mat_up = np.zeros((2, 2, 2, 2), dtype=complex)
        mat_dn = np.zeros((2, 2, 2, 2), dtype=complex)
        mat_up[0, 0, 0, 1] = mat_up[0, 0, 1, 0] = 1
        mat_dn[1, 1, 0, 1] = mat_dn[1, 1, 1, 0] = 1

        bath = solver.bath
        bath.l[0][self.find_basis_mat(bath.hvec, mat_up)] = 0.2
        bath.l[0][self.find_basis_mat(bath.hvec, mat_dn)] = -0.2
        bath.V[0][:] = mul.outer([1, 0.9], V[:, 0])
        bath.V[1][:] = mul.outer([1, 0.9], V[:, 1])

        solve_params = {
            "beta": 60.0,
            "n_iw": 20,
            "energy_window": (-1.0, 1.0),
            "n_w": 40,
            "broadening": 0.03
        }
        solver.solve(**solve_params)

        ## Reference solution
        h_bath = self.make_h_bath(mul.outer(sz, h),
                                  mul.outer([1, 0.9], V),
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

        fops_imp_up, fops_imp_dn = self.make_fops_imp(spin_blocks=False)
        fops = fops_imp_up + fops_imp_dn + self.fops_bath_up + self.fops_bath_dn

        h = np.moveaxis(np.array([[[0.5, 0.1 - 0.1j],
                                   [0.1 + 0.1j, 0.6]],
                                  [[-0.5, 0.0],
                                   [0.0, -0.6]]]), 0, 2)
        V = np.array([[0.1, 0.2],
                      [0.3, 0.4]])
        h_bath = self.make_h_bath(mul.outer(sz + 0.2 * sx, h),
                                  mul.outer([1, 0.9], V),
                                  spin_blocks=False)

        h = h_loc + h_int + h_bath
        solver = EDIpackSolver(
            h,
            fops_imp_up, fops_imp_dn,
            self.fops_bath_up, self.fops_bath_dn,
            lanc_nstates_sector=4,
            lanc_nstates_total=14,
            verbose=0
        )

        self.assertEqual(solver.h_params.ed_mode, EDMode.NONSU2)
        self.assertEqual(solver.nspin, 2)
        self.assertEqual(solver.norb, 2)
        self.assertEqual(solver.hloc, h_loc)
        assert_equal(solver.hloc_mat, h_loc_mat)
        assert_equal(solver.hloc_an_mat, np.zeros((1, 1, 2, 2)))
        self.assertEqual(solver.bath.name, "general")
        self.assertEqual(solver.bath.nbath, 2)

        # Part I: Initial solve()
        solve_params = {
            "beta": 60.0,
            "n_iw": 10,
            "energy_window": (-1.0, 1.0),
            "n_w": 40,
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
            "energy_window": (-1.0, 1.0),
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
        h = np.moveaxis(np.array([[[0.5, 0.2 - 0.1j],
                                   [0.2 + 0.1j, 0.6]],
                                  [[-0.5, 0.0],
                                   [0.0, -0.6]]]), 0, 2)
        V = np.array([[0.1, 0.2],
                      [0.35, 0.4]])

        mat_up = np.zeros((2, 2, 2, 2), dtype=complex)
        mat_dn = np.zeros((2, 2, 2, 2), dtype=complex)
        mat_up[0, 0, 0, 1] = mat_up[0, 0, 1, 0] = 1
        mat_dn[1, 1, 0, 1] = mat_dn[1, 1, 1, 0] = 1
        mat_updn1 = np.zeros((2, 2, 2, 2), dtype=complex)
        mat_updn2 = np.zeros((2, 2, 2, 2), dtype=complex)
        mat_updn1[0, 1, 0, 1] = mat_updn1[1, 0, 1, 0] = 1
        mat_updn2[0, 1, 1, 0] = mat_updn2[1, 0, 0, 1] = 1

        bath = solver.bath
        bath.l[0][self.find_basis_mat(bath.hvec, mat_up)] = 0.2
        bath.l[0][self.find_basis_mat(bath.hvec, mat_dn)] = -0.2
        bath.l[0][self.find_basis_mat(bath.hvec, mat_updn1)] = 0.2 * 0.2
        bath.l[0][self.find_basis_mat(bath.hvec, mat_updn2)] = 0.2 * 0.2
        bath.V[0][:] = mul.outer([1, 0.9], V[:, 0])
        bath.V[1][:] = mul.outer([1, 0.9], V[:, 1])

        solve_params = {
            "beta": 60.0,
            "n_iw": 20,
            "energy_window": (-1.0, 1.0),
            "n_w": 40,
            "broadening": 0.03
        }
        solver.solve(**solve_params)

        ## Reference solution
        h_bath = self.make_h_bath(mul.outer(sz + 0.2 * sx, h),
                                  mul.outer([1, 0.9], V),
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

        h = np.moveaxis(np.array([[[0.5, 0.1],
                                   [0.1, 0.6]],
                                  [[-0.5, 0.0],
                                   [0.0, -0.6]]]), 0, 2)
        V = np.array([[0.2, -0.2],
                      [0.4, -0.4]])
        h_bath = self.make_h_bath(mul.outer(s0, h), mul.outer([1, 1], V))

        h_loc_an_mat = np.array([[0.1, 0.6j],
                                 [0.6j, 0.15]])
        h_loc_an = self.make_h_loc_an(h_loc_an_mat)

        h = h_loc + h_loc_an + h_int + h_bath
        solver = EDIpackSolver(
            h,
            fops_imp_up, fops_imp_dn,
            self.fops_bath_up, self.fops_bath_dn,
            lanc_nstates_sector=4,
            lanc_nstates_total=14,
            verbose=0
        )

        self.assertEqual(solver.h_params.ed_mode, EDMode.SUPERC)
        self.assertEqual(solver.nspin, 1)
        self.assertEqual(solver.norb, 2)
        self.assertEqual(solver.hloc, h_loc + h_loc_an)
        assert_equal(solver.hloc_mat, h_loc_mat[:1, :1, ...])
        assert_equal(solver.hloc_an_mat, h_loc_an_mat.reshape((1, 1, 2, 2)))
        self.assertEqual(solver.bath.name, "general")
        self.assertEqual(solver.bath.nbath, 2)

        # Part I: Initial solve()
        solve_params = {
            "beta": 60.0,
            "n_iw": 10,
            "energy_window": (-1.0, 1.0),
            "n_w": 40,
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
        h_loc_an_mat = np.array([[0.2, 0.5j],
                                 [0.5j, 0.3]])
        h_loc_an = self.make_h_loc_an(h_loc_an_mat)
        solver.hloc = h_loc + h_loc_an
        solver.comm.barrier()

        solve_params = {
            "beta": 70.0,
            "n_iw": 10,
            "energy_window": (-1.0, 1.0),
            "n_w": 40,
            "broadening": 0.03
        }
        solver.solve(**solve_params)

        ## Reference solution
        h_int = self.make_h_int(**new_int_params)
        h_loc_an = self.make_h_loc_an(h_loc_an_mat)
        h = h_loc + h_loc_an + h_int + h_bath
        refs = self.ref_results("h_loc_an_2", h=h, fops=fops,
                                superc=True, **solve_params)
        self.assert_all(solver, **refs)

        # Part III: Updated bath parameters
        h = np.moveaxis(np.array([[[0.5, 0.2],
                                   [0.2, 0.6]],
                                  [[-0.5, 0.0],
                                   [0.0, -0.6]]]), 0, 2)
        V = np.array([[0.1, 0.2],
                      [0.5, 0.4]])

        mat = np.zeros((2, 2, 2, 2), dtype=complex)
        mat[0, 0, 0, 1] = mat[0, 0, 1, 0] = 1
        mat[1, 1, 0, 1] = mat[1, 1, 1, 0] = -1

        bath = solver.bath
        bath.l[0][self.find_basis_mat(bath.hvec, mat)] = 0.2
        bath.V[0][:] = V[:, 0]
        bath.V[1][:] = V[:, 1]

        solve_params = {
            "beta": 60.0,
            "n_iw": 10,
            "energy_window": (-1.0, 1.0),
            "n_w": 40,
            "broadening": 0.02
        }
        solver.solve(**solve_params)

        ## Reference solution
        h_bath = self.make_h_bath(mul.outer(s0, h), mul.outer([1, 1], V))
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

        h = np.moveaxis(np.array([[[0.5, 0.1],
                                   [0.1, 0.6]],
                                  [[-0.5, 0.0],
                                   [0.0, -0.6]]]), 0, 2)
        V = np.array([[0.2, -0.2],
                      [0.4, -0.4]])
        h_bath = self.make_h_bath(mul.outer(s0, h), mul.outer([1, 1], V))
        Delta = np.moveaxis(np.array([[[0.3, 0.2j],
                                       [0.2j, 0.4]],
                                      [[0.3, 0.0],
                                       [0.0, 0.5]]]), 0, 2)
        h_sc = self.make_h_sc(Delta)

        h = h_loc + h_int + h_bath + h_sc
        solver = EDIpackSolver(
            h,
            fops_imp_up, fops_imp_dn,
            self.fops_bath_up, self.fops_bath_dn,
            lanc_nstates_sector=4,
            lanc_nstates_total=14,
            verbose=0
        )

        self.assertEqual(solver.h_params.ed_mode, EDMode.SUPERC)
        self.assertEqual(solver.nspin, 1)
        self.assertEqual(solver.norb, 2)
        self.assertEqual(solver.hloc, h_loc)
        assert_equal(solver.hloc_mat, h_loc_mat[:1, :1, ...])
        assert_equal(solver.hloc_an_mat, np.zeros((1, 1, 2, 2)))
        self.assertEqual(solver.bath.name, "general")
        self.assertEqual(solver.bath.nbath, 2)

        # Part I: Initial solve()
        solve_params = {
            "beta": 60.0,
            "n_iw": 10,
            "energy_window": (-1.0, 1.0),
            "n_w": 40,
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
            "n_iw": 10,
            "energy_window": (-1.0, 1.0),
            "n_w": 40,
            "broadening": 0.03
        }
        solver.solve(**solve_params)

        ## Reference solution
        h_int = self.make_h_int(**new_int_params)
        h = h_loc + h_int + h_bath + h_sc
        refs = self.ref_results("superc_2", h=h, fops=fops,
                                superc=True, **solve_params)
        self.assert_all(solver, **refs)

        # Part III: Updated bath parameters
        h = np.moveaxis(np.array([[[0.5, 0.2],
                                   [0.2, 0.6]],
                                  [[-0.5, 0.0],
                                   [0.0, -0.6]]]), 0, 2)
        V = np.array([[0.1, 0.2],
                      [0.5, 0.4]])

        Delta = np.moveaxis(np.array([[[0.3, 0.1j],
                                       [0.1j, 0.4]],
                                      [[0.3, 0.0],
                                       [0.0, 0.5]]]), 0, 2)

        mat = np.zeros((2, 2, 2, 2), dtype=complex)
        mat[0, 0, 0, 1] = mat[0, 0, 1, 0] = 1
        mat[1, 1, 0, 1] = mat[1, 1, 1, 0] = -1
        mat_sc1 = np.zeros((2, 2, 2, 2), dtype=complex)
        mat_sc1[0, 1, 0, 1] = -1j
        mat_sc1[1, 0, 1, 0] = 1j
        mat_sc2 = np.zeros((2, 2, 2, 2), dtype=complex)
        mat_sc2[0, 1, 1, 0] = -1j
        mat_sc2[1, 0, 0, 1] = 1j

        bath = solver.bath
        bath.l[0][self.find_basis_mat(bath.hvec, mat)] = 0.2
        bath.l[0][self.find_basis_mat(bath.hvec, mat_sc1)] = -0.1
        bath.l[0][self.find_basis_mat(bath.hvec, mat_sc2)] = -0.1
        bath.V[0][:] = V[:, 0]
        bath.V[1][:] = V[:, 1]

        solve_params = {
            "beta": 60.0,
            "n_iw": 10,
            "energy_window": (-1.0, 1.0),
            "n_w": 40,
            "broadening": 0.02
        }
        solver.solve(**solve_params)

        ## Reference solution
        h_bath = self.make_h_bath(mul.outer(s0, h), mul.outer([1, 1], V))
        h_sc = self.make_h_sc(Delta)
        h = h_loc + h_int + h_bath + h_sc
        refs = self.ref_results("superc_3", h=h, fops=fops,
                                superc=True, **solve_params)
        self.assert_all(solver, **refs)

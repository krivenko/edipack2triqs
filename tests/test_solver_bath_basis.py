from itertools import product

import numpy as np
from numpy import multiply as mul
from numpy.testing import assert_equal, assert_allclose

import triqs.operators as op

from edipack2triqs import EDMode
from edipack2triqs.solver import EDIpackSolver

from .test_solver import TestSolver


s0 = np.eye(2)
sx = np.array([[0, 1], [1, 0]])
sy = np.array([[0, -1j], [1j, 0]])
sz = np.array([[1, 0], [0, -1]])


class TestEDIpackSolverBathBasis(TestSolver):

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
                                cls.orbs, cls.orbs, cls.bsites))
        h_bath += sum(V[s, o, nu] * (
                      d_dag(spin, o) * a(spin, nu, o)
                      + a_dag(spin, nu, o) * d(spin, o))
                      for (s, spin), o, nu
                      in product(enumerate(cls.spins), cls.orbs, cls.bsites))
        return h_bath

    @classmethod
    def make_h_sc(cls, Delta):
        a_dag, a = [cls.make_bath_op(o) for o in (op.c_dag, op.c)]
        h_sc = sum(Delta[o1, o2, nu] * a_dag('up', nu, o1) * a_dag('dn', nu, o2)
                   for o1, o2, nu in product(cls.orbs, cls.orbs, cls.bsites))
        return h_sc + op.dagger(h_sc)

    @classmethod
    def make_bath_basis(cls, basis_mats, basis_mats_an=None):
        spins = cls.spins
        orbs = cls.orbs
        if basis_mats_an is None:
            basis_mats_an = [np.zeros((cls.norb, cls.norb))
                             for _ in range(len(basis_mats))]

        a_dag, a = [cls.make_bath_op(o) for o in (op.c_dag, op.c)]
        # Interleave replicas to make test less trivial
        basis = [
            sum(mat[s1, s2, o1, o2] * a_dag(spin1, nu, o1) * a(spin2, nu, o2)
                for (s1, spin1), (s2, spin2), o1, o2
                in product(enumerate(spins), enumerate(spins), orbs, orbs))
            + sum(mat_an[o1, o2] * a_dag('up', nu, o1) * a_dag('dn', nu, o2)
                  + np.conj(mat_an[o1, o2]) * a('dn', nu, o2) * a('up', nu, o1)
                  for o1, o2 in product(cls.orbs, cls.orbs))
            for (mat, mat_an), nu
            in product(zip(basis_mats, basis_mats_an), cls.bsites)
        ]

        return basis

    @classmethod
    def assert_all(cls, s, **refs):
        cls.assert_static_obs(s, 1e-8, **refs)
        cls.assert_gfs(s, **refs)
        cls.assert_chi(s, **refs)

    def test_nspin1(self):
        h_loc_mat = mul.outer(s0, np.diag([-0.5, -0.6]))
        h_loc = self.make_h_loc(h_loc_mat)
        h_int = self.make_h_int(
            Uloc=[0.1, 0.2], Ust=0.4, Jh=0.2, Jx=0.1, Jp=0.15
        )

        fops_imp_up, fops_imp_dn = self.make_fops_imp()
        fops = fops_imp_up + fops_imp_dn + self.fops_bath_up + self.fops_bath_dn

        basis_mats = [mul.outer(s0, 2 * s0 + sx),
                      mul.outer(s0, s0 + 0.5 * sz),
                      mul.outer(s0, s0 - 0.5 * sz)]
        bath_basis = self.make_bath_basis(basis_mats)

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
            bath_basis=bath_basis,
            lanc_nstates_sector=4,
            lanc_nstates_total=14,
            verbose=0
        )

        self.assertEqual(solver.h_params.ed_mode, EDMode.NORMAL)
        self.assertEqual(solver.nspin, 1)
        self.assertEqual(solver.norb, self.norb)
        self.assertEqual(solver.hloc, h_loc)
        assert_equal(solver.hloc_mat, h_loc_mat[:1, :1, ...])
        assert_equal(solver.hloc_an_mat, np.zeros((1, 1, self.norb, self.norb)))
        self.assertEqual(solver.bath.name, "general")
        self.assertEqual(solver.bath.nbath, self.nbath)

        # Check constructed bath
        hvec_ref = np.zeros((1, 1, self.norb, self.norb, len(basis_mats)),
                            dtype=complex)
        for isym, mat in enumerate(basis_mats):
            hvec_ref[:, :, :, :, isym] = \
                mat[0, 0, :, :].reshape(1, 1, self.norb, self.norb)
        assert_equal(solver.bath.hvec, hvec_ref)
        l_ref = [[0.1, 0.125, 0.225],
                 [0, -0.225, -0.325]]
        assert_allclose(solver.bath.l, l_ref, atol=1e-10)

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

        # Reference solution
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

        basis_mats = [mul.outer(0.5 * s0 + sz, 2 * s0 + sx),
                      mul.outer(0.5 * s0 + sz, s0 + 0.5 * sz),
                      mul.outer(0.5 * s0 + sz, s0 - 0.5 * sz),
                      mul.outer(0.5 * s0 - sz, 2 * s0 + sx),
                      mul.outer(0.5 * s0 - sz, s0 + 0.5 * sz),
                      mul.outer(0.5 * s0 - sz, s0 - 0.5 * sz)]
        bath_basis = self.make_bath_basis(basis_mats)

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
            bath_basis=bath_basis,
            lanc_nstates_sector=4,
            lanc_nstates_total=14,
            verbose=0
        )

        self.assertEqual(solver.h_params.ed_mode, EDMode.NORMAL)
        self.assertEqual(solver.nspin, 2)
        self.assertEqual(solver.norb, self.norb)
        self.assertEqual(solver.hloc, h_loc)
        assert_equal(solver.hloc_mat, h_loc_mat)
        assert_equal(solver.hloc_an_mat, np.zeros((1, 1, self.norb, self.norb)))
        self.assertEqual(solver.bath.name, "general")
        self.assertEqual(solver.bath.nbath, self.nbath)

        # Check constructed bath
        hvec_ref = np.zeros((2, 2, self.norb, self.norb, len(basis_mats)),
                            dtype=complex)
        for isym, mat in enumerate(basis_mats):
            hvec_ref[:, :, :, :, isym] = mat
        assert_equal(solver.bath.hvec, hvec_ref)
        l_ref = [[0.05, 0.0625, 0.1125, -0.05, -0.0625, -0.1125],
                 [0, -0.1125, -0.1625, 0, 0.1125, 0.1625]]
        assert_allclose(solver.bath.l, l_ref, atol=1e-10)

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

        fops_imp_up, fops_imp_dn = self.make_fops_imp(spin_blocks=False)
        fops = fops_imp_up + fops_imp_dn + self.fops_bath_up + self.fops_bath_dn

        basis_mats = [mul.outer(0.5 * s0 + sz, 2 * s0 + sx),
                      mul.outer(0.5 * s0 + sz, s0 + sy),
                      mul.outer(0.5 * s0 + sz, s0 + 0.5 * sz),
                      mul.outer(0.5 * s0 + sz, s0 - 0.5 * sz),
                      mul.outer(0.5 * s0 - sz, 2 * s0 + sx),
                      mul.outer(0.5 * s0 - sz, s0 + sy),
                      mul.outer(0.5 * s0 - sz, s0 + 0.5 * sz),
                      mul.outer(0.5 * s0 - sz, s0 - 0.5 * sz)]
        bath_basis = self.make_bath_basis(basis_mats)

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
            bath_basis=bath_basis,
            lanc_nstates_sector=4,
            lanc_nstates_total=14,
            verbose=0
        )

        self.assertEqual(solver.h_params.ed_mode, EDMode.NONSU2)
        self.assertEqual(solver.nspin, 2)
        self.assertEqual(solver.norb, self.norb)
        self.assertEqual(solver.hloc, h_loc)
        assert_equal(solver.hloc_mat, h_loc_mat)
        assert_equal(solver.hloc_an_mat, np.zeros((1, 1, self.norb, self.norb)))
        self.assertEqual(solver.bath.name, "general")
        self.assertEqual(solver.bath.nbath, self.nbath)

        # Check constructed bath
        hvec_ref = np.zeros((2, 2, 2, 2, len(basis_mats)), dtype=complex)
        for isym, mat in enumerate(basis_mats):
            hvec_ref[:, :, :, :, isym] = mat
        assert_equal(solver.bath.hvec, hvec_ref)
        l_ref = [[0.05, 0.05, 0.0375, 0.0875, -0.05, -0.05, -0.0375, -0.0875],
                 [0, 0, -0.1125, -0.1625, 0, 0, 0.1125, 0.1625]]
        assert_allclose(solver.bath.l, l_ref, atol=1e-10)

        solve_params = {
            "beta": 60.0,
            "n_iw": 10,
            "energy_window": (-1.0, 1.0),
            "n_w": 40,
            "broadening": 0.05
        }
        solver.solve(**solve_params)

        # Reference solution
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

        fops_imp_up, fops_imp_dn = self.make_fops_imp(spin_blocks=False)
        fops = fops_imp_up + fops_imp_dn + self.fops_bath_up + self.fops_bath_dn

        basis_mats = [mul.outer(0.5 * sz, 2 * s0 + sx),
                      mul.outer(0.5 * sz, s0 + sy),
                      mul.outer(0.5 * sz, s0 + 0.5 * sz),
                      mul.outer(0.5 * sz, s0 - 0.5 * sz),
                      mul.outer(sx - sz, 2 * s0 + sx),
                      mul.outer(sx - sz, s0 + sy),
                      mul.outer(sx - sz, s0 + 0.5 * sz),
                      mul.outer(sx - sz, s0 - 0.5 * sz)]
        bath_basis = self.make_bath_basis(basis_mats)

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
            bath_basis=bath_basis,
            lanc_nstates_sector=4,
            lanc_nstates_total=14,
            verbose=0
        )

        self.assertEqual(solver.h_params.ed_mode, EDMode.NONSU2)
        self.assertEqual(solver.nspin, 2)
        self.assertEqual(solver.norb, self.norb)
        self.assertEqual(solver.hloc, h_loc)
        assert_equal(solver.hloc_mat, h_loc_mat)
        assert_equal(solver.hloc_an_mat, np.zeros((1, 1, self.norb, self.norb)))
        self.assertEqual(solver.bath.name, "general")
        self.assertEqual(solver.bath.nbath, self.nbath)

        # Check constructed bath
        hvec_ref = np.zeros((2, 2, self.norb, self.norb, len(basis_mats)),
                            dtype=complex)
        for isym, mat in enumerate(basis_mats):
            hvec_ref[:, :, :, :, isym] = mat
        assert_equal(solver.bath.hvec, hvec_ref)
        l_ref = [[0.24, 0.24, 0.18, 0.42, 0.02, 0.02, 0.015, 0.035],
                 [0, 0, -0.54, -0.78, 0, 0, -0.045, -0.065]]
        assert_allclose(solver.bath.l, l_ref, atol=1e-10)

        solve_params = {
            "beta": 60.0,
            "n_iw": 10,
            "energy_window": (-1.0, 1.0),
            "n_w": 40,
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

        basis_mats = [mul.outer(s0, 2 * s0 + sx),
                      mul.outer(s0, s0 + 0.5 * sz),
                      mul.outer(s0, s0 - 0.5 * sz)]
        bath_basis = self.make_bath_basis(basis_mats)

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
            bath_basis=bath_basis,
            lanc_nstates_sector=4,
            lanc_nstates_total=14,
            verbose=0
        )

        self.assertEqual(solver.h_params.ed_mode, EDMode.SUPERC)
        self.assertEqual(solver.nspin, 1)
        self.assertEqual(solver.norb, self.norb)
        self.assertEqual(solver.hloc, h_loc + h_loc_an)
        assert_equal(solver.hloc_mat, h_loc_mat[:1, :1, ...])
        assert_equal(solver.hloc_an_mat,
                     h_loc_an_mat.reshape((1, 1, self.norb, self.norb)))
        self.assertEqual(solver.bath.name, "general")
        self.assertEqual(solver.bath.nbath, self.nbath)

        # Check constructed bath
        hvec_ref = np.zeros((2, 2, self.norb, self.norb, len(basis_mats)),
                            dtype=complex)
        for isym, mat in enumerate(basis_mats):
            hvec_ref[:, :, :, :, isym] = mul.outer(sz, mat[0, 0, :, :])
        assert_equal(solver.bath.hvec, hvec_ref)
        l_ref = [[0.1, 0.125, 0.225],
                 [0, -0.225, -0.325]]
        assert_allclose(solver.bath.l, l_ref, atol=1e-10)

        solve_params = {
            "beta": 60.0,
            "n_iw": 10,
            "energy_window": (-1.0, 1.0),
            "n_w": 40,
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

        zero = np.zeros((self.norb, self.norb))
        basis_mats = [mul.outer(s0, s0 / 2 + sz),
                      mul.outer(s0, s0 / 2 - sz),
                      mul.outer(s0, sx - sz),
                      mul.outer(s0, zero),
                      mul.outer(s0, zero),
                      mul.outer(s0, zero)]
        basis_mats_an = [zero, zero, zero, s0 / 2 + sz, s0 / 2 - sz, 1j * sx]
        bath_basis = self.make_bath_basis(basis_mats, basis_mats_an)

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
            bath_basis=bath_basis,
            lanc_nstates_sector=4,
            lanc_nstates_total=14,
            verbose=0
        )

        self.assertEqual(solver.h_params.ed_mode, EDMode.SUPERC)
        self.assertEqual(solver.nspin, 1)
        self.assertEqual(solver.norb, self.norb)
        self.assertEqual(solver.hloc, h_loc)
        assert_equal(solver.hloc_mat, h_loc_mat[:1, :1, ...])
        assert_equal(solver.hloc_an_mat, np.zeros((1, 1, self.norb, self.norb)))
        self.assertEqual(solver.bath.name, "general")
        self.assertEqual(solver.bath.nbath, self.nbath)

        # Check constructed bath
        hvec_ref = np.zeros((2, 2, self.norb, self.norb, len(basis_mats)),
                            dtype=complex)
        for isym, (mat, mat_an) in enumerate(zip(basis_mats, basis_mats_an)):
            hvec_ref[0, 0, :, :, isym] = mat[0, 0, :, :]
            hvec_ref[0, 1, :, :, isym] = mat_an
            hvec_ref[1, 0, :, :, isym] = np.conj(mat_an.T)
            hvec_ref[1, 1, :, :, isym] = -mat[1, 1, :, :]
        assert_equal(solver.bath.hvec, hvec_ref)
        l_ref = [[0.575, 0.525, 0.1, 0.325, 0.375, 0.2],
                 [-0.525, -0.575, 0, 0.35, 0.45, 0]]
        assert_allclose(solver.bath.l, l_ref, atol=1e-10)

        solve_params = {
            "beta": 60.0,
            "n_iw": 10,
            "energy_window": (-1.0, 1.0),
            "n_w": 40,
            "broadening": 0.05
        }
        solver.solve(**solve_params)

        ## Reference solution
        refs = self.ref_results("superc", h=h, fops=fops,
                                superc=True, **solve_params)
        self.assert_all(solver, **refs)

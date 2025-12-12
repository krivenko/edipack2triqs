import unittest
import gc
from itertools import product

import numpy as np
from numpy.testing import assert_allclose
from numpy import multiply as mul

import triqs.operators as op
from triqs.utility.comparison_tests import (assert_gfs_are_close,
                                            assert_block_gfs_are_close)

from edipack2triqs.solver import EDIpackSolver

from . import reference as ref


s0 = np.eye(2)
sx = np.array([[0, 1], [1, 0]])
sz = np.array([[1, 0], [0, -1]])


# If pytest is available, detect a use of --generate-ref-data
try:
    import pytest

    @pytest.fixture(scope="session", autouse=True)
    def generate_ref_data(request):
        ref.generate_ref_data = request.config.getoption("generate_ref_data",
                                                         default=False)
except ImportError:
    pass


class TestEDIpackSolverBathGeneral(unittest.TestCase):

    spins = ('up', 'dn')
    orbs = range(2)

    fops_bath_up = [('B_up', nu * 2 + o) for nu, o in product(range(2), orbs)]
    fops_bath_dn = [('B_dn', nu * 2 + o) for nu, o in product(range(2), orbs)]

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
    def make_h_bath(cls, h, V, spin_blocks=True):
        mki = ref.make_mkind(cls.spins, cls.orbs, spin_blocks)
        h_bath = sum(h[s1, s2, o1, o2, nu]
                     * op.c_dag("B_" + spin1, nu * 2 + o1)
                     * op.c("B_" + spin2, nu * 2 + o2)
                     for (s1, spin1), (s2, spin2), o1, o2, nu
                     in product(enumerate(cls.spins), enumerate(cls.spins),
                                cls.orbs, cls.orbs, range(2)))
        h_bath += sum(V[s, o, nu] * (
                      op.c_dag(*mki(spin, o))
                      * op.c("B_" + spin, nu * 2 + o)
                      + op.c_dag("B_" + spin, nu * 2 + o)
                      * op.c(*mki(spin, o)))
                      for (s, spin), o, nu
                      in product(enumerate(cls.spins), cls.orbs, range(2)))
        return h_bath

    @classmethod
    def make_h_loc_an(cls, h_loc_an):
        return sum(h_loc_an[o1, o2] * op.c_dag('up', o1) * op.c_dag('dn', o2)
                   + np.conj(h_loc_an[o1, o2]) * op.c('dn', o2) * op.c('up', o1)
                   for o1, o2 in product(cls.orbs, cls.orbs))

    @classmethod
    def make_h_sc(cls, Delta):
        h_sc = sum(Delta[o1, o2, nu]
                   * op.c_dag('B_up', nu * 2 + o1)
                   * op.c_dag('B_dn', nu * 2 + o2)
                   for o1, o2, nu in product(cls.orbs, cls.orbs, range(2)))
        return h_sc + op.dagger(h_sc)

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
        if 'phi' in refs:
            assert_allclose(
                s.superconductive_phi * np.exp(1j * s.superconductive_phi_arg),
                refs['phi'],
                atol=1e-8
            )
        for gf in ('g_iw', 'g_an_iw', 'g_w', 'g_an_w',
                   'g0_iw', 'g0_an_iw', 'g0_w', 'g0_an_w',
                   'Sigma_iw', 'Sigma_an_iw', 'Sigma_w', 'Sigma_an_w',
                   'Delta_iw', 'Delta_an_iw', 'Delta_w', 'Delta_an_w'):
            if gf in refs:
                try:
                    assert_block_gfs_are_close(getattr(s, gf), refs[gf])
                except AssertionError as error:
                    print(f"Failed check for {gf}:")
                    raise error
        for axis, chan in product(['iw', 'w', 'tau'],
                                  ['spin', 'dens', 'pair', 'exct']):
            chi = f"chi_{chan}_{axis}"
            if chi in refs:
                try:
                    assert_gfs_are_close(getattr(s, chi), refs[chi])
                except AssertionError as error:
                    print(f"Failed check for {chi}:")
                    raise error

    @classmethod
    def find_basis_mat(cls, hvec, mat):
        for isym in range(hvec.shape[-1]):
            if (hvec[:, :, :, :, isym] == mat).all():
                return isym
        raise ValueError(f"Basis matrix {mat} not found")

    def test_zerotemp(self):
        h_loc = self.make_h_loc(mul.outer(s0, np.diag([-0.5, -0.6])))
        h_int = self.make_h_int(Uloc=np.array([0.1, 0.2]),
                                Ust=0.4,
                                Jh=0.2,
                                Jx=0.1,
                                Jp=0.15)

        fops_imp_up, fops_imp_dn = self.make_fops_imp()
        fops = fops_imp_up + fops_imp_dn + self.fops_bath_up + self.fops_bath_dn
        struct_params = {"spins": self.spins, "orbs": self.orbs, "fops": fops}

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

        self.assertEqual(solver.nspin, 1)
        self.assertEqual(solver.norb, 2)
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
        refs = ref.ref_results("zerotemp", h=h, beta=10000,
                               zerotemp=True, **struct_params, **solve_params)
        self.assert_all(solver, **refs)

    def test_nspin1(self):
        h_loc = self.make_h_loc(mul.outer(s0, np.diag([-0.5, -0.6])))
        h_int = self.make_h_int(Uloc=np.array([0.1, 0.2]),
                                Ust=0.4,
                                Jh=0.2,
                                Jx=0.1,
                                Jp=0.15)

        fops_imp_up, fops_imp_dn = self.make_fops_imp()
        fops = fops_imp_up + fops_imp_dn + self.fops_bath_up + self.fops_bath_dn
        struct_params = {"spins": self.spins, "orbs": self.orbs, "fops": fops}

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

        self.assertEqual(solver.nspin, 1)
        self.assertEqual(solver.norb, 2)
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
            "chi_spin": True,
            "chi_dens": True,
            "chi_pair": True,
            "chi_exct": True
        }
        solver.solve(**solve_params)

        ## Reference solution
        refs = ref.ref_results("nspin1_1", h=h, **struct_params, **solve_params)
        self.assert_all(solver, **refs)

        # Part II: update interaction parameters
        new_int_params = {'Uloc': np.array([0.15, 0.25]),
                          'Ust': 0.35,
                          'Jh': 0.1,
                          'Jx': 0.2,
                          'Jp': 0.0}
        self.change_int_params(solver.U, new_int_params)
        solver.comm.barrier()

        solve_params = {
            "beta": 80.0,
            "n_iw": 10,
            "energy_window": (-1.0, 1.0),
            "n_w": 40,
            "broadening": 0.03,
            "n_tau": 11,
            "chi_spin": True,
            "chi_dens": True,
            "chi_pair": True,
            "chi_exct": True
        }
        solver.solve(**solve_params)

        ## Reference solution
        h_int = self.make_h_int(**new_int_params)
        h = h_loc + h_int + h_bath
        refs = ref.ref_results("nspin1_2", h=h, **struct_params, **solve_params)
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
            "chi_spin": True,
            "chi_dens": True,
            "chi_pair": True,
            "chi_exct": True
        }
        solver.solve(**solve_params)

        ## Reference solution
        h_bath = self.make_h_bath(mul.outer(s0, h), mul.outer([1, 1], V))
        h = h_loc + h_int + h_bath
        refs = ref.ref_results("nspin1_3", h=h, **struct_params, **solve_params)
        self.assert_all(solver, **refs)

    def test_nspin2(self):
        h_loc = self.make_h_loc(mul.outer(np.diag([0.8, 1.2]),
                                          np.diag([-0.5, -0.6])))
        h_int = self.make_h_int(Uloc=np.array([0.1, 0.2]),
                                Ust=0.4,
                                Jh=0.2,
                                Jx=0.1,
                                Jp=0.15)

        fops_imp_up, fops_imp_dn = self.make_fops_imp()
        fops = fops_imp_up + fops_imp_dn + self.fops_bath_up + self.fops_bath_dn
        struct_params = {"spins": self.spins, "orbs": self.orbs, "fops": fops}

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

        self.assertEqual(solver.nspin, 2)
        self.assertEqual(solver.norb, 2)
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
            "chi_spin": True,
            "chi_dens": True,
            "chi_pair": True,
            "chi_exct": True
        }
        solver.solve(**solve_params)

        ## Reference solution
        refs = ref.ref_results("nspin2_1", h=h, **struct_params, **solve_params)
        self.assert_all(solver, **refs)

        # Part II: update interaction parameters
        new_int_params = {'Uloc': np.array([0.15, 0.25]),
                          'Ust': 0.35,
                          'Jh': 0.1,
                          'Jx': 0.2,
                          'Jp': 0.0}
        self.change_int_params(solver.U, new_int_params)
        solver.comm.barrier()

        solve_params = {
            "beta": 70.0,
            "n_iw": 10,
            "energy_window": (-1.0, 1.0),
            "n_w": 40,
            "broadening": 0.03,
            "n_tau": 11,
            "chi_spin": True,
            "chi_dens": True,
            "chi_pair": True,
            "chi_exct": True
        }
        solver.solve(**solve_params)

        ## Reference solution
        h_int = self.make_h_int(**new_int_params)
        h = h_loc + h_int + h_bath
        refs = ref.ref_results("nspin2_2", h=h, **struct_params, **solve_params)
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
            "chi_spin": True,
            "chi_dens": True,
            "chi_pair": True,
            "chi_exct": True
        }
        solver.solve(**solve_params)

        ## Reference solution
        h_bath = self.make_h_bath(mul.outer(sz, h), mul.outer([1, 0.9], V))
        h = h_loc + h_int + h_bath
        refs = ref.ref_results("nspin2_3", h=h, **struct_params, **solve_params)
        self.assert_all(solver, **refs)

    def test_nonsu2_hloc(self):
        h_loc = self.make_h_loc(mul.outer(np.array([[0.8, 0.2],
                                                    [0.2, 1.2]]),
                                          np.diag([-0.5, -0.6])),
                                spin_blocks=False)
        h_int = self.make_h_int(Uloc=np.array([0.1, 0.2]),
                                Ust=0.4,
                                Jh=0.2,
                                Jx=0.1,
                                Jp=0.15,
                                spin_blocks=False)

        fops_imp_up, fops_imp_dn = self.make_fops_imp(spin_blocks=False)
        fops = fops_imp_up + fops_imp_dn + self.fops_bath_up + self.fops_bath_dn
        struct_params = {"spins": self.spins, "orbs": self.orbs, "fops": fops}

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

        self.assertEqual(solver.nspin, 2)
        self.assertEqual(solver.norb, 2)
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
        refs = ref.ref_results("nonsu2_hloc_1", h=h, spin_blocks=False,
                               **struct_params, **solve_params)
        self.assert_all(solver, **refs)

        # Part II: update interaction parameters
        new_int_params = {'Uloc': np.array([0.15, 0.25]),
                          'Ust': 0.35,
                          'Jh': 0.1,
                          'Jx': 0.2,
                          'Jp': 0.0}
        self.change_int_params(solver.U, new_int_params)
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
        refs = ref.ref_results("nonsu2_hloc_2", h=h, spin_blocks=False,
                               **struct_params, **solve_params)
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
        refs = ref.ref_results("nonsu2_hloc_3", h=h, spin_blocks=False,
                               **struct_params, **solve_params)
        self.assert_all(solver, **refs)

    def test_nonsu2_bath(self):
        h_loc = self.make_h_loc(mul.outer(np.diag([0.8, 1.2]),
                                          np.diag([-0.5, -0.6])),
                                spin_blocks=False)
        h_int = self.make_h_int(Uloc=np.array([0.1, 0.2]),
                                Ust=0.4,
                                Jh=0.2,
                                Jx=0.1,
                                Jp=0.15,
                                spin_blocks=False)

        fops_imp_up, fops_imp_dn = self.make_fops_imp(spin_blocks=False)
        fops = fops_imp_up + fops_imp_dn + self.fops_bath_up + self.fops_bath_dn
        struct_params = {"spins": self.spins, "orbs": self.orbs, "fops": fops}

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

        self.assertEqual(solver.nspin, 2)
        self.assertEqual(solver.norb, 2)
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
        refs = ref.ref_results("nonsu2_bath_1", h=h, spin_blocks=False,
                               **struct_params, **solve_params)
        self.assert_all(solver, **refs)

        # Part II: update interaction parameters
        new_int_params = {'Uloc': np.array([0.15, 0.25]),
                          'Ust': 0.35,
                          'Jh': 0.1,
                          'Jx': 0.2,
                          'Jp': 0.0}
        self.change_int_params(solver.U, new_int_params)
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
        refs = ref.ref_results("nonsu2_bath_2", h=h, spin_blocks=False,
                               **struct_params, **solve_params)
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
        refs = ref.ref_results("nonsu2_bath_3", h=h, spin_blocks=False,
                               **struct_params, **solve_params)
        self.assert_all(solver, **refs)

    def test_h_loc_an(self):
        h_loc = self.make_h_loc(mul.outer(s0, np.diag([-0.5, -0.6])))
        h_int = self.make_h_int(Uloc=np.array([0.1, 0.2]),
                                Ust=0.4,
                                Jh=0.2,
                                Jx=0.1,
                                Jp=0.15)

        fops_imp_up, fops_imp_dn = self.make_fops_imp()
        fops = fops_imp_up + fops_imp_dn + self.fops_bath_up + self.fops_bath_dn
        struct_params = {"spins": self.spins, "orbs": self.orbs, "fops": fops}

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

        self.assertEqual(solver.nspin, 1)
        self.assertEqual(solver.norb, 2)
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
        refs = ref.ref_results("h_loc_an_1", h=h, superc=True,
                               **struct_params, **solve_params)
        self.assert_all(solver, **refs)

        # Part II: update interaction parameters and pairing field
        new_int_params = {'Uloc': np.array([0.15, 0.25]),
                          'Ust': 0.35,
                          'Jh': 0.1,
                          'Jx': 0.2,
                          'Jp': 0.0}
        self.change_int_params(solver.U, new_int_params)
        h_loc_an_mat = np.array([[0.2, 0.5j],
                                 [0.5j, 0.3]])
        solver.hloc_an[0, 0, :, :] = h_loc_an_mat
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
        refs = ref.ref_results("h_loc_an_2", h=h, superc=True,
                               **struct_params, **solve_params)
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
        refs = ref.ref_results("h_loc_an_3", h=h, superc=True,
                               **struct_params, **solve_params)
        self.assert_all(solver, **refs)

    def test_superc(self):
        h_loc = self.make_h_loc(mul.outer(s0, np.diag([-0.5, -0.6])))
        h_int = self.make_h_int(Uloc=np.array([0.1, 0.2]),
                                Ust=0.4,
                                Jh=0.2,
                                Jx=0.1,
                                Jp=0.15)

        fops_imp_up, fops_imp_dn = self.make_fops_imp()
        fops = fops_imp_up + fops_imp_dn + self.fops_bath_up + self.fops_bath_dn
        struct_params = {"spins": self.spins, "orbs": self.orbs, "fops": fops}

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

        self.assertEqual(solver.nspin, 1)
        self.assertEqual(solver.norb, 2)
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
        refs = ref.ref_results("superc_1", h=h, superc=True,
                               **struct_params, **solve_params)
        self.assert_all(solver, **refs)

        # Part II: update interaction parameters
        new_int_params = {'Uloc': np.array([0.15, 0.25]),
                          'Ust': 0.35,
                          'Jh': 0.1,
                          'Jx': 0.2,
                          'Jp': 0.0}
        self.change_int_params(solver.U, new_int_params)
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
        refs = ref.ref_results("superc_2", h=h, superc=True,
                               **struct_params, **solve_params)
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
        refs = ref.ref_results("superc_3", h=h, superc=True,
                               **struct_params, **solve_params)
        self.assert_all(solver, **refs)

    def tearDown(self):
        # Make sure EDIpackSolver.__del__() is called
        gc.collect()


if __name__ == '__main__':
    unittest.main()

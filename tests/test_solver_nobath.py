from itertools import product

import numpy as np
from numpy.testing import assert_allclose
from numpy import multiply as mul

from triqs.utility.comparison_tests import (assert_gfs_are_close,
                                            assert_block_gfs_are_close)

from edipack2triqs import EDMode
from edipack2triqs.solver import EDIpackSolver

from . import reference as ref
from .test_solver import TestSolver


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


class TestEDIpackSolverNoBath(TestSolver):

    @classmethod
    def assert_all(cls, s, **refs):
        assert_allclose(s.densities, refs['densities'], atol=1e-7)
        assert_allclose(s.double_occ, refs['double_occ'], atol=1e-7)
        assert_allclose(s.magnetization[:, 0], refs['magn_x'], atol=1e-7)
        assert_allclose(s.magnetization[:, 1], refs['magn_y'], atol=1e-7)
        assert_allclose(s.magnetization[:, 2], refs['magn_z'], atol=1e-7)
        assert_block_gfs_are_close(s.g_w, refs['g_w'])
        for gf in ('g_iw', 'Sigma_iw', 'g_w', 'Sigma_w'):
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

    def test_zerotemp(self):
        h_loc = self.make_h_loc(mul.outer(s0, np.diag([-0.5, -0.6])))
        h_int = self.make_h_int(
            Uloc=[0.1, 0.2], Ust=0.4, Jh=0.2, Jx=0.1, Jp=0.15
        )

        fops_imp_up, fops_imp_dn = self.make_fops_imp()
        fops = fops_imp_up + fops_imp_dn
        struct_params = {"spins": self.spins, "orbs": self.orbs, "fops": fops}

        h = h_loc + h_int
        solver = EDIpackSolver(
            h,
            fops_imp_up, fops_imp_dn,
            zerotemp=True,
            lanc_nstates_total=17,
            verbose=0,
        )

        self.assertEqual(solver.nspin, 1)
        self.assertEqual(solver.norb, 2)
        self.assertIsNone(solver.bath)

        solve_params = {
            "n_iw": 10,
            "energy_window": (-2.0, 2.0),
            "n_w": 60,
            "broadening": 0.05
        }
        solver.solve(**solve_params)

        ## Reference solution
        refs = ref.ref_results("zerotemp", h=h, beta=10000, zerotemp=True,
                               **struct_params, **solve_params)
        self.assert_all(solver, **refs)

    def test_nspin1(self):
        h_loc = self.make_h_loc(mul.outer(s0, np.diag([-0.5, -0.6])))
        h_int = self.make_h_int(
            Uloc=np.array([0.1, 0.2]), Ust=0.4, Jh=0.2, Jx=0.1, Jp=0.15
        )

        fops_imp_up, fops_imp_dn = self.make_fops_imp()
        fops = fops_imp_up + fops_imp_dn
        struct_params = {"spins": self.spins, "orbs": self.orbs, "fops": fops}

        h = h_loc + h_int
        solver = EDIpackSolver(
            h,
            fops_imp_up, fops_imp_dn,
            lanc_nstates_total=8,
            verbose=0
        )

        self.assertEqual(solver.nspin, 1)
        self.assertEqual(solver.norb, 2)
        self.assertIsNone(solver.bath)

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
        refs = ref.ref_results("nspin1_1", h=h, **struct_params, **solve_params)
        self.assert_all(solver, **refs)

        # Part II: update interaction parameters
        new_int_params = {
            'Uloc': [0.15, 0.25], 'Ust': 0.35, 'Jh': 0.1, 'Jx': 0.2, 'Jp': 0.0
        }
        self.change_int_params(solver.U, new_int_params)
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
        h = h_loc + h_int
        refs = ref.ref_results("nspin1_2", h=h, **struct_params, **solve_params)
        self.assert_all(solver, **refs)

    def test_nspin2(self):
        h_loc = self.make_h_loc(mul.outer(np.diag([0.8, 1.2]),
                                          np.diag([-0.5, -0.6])))
        h_int = self.make_h_int(
            Uloc=[0.1, 0.2], Ust=0.4, Jh=0.2, Jx=0.1, Jp=0.15
        )

        fops_imp_up, fops_imp_dn = self.make_fops_imp()
        fops = fops_imp_up + fops_imp_dn
        struct_params = {"spins": self.spins, "orbs": self.orbs, "fops": fops}

        h = h_loc + h_int
        solver = EDIpackSolver(
            h,
            fops_imp_up, fops_imp_dn,
            lanc_nstates_total=8,
            verbose=0
        )

        self.assertEqual(solver.nspin, 2)
        self.assertEqual(solver.norb, 2)
        self.assertIsNone(solver.bath)

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
        refs = ref.ref_results("nspin2_1", h=h, **struct_params, **solve_params)
        self.assert_all(solver, **refs)

        # Part II: update interaction parameters
        new_int_params = {
            'Uloc': [0.15, 0.25], 'Ust': 0.35, 'Jh': 0.1, 'Jx': 0.2, 'Jp': 0.0
        }
        self.change_int_params(solver.U, new_int_params)
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
        h = h_loc + h_int
        refs = ref.ref_results("nspin2_2", h=h, **struct_params, **solve_params)
        self.assert_all(solver, **refs)

    def test_nonsu2(self):
        h_loc = self.make_h_loc(mul.outer(np.array([[0.8, 0.2],
                                                    [0.2, 1.2]]),
                                          np.diag([-0.5, -0.6])),
                                spin_blocks=False)
        h_int = self.make_h_int(
            Uloc=[0.1, 0.2], Ust=0.4, Jh=0.2, Jx=0.1, Jp=0.15, spin_blocks=False
        )

        fops_imp_up, fops_imp_dn = self.make_fops_imp(False)
        fops = fops_imp_up + fops_imp_dn
        struct_params = {"spins": self.spins, "orbs": self.orbs, "fops": fops}

        h = h_loc + h_int
        solver = EDIpackSolver(
            h,
            fops_imp_up, fops_imp_dn,
            lanc_nstates_total=8,
            verbose=0,
        )

        self.assertEqual(solver.nspin, 2)
        self.assertEqual(solver.norb, 2)
        self.assertIsNone(solver.bath)

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
        refs = ref.ref_results("nonsu2_hloc_1", h=h, spin_blocks=False,
                               **struct_params, **solve_params)
        self.assert_all(solver, **refs)

        # Part II: update interaction parameters
        new_int_params = {
            'Uloc': [0.15, 0.25], 'Ust': 0.35, 'Jh': 0.1, 'Jx': 0.2, 'Jp': 0.0
        }
        self.change_int_params(solver.U, new_int_params)
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
        h = h_loc + h_int
        refs = ref.ref_results("nonsu2_hloc_2", h=h, spin_blocks=False,
                               **struct_params, **solve_params)
        self.assert_all(solver, **refs)

    def test_h_loc_an(self):
        h_loc = self.make_h_loc(mul.outer(s0, np.array([[-0.5, 0.1],
                                                        [0.1, -0.6]])))
        h_int = self.make_h_int(
            Uloc=[0.1, 0.2], Ust=0.4, Jh=0.2, Jx=0.1, Jp=0.15
        )

        fops_imp_up, fops_imp_dn = self.make_fops_imp()
        fops = fops_imp_up + fops_imp_dn
        struct_params = {"spins": self.spins, "orbs": self.orbs, "fops": fops}

        h_loc_an_mat = np.array([[0.1, 0.6j],
                                 [0.6j, 0.15]])
        h_loc_an = self.make_h_loc_an(h_loc_an_mat)

        h = h_loc + h_loc_an + h_int
        solver = EDIpackSolver(
            h,
            fops_imp_up, fops_imp_dn,
            lanc_nstates_sector=4,
            lanc_nstates_total=10,
            verbose=0
        )

        self.assertEqual(solver.h_params.ed_mode, EDMode.SUPERC)
        self.assertEqual(solver.nspin, 1)
        self.assertEqual(solver.norb, 2)
        self.assertIsNone(solver.bath)

        # Part I: Initial solve()
        solve_params = {
            "beta": 60.0,
            "n_iw": 10,
            "energy_window": (-2.0, 2.0),
            "n_w": 60,
            "broadening": 0.05
        }
        solver.solve(**solve_params)

        # Reference solution
        refs = ref.ref_results("h_loc_an_1", h=h, superc=True,
                               **struct_params, **solve_params)
        self.assert_all(solver, **refs)

        # Part II: update interaction parameters and pairing field
        new_int_params = {
            'Uloc': [0.15, 0.25], 'Ust': 0.35, 'Jh': 0.1, 'Jx': 0.2, 'Jp': 0.0
        }
        self.change_int_params(solver.U, new_int_params)
        h_loc_an_mat = np.array([[0.2, 0.4j],
                                 [0.4j, 0.3]])
        solver.hloc_an[0, 0, :, :] = h_loc_an_mat
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
        h = h_loc + h_loc_an + h_int
        refs = ref.ref_results("h_loc_an_2", h=h, superc=True,
                               **struct_params, **solve_params)
        self.assert_all(solver, **refs)


if __name__ == '__main__':
    # TODO: Test direct invocation of this file
    import unittest
    unittest.main()

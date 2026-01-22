import numpy as np
from numpy import multiply as mul
from numpy.testing import assert_equal

from edipack2triqs import EDMode
from edipack2triqs.solver import EDIpackSolver

from .test_solver import TestSolver


s0 = np.eye(2)
sx = np.array([[0, 1], [1, 0]])
sz = np.array([[1, 0], [0, -1]])


class TestEDIpackSolverNoBath(TestSolver):

    @classmethod
    def assert_all(cls, s, **refs):
        cls.assert_static_obs(s, 1e-7, **refs)
        cls.assert_gfs(s, has_bath=False, **refs)
        cls.assert_chi(s, **refs)

    def test_zerotemp(self):
        h_loc_mat = mul.outer(s0, np.diag([-0.5, -0.6]))
        h_loc = self.make_h_loc(h_loc_mat)
        h_int = self.make_h_int(
            Uloc=[0.1, 0.2], Ust=0.4, Jh=0.2, Jx=0.1, Jp=0.15
        )

        fops_imp_up, fops_imp_dn = self.make_fops_imp()
        fops = fops_imp_up + fops_imp_dn

        h = h_loc + h_int
        solver = EDIpackSolver(
            h,
            fops_imp_up, fops_imp_dn,
            zerotemp=True,
            lanc_nstates_total=17,
            verbose=0,
        )

        self.assertEqual(solver.h_params.ed_mode, EDMode.NORMAL)
        self.assertEqual(solver.nspin, 1)
        self.assertEqual(solver.norb, self.norb)
        self.assertEqual(solver.hloc, h_loc)
        assert_equal(solver.hloc_mat, h_loc_mat[:1, :1, ...])
        assert_equal(solver.hloc_an_mat, np.zeros((1, 1, self.norb, self.norb)))
        self.assertIsNone(solver.bath)

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
        fops = fops_imp_up + fops_imp_dn

        h = h_loc + h_int
        solver = EDIpackSolver(
            h,
            fops_imp_up, fops_imp_dn,
            lanc_nstates_total=8,
            verbose=0
        )

        self.assertEqual(solver.h_params.ed_mode, EDMode.NORMAL)
        self.assertEqual(solver.nspin, 1)
        self.assertEqual(solver.norb, self.norb)
        self.assertEqual(solver.hloc, h_loc)
        assert_equal(solver.hloc_mat, h_loc_mat[:1, :1, ...])
        assert_equal(solver.hloc_an_mat, np.zeros((1, 1, self.norb, self.norb)))
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
        h = h_loc + h_int
        refs = self.ref_results("nspin1_2", h=h, fops=fops, **solve_params)
        self.assert_all(solver, **refs)

    def test_nspin2(self):
        h_loc_mat = mul.outer(np.diag([0.8, 1.2]), np.diag([-0.5, -0.6]))
        h_loc = self.make_h_loc(h_loc_mat)
        h_int = self.make_h_int(
            Uloc=[0.1, 0.2], Ust=0.4, Jh=0.2, Jx=0.1, Jp=0.15
        )

        fops_imp_up, fops_imp_dn = self.make_fops_imp()
        fops = fops_imp_up + fops_imp_dn

        h = h_loc + h_int
        solver = EDIpackSolver(
            h,
            fops_imp_up, fops_imp_dn,
            lanc_nstates_total=8,
            verbose=0
        )

        self.assertEqual(solver.h_params.ed_mode, EDMode.NORMAL)
        self.assertEqual(solver.nspin, 2)
        self.assertEqual(solver.norb, self.norb)
        self.assertEqual(solver.hloc, h_loc)
        assert_equal(solver.hloc_mat, h_loc_mat)
        assert_equal(solver.hloc_an_mat, np.zeros((1, 1, self.norb, self.norb)))
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
        refs = self.ref_results("nspin2_2", h=h, fops=fops, **solve_params)
        self.assert_all(solver, **refs)

    def test_nonsu2(self):
        h_loc_mat = mul.outer(np.array([[0.8, 0.2],
                                        [0.2, 1.2]]),
                              np.diag([-0.5, -0.6]))
        h_loc = self.make_h_loc(h_loc_mat, spin_blocks=False)
        h_int = self.make_h_int(
            Uloc=[0.1, 0.2], Ust=0.4, Jh=0.2, Jx=0.1, Jp=0.15, spin_blocks=False
        )

        fops_imp_up, fops_imp_dn = self.make_fops_imp(False)
        fops = fops_imp_up + fops_imp_dn

        h = h_loc + h_int
        solver = EDIpackSolver(
            h,
            fops_imp_up, fops_imp_dn,
            lanc_nstates_total=8,
            verbose=0,
        )

        self.assertEqual(solver.h_params.ed_mode, EDMode.NONSU2)
        self.assertEqual(solver.nspin, 2)
        self.assertEqual(solver.norb, self.norb)
        self.assertEqual(solver.hloc, h_loc)
        assert_equal(solver.hloc_mat, h_loc_mat)
        assert_equal(solver.hloc_an_mat, np.zeros((1, 1, self.norb, self.norb)))
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
            "energy_window": (-1.5, 1.5),
            "n_w": 40,
            "broadening": 0.03
        }
        solver.solve(**solve_params)

        ## Reference solution
        h_int = self.make_h_int(**new_int_params, spin_blocks=False)
        h = h_loc + h_int
        refs = self.ref_results("nonsu2_hloc_2", h=h, fops=fops,
                                spin_blocks=False, **solve_params)
        self.assert_all(solver, **refs)

    def test_h_loc_an(self):
        h_loc_mat = mul.outer(s0, np.array([[-0.5, 0.1],
                                            [0.1, -0.6]]))
        h_loc = self.make_h_loc(h_loc_mat)
        h_int = self.make_h_int(
            Uloc=[0.1, 0.2], Ust=0.4, Jh=0.2, Jx=0.1, Jp=0.15
        )

        fops_imp_up, fops_imp_dn = self.make_fops_imp()
        fops = fops_imp_up + fops_imp_dn

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
        self.assertEqual(solver.norb, self.norb)
        self.assertEqual(solver.hloc, h_loc + h_loc_an)
        assert_equal(solver.hloc_mat, h_loc_mat[:1, :1, ...])
        assert_equal(solver.hloc_an_mat,
                     h_loc_an_mat.reshape((1, 1, self.norb, self.norb)))
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
        refs = self.ref_results("h_loc_an_1", h=h, fops=fops, superc=True,
                                **solve_params)
        self.assert_all(solver, **refs)

        # Part II: update interaction parameters and pairing field
        new_int_params = {
            'Uloc': [0.15, 0.25], 'Ust': 0.35, 'Jh': 0.1, 'Jx': 0.2, 'Jp': 0.0
        }
        self.change_int_params(solver.U, **new_int_params)
        h_loc_an_mat = np.array([[0.2, 0.4j],
                                 [0.4j, 0.3]])
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
        h = h_loc + h_loc_an + h_int
        refs = self.ref_results("h_loc_an_2", h=h, fops=fops, superc=True,
                                **solve_params)
        self.assert_all(solver, **refs)

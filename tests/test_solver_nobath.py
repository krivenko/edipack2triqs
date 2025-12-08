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


class TestEDIpackSolverNoBath(unittest.TestCase):

    spins = ('up', 'dn')
    orbs = range(2)

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
    def make_H_pair_field(cls, pair_field):
        return sum(pair_field[o] * (op.c_dag('up', o) * op.c_dag('dn', o)
                                    + op.c('dn', o) * op.c('up', o))
                   for o in cls.orbs)

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
        h_int = self.make_h_int(Uloc=np.array([0.1, 0.2]),
                                Ust=0.4,
                                Jh=0.2,
                                Jx=0.1,
                                Jp=0.15)

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
        h_int = self.make_h_int(Uloc=np.array([0.1, 0.2]),
                                Ust=0.4,
                                Jh=0.2,
                                Jx=0.1,
                                Jp=0.15)

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
            "beta": 70.0,
            "n_iw": 20,
            "energy_window": (-1.5, 1.5),
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
        h = h_loc + h_int
        refs = ref.ref_results("nspin1_2", h=h, **struct_params, **solve_params)
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
            "n_iw": 20,
            "energy_window": (-1.5, 1.5),
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
        h = h_loc + h_int
        refs = ref.ref_results("nspin2_2", h=h, **struct_params, **solve_params)
        self.assert_all(solver, **refs)

    def test_nonsu2(self):
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

    def test_pair_field(self):
        h_loc = self.make_h_loc(mul.outer(s0, np.diag([-0.5, -0.6])))
        h_int = self.make_h_int(Uloc=np.array([0.1, 0.2]),
                                Ust=0.4,
                                Jh=0.2,
                                Jx=0.1,
                                Jp=0.15)

        fops_imp_up, fops_imp_dn = self.make_fops_imp()
        fops = fops_imp_up + fops_imp_dn
        struct_params = {"spins": self.spins, "orbs": self.orbs, "fops": fops}

        pair_field = np.array([0.1, 0.2])
        h_pair_field = self.make_H_pair_field(pair_field)

        h = h_loc + h_pair_field + h_int
        solver = EDIpackSolver(
            h,
            fops_imp_up, fops_imp_dn,
            lanc_nstates_sector=4,
            lanc_nstates_total=10,
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
            "broadening": 0.05
        }
        solver.solve(**solve_params)

        # Reference solution
        refs = ref.ref_results("pair_field_1", h=h, superc=True,
                               **struct_params, **solve_params)
        self.assert_all(solver, **refs)

        # Part II: update interaction parameters and pairing field
        new_int_params = {'Uloc': np.array([0.15, 0.25]),
                          'Ust': 0.35,
                          'Jh': 0.1,
                          'Jx': 0.2,
                          'Jp': 0.0}
        self.change_int_params(solver.U, new_int_params)
        pair_field = np.array([0.2, 0.3])
        solver.pair_field[:] = pair_field
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
        h_pair_field = self.make_H_pair_field(pair_field)
        h = h_loc + h_int + h_pair_field
        refs = ref.ref_results("pair_field_2", h=h, superc=True,
                               **struct_params, **solve_params)
        self.assert_all(solver, **refs)

    def tearDown(self):
        # Make sure EDIpackSolver.__del__() is called
        gc.collect()


if __name__ == '__main__':
    unittest.main()

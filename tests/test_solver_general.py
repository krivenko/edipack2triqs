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
from triqs.gf.tools import dyson
from triqs.utility.comparison_tests import assert_block_gfs_are_close

from edipack2triqs.solver import EDIpackSolver
from edipack2triqs.util import non_int_part


s0 = np.eye(2)
sx = np.array([[0, 1], [1, 0]])
sz = np.array([[1, 0], [0, -1]])


class TestEDIpackSolverBathGeneral(unittest.TestCase):

    spins = ('up', 'dn')
    orbs = range(2)

    fops_bath_up = [('B_up', nu * 2 + o) for nu, o in product(range(2), orbs)]
    fops_bath_dn = [('B_dn', nu * 2 + o) for nu, o in product(range(2), orbs)]

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
    def make_h_bath(cls, h, V, spin_blocks):
        mki = cls.make_mkind(spin_blocks)
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
    def make_h_sc(cls, Delta):
        h_sc = sum(Delta[o1, o2, nu]
                   * op.c_dag('B_up', nu * 2 + o1)
                   * op.c_dag('B_dn', nu * 2 + o2)
                   for o1, o2, nu in product(cls.orbs, cls.orbs, range(2)))
        return h_sc + op.dagger(h_sc)

    @classmethod
    def make_ref_results(
        cls, h, fops, beta, n_iw, energy_window, n_w, eta, spin_blocks, superc
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
        g_iw = atomic_g_iw(ad, beta, gf_struct, n_iw)
        g_w = atomic_g_w(ad, beta, gf_struct, energy_window, n_w, eta)

        def avg(ops):
            return np.array([trace_rho_op(rho, ops[o], ad) for o in cls.orbs])

        if not superc:
            h0 = non_int_part(h)
            ad0 = AtomDiag(h0, fops)
            g0_iw = atomic_g_iw(ad0, beta, gf_struct, n_iw)
            g0_w = atomic_g_w(ad0, beta, gf_struct, energy_window, n_w, eta)
            Sigma_iw = dyson(G0_iw=g0_iw, G_iw=g_iw)
            Sigma_w = dyson(G0_iw=g0_w, G_iw=g_w)
        else:
            Sigma_iw = None
            Sigma_w = None

        return {'densities': avg(N),
                'double_occ': avg(D),
                'magn_x': avg(S_x),
                'magn_y': 1j * avg(S_y),
                'magn_z': avg(S_z),
                'g_iw': g_iw,
                'g_w': g_w,
                'Sigma_iw': Sigma_iw,
                'Sigma_w': Sigma_w}

    @classmethod
    def assert_all(cls, s, **refs):
        assert_allclose(s.densities, refs['densities'], atol=1e-8)
        assert_allclose(s.double_occ, refs['double_occ'], atol=1e-8)
        assert_allclose(s.magnetization(comp='x'), refs['magn_x'], atol=1e-8)
        assert_allclose(s.magnetization(comp='y'), refs['magn_y'], atol=1e-8)
        assert_allclose(s.magnetization(comp='z'), refs['magn_z'], atol=1e-8)
        assert_block_gfs_are_close(s.g_iw, refs['g_iw'])
        assert_block_gfs_are_close(s.g_w, refs['g_w'])
        if not refs['Sigma_iw'] is None:
            assert_block_gfs_are_close(s.Sigma_iw, refs['Sigma_iw'])
        if not refs['Sigma_w'] is None:
            assert_block_gfs_are_close(s.Sigma_w, refs['Sigma_w'])

    @classmethod
    def find_basis_mat(cls, hvec, mat):
        for isym in range(hvec.shape[-1]):
            if (hvec[:, :, :, :, isym] == mat).all():
                return isym
        raise ValueError(f"Basis matrix {mat} not found")

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

        h = np.moveaxis(np.array([[[0.5, 0.1],
                                   [0.1, 0.6]],
                                  [[-0.5, 0.0],
                                   [0.0, -0.6]]]), 0, 2)
        V = np.array([[0.2, -0.2],
                      [0.4, -0.4]])
        h_bath = self.make_h_bath(mul.outer(s0, h), mul.outer([1, 1], V), True)

        h = h_loc + h_int + h_bath
        solver = EDIpackSolver(
            h,
            fops_imp_up, fops_imp_dn,
            self.fops_bath_up, self.fops_bath_dn,
            verbose=0
        )

        self.assertEqual(solver.nspin, 1)
        self.assertEqual(solver.norb, 2)
        self.assertEqual(solver.bath.name, "general")
        self.assertEqual(solver.bath.nbath, 2)

        # Part I: Initial solve()
        beta = 100.0
        n_iw = 100
        energy_window = (-1.0, 1.0)
        n_w = 400
        broadening = 0.05
        solver.solve(beta=beta,
                     n_iw=n_iw,
                     energy_window=energy_window,
                     n_w=n_w,
                     broadening=broadening)

        ## Reference solution
        refs = self.make_ref_results(h, fops, beta,
                                     n_iw,
                                     energy_window, n_w, broadening,
                                     True,
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
        n_iw = 100
        energy_window = (-1.0, 1.0)
        n_w = 400
        broadening = 0.03
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
                                     True,
                                     False)
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

        beta = 100.0
        n_iw = 100
        energy_window = (-1.0, 1.0)
        n_w = 400
        broadening = 0.02
        solver.solve(beta=beta,
                     n_iw=n_iw,
                     energy_window=energy_window,
                     n_w=n_w,
                     broadening=broadening)

        ## Reference solution
        h_bath = self.make_h_bath(mul.outer(s0, h), mul.outer([1, 1], V), True)
        h = h_loc + h_int + h_bath
        refs = self.make_ref_results(h, fops, beta,
                                     n_iw,
                                     energy_window, n_w, broadening,
                                     True,
                                     False)
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

        h = np.moveaxis(np.array([[[0.5, 0.1],
                                   [0.1, 0.6]],
                                  [[-0.5, 0.0],
                                   [0.0, -0.6]]]), 0, 2)
        V = np.array([[0.2, -0.2],
                      [0.4, -0.4]])
        h_bath = self.make_h_bath(mul.outer(sz, h),
                                  mul.outer([1, 0.9], V),
                                  True)

        h = h_loc + h_int + h_bath
        solver = EDIpackSolver(
            h,
            fops_imp_up, fops_imp_dn,
            self.fops_bath_up, self.fops_bath_dn,
            verbose=0
        )

        self.assertEqual(solver.nspin, 2)
        self.assertEqual(solver.norb, 2)
        self.assertEqual(solver.bath.name, "general")
        self.assertEqual(solver.bath.nbath, 2)

        # Part I: Initial solve()
        beta = 100.0
        n_iw = 100
        energy_window = (-1.0, 1.0)
        n_w = 400
        broadening = 0.05
        solver.solve(beta=beta,
                     n_iw=n_iw,
                     energy_window=energy_window,
                     n_w=n_w,
                     broadening=broadening)

        ## Reference solution
        refs = self.make_ref_results(h, fops, beta,
                                     n_iw,
                                     energy_window, n_w, broadening,
                                     True,
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
        n_iw = 100
        energy_window = (-1.0, 1.0)
        n_w = 400
        broadening = 0.03
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
                                     True,
                                     False)
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

        beta = 100.0
        n_iw = 100
        energy_window = (-1.0, 1.0)
        n_w = 400
        broadening = 0.02
        solver.solve(beta=beta,
                     n_iw=n_iw,
                     energy_window=energy_window,
                     n_w=n_w,
                     broadening=broadening)

        ## Reference solution
        h_bath = self.make_h_bath(mul.outer(sz, h),
                                  mul.outer([1, 0.9], V), True)
        h = h_loc + h_int + h_bath
        refs = self.make_ref_results(h, fops, beta,
                                     n_iw,
                                     energy_window, n_w, broadening,
                                     True,
                                     False)
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

        h = np.moveaxis(np.array([[[0.5, 0.1 - 0.1j],
                                   [0.1 + 0.1j, 0.6]],
                                  [[-0.5, 0.0],
                                   [0.0, -0.6]]]), 0, 2)
        V = np.array([[0.1, 0.2],
                      [0.3, 0.4]])
        h_bath = self.make_h_bath(mul.outer(sz, h),
                                  mul.outer([1, 0.9], V),
                                  False)

        h = h_loc + h_int + h_bath
        solver = EDIpackSolver(
            h,
            fops_imp_up, fops_imp_dn,
            self.fops_bath_up, self.fops_bath_dn,
            verbose=0
        )

        self.assertEqual(solver.nspin, 2)
        self.assertEqual(solver.norb, 2)
        self.assertEqual(solver.bath.name, "general")
        self.assertEqual(solver.bath.nbath, 2)

        # Part I: Initial solve()
        beta = 100.0
        n_iw = 100
        energy_window = (-1.0, 1.0)
        n_w = 400
        broadening = 0.05
        solver.solve(beta=beta,
                     n_iw=n_iw,
                     energy_window=energy_window,
                     n_w=n_w,
                     broadening=broadening)

        # Reference solution
        refs = self.make_ref_results(h, fops, beta,
                                     n_iw,
                                     energy_window, n_w, broadening,
                                     False,
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
        broadening = 0.03
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
                                     False,
                                     False)
        self.assert_all(solver, **refs)

        # Part III: Updated bath parameters
        h = np.moveaxis(np.array([[[0.5, 0.2 - 0.1j],
                                   [0.2 + 0.1j, 0.6]],
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

        beta = 120.0
        n_iw = 200
        energy_window = (-1.0, 1.0)
        n_w = 400
        broadening = 0.03
        solver.solve(beta=beta,
                     n_iw=n_iw,
                     energy_window=energy_window,
                     n_w=n_w,
                     broadening=broadening)

        ## Reference solution
        h_bath = self.make_h_bath(mul.outer(sz, h),
                                  mul.outer([1, 0.9], V),
                                  False)
        h = h_loc + h_int + h_bath
        refs = self.make_ref_results(h, fops, beta,
                                     n_iw,
                                     energy_window, n_w, broadening,
                                     False,
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

        h = np.moveaxis(np.array([[[0.5, 0.1 - 0.1j],
                                   [0.1 + 0.1j, 0.6]],
                                  [[-0.5, 0.0],
                                   [0.0, -0.6]]]), 0, 2)
        V = np.array([[0.1, 0.2],
                      [0.3, 0.4]])
        h_bath = self.make_h_bath(mul.outer(sz + 0.2 * sx, h),
                                  mul.outer([1, 0.9], V),
                                  False)

        h = h_loc + h_int + h_bath
        solver = EDIpackSolver(
            h,
            fops_imp_up, fops_imp_dn,
            self.fops_bath_up, self.fops_bath_dn,
            verbose=0
        )

        self.assertEqual(solver.nspin, 2)
        self.assertEqual(solver.norb, 2)
        self.assertEqual(solver.bath.name, "general")
        self.assertEqual(solver.bath.nbath, 2)

        # Part I: Initial solve()
        beta = 100.0
        n_iw = 100
        energy_window = (-1.0, 1.0)
        n_w = 400
        broadening = 0.05
        solver.solve(beta=beta,
                     n_iw=n_iw,
                     energy_window=energy_window,
                     n_w=n_w,
                     broadening=broadening)

        ## Reference solution
        refs = self.make_ref_results(h, fops, beta,
                                     n_iw,
                                     energy_window, n_w, broadening,
                                     False,
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
        broadening = 0.03
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
                                     False,
                                     False)
        self.assert_all(solver, **refs)

        # Part III: Updated bath parameters
        h = np.moveaxis(np.array([[[0.5, 0.2 - 0.1j],
                                   [0.2 + 0.1j, 0.6]],
                                  [[-0.5, 0.0],
                                   [0.0, -0.6]]]), 0, 2)
        V = np.array([[0.1, 0.2],
                      [0.5, 0.4]])

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

        beta = 120.0
        n_iw = 200
        energy_window = (-1.0, 1.0)
        n_w = 400
        broadening = 0.03
        solver.solve(beta=beta,
                     n_iw=n_iw,
                     energy_window=energy_window,
                     n_w=n_w,
                     broadening=broadening)

        ## Reference solution
        h_bath = self.make_h_bath(mul.outer(sz + 0.2 * sx, h),
                                  mul.outer([1, 0.9], V),
                                  False)
        h = h_loc + h_int + h_bath
        refs = self.make_ref_results(h, fops, beta,
                                     n_iw,
                                     energy_window, n_w, broadening,
                                     False,
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

        h = np.moveaxis(np.array([[[0.5, 0.1],
                                   [0.1, 0.6]],
                                  [[-0.5, 0.0],
                                   [0.0, -0.6]]]), 0, 2)
        V = np.array([[0.2, -0.2],
                      [0.4, -0.4]])
        h_bath = self.make_h_bath(mul.outer(s0, h), mul.outer([1, 1], V), True)

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
            verbose=0
        )

        self.assertEqual(solver.nspin, 1)
        self.assertEqual(solver.norb, 2)
        self.assertEqual(solver.bath.name, "general")
        self.assertEqual(solver.bath.nbath, 2)

        # Part I: Initial solve()
        beta = 100.0
        n_iw = 100
        energy_window = (-1.0, 1.0)
        n_w = 400
        broadening = 0.05
        solver.solve(beta=beta,
                     n_iw=n_iw,
                     energy_window=energy_window,
                     n_w=n_w,
                     broadening=broadening)

        ## Reference solution
        refs = self.make_ref_results(h, fops, beta,
                                     n_iw,
                                     energy_window, n_w, broadening,
                                     True,
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
        n_iw = 100
        energy_window = (-1.0, 1.0)
        n_w = 400
        broadening = 0.03
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
                                     True,
                                     True)
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

        beta = 100.0
        n_iw = 100
        energy_window = (-1.0, 1.0)
        n_w = 400
        broadening = 0.02
        solver.solve(beta=beta,
                     n_iw=n_iw,
                     energy_window=energy_window,
                     n_w=n_w,
                     broadening=broadening)

        ## Reference solution
        h_bath = self.make_h_bath(mul.outer(s0, h), mul.outer([1, 1], V), True)
        h_sc = self.make_h_sc(Delta)
        h = h_loc + h_int + h_bath + h_sc
        refs = self.make_ref_results(h, fops, beta,
                                     n_iw,
                                     energy_window, n_w, broadening,
                                     True,
                                     True)
        self.assert_all(solver, **refs)

        solver.g_an_iw
        solver.Sigma_an_iw
        solver.g_an_w
        solver.Sigma_an_w

    def tearDown(self):
        # Make sure EDIpackSolver.__del__() is called
        gc.collect()


if __name__ == '__main__':
    unittest.main()

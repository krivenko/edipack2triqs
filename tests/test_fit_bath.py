import unittest
import gc
from itertools import product

import numpy as np
from numpy import multiply as mul

import triqs.operators as op
from triqs.gf import BlockGf, MeshImFreq
from triqs.gf.descriptors import SemiCircular, iOmega_n
from triqs.gf.tools import inverse
from triqs.utility.comparison_tests import assert_block_gfs_are_close

from edipack2triqs import EDMode
from edipack2triqs.solver import EDIpackSolver
from edipack2triqs.bath import BathHybrid
from edipack2triqs.fit import BathFittingParams


s0 = np.eye(2)
sx = np.array([[0, 1], [1, 0]])
sz = np.array([[1, 0], [0, -1]])


class TestFitBath(unittest.TestCase):

    spins = ('up', 'dn')
    norb = 2

    @classmethod
    def make_fops_bath(cls, nbath):
        return ([('B_up', nu) for nu in range(nbath)],
                [('B_dn', nu) for nu in range(nbath)])

    @classmethod
    def make_fops_imp(cls, spin_blocks=True):
        if spin_blocks:
            return ([('up', o) for o in range(cls.norb)],
                    [('dn', o) for o in range(cls.norb)])
        else:
            return ([('up_dn', so) for so in range(cls.norb)],
                    [('up_dn', so + cls.norb) for so in range(cls.norb)])

    @classmethod
    def make_mkind(cls, spin_blocks):
        if spin_blocks:
            return lambda spin, o: (spin, o)
        else:
            return lambda spin, o: ('up_dn',
                                    cls.norb * cls.spins.index(spin) + o)

    @classmethod
    def make_h_loc(cls, h_loc, spin_blocks=True):
        mki = cls.make_mkind(spin_blocks)
        return sum(h_loc[s1, s2, o1, o2]
                   * op.c_dag(*mki(spin1, o1)) * op.c(*mki(spin2, o2))
                   for (s1, spin1), (s2, spin2), o1, o2
                   in product(enumerate(cls.spins), enumerate(cls.spins),
                              range(cls.norb), range(cls.norb)))

    @classmethod
    def make_h_bath(cls, eps, V, spin_blocks=True):
        nbath = eps.shape[-1]
        mki = cls.make_mkind(spin_blocks)
        h_bath = sum(eps[s, nu]
                     * op.c_dag("B_" + spin, nu) * op.c("B_" + spin, nu)
                     for nu, (s, spin)
                     in product(range(nbath), enumerate(cls.spins)))
        h_bath += sum(V[s1, s2, o, nu] * (
                      op.c_dag(*mki(spin1, o)) * op.c("B_" + spin2, nu)
                      + op.c_dag("B_" + spin2, nu) * op.c(*mki(spin1, o)))
                      for (s1, spin1), (s2, spin2), o, nu
                      in product(enumerate(cls.spins), enumerate(cls.spins),
                                 range(cls.norb), range(nbath)))
        return h_bath

    @classmethod
    def make_h_sc(cls, Delta):
        nbath = len(Delta)
        return sum(Delta[nu] * (
            op.c_dag('B_up', nu) * op.c_dag('B_dn', nu)
            + op.c('B_dn', nu) * op.c('B_up', nu))
            for nu in range(nbath)
        )

    def test_nspin1(self):
        nbath = 20
        fops_imp_up, fops_imp_dn = self.make_fops_imp()
        fops_bath_up, fops_bath_dn = self.make_fops_bath(nbath)
        h_loc = self.make_h_loc(mul.outer(s0, np.diag([-1.0, 1.0])))
        h_bath = self.make_h_bath(
            mul.outer([1, 1], np.ones(nbath)),
            mul.outer(s0, np.ones((self.norb, nbath)))
        )

        fit_params = BathFittingParams(scheme="delta", n_iw=100, niter=1000)
        solver = EDIpackSolver(h_loc + h_bath,
                               fops_imp_up,
                               fops_imp_dn,
                               fops_bath_up,
                               fops_bath_dn,
                               bath_fitting_params=fit_params,
                               verbose=0)
        self.assertEqual(solver.h_params.ed_mode, EDMode.NORMAL)
        self.assertEqual(solver.nspin, 1)

        mesh = MeshImFreq(beta=50.0, S="Fermion", n_iw=200)
        Delta = BlockGf(gf_struct=[("up", 2), ("dn", 2)], mesh=mesh)
        V = 2.0 * s0 + 0.5 * sx
        for bn, d in Delta:
            d[0, 0] << SemiCircular(1.8, 0.5)
            d[1, 1] << SemiCircular(2.2, -0.5)
            d << V @ d @ V.T

        fitted_bath, Delta_fit = solver.chi2_fit_bath(Delta)
        self.assertIsInstance(fitted_bath, BathHybrid)
        assert_block_gfs_are_close(Delta_fit, Delta, precision=0.005)

    def test_nspin2(self):
        nbath = 20
        fops_imp_up, fops_imp_dn = self.make_fops_imp()
        fops_bath_up, fops_bath_dn = self.make_fops_bath(nbath)
        h_loc = self.make_h_loc(mul.outer(s0 + 0.1 * sz, np.diag([-1.0, 1.0])))
        h_bath = self.make_h_bath(
            mul.outer([1, 1], np.ones(nbath)),
            mul.outer(s0, np.ones((self.norb, nbath)))
        )

        fit_params = BathFittingParams(scheme="delta", n_iw=100, niter=1000)
        solver = EDIpackSolver(h_loc + h_bath,
                               fops_imp_up,
                               fops_imp_dn,
                               fops_bath_up,
                               fops_bath_dn,
                               bath_fitting_params=fit_params,
                               verbose=0)
        self.assertEqual(solver.h_params.ed_mode, EDMode.NORMAL)
        self.assertEqual(solver.nspin, 2)

        mesh = MeshImFreq(beta=50.0, S="Fermion", n_iw=200)
        Delta = BlockGf(gf_struct=[("up", 2), ("dn", 2)], mesh=mesh)
        V = 2.0 * s0 + 0.5 * sx
        for bn, d in Delta:
            s_coeff = 1.2 if bn == "up" else 0.8
            d[0, 0] << s_coeff * SemiCircular(1.8, 0.5)
            d[1, 1] << s_coeff * SemiCircular(2.2, -0.5)
            d << V @ d @ V.T

        fitted_bath, Delta_fit = solver.chi2_fit_bath(Delta)
        self.assertIsInstance(fitted_bath, BathHybrid)
        assert_block_gfs_are_close(Delta_fit, Delta, precision=0.005)

    def test_nonsu2(self):
        nbath = 8
        fops_imp_up, fops_imp_dn = self.make_fops_imp(spin_blocks=False)
        fops_bath_up, fops_bath_dn = self.make_fops_bath(nbath)
        h_loc = self.make_h_loc(mul.outer(s0, np.diag([-1.0, 1.0])),
                                spin_blocks=False)
        h_bath = self.make_h_bath(
            mul.outer([1, -1], np.ones(nbath)),
            mul.outer(s0 + 0.1 * sx, np.ones((self.norb, nbath))),
            spin_blocks=False
        )
        fit_params = BathFittingParams(scheme="delta", n_iw=100, niter=1000)
        solver = EDIpackSolver(h_loc + h_bath,
                               fops_imp_up,
                               fops_imp_dn,
                               fops_bath_up,
                               fops_bath_dn,
                               bath_fitting_params=fit_params,
                               verbose=3)
        self.assertEqual(solver.h_params.ed_mode, EDMode.NONSU2)

        mesh = MeshImFreq(beta=50.0, S="Fermion", n_iw=200)
        Delta = BlockGf(gf_struct=[("up_dn", 4)], mesh=mesh)
        d1 = inverse(iOmega_n + 0.4) + inverse(iOmega_n - 1.4)
        d2 = inverse(iOmega_n + 1.6) + inverse(iOmega_n - 0.6)
        Delta["up_dn"][0, 0] << 1.1 * d1
        Delta["up_dn"][1, 1] << 1.1 * d2
        Delta["up_dn"][2, 2] << 0.9 * d1
        Delta["up_dn"][3, 3] << 0.9 * d2
        V = np.array([[2.0, 0.5, 0.4, 0.1],
                      [0.5, 2.0, 0.1, 0.4],
                      [0.4, 0.1, 2.0, 0.5],
                      [0.1, 0.4, 0.5, 2.0]])
        Delta["up_dn"] << V @ Delta["up_dn"] @ V.T

        fitted_bath, Delta_fit = solver.chi2_fit_bath(Delta)
        self.assertIsInstance(fitted_bath, BathHybrid)
        assert_block_gfs_are_close(Delta_fit, Delta, precision=1e-3)

    def test_superc(self):
        nbath = 20
        fops_imp_up, fops_imp_dn = self.make_fops_imp()
        fops_bath_up, fops_bath_dn = self.make_fops_bath(nbath)
        h_loc = self.make_h_loc(mul.outer(s0, np.diag([-1.0, 1.0])))
        h_bath = self.make_h_bath(
            mul.outer([1, 1], np.linspace(-1.0, 1.0, nbath)),
            mul.outer(s0, np.outer([1, 1], np.linspace(0.1, 1.0, nbath)))
        )
        h_sc = self.make_h_sc(np.ones(nbath))
        fit_params = BathFittingParams(scheme="delta", n_iw=100, niter=1000)
        solver = EDIpackSolver(h_loc + h_bath + h_sc,
                               fops_imp_up,
                               fops_imp_dn,
                               fops_bath_up,
                               fops_bath_dn,
                               bath_fitting_params=fit_params,
                               verbose=0)
        self.assertEqual(solver.h_params.ed_mode, EDMode.SUPERC)

        mesh = MeshImFreq(beta=50.0, S="Fermion", n_iw=200)
        Delta = BlockGf(gf_struct=[("up", 2), ("dn", 2)], mesh=mesh)
        Delta_an = BlockGf(gf_struct=[("up_dn", 2)], mesh=mesh)

        eps = np.array([-1.0, 0.5])
        D = np.array([0.2, 0.3])
        E = np.sqrt(eps ** 2 + D ** 2)
        V = np.array([[0.4, 0.6], [0.6, 0.4]])

        d1 = (inverse(iOmega_n + E[0]) - inverse(iOmega_n - E[0])) / (2 * E[0])
        d2 = (inverse(iOmega_n + E[1]) - inverse(iOmega_n - E[1])) / (2 * E[1])

        for i, j in product(range(2), repeat=2):
            d1coeff = V[0, i] * V[0, j]
            d2coeff = V[1, i] * V[1, j]
            for bn, d in Delta:
                d[i, j] << -d1coeff * (iOmega_n + eps[0]) * d1 + \
                           -d2coeff * (iOmega_n + eps[1]) * d2
            Delta_an["up_dn"][i, j] << D[0] * d1coeff * d1 + D[1] * d2coeff * d2

        fitted_bath, Delta_fit, Delta_an_fit = \
            solver.chi2_fit_bath(Delta, Delta_an)
        self.assertIsInstance(fitted_bath, BathHybrid)

        assert_block_gfs_are_close(Delta_fit, Delta, precision=1e-3)
        assert_block_gfs_are_close(Delta_an_fit, Delta_an, precision=1e-3)

    def tearDown(self):
        # Make sure EDIpackSolver.__del__() is called
        gc.collect()

import unittest
from itertools import product
from copy import deepcopy

import numpy as np
from numpy.testing import assert_equal, assert_allclose
from numpy import multiply as mul
from numpy.linalg import eigh
from mpi4py import MPI

import triqs.operators as op
from triqs.operators.util.hamiltonians import h_int_kanamori
from h5 import HDFArchive

from edipack2triqs import EDMode
from edipack2triqs.hamiltonian import parse_hamiltonian
from edipack2triqs.bath import BathNormal, BathHybrid, BathGeneral


s0 = np.eye(2)
sx = np.array([[0, 1], [1, 0]])
sz = np.array([[1, 0], [0, -1]])


def assert_equal_and_not_id(test, a, b):
    assert_equal(a, b)
    test.assertFalse(a is b)


class TestHamiltonian(unittest.TestCase):

    spins = ('up', 'dn')
    norb = 3
    orbs = list(range(norb))
    fops_imp_up = [('up', o) for o in orbs]
    fops_imp_dn = [('dn', o) for o in orbs]

    # Interaction parameters for make_H_int()
    Uloc = np.array([1.0, 2.0, 3.0])
    Ust = 0.6
    Jh = 0.15
    Jx = 0.01
    Jp = 0.03

    @classmethod
    def make_H_loc(cls, h_loc):
        return sum(h_loc[s1, s2, o1, o2] * op.c_dag(spin1, o1) * op.c(spin2, o2)
                   for (s1, spin1), (s2, spin2), o1, o2
                   in product(enumerate(cls.spins), enumerate(cls.spins),
                              cls.orbs, cls.orbs))

    @classmethod
    def make_H_int(cls):
        h_int = sum(cls.Uloc[o] * op.n('up', o) * op.n('dn', o)
                    for o in cls.orbs)
        h_int += cls.Ust * sum(int(o1 != o2) * op.n('up', o1) * op.n('dn', o2)
                               for o1, o2 in product(cls.orbs, cls.orbs))
        h_int += (cls.Ust - cls.Jh) * \
            sum(int(o1 < o2) * op.n(s, o1) * op.n(s, o2)
                for s, o1, o2 in product(cls.spins, cls.orbs, cls.orbs))
        h_int -= cls.Jx * sum(int(o1 != o2)
                              * op.c_dag('up', o1) * op.c('dn', o1)
                              * op.c_dag('dn', o2) * op.c('up', o2)
                              for o1, o2 in product(cls.orbs, cls.orbs))
        h_int += cls.Jp * sum(int(o1 != o2)
                              * op.c_dag('up', o1) * op.c_dag('dn', o1)
                              * op.c('dn', o2) * op.c('up', o2)
                              for o1, o2 in product(cls.orbs, cls.orbs))
        return h_int

    @classmethod
    def make_H_loc_an(cls, h_loc_an):
        return sum(h_loc_an[o1, o2] * op.c_dag('up', o1) * op.c_dag('dn', o2)
                   + np.conj(h_loc_an[o1, o2]) * op.c('dn', o2) * op.c('up', o1)
                   for o1, o2 in product(cls.orbs, cls.orbs))

    def check_int_params(self, params, **ref):
        Uloc = ref.get("Uloc", self.Uloc)
        U_ref = np.zeros((self.norb, 2) * 4, dtype=float)
        Ust = ref.get("Ust", self.Ust)
        Jh = ref.get("Jh", self.Jh)
        Jx = ref.get("Jx", self.Jx)
        Jp = ref.get("Jp", self.Jp)
        # Uloc
        for s, o in product(range(2), self.orbs):
            U_ref[o, s, o, 1 - s, o, s, o, 1 - s] = 0.5 * Uloc[o]
            U_ref[o, s, o, 1 - s, o, 1 - s, o, s] = -0.5 * Uloc[o]
        for s, o1, o2 in product(range(2), self.orbs, self.orbs):
            if o1 == o2:
                continue
            # Ust
            U_ref[o1, s, o2, 1 - s, o1, s, o2, 1 - s] = 0.5 * Ust
            U_ref[o1, s, o2, 1 - s, o2, 1 - s, o1, s] = -0.5 * Ust
            # Ust - Jh
            U_ref[o1, s, o2, s, o1, s, o2, s] = 0.5 * (Ust - Jh)
            U_ref[o1, s, o2, s, o2, s, o1, s] = -0.5 * (Ust - Jh)
            # Jx
            U_ref[o1, s, o2, 1 - s, o2, s, o1, 1 - s] = 0.5 * Jx
            U_ref[o1, s, o2, 1 - s, o1, 1 - s, o2, s] = -0.5 * Jx
            # Jp
            U_ref[o1, s, o1, 1 - s, o2, s, o2, 1 - s] = 0.5 * Jp
            U_ref[o1, s, o1, 1 - s, o2, 1 - s, o2, s] = -0.5 * Jp

        assert_equal(params.U, U_ref)


class TestHamiltonianNoBath(TestHamiltonian):
    def test_h_int_kanamori(self):
        norb = self.norb
        Uloc = [1.0, 2.0, 3.0]
        Ust = 0.6
        J = 0.15

        Ud_mat = (Ust - J) * (np.ones((norb, norb)) - np.eye(norb))
        Uod_mat = np.diag(Uloc) + Ust * (np.ones((norb, norb)) - np.eye(norb))

        h = h_int_kanamori(self.spins, norb, Ud_mat, Uod_mat, J, off_diag=True)

        params = parse_hamiltonian(
            h, self.fops_imp_up, self.fops_imp_dn, [], []
        )

        self.check_int_params(params, Uloc=Uloc, Ust=Ust, Jh=J, Jx=J, Jp=J)


class TestHamiltonianBathNormal(TestHamiltonian):

    nbath = 2
    bsites = list(range(nbath))
    fops_bath_up = [('B_up', nu * TestHamiltonian.norb + o)
                    for nu, o in product(bsites, TestHamiltonian.orbs)]
    fops_bath_dn = [('B_dn', nu * TestHamiltonian.norb + o)
                    for nu, o in product(bsites, TestHamiltonian.orbs)]

    h_loc = np.diag([1.5, 2.0, 2.5])
    eps = np.array([[-0.1, 0.1],
                    [-0.2, 0.2],
                    [-0.3, 0.3]])
    V = np.array([[0.4, 0.7],
                  [0.0, 0.5],
                  [0.7, 0.0]])

    @classmethod
    def make_H_bath(cls, eps, V):
        h_bath = sum(
            eps[s, o, nu]
            * op.c_dag("B_" + spin, nu * cls.norb + o)
            * op.c("B_" + spin, nu * cls.norb + o)
            for (s, spin), o, nu
            in product(enumerate(cls.spins), cls.orbs, cls.bsites)
        )
        h_bath += sum(V[s1, s2, o, nu] * (
            op.c_dag(spin1, o) * op.c("B_" + spin2, nu * cls.norb + o)
            + op.c_dag("B_" + spin2, nu * cls.norb + o) * op.c(spin1, o))
            for (s1, spin1), (s2, spin2), o, nu
            in product(enumerate(cls.spins), enumerate(cls.spins),
                       cls.orbs, cls.bsites)
        )
        return h_bath

    def check_bath_arithmetics(self, b1, b2):
        res1 = -(2 * b1 + 3 * b2)
        assert_allclose(res1.data, -5 * b1.data, atol=1e-10)
        res2 = -(3 * b1 - 5 * b2)
        assert_allclose(res2.data, 2 * b1.data, atol=1e-10)

    def check_bath_h5(self, b, h5_name):
        if MPI.COMM_WORLD.Get_rank() != 0:
            return

        archive_name = __file__[:-2] + "h5"

        with HDFArchive(archive_name, 'a') as ar:
            ar["normal_" + h5_name] = b

        with HDFArchive(archive_name, 'r') as ar:
            b2 = ar["normal_" + h5_name]
            self.assertEqual(b2.ed_mode, b.ed_mode)
            assert_equal(b2.data, b.data)
            assert_equal(b2.eps.base, b2.data)
            assert_equal(b2.V.base, b2.data)
            if hasattr(b2, 'U'):
                self.assertIs(b2.U.base, b2.data)
            if hasattr(b2, 'Delta'):
                self.assertIs(b2.Delta.base, b2.data)

    def test_parse_hamiltonian_nspin1(self):
        h_loc = self.make_H_loc(mul.outer(s0, self.h_loc))
        h_int = self.make_H_int()
        h_bath = self.make_H_bath(mul.outer([1, 1], self.eps),
                                  mul.outer(s0, self.V))

        h = h_loc + h_int + h_bath
        params = parse_hamiltonian(
            h,
            self.fops_imp_up, self.fops_imp_dn,
            self.fops_bath_up, self.fops_bath_dn
        )

        self.assertEqual(params.ed_mode, EDMode.NORMAL)
        assert_allclose(params.Hloc,
                        self.h_loc.reshape((1, 1, self.norb, self.norb)))
        assert_allclose(params.Hloc_an, np.zeros((1, 1, self.norb, self.norb)))
        self.assertEqual(params.Hloc_op(self.fops_imp_up, self.fops_imp_dn),
                         h_loc)
        self.assertEqual(params.Hloc_an_op(self.fops_imp_up, self.fops_imp_dn),
                         op.Operator())
        b = params.bath
        self.assertTrue(isinstance(b, BathNormal))
        self.assertEqual(b.nbath, self.nbath)
        self.assertEqual(b.eps.shape, (1, self.norb, self.nbath))
        self.assertFalse(hasattr(b, 'Delta'))
        self.assertEqual(b.V.shape, (1, self.norb, self.nbath))
        self.assertFalse(hasattr(b, 'U'))
        # Check connected bath states
        for o in self.orbs:
            self.assertEqual(
                set((e, v) for e, v in zip(b.eps[0, o], b.V[0, o]) if v != 0),
                set((e, v) for e, v in zip(self.eps[o], self.V[o]) if v != 0)
            )
        # Check disconnected bath states
        self.assertEqual(
            set(e for e, v in zip(b.eps[0].flat, b.V[0].flat) if v == 0),
            set(e for e, v in zip(self.eps.flat, self.V.flat) if v == 0)
        )
        self.check_int_params(params)
        # Check deepcopy of bath
        b2 = deepcopy(b)
        assert_equal_and_not_id(self, b2.data, b.data)
        assert_equal_and_not_id(self, b2.eps, b.eps)
        assert_equal_and_not_id(self, b2.V, b.V)
        self.assertTrue(b2.eps.base is b2.data)
        self.assertTrue(b2.V.base is b2.data)

        self.check_bath_arithmetics(b, b2)
        self.check_bath_h5(b, "nspin1")

    def test_parse_hamiltonian_nspin2(self):
        h_loc = self.make_H_loc(mul.outer(sz, self.h_loc))
        h_int = self.make_H_int()
        h_bath = self.make_H_bath(mul.outer([1, -1], self.eps),
                                  mul.outer(sz, self.V))

        h = h_loc + h_int + h_bath
        params = parse_hamiltonian(
            h,
            self.fops_imp_up, self.fops_imp_dn,
            self.fops_bath_up, self.fops_bath_dn
        )

        self.assertEqual(params.ed_mode, EDMode.NORMAL)
        assert_allclose(params.Hloc, mul.outer(sz, self.h_loc))
        assert_allclose(params.Hloc_an, np.zeros((1, 1, self.norb, self.norb)))
        self.assertEqual(params.Hloc_op(self.fops_imp_up, self.fops_imp_dn),
                         h_loc)
        self.assertEqual(params.Hloc_an_op(self.fops_imp_up, self.fops_imp_dn),
                         op.Operator())
        b = params.bath
        self.assertTrue(isinstance(b, BathNormal))
        self.assertEqual(b.nbath, self.nbath)
        self.assertEqual(b.eps.shape, (2, self.norb, self.nbath))
        self.assertFalse(hasattr(b, 'Delta'))
        self.assertEqual(b.V.shape, (2, self.norb, self.nbath))
        self.assertFalse(hasattr(b, 'U'))
        for s in range(2):
            eps_s = self.eps * (1 - 2 * s)
            V_s = self.V * (1 - 2 * s)
            # Check connected bath states
            for o in self.orbs:
                self.assertEqual(
                    set((e, v) for e, v
                        in zip(b.eps[s, o], b.V[s, o]) if v != 0),
                    set((e, v) for e, v in zip(eps_s[o], V_s[o]) if v != 0)
                )
            # Check disconnected bath states
            self.assertEqual(
                set(e for e, v in zip(b.eps[s].flat, b.V[s].flat) if v == 0),
                set(e for e, v in zip(eps_s.flat, V_s.flat) if v == 0)
            )
        self.check_int_params(params)
        # Check deepcopy of bath
        b2 = deepcopy(b)
        assert_equal_and_not_id(self, b2.data, b.data)
        assert_equal_and_not_id(self, b2.eps, b.eps)
        assert_equal_and_not_id(self, b2.V, b.V)
        self.assertTrue(b2.eps.base is b2.data)
        self.assertTrue(b2.V.base is b2.data)

        self.check_bath_arithmetics(b, b2)
        self.check_bath_h5(b, "nspin2")

    def test_parse_hamiltonian_nonsu2_hloc(self):
        h_loc = self.make_H_loc(mul.outer(sz + 0.2 * sx, self.h_loc))
        h_int = self.make_H_int()
        h_bath = self.make_H_bath(mul.outer([1, 1], self.eps),
                                  mul.outer(s0, self.V))

        h = h_loc + h_int + h_bath
        params = parse_hamiltonian(
            h,
            self.fops_imp_up, self.fops_imp_dn,
            self.fops_bath_up, self.fops_bath_dn
        )

        self.assertEqual(params.ed_mode, EDMode.NONSU2)
        assert_allclose(params.Hloc, mul.outer(sz + 0.2 * sx, self.h_loc))
        assert_allclose(params.Hloc_an, np.zeros((1, 1, self.norb, self.norb)))
        self.assertEqual(params.Hloc_op(self.fops_imp_up, self.fops_imp_dn),
                         h_loc)
        self.assertEqual(params.Hloc_an_op(self.fops_imp_up, self.fops_imp_dn),
                         op.Operator())
        b = params.bath
        self.assertTrue(isinstance(b, BathNormal))
        self.assertEqual(b.nbath, self.nbath)
        self.assertEqual(b.eps.shape, (2, self.norb, self.nbath))
        self.assertFalse(hasattr(b, 'Delta'))
        self.assertEqual(b.V.shape, (2, self.norb, self.nbath))
        assert_equal(b.U, np.zeros((2, self.norb, self.nbath)))
        for s in range(2):
            # Check connected bath states
            for o in self.orbs:
                self.assertEqual(
                    set((e, v) for e, v
                        in zip(b.eps[s, o], b.V[s, o]) if v != 0),
                    set((e, v) for e, v in zip(self.eps[o], self.V[o])
                        if v != 0)
                )
            # Check disconnected bath states
            self.assertEqual(
                set(e for e, v in zip(b.eps[s].flat, b.V[s].flat) if v == 0),
                set(e for e, v in zip(self.eps.flat, self.V.flat) if v == 0)
            )
        self.check_int_params(params)
        # Check deepcopy of bath
        b2 = deepcopy(b)
        assert_equal_and_not_id(self, b2.data, b.data)
        assert_equal_and_not_id(self, b2.eps, b.eps)
        assert_equal_and_not_id(self, b2.V, b.V)
        assert_equal_and_not_id(self, b2.U, b.U)
        self.assertTrue(b2.eps.base is b2.data)
        self.assertTrue(b2.V.base is b2.data)
        self.assertTrue(b2.U.base is b2.data)

        self.check_bath_arithmetics(b, b2)
        self.check_bath_h5(b, "nonsu2_hloc")

    def test_parse_hamiltonian_nonsu2_bath(self):
        h_loc = self.make_H_loc(mul.outer(s0, self.h_loc))
        h_int = self.make_H_int()
        h_bath = self.make_H_bath(mul.outer([1, -1], self.eps),
                                  mul.outer(sz + 0.2 * sx, self.V))

        h = h_loc + h_int + h_bath
        params = parse_hamiltonian(
            h,
            self.fops_imp_up, self.fops_imp_dn,
            self.fops_bath_up, self.fops_bath_dn
        )

        self.assertEqual(params.ed_mode, EDMode.NONSU2)
        assert_allclose(params.Hloc, mul.outer(s0, self.h_loc))
        assert_allclose(params.Hloc_an, np.zeros((1, 1, self.norb, self.norb)))
        self.assertEqual(params.Hloc_op(self.fops_imp_up, self.fops_imp_dn),
                         h_loc)
        self.assertEqual(params.Hloc_an_op(self.fops_imp_up, self.fops_imp_dn),
                         op.Operator())
        b = params.bath
        self.assertTrue(isinstance(b, BathNormal))
        self.assertEqual(b.nbath, self.nbath)
        self.assertEqual(b.eps.shape, (2, self.norb, self.nbath))
        self.assertFalse(hasattr(b, 'Delta'))
        self.assertEqual(b.V.shape, (2, self.norb, self.nbath))
        self.assertEqual(b.U.shape, (2, self.norb, self.nbath))
        for s in range(2):
            eps_s = self.eps * (1 - 2 * s)
            V_s = self.V * (1 - 2 * s)
            # Check connected bath states
            for o in self.orbs:
                self.assertEqual(
                    set((e, v, u) for e, v, u
                        in zip(b.eps[s, o], b.V[s, o], b.U[s, o]) if v != 0),
                    set((e, v, u) for e, v, u
                        in zip(eps_s[o], V_s[o], 0.2 * self.V[o]) if v != 0)
                )
            # Check disconnected bath states
            self.assertEqual(
                set(e for e, v in zip(b.eps[s].flat, b.V[s].flat) if v == 0),
                set(e for e, v in zip(eps_s.flat, V_s.flat) if v == 0)
            )
        self.check_int_params(params)
        # Check deepcopy of bath
        b2 = deepcopy(b)
        assert_equal_and_not_id(self, b2.data, b.data)
        assert_equal_and_not_id(self, b2.eps, b.eps)
        assert_equal_and_not_id(self, b2.V, b.V)
        assert_equal_and_not_id(self, b2.U, b.U)
        self.assertTrue(b2.eps.base is b2.data)
        self.assertTrue(b2.V.base is b2.data)
        self.assertTrue(b2.U.base is b2.data)

        self.check_bath_arithmetics(b, b2)
        self.check_bath_h5(b, "nonsu2_bath")

    def test_parse_hamiltonian_superc_hloc_an(self):
        h_loc = self.make_H_loc(mul.outer(s0, self.h_loc))
        h_int = self.make_H_int()
        h_bath = self.make_H_bath(mul.outer([1, 1], self.eps),
                                  mul.outer(s0, self.V))

        Delta = np.diag([0.1, 0.2, 0.3])
        h_loc_an = self.make_H_loc_an(Delta)

        h = h_loc + h_loc_an + h_int + h_bath
        params = parse_hamiltonian(
            h,
            self.fops_imp_up, self.fops_imp_dn,
            self.fops_bath_up, self.fops_bath_dn
        )

        self.assertEqual(params.ed_mode, EDMode.SUPERC)
        assert_allclose(params.Hloc,
                        self.h_loc.reshape(1, 1, self.norb, self.norb))
        assert_allclose(params.Hloc_an,
                        Delta.reshape(1, 1, self.norb, self.norb))
        self.assertEqual(params.Hloc_op(self.fops_imp_up, self.fops_imp_dn),
                         h_loc)
        self.assertEqual(params.Hloc_an_op(self.fops_imp_up, self.fops_imp_dn),
                         h_loc_an)
        b = params.bath
        self.assertTrue(isinstance(b, BathNormal))
        self.assertEqual(b.nbath, self.nbath)
        self.assertEqual(b.eps.shape, (1, self.norb, self.nbath))
        assert_equal(b.Delta, np.zeros((1, self.norb, self.nbath)))
        self.assertEqual(b.V.shape, (1, self.norb, self.nbath))
        self.assertFalse(hasattr(b, 'U'))
        # Check connected bath states
        for o in self.orbs:
            self.assertEqual(
                set((e, v) for e, v in zip(b.eps[0, o], b.V[0, o]) if v != 0),
                set((e, v) for e, v in zip(self.eps[o], self.V[o]) if v != 0)
            )
        # Check disconnected bath states
        self.assertEqual(
            set(e for e, v in zip(b.eps[0].flat, b.V[0].flat) if v == 0),
            set(e for e, v in zip(self.eps.flat, self.V.flat) if v == 0)
        )
        self.check_int_params(params)
        # Check deepcopy of bath
        b2 = deepcopy(b)
        assert_equal_and_not_id(self, b2.data, b.data)
        assert_equal_and_not_id(self, b2.eps, b.eps)
        assert_equal_and_not_id(self, b2.V, b.V)
        assert_equal_and_not_id(self, b2.Delta, b.Delta)
        self.assertTrue(b2.eps.base is b2.data)
        self.assertTrue(b2.V.base is b2.data)
        self.assertTrue(b2.Delta.base is b2.data)

        self.check_bath_arithmetics(b, b2)
        self.check_bath_h5(b, "superc")

    def test_parse_hamiltonian_superc_bath(self):
        h_loc = self.make_H_loc(mul.outer(s0, self.h_loc))
        h_int = self.make_H_int()
        h_bath = self.make_H_bath(mul.outer([1, 1], self.eps),
                                  mul.outer(s0, self.V))

        Delta = np.array([[0.6, 0.7],
                          [0.8, 0.6],
                          [0.9, 0.7]])
        h_sc = sum(Delta[o, nu] * (op.c_dag('B_up', nu * self.norb + o)
                                   * op.c_dag('B_dn', nu * self.norb + o)
                                   + op.c('B_dn', nu * self.norb + o)
                                   * op.c('B_up', nu * self.norb + o))
                   for o, nu in product(self.orbs, self.bsites))

        h = h_loc + h_int + h_bath + h_sc
        params = parse_hamiltonian(
            h,
            self.fops_imp_up, self.fops_imp_dn,
            self.fops_bath_up, self.fops_bath_dn
        )

        self.assertEqual(params.ed_mode, EDMode.SUPERC)
        assert_allclose(params.Hloc,
                        self.h_loc.reshape(1, 1, self.norb, self.norb))
        assert_allclose(params.Hloc_an,
                        np.zeros((1, 1, self.norb, self.norb)))
        self.assertEqual(params.Hloc_op(self.fops_imp_up, self.fops_imp_dn),
                         h_loc)
        self.assertEqual(params.Hloc_an_op(self.fops_imp_up, self.fops_imp_dn),
                         op.Operator())
        b = params.bath
        self.assertTrue(isinstance(b, BathNormal))
        self.assertEqual(b.nbath, self.nbath)
        self.assertEqual(b.eps.shape, (1, self.norb, self.nbath))
        self.assertEqual(b.Delta.shape, (1, self.norb, self.nbath))
        self.assertEqual(b.V.shape, (1, self.norb, self.nbath))
        self.assertFalse(hasattr(b, 'U'))
        # Check connected bath states
        for o in self.orbs:
            self.assertEqual(
                set((e, d, v) for e, d, v
                    in zip(b.eps[0, o], b.Delta[0, o], b.V[0, o])
                    if v != 0),
                set((e, d, v) for e, d, v
                    in zip(self.eps[o], Delta[o], self.V[o])
                    if v != 0)
            )
        # Check disconnected bath states
        self.assertEqual(
            set((e, d) for e, d, v
                in zip(b.eps[0].flat, b.Delta[0].flat, b.V[0].flat)
                if v == 0),
            set((e, d) for e, d, v
                in zip(self.eps.flat, Delta.flat, self.V.flat)
                if v == 0)
        )
        self.check_int_params(params)
        # Check deepcopy of bath
        b2 = deepcopy(b)
        assert_equal_and_not_id(self, b2.data, b.data)
        assert_equal_and_not_id(self, b2.eps, b.eps)
        assert_equal_and_not_id(self, b2.V, b.V)
        assert_equal_and_not_id(self, b2.Delta, b.Delta)
        self.assertTrue(b2.eps.base is b2.data)
        self.assertTrue(b2.V.base is b2.data)
        self.assertTrue(b2.Delta.base is b2.data)

        self.check_bath_arithmetics(b, b2)
        self.check_bath_h5(b, "superc")


class TestHamiltonianBathHybrid(TestHamiltonian):

    nbath = 4
    bsites = list(range(nbath))
    fops_bath_up = [('B_up', nu) for nu in bsites]
    fops_bath_dn = [('B_dn', nu) for nu in bsites]

    h_loc = np.diag([1.5, 2.0, 2.5])
    eps = np.array([-0.1, -0.2, -0.3, -0.4])
    V = np.array([[0.4, 0.7, 0.1, 0.4],
                  [0.0, 0.5, 0.2, 0.5],
                  [0.7, 0.4, 0.3, 0.0]])

    @classmethod
    def make_H_bath(cls, eps, V):
        h_bath = sum(
            eps[s, nu] * op.c_dag("B_" + spin, nu) * op.c("B_" + spin, nu)
            for (s, spin), nu
            in product(enumerate(cls.spins), cls.bsites)
        )
        h_bath += sum(V[s1, s2, o, nu] * (
            op.c_dag(spin1, o) * op.c("B_" + spin2, nu)
            + op.c_dag("B_" + spin2, nu) * op.c(spin1, o))
            for (s1, spin1), (s2, spin2), o, nu
            in product(enumerate(cls.spins), enumerate(cls.spins),
                       TestHamiltonian.orbs, cls.bsites)
        )
        return h_bath

    def check_bath_arithmetics(self, b1, b2):
        res1 = -(2 * b1 + 3 * b2)
        assert_allclose(res1.data, -5 * b1.data, atol=1e-10)
        res2 = -(3 * b1 - 5 * b2)
        assert_allclose(res2.data, 2 * b1.data, atol=1e-10)

    def check_bath_h5(self, b, h5_name):
        if MPI.COMM_WORLD.Get_rank() != 0:
            return

        archive_name = __file__[:-2] + "h5"

        with HDFArchive(archive_name, 'a') as ar:
            ar["hybrid_" + h5_name] = b

        with HDFArchive(archive_name, 'r') as ar:
            b2 = ar["hybrid_" + h5_name]
            self.assertEqual(b2.ed_mode, b.ed_mode)
            assert_equal(b2.data, b.data)
            assert_equal(b2.eps.base, b2.data)
            assert_equal(b2.V.base, b2.data)
            if hasattr(b2, 'U'):
                self.assertIs(b2.U.base, b2.data)
            if hasattr(b2, 'Delta'):
                self.assertIs(b2.Delta.base, b2.data)

    def test_parse_hamiltonian_nspin1(self):
        h_loc = self.make_H_loc(mul.outer(s0, self.h_loc))
        h_int = self.make_H_int()
        h_bath = self.make_H_bath(mul.outer([1, 1], self.eps),
                                  mul.outer(s0, self.V))

        h = h_loc + h_int + h_bath
        params = parse_hamiltonian(
            h,
            self.fops_imp_up, self.fops_imp_dn,
            self.fops_bath_up, self.fops_bath_dn
        )

        self.assertEqual(params.ed_mode, EDMode.NORMAL)
        assert_allclose(params.Hloc,
                        self.h_loc.reshape((1, 1, self.norb, self.norb)))
        assert_allclose(params.Hloc_an, np.zeros((1, 1, self.norb, self.norb)))
        self.assertEqual(params.Hloc_op(self.fops_imp_up, self.fops_imp_dn),
                         h_loc)
        self.assertEqual(params.Hloc_an_op(self.fops_imp_up, self.fops_imp_dn),
                         op.Operator())
        self.assertTrue(isinstance(params.bath, BathHybrid))
        self.assertEqual(params.bath.nbath, self.nbath)
        assert_allclose(params.bath.eps, self.eps.reshape(1, self.nbath))
        self.assertFalse(hasattr(params.bath, 'Delta'))
        assert_allclose(params.bath.V, self.V.reshape(1, self.norb, self.nbath))
        self.assertFalse(hasattr(params.bath, 'U'))
        self.check_int_params(params)

        # Check deepcopy of bath
        b2 = deepcopy(params.bath)
        assert_equal_and_not_id(self, b2.data, params.bath.data)
        assert_equal_and_not_id(self, b2.eps, params.bath.eps)
        assert_equal_and_not_id(self, b2.V, params.bath.V)
        self.assertTrue(b2.eps.base is b2.data)
        self.assertTrue(b2.V.base is b2.data)

        self.check_bath_arithmetics(params.bath, b2)
        self.check_bath_h5(b2, "nspin1")

    def test_parse_hamiltonian_nspin2(self):
        h_loc = self.make_H_loc(mul.outer(sz, self.h_loc))
        h_int = self.make_H_int()
        h_bath = self.make_H_bath(mul.outer([1, -1], self.eps),
                                  mul.outer(sz, self.V))

        h = h_loc + h_int + h_bath
        params = parse_hamiltonian(
            h,
            self.fops_imp_up, self.fops_imp_dn,
            self.fops_bath_up, self.fops_bath_dn
        )

        self.assertEqual(params.ed_mode, EDMode.NORMAL)
        assert_allclose(params.Hloc, mul.outer(sz, self.h_loc))
        assert_allclose(params.Hloc_an, np.zeros((1, 1, self.norb, self.norb)))
        self.assertEqual(params.Hloc_op(self.fops_imp_up, self.fops_imp_dn),
                         h_loc)
        self.assertEqual(params.Hloc_an_op(self.fops_imp_up, self.fops_imp_dn),
                         op.Operator())
        self.assertTrue(isinstance(params.bath, BathHybrid))
        self.assertEqual(params.bath.nbath, self.nbath)
        assert_allclose(params.bath.eps,
                        mul.outer([1, -1], self.eps.reshape(self.nbath)))
        self.assertFalse(hasattr(params.bath, 'Delta'))
        assert_allclose(
            params.bath.V,
            mul.outer([1, -1], self.V.reshape(self.norb, self.nbath))
        )
        self.assertFalse(hasattr(params.bath, 'U'))
        self.check_int_params(params)

        # Check deepcopy of bath
        b2 = deepcopy(params.bath)
        assert_equal_and_not_id(self, b2.data, params.bath.data)
        assert_equal_and_not_id(self, b2.eps, params.bath.eps)
        assert_equal_and_not_id(self, b2.V, params.bath.V)
        self.assertTrue(b2.eps.base is b2.data)
        self.assertTrue(b2.V.base is b2.data)

        self.check_bath_arithmetics(params.bath, b2)
        self.check_bath_h5(b2, "nspin2")

    def test_parse_hamiltonian_nonsu2_hloc(self):
        h_loc = self.make_H_loc(mul.outer(sz + 0.2 * sx, self.h_loc))
        h_int = self.make_H_int()
        h_bath = self.make_H_bath(mul.outer([1, 1], self.eps),
                                  mul.outer(s0, self.V))

        h = h_loc + h_int + h_bath
        params = parse_hamiltonian(
            h,
            self.fops_imp_up, self.fops_imp_dn,
            self.fops_bath_up, self.fops_bath_dn
        )

        self.assertEqual(params.ed_mode, EDMode.NONSU2)
        assert_allclose(params.Hloc, mul.outer(sz + 0.2 * sx, self.h_loc))
        assert_allclose(params.Hloc_an, np.zeros((1, 1, self.norb, self.norb)))
        self.assertEqual(params.Hloc_op(self.fops_imp_up, self.fops_imp_dn),
                         h_loc)
        self.assertEqual(params.Hloc_an_op(self.fops_imp_up, self.fops_imp_dn),
                         op.Operator())
        self.assertTrue(isinstance(params.bath, BathHybrid))
        self.assertEqual(params.bath.nbath, self.nbath)
        assert_allclose(params.bath.eps, mul.outer([1, 1], self.eps))
        self.assertFalse(hasattr(params.bath, 'Delta'))
        assert_allclose(params.bath.V, mul.outer([1, 1], self.V))
        assert_allclose(params.bath.U, np.zeros((2, self.norb, self.nbath)))
        self.check_int_params(params)

        # Check deepcopy of bath
        b2 = deepcopy(params.bath)
        assert_equal_and_not_id(self, b2.data, params.bath.data)
        assert_equal_and_not_id(self, b2.eps, params.bath.eps)
        assert_equal_and_not_id(self, b2.V, params.bath.V)
        assert_equal_and_not_id(self, b2.U, params.bath.U)
        self.assertTrue(b2.eps.base is b2.data)
        self.assertTrue(b2.V.base is b2.data)
        self.assertTrue(b2.U.base is b2.data)

        self.check_bath_arithmetics(params.bath, b2)
        self.check_bath_h5(b2, "nonsu2_hloc")

    def test_parse_hamiltonian_nonsu2_bath(self):
        h_loc = self.make_H_loc(mul.outer(s0, self.h_loc))
        h_int = self.make_H_int()
        h_bath = self.make_H_bath(mul.outer([1, -1], self.eps),
                                  mul.outer(sz + 0.2 * sx, self.V))

        h = h_loc + h_int + h_bath
        params = parse_hamiltonian(
            h,
            self.fops_imp_up, self.fops_imp_dn,
            self.fops_bath_up, self.fops_bath_dn
        )

        self.assertEqual(params.ed_mode, EDMode.NONSU2)
        assert_allclose(params.Hloc, mul.outer(s0, self.h_loc))
        assert_allclose(params.Hloc_an, np.zeros((1, 1, self.norb, self.norb)))
        self.assertEqual(params.Hloc_op(self.fops_imp_up, self.fops_imp_dn),
                         h_loc)
        self.assertEqual(params.Hloc_an_op(self.fops_imp_up, self.fops_imp_dn),
                         op.Operator())
        self.assertTrue(isinstance(params.bath, BathHybrid))
        self.assertEqual(params.bath.nbath, self.nbath)
        assert_allclose(params.bath.eps, mul.outer([1, -1], self.eps))
        self.assertFalse(hasattr(params.bath, 'Delta'))
        assert_allclose(params.bath.V, mul.outer([1, -1], self.V))
        assert_allclose(params.bath.U, mul.outer([0.2, 0.2], self.V))

        # Check deepcopy of bath
        b2 = deepcopy(params.bath)
        assert_equal_and_not_id(self, b2.data, params.bath.data)
        assert_equal_and_not_id(self, b2.eps, params.bath.eps)
        assert_equal_and_not_id(self, b2.V, params.bath.V)
        assert_equal_and_not_id(self, b2.U, params.bath.U)
        self.assertTrue(b2.eps.base is b2.data)
        self.assertTrue(b2.V.base is b2.data)
        self.assertTrue(b2.U.base is b2.data)

        self.check_bath_arithmetics(params.bath, b2)
        self.check_bath_h5(b2, "nonsu2_bath")

    def test_parse_hamiltonian_superc_hloc_an(self):
        h_loc = self.make_H_loc(mul.outer(s0, self.h_loc))
        h_int = self.make_H_int()
        h_bath = self.make_H_bath(mul.outer([1, 1], self.eps),
                                  mul.outer(s0, self.V))

        Delta = np.array([[0.1, 0.04, 0.05j],
                          [0.04, 0.2, 0.06],
                          [0.05j, 0.06, 0.3]])
        h_loc_an = self.make_H_loc_an(Delta)

        h = h_loc + h_loc_an + h_int + h_bath
        params = parse_hamiltonian(
            h,
            self.fops_imp_up, self.fops_imp_dn,
            self.fops_bath_up, self.fops_bath_dn
        )

        self.assertEqual(params.ed_mode, EDMode.SUPERC)
        assert_allclose(params.Hloc,
                        self.h_loc.reshape(1, 1, self.norb, self.norb))
        assert_allclose(params.Hloc_an,
                        Delta.reshape(1, 1, self.norb, self.norb))
        self.assertEqual(params.Hloc_op(self.fops_imp_up, self.fops_imp_dn),
                         h_loc)
        self.assertEqual(params.Hloc_an_op(self.fops_imp_up, self.fops_imp_dn),
                         h_loc_an)
        self.assertTrue(isinstance(params.bath, BathHybrid))
        self.assertEqual(params.bath.nbath, self.nbath)
        assert_allclose(params.bath.eps, self.eps.reshape(1, self.nbath))
        assert_equal(params.bath.Delta, np.zeros((1, self.nbath)))
        assert_allclose(params.bath.V, self.V.reshape(1, self.norb, self.nbath))
        self.assertFalse(hasattr(params.bath, 'U'))
        self.check_int_params(params)

        # Check deepcopy of bath
        b2 = deepcopy(params.bath)
        assert_equal_and_not_id(self, b2.data, params.bath.data)
        assert_equal_and_not_id(self, b2.eps, params.bath.eps)
        assert_equal_and_not_id(self, b2.V, params.bath.V)
        assert_equal_and_not_id(self, b2.Delta, params.bath.Delta)
        self.assertTrue(b2.eps.base is b2.data)
        self.assertTrue(b2.V.base is b2.data)
        self.assertTrue(b2.Delta.base is b2.data)

        self.check_bath_arithmetics(params.bath, b2)
        self.check_bath_h5(b2, "superc")

    def test_parse_hamiltonian_superc_bath(self):
        h_loc = self.make_H_loc(mul.outer(s0, self.h_loc))
        h_int = self.make_H_int()
        h_bath = self.make_H_bath(mul.outer([1, 1], self.eps),
                                  mul.outer(s0, self.V))

        Delta = np.array([0.6, 0.7, 0.8, 0.9])
        h_sc = sum(Delta[nu] * (op.c_dag('B_up', nu) * op.c_dag('B_dn', nu)
                                + op.c('B_dn', nu) * op.c('B_up', nu))
                   for nu in self.bsites)

        h = h_loc + h_int + h_bath + h_sc
        params = parse_hamiltonian(
            h,
            self.fops_imp_up, self.fops_imp_dn,
            self.fops_bath_up, self.fops_bath_dn
        )

        self.assertEqual(params.ed_mode, EDMode.SUPERC)
        assert_allclose(params.Hloc,
                        self.h_loc.reshape(1, 1, self.norb, self.norb))
        assert_allclose(params.Hloc_an, np.zeros((1, 1, self.norb, self.norb)))
        self.assertEqual(params.Hloc_op(self.fops_imp_up, self.fops_imp_dn),
                         h_loc)
        self.assertEqual(params.Hloc_an_op(self.fops_imp_up, self.fops_imp_dn),
                         op.Operator())
        self.assertTrue(isinstance(params.bath, BathHybrid))
        self.assertEqual(params.bath.nbath, self.nbath)
        assert_allclose(params.bath.eps, self.eps.reshape(1, self.nbath))
        assert_allclose(params.bath.Delta, Delta.reshape(1, self.nbath))
        assert_allclose(params.bath.V, self.V.reshape(1, self.norb, self.nbath))
        self.assertFalse(hasattr(params.bath, 'U'))
        self.check_int_params(params)

        # Check deepcopy of bath
        b2 = deepcopy(params.bath)
        assert_equal_and_not_id(self, b2.data, params.bath.data)
        assert_equal_and_not_id(self, b2.eps, params.bath.eps)
        assert_equal_and_not_id(self, b2.V, params.bath.V)
        assert_equal_and_not_id(self, b2.Delta, params.bath.Delta)
        self.assertTrue(b2.eps.base is b2.data)
        self.assertTrue(b2.V.base is b2.data)
        self.assertTrue(b2.Delta.base is b2.data)

        self.check_bath_arithmetics(params.bath, b2)
        self.check_bath_h5(b2, "superc")


class TestHamiltonianBathGeneral(TestHamiltonian):

    nbath = 4
    bsites = list(range(nbath))
    fops_bath_up = [('B_up', nu * TestHamiltonian.norb + o)
                    for o, nu in product(TestHamiltonian.orbs, bsites)]
    fops_bath_dn = [('B_dn', nu * TestHamiltonian.norb + o)
                    for o, nu in product(TestHamiltonian.orbs, bsites)]

    h_loc = np.array([[1.5, 1.0, 0.5],
                      [1.0, 2.0, 1.0],
                      [0.5, 1.0, 2.5]])
    h = np.moveaxis(np.array([[[0.5, -0.2, 0.0],
                               [-0.2, 0.6, -0.2],
                               [0.0, -0.2, 0.7]],
                              [[0.6, -0.2, 0.0],
                               [-0.2, 0.7, 0.0],
                               [0.0, 0.0, 0.85]],
                              [[0.65, 0.0, 0.0],
                               [0.0, 0.7, -0.2j],
                               [0.0, 0.2j, 0.8]],
                              [[0.6, 0.1, 0.0],
                               [0.1, 0.7, 0.0],
                               [0.0, 0.0, 0.8]]]), 0, 2)
    Delta = 0.1 * np.moveaxis(np.array([[[0.5, -0.2, 0.0],
                                         [-0.2, 0.6, -0.2],
                                         [0.0, -0.2, 0.7]],
                                        [[0.6, -0.2, 0.0],
                                         [-0.2, 0.7, 0.0],
                                         [0.0, 0.0, 0.85]],
                                        [[0.65, 0.0, 0.0],
                                         [0.0, 0.7, 0.2j],
                                         [0.0, 0.2j, 0.8]],
                                        [[0.6, 0.1, 0.0],
                                         [0.1, 0.7, 0.0],
                                         [0.0, 0.0, 0.8]]]), 0, 2)

    V = np.array([[0.4, 0.75, 0.2, 0.0],
                  [0.0, 0.5, 0.45, 0.3],
                  [0.7, 0.0, 0.0, 0.1]])

    @classmethod
    def make_H_bath(cls, h, V):
        h_bath = sum(
            h[s1, s2, o1, o2, nu]
            * op.c_dag("B_" + spin1, nu * cls.norb + o1)
            * op.c("B_" + spin2, nu * cls.norb + o2)
            for (s1, spin1), (s2, spin2), o1, o2, nu
            in product(enumerate(cls.spins), enumerate(cls.spins),
                       TestHamiltonian.orbs, TestHamiltonian.orbs,
                       cls.bsites)
        )
        h_bath += sum(V[s, o, nu] * (
            op.c_dag(spin, o) * op.c("B_" + spin, nu * cls.norb + o)
            + op.c_dag("B_" + spin, nu * cls.norb + o) * op.c(spin, o))
            for (s, spin), o, nu
            in product(enumerate(cls.spins), TestHamiltonian.orbs, cls.bsites)
        )
        return h_bath

    @classmethod
    def check_bath(cls, hvec, lambdavec, V, h_ref, V_ref, is_nambu=False):
        # Checking the bath Hamiltonian and the hopping amplitude matrix derived
        # by BathGeneral is not trivial because the way bath states are
        # distributed over replicas is not necessarily unique.
        # Here, we perform two indirect tests to check consistency of the bath
        # parameters with the reference data.
        #
        # - Eigenvalues of h and h_ref must agree.
        # - Matrix products V @ h @ V^T and V_ref @ h_ref @ V_ref^T must
        #   coincide. This effectively checks that the tested and reference
        #   baths have the same effect on the impurity.

        nspin, norb = h_ref.shape[0], h_ref.shape[2]

        # Build and diagonalize Hamiltonian
        h_mat = np.zeros((cls.nbath, nspin, norb, cls.nbath, nspin, norb),
                         dtype=complex)
        for nu in cls.bsites:
            for spin1, spin2, orb1, orb2, isym in np.ndindex(hvec.shape):
                h_mat[nu, spin1, orb1, nu, spin2, orb2] += \
                    lambdavec[nu][isym] * hvec[spin1, spin2, orb1, orb2, isym]
        h_mat = h_mat.reshape((cls.nbath * nspin * norb,
                               cls.nbath * nspin * norb))
        eps = eigh(h_mat)[0]

        # Build and diagonalize reference Hamiltonian
        h_ref_mat = np.zeros((cls.nbath, nspin, norb, cls.nbath, nspin, norb),
                             dtype=complex)
        for spin1, spin2, orb1, orb2, nu in np.ndindex(h_ref.shape):
            h_ref_mat[nu, spin1, orb1, nu, spin2, orb2] = \
                h_ref[spin1, spin2, orb1, orb2, nu]
        h_ref_mat = h_ref_mat.reshape((cls.nbath * nspin * norb,
                                       cls.nbath * nspin * norb))
        eps_ref = eigh(h_ref_mat)[0]

        # Compare the eigenvalues
        assert_allclose(eps, eps_ref, atol=1e-10)

        # Compare V @ h @ V^T and V_ref @ h_ref @ V_ref^T
        V_mat = np.zeros((nspin, norb, cls.nbath, nspin, norb))
        for nu in cls.bsites:
            for spin, orb in np.ndindex(V[nu].shape):
                if is_nambu:
                    V_mat[0, orb, nu, 0, orb] = V[nu][0, orb]
                    V_mat[1, orb, nu, 1, orb] = -V[nu][0, orb]
                else:
                    V_mat[spin, orb, nu, spin, orb] = V[nu][spin, orb]
        V_mat = V_mat.reshape((nspin * norb, cls.nbath * nspin * norb))

        V_ref_mat = np.zeros((nspin, norb, cls.nbath, nspin, norb))
        for spin, orb, nu in np.ndindex(V_ref.shape):
            if is_nambu:
                V_ref_mat[0, orb, nu, 0, orb] = V_ref[0, orb, nu]
                V_ref_mat[1, orb, nu, 1, orb] = -V_ref[0, orb, nu]
            else:
                V_ref_mat[spin, orb, nu, spin, orb] = V_ref[spin, orb, nu]
        V_ref_mat = V_ref_mat.reshape((nspin * norb, cls.nbath * nspin * norb))

        assert_allclose(V_mat @ h_mat @ V_mat.T,
                        V_ref_mat @ h_ref_mat @ V_ref_mat.T,
                        atol=1e-10)

    def check_bath_arithmetics(self, b1, b2):
        res1 = -(2 * b1 + 3 * b2)
        self.assertEqual(res1.data[0], b1.data[0])
        assert_allclose(res1.data[1:], -5 * b1.data[1:], atol=1e-10)
        assert_allclose(res1.lambdavec, -5 * b1.lambdavec, atol=1e-10)
        res2 = -(3 * b1 - 5 * b2)
        self.assertEqual(res2.data[0], b1.data[0])
        assert_allclose(res2.data[1:], 2 * b1.data[1:], atol=1e-10)
        assert_allclose(res2.lambdavec, 2 * b1.lambdavec, atol=1e-10)

    def check_bath_h5(self, b, h5_name):
        if MPI.COMM_WORLD.Get_rank() != 0:
            return

        archive_name = __file__[:-2] + "h5"

        with HDFArchive(archive_name, 'a') as ar:
            ar["general_" + h5_name] = b

        with HDFArchive(archive_name, 'r') as ar:
            b2 = ar["general_" + h5_name]
            assert_equal(b2.hvec, b.hvec)
            assert_equal(b2.lambdavec, b.lambdavec)
            assert_equal(b2.data, b.data)
            for l_nu, V_nu in zip(b2.l, b2.V):
                self.assertIs(l_nu.base, b2.data)
                self.assertIs(V_nu.base, b2.data)

    def test_parse_hamiltonian_nspin1(self):
        h_loc = self.make_H_loc(mul.outer(s0, self.h_loc))
        h_int = self.make_H_int()
        h_bath = self.make_H_bath(mul.outer(s0, self.h),
                                  mul.outer([1, 1], self.V))

        h = h_loc + h_int + h_bath
        params = parse_hamiltonian(
            h,
            self.fops_imp_up, self.fops_imp_dn,
            self.fops_bath_up, self.fops_bath_dn
        )

        self.assertEqual(params.ed_mode, EDMode.NORMAL)
        assert_allclose(params.Hloc,
                        self.h_loc.reshape((1, 1, self.norb, self.norb)))
        assert_allclose(params.Hloc_an, np.zeros((1, 1, self.norb, self.norb)))
        self.assertEqual(params.Hloc_op(self.fops_imp_up, self.fops_imp_dn),
                         h_loc)
        self.assertEqual(params.Hloc_an_op(self.fops_imp_up, self.fops_imp_dn),
                         op.Operator())
        self.assertTrue(isinstance(params.bath, BathGeneral))
        self.assertEqual(params.bath.nbath, self.nbath)
        self.assertFalse(hasattr(params.bath, 'Delta'))
        self.assertEqual(params.bath.nsym, 6)
        self.assertEqual(params.bath.hvec.shape,
                         (1, 1, self.norb, self.norb, 6))
        self.assertEqual(len(params.bath.l), self.nbath)
        self.assertEqual(len(params.bath.V), self.nbath)
        self.check_bath(params.bath.hvec, params.bath.l, params.bath.V,
                        self.h.reshape(1, 1, self.norb, self.norb, self.nbath),
                        self.V.reshape(1, self.norb, self.nbath))
        self.assertFalse(hasattr(params.bath, 'U'))
        self.check_int_params(params)

        # Check deepcopy of bath
        b2 = deepcopy(params.bath)
        assert_equal_and_not_id(self, b2.data, params.bath.data)
        assert_equal_and_not_id(self, b2.hvec, params.bath.hvec)
        assert_equal_and_not_id(self, b2.lambdavec, params.bath.lambdavec)
        assert_equal_and_not_id(self, b2.V, params.bath.V)
        assert_equal_and_not_id(self, b2.l, params.bath.l)
        self.assertTrue(all(V_nu.base is b2.data for V_nu in b2.V))
        self.assertTrue(all(l_nu.base is b2.data for l_nu in b2.l))

        self.check_bath_arithmetics(params.bath, b2)
        self.check_bath_h5(params.bath, "nspin1")

    def test_parse_hamiltonian_nspin2(self):
        h_loc = self.make_H_loc(mul.outer(sz, self.h_loc))
        h_int = self.make_H_int()
        h_bath = self.make_H_bath(mul.outer(sz, self.h),
                                  mul.outer([1, -1], self.V))

        h = h_loc + h_int + h_bath
        params = parse_hamiltonian(
            h,
            self.fops_imp_up, self.fops_imp_dn,
            self.fops_bath_up, self.fops_bath_dn
        )

        self.assertEqual(params.ed_mode, EDMode.NORMAL)
        assert_allclose(params.Hloc, mul.outer(sz, self.h_loc))
        assert_allclose(params.Hloc_an, np.zeros((1, 1, self.norb, self.norb)))
        self.assertEqual(params.Hloc_op(self.fops_imp_up, self.fops_imp_dn),
                         h_loc)
        self.assertEqual(params.Hloc_an_op(self.fops_imp_up, self.fops_imp_dn),
                         op.Operator())
        self.assertTrue(isinstance(params.bath, BathGeneral))
        self.assertEqual(params.bath.nbath, self.nbath)
        self.assertFalse(hasattr(params.bath, 'Delta'))
        self.assertEqual(params.bath.nsym, 12)
        self.assertEqual(params.bath.hvec.shape,
                         (2, 2, self.norb, self.norb, 12))
        self.assertEqual(len(params.bath.l), self.nbath)
        self.assertEqual(len(params.bath.V), self.nbath)
        self.check_bath(params.bath.hvec, params.bath.l, params.bath.V,
                        mul.outer(sz, self.h), mul.outer([1, -1], self.V))
        self.assertFalse(hasattr(params.bath, 'U'))
        self.check_int_params(params)

        # Check deepcopy of bath
        b2 = deepcopy(params.bath)
        assert_equal_and_not_id(self, b2.data, params.bath.data)
        assert_equal_and_not_id(self, b2.hvec, params.bath.hvec)
        assert_equal_and_not_id(self, b2.lambdavec, params.bath.lambdavec)
        assert_equal_and_not_id(self, b2.V, params.bath.V)
        assert_equal_and_not_id(self, b2.l, params.bath.l)
        self.assertTrue(all(V_nu.base is b2.data for V_nu in b2.V))
        self.assertTrue(all(l_nu.base is b2.data for l_nu in b2.l))

        self.check_bath_arithmetics(params.bath, b2)
        self.check_bath_h5(params.bath, "nspin2")

    def test_parse_hamiltonian_nonsu2_hloc(self):
        h_loc = self.make_H_loc(mul.outer(sz + 0.2 * sx, self.h_loc))
        h_int = self.make_H_int()
        h_bath = self.make_H_bath(mul.outer(s0, self.h),
                                  mul.outer([1, 1], self.V))

        h = h_loc + h_int + h_bath
        params = parse_hamiltonian(
            h,
            self.fops_imp_up, self.fops_imp_dn,
            self.fops_bath_up, self.fops_bath_dn
        )

        self.assertEqual(params.ed_mode, EDMode.NONSU2)
        assert_allclose(params.Hloc, mul.outer(sz + 0.2 * sx, self.h_loc))
        assert_allclose(params.Hloc_an, np.zeros((1, 1, self.norb, self.norb)))
        self.assertEqual(params.Hloc_op(self.fops_imp_up, self.fops_imp_dn),
                         h_loc)
        self.assertEqual(params.Hloc_an_op(self.fops_imp_up, self.fops_imp_dn),
                         op.Operator())
        self.assertTrue(isinstance(params.bath, BathGeneral))
        self.assertEqual(params.bath.nbath, self.nbath)
        self.assertFalse(hasattr(params.bath, 'Delta'))
        self.assertEqual(params.bath.nsym, 12)
        self.assertEqual(params.bath.hvec.shape,
                         (2, 2, self.norb, self.norb, 12))
        self.assertEqual(len(params.bath.l), self.nbath)
        self.assertEqual(len(params.bath.V), self.nbath)
        self.check_bath(params.bath.hvec, params.bath.l, params.bath.V,
                        mul.outer(s0, self.h), mul.outer([1, 1], self.V))
        self.assertFalse(hasattr(params.bath, 'U'))
        self.check_int_params(params)

        # Check deepcopy of bath
        b2 = deepcopy(params.bath)
        assert_equal_and_not_id(self, b2.data, params.bath.data)
        assert_equal_and_not_id(self, b2.hvec, params.bath.hvec)
        assert_equal_and_not_id(self, b2.lambdavec, params.bath.lambdavec)
        assert_equal_and_not_id(self, b2.V, params.bath.V)
        assert_equal_and_not_id(self, b2.l, params.bath.l)
        self.assertTrue(all(V_nu.base is b2.data for V_nu in b2.V))
        self.assertTrue(all(l_nu.base is b2.data for l_nu in b2.l))

        self.check_bath_arithmetics(params.bath, b2)
        self.check_bath_h5(params.bath, "nonsu2_hloc")

    def test_parse_hamiltonian_nonsu2_bath(self):
        h_loc = self.make_H_loc(mul.outer(s0, self.h_loc))
        h_int = self.make_H_int()
        h_bath = self.make_H_bath(mul.outer(sz + 0.2 * sx, self.h),
                                  mul.outer([1, -1], self.V))

        h = h_loc + h_int + h_bath
        params = parse_hamiltonian(
            h,
            self.fops_imp_up, self.fops_imp_dn,
            self.fops_bath_up, self.fops_bath_dn
        )

        self.assertEqual(params.ed_mode, EDMode.NONSU2)
        assert_allclose(params.Hloc, mul.outer(s0, self.h_loc))
        assert_allclose(params.Hloc_an, np.zeros((1, 1, self.norb, self.norb)))
        self.assertEqual(params.Hloc_op(self.fops_imp_up, self.fops_imp_dn),
                         h_loc)
        self.assertEqual(params.Hloc_an_op(self.fops_imp_up, self.fops_imp_dn),
                         op.Operator())
        self.assertTrue(isinstance(params.bath, BathGeneral))
        self.assertEqual(params.bath.nbath, self.nbath)
        self.assertFalse(hasattr(params.bath, 'Delta'))
        self.assertEqual(params.bath.nsym, 21)
        self.assertEqual(params.bath.hvec.shape,
                         (2, 2, self.norb, self.norb, 21))
        self.assertEqual(len(params.bath.l), self.nbath)
        self.assertEqual(len(params.bath.V), self.nbath)
        self.check_bath(params.bath.hvec, params.bath.l, params.bath.V,
                        mul.outer(sz + 0.2 * sx, self.h),
                        mul.outer([1, -1], self.V))
        self.assertFalse(hasattr(params.bath, 'U'))
        self.check_int_params(params)

        # Check deepcopy of bath
        b2 = deepcopy(params.bath)
        assert_equal_and_not_id(self, b2.data, params.bath.data)
        assert_equal_and_not_id(self, b2.hvec, params.bath.hvec)
        assert_equal_and_not_id(self, b2.lambdavec, params.bath.lambdavec)
        assert_equal_and_not_id(self, b2.V, params.bath.V)
        assert_equal_and_not_id(self, b2.l, params.bath.l)
        self.assertTrue(all(V_nu.base is b2.data for V_nu in b2.V))
        self.assertTrue(all(l_nu.base is b2.data for l_nu in b2.l))

        self.check_bath_arithmetics(params.bath, b2)
        self.check_bath_h5(params.bath, "nonsu2_bath")

    def test_parse_hamiltonian_superc_hloc_an(self):
        h_loc = self.make_H_loc(mul.outer(s0, self.h_loc))
        h_int = self.make_H_int()
        h_bath = self.make_H_bath(mul.outer(s0, self.h),
                                  mul.outer([1, 1], self.V))

        Delta = np.array([[0.1, 0.04, 0.05j],
                          [0.04, 0.2, 0.06],
                          [0.05j, 0.06, 0.3]])
        h_loc_an = self.make_H_loc_an(Delta)

        h = h_loc + h_loc_an + h_int + h_bath
        params = parse_hamiltonian(
            h,
            self.fops_imp_up, self.fops_imp_dn,
            self.fops_bath_up, self.fops_bath_dn
        )

        self.assertEqual(params.ed_mode, EDMode.SUPERC)
        assert_allclose(params.Hloc,
                        self.h_loc.reshape(1, 1, self.norb, self.norb))
        assert_allclose(params.Hloc_an,
                        Delta.reshape(1, 1, self.norb, self.norb))
        self.assertEqual(params.Hloc_op(self.fops_imp_up, self.fops_imp_dn),
                         h_loc)
        self.assertEqual(params.Hloc_an_op(self.fops_imp_up, self.fops_imp_dn),
                         h_loc_an)
        self.assertTrue(isinstance(params.bath, BathGeneral))
        self.assertEqual(params.bath.nbath, self.nbath)
        self.assertEqual(params.bath.nsym, 6)
        self.assertEqual(params.bath.hvec.shape,
                         (2, 2, self.norb, self.norb, 6))
        self.assertEqual(len(params.bath.l), self.nbath)
        self.assertEqual(len(params.bath.V), self.nbath)

        h_ref = np.zeros((2, 2, self.norb, self.norb, self.nbath),
                         dtype=complex)
        for nu in self.bsites:
            h_ref[0, 0, ..., nu] = self.h[:, :, nu]
            h_ref[1, 1, ..., nu] = -self.h[:, :, nu]

        self.check_bath(params.bath.hvec, params.bath.l, params.bath.V,
                        h_ref, mul.outer([1, 1], self.V), is_nambu=True)
        self.assertFalse(hasattr(params.bath, 'U'))
        self.check_int_params(params)

        # Check deepcopy of bath
        b2 = deepcopy(params.bath)
        assert_equal_and_not_id(self, b2.data, params.bath.data)
        assert_equal_and_not_id(self, b2.hvec, params.bath.hvec)
        assert_equal_and_not_id(self, b2.lambdavec, params.bath.lambdavec)
        assert_equal_and_not_id(self, b2.V, params.bath.V)
        assert_equal_and_not_id(self, b2.l, params.bath.l)
        self.assertTrue(all(V_nu.base is b2.data for V_nu in b2.V))
        self.assertTrue(all(l_nu.base is b2.data for l_nu in b2.l))

        self.check_bath_arithmetics(params.bath, b2)
        self.check_bath_h5(params.bath, "superc")

    def test_parse_hamiltonian_superc_bath(self):
        h_loc = self.make_H_loc(mul.outer(s0, self.h_loc))
        h_int = self.make_H_int()
        h_bath = self.make_H_bath(mul.outer(s0, self.h),
                                  mul.outer([1, 1], self.V))

        h_sc = sum(self.Delta[o1, o2, nu]
                   * op.c_dag('B_up', nu * self.norb + o1)
                   * op.c_dag('B_dn', nu * self.norb + o2)
                   for o1, o2, nu in product(self.orbs, self.orbs, self.bsites))
        h_sc = h_sc + op.dagger(h_sc)

        h = h_loc + h_int + h_bath + h_sc
        params = parse_hamiltonian(
            h,
            self.fops_imp_up, self.fops_imp_dn,
            self.fops_bath_up, self.fops_bath_dn
        )

        self.assertEqual(params.ed_mode, EDMode.SUPERC)
        assert_allclose(params.Hloc,
                        self.h_loc.reshape(1, 1, self.norb, self.norb))
        assert_allclose(params.Hloc_an, np.zeros((1, 1, self.norb, self.norb)))
        self.assertEqual(params.Hloc_op(self.fops_imp_up, self.fops_imp_dn),
                         h_loc)
        self.assertEqual(params.Hloc_an_op(self.fops_imp_up, self.fops_imp_dn),
                         op.Operator())
        self.assertTrue(isinstance(params.bath, BathGeneral))
        self.assertEqual(params.bath.nbath, self.nbath)
        self.assertEqual(params.bath.nsym, 15)
        self.assertEqual(params.bath.hvec.shape,
                         (2, 2, self.norb, self.norb, 15))
        self.assertEqual(len(params.bath.l), self.nbath)
        self.assertEqual(len(params.bath.V), self.nbath)

        h_ref = np.zeros((2, 2, self.norb, self.norb, self.nbath),
                         dtype=complex)
        for nu in self.bsites:
            h_ref[0, 0, ..., nu] = self.h[:, :, nu]
            h_ref[1, 1, ..., nu] = -self.h[:, :, nu]
            h_ref[0, 1, ..., nu] = self.Delta[:, :, nu]
            h_ref[1, 0, ..., nu] = np.conj(self.Delta[:, :, nu].T)

        self.check_bath(params.bath.hvec, params.bath.l, params.bath.V,
                        h_ref, mul.outer([1, 1], self.V), is_nambu=True)
        self.assertFalse(hasattr(params.bath, 'U'))
        self.check_int_params(params)

        # Check deepcopy of bath
        b2 = deepcopy(params.bath)
        assert_equal_and_not_id(self, b2.data, params.bath.data)
        assert_equal_and_not_id(self, b2.hvec, params.bath.hvec)
        assert_equal_and_not_id(self, b2.lambdavec, params.bath.lambdavec)
        assert_equal_and_not_id(self, b2.V, params.bath.V)
        assert_equal_and_not_id(self, b2.l, params.bath.l)
        self.assertTrue(all(V_nu.base is b2.data for V_nu in b2.V))
        self.assertTrue(all(l_nu.base is b2.data for l_nu in b2.l))

        self.check_bath_arithmetics(params.bath, b2)
        self.check_bath_h5(params.bath, "superc")


if __name__ == '__main__':
    unittest.main()

import unittest
from itertools import product

import numpy as np
from numpy.testing import assert_equal, assert_allclose
from numpy import multiply as mul

import triqs.operators as op
from triqs.operators.util.hamiltonians import h_int_kanamori

from edipack2triqs.hamiltonian import parse_hamiltonian


s0 = np.eye(2)
sx = np.array([[0, 1], [1, 0]])
sz = np.array([[1, 0], [0, -1]])


class TestHamiltonian(unittest.TestCase):

    spins = ('up', 'dn')
    orbs = range(3)
    fops_imp_up = [('up', o) for o in orbs]
    fops_imp_dn = [('dn', o) for o in orbs]

    # Normal bath
    fops_bath_n_up = [('B_up', nu * 3 + o) for nu, o in product(range(2), orbs)]
    fops_bath_n_dn = [('B_dn', nu * 3 + o) for nu, o in product(range(2), orbs)]

    # Hybrid bath
    fops_bath_h_up = [('B_up', nu) for nu in range(4)]
    fops_bath_h_dn = [('B_dn', nu) for nu in range(4)]

    # Interaction parameters for make_H_int()
    Uloc = np.array([1.0, 2.0, 3.0, .0, .0])
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
    def make_H_bath(cls, eps, V, mkind):
        h_bath = sum(
            eps[s, *ind]
            * op.c_dag("B_" + spin, mkind(*ind))
            * op.c("B_" + spin, mkind(*ind))
            for (s, spin), *ind
            in product(enumerate(cls.spins), *map(range, eps.shape[1:]))
        )
        h_bath += sum(V[s1, s2, o, nu] * (
            op.c_dag(spin1, o) * op.c("B_" + spin2, mkind(o, nu))
            + op.c_dag("B_" + spin2, mkind(o, nu)) * op.c(spin1, o))
            for (s1, spin1), (s2, spin2), o, nu
            in product(enumerate(cls.spins), enumerate(cls.spins),
                       range(V.shape[2]), range(V.shape[3]))
        )
        return h_bath

    @classmethod
    def check_int_params(cls, params):
        assert_allclose(params.Uloc, cls.Uloc)
        assert_allclose(params.Ust, cls.Ust)
        assert_allclose(params.Jh, cls.Jh)
        assert_allclose(params.Jx, cls.Jx)
        assert_allclose(params.Jp, cls.Jp)

    def test_h_int_kanamori(self):
        Uloc = [1.0, 2.0, 3.0]
        Ust = 0.6
        J = 0.15

        Ud_mat = (Ust - J) * (np.ones((3, 3)) - np.eye(3))
        Uod_mat = np.diag(Uloc) + Ust * (np.ones((3, 3)) - np.eye(3))

        h = h_int_kanamori(self.spins, 3, Ud_mat, Uod_mat, J, off_diag=True)

        params = parse_hamiltonian(
            h, self.fops_imp_up, self.fops_imp_dn, [], []
        )

        assert_allclose(params.Uloc[:3], Uloc)
        assert_allclose(params.Ust, Ust)
        assert_allclose(params.Jh, J)
        assert_allclose(params.Jx, J)
        assert_allclose(params.Jp, J)

    def test_parse_hamiltonian_nspin1_normal(self):
        h_loc = np.diag([1.5, 2.0, 2.5])
        h = self.make_H_loc(mul.outer(s0, h_loc)) + self.make_H_int()

        eps = np.array([[-0.1, 0.1],
                        [-0.2, 0.2],
                        [-0.3, 0.3]])
        V = np.array([[0.4, 0.7],
                      [0.0, 0.5],
                      [0.7, 0.0]])

        def bath_mkind(o, nu):
            return nu * 3 + o
        h += self.make_H_bath(mul.outer([1, 1], eps),
                              mul.outer(s0, V),
                              bath_mkind)

        params = parse_hamiltonian(
            h,
            self.fops_imp_up, self.fops_imp_dn,
            self.fops_bath_n_up, self.fops_bath_n_dn
        )

        self.assertEqual(params.ed_mode, "normal")
        assert_allclose(params.Hloc, h_loc.reshape((1, 1, 3, 3)))
        b = params.bath
        self.assertEqual(b.nbath, 2)
        self.assertEqual(b.name, "normal")
        self.assertEqual(b.eps.shape, (1, 3, 2))
        self.assertFalse(hasattr(b, 'Delta'))
        self.assertEqual(b.V.shape, (1, 3, 2))
        self.assertFalse(hasattr(b, 'U'))
        # Check connected bath states
        for o in self.orbs:
            self.assertEqual(
                set((e, v) for e, v in zip(b.eps[0, o], b.V[0, o]) if v != 0),
                set((e, v) for e, v in zip(eps[o], V[o]) if v != 0)
            )
        # Check disconnected bath states
        self.assertEqual(
            set(e for e, v in zip(b.eps[0].flat, b.V[0].flat) if v == 0),
            set(e for e, v in zip(eps.flat, V.flat) if v == 0)
        )
        self.check_int_params(params)

    def test_parse_hamiltonian_nspin2_normal(self):
        h_loc = np.diag([1.5, 2.0, 2.5])
        h = self.make_H_loc(mul.outer(sz, h_loc)) + self.make_H_int()

        eps = np.array([[-0.1, 0.1],
                        [-0.2, 0.2],
                        [-0.3, 0.3]])
        V = np.array([[0.4, 0.7],
                      [0.0, 0.5],
                      [0.7, 0.0]])

        def bath_mkind(o, nu):
            return nu * 3 + o
        h += self.make_H_bath(mul.outer([1, -1], eps),
                              mul.outer(sz, V),
                              bath_mkind)

        params = parse_hamiltonian(
            h,
            self.fops_imp_up, self.fops_imp_dn,
            self.fops_bath_n_up, self.fops_bath_n_dn
        )

        self.assertEqual(params.ed_mode, "normal")
        assert_allclose(params.Hloc, mul.outer(sz, h_loc))
        b = params.bath
        self.assertEqual(b.nbath, 2)
        self.assertEqual(b.name, "normal")
        self.assertEqual(b.eps.shape, (2, 3, 2))
        self.assertFalse(hasattr(b, 'Delta'))
        self.assertEqual(b.V.shape, (2, 3, 2))
        self.assertFalse(hasattr(b, 'U'))
        for s in range(2):
            eps_s = eps * (1 - 2 * s)
            V_s = V * (1 - 2 * s)
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

    def test_parse_hamiltonian_nonsu2_hloc_normal(self):
        h_loc = np.diag([1.5, 2.0, 2.5])
        h = self.make_H_loc(mul.outer(sz + 0.2 * sx, h_loc)) + self.make_H_int()

        eps = np.array([[-0.1, 0.1],
                        [-0.2, 0.2],
                        [-0.3, 0.3]])
        V = np.array([[0.4, 0.7],
                      [0.0, 0.5],
                      [0.7, 0.0]])

        def bath_mkind(o, nu):
            return nu * 3 + o
        h += self.make_H_bath(mul.outer([1, 1], eps),
                              mul.outer(s0, V),
                              bath_mkind)

        params = parse_hamiltonian(
            h,
            self.fops_imp_up, self.fops_imp_dn,
            self.fops_bath_n_up, self.fops_bath_n_dn
        )

        self.assertEqual(params.ed_mode, "nonsu2")
        assert_allclose(params.Hloc, mul.outer(sz + 0.2 * sx, h_loc))
        b = params.bath
        self.assertEqual(b.nbath, 2)
        self.assertEqual(b.name, "normal")
        self.assertEqual(b.eps.shape, (2, 3, 2))
        self.assertFalse(hasattr(b, 'Delta'))
        self.assertEqual(b.V.shape, (2, 3, 2))
        assert_equal(b.U, np.zeros((2, 3, 2)))
        for s in range(2):
            # Check connected bath states
            for o in self.orbs:
                self.assertEqual(
                    set((e, v) for e, v
                        in zip(b.eps[s, o], b.V[s, o]) if v != 0),
                    set((e, v) for e, v in zip(eps[o], V[o]) if v != 0)
                )
            # Check disconnected bath states
            self.assertEqual(
                set(e for e, v in zip(b.eps[s].flat, b.V[s].flat) if v == 0),
                set(e for e, v in zip(eps.flat, V.flat) if v == 0)
            )
        self.check_int_params(params)

    def test_parse_hamiltonian_nonsu2_bath_normal(self):
        h_loc = np.diag([1.5, 2.0, 2.5])
        h = self.make_H_loc(mul.outer(s0, h_loc)) + self.make_H_int()

        eps = np.array([[-0.1, 0.1],
                        [-0.2, 0.2],
                        [-0.3, 0.3]])
        V = np.array([[0.4, 0.7],
                      [0.0, 0.5],
                      [0.7, 0.0]])

        def bath_mkind(o, nu):
            return nu * 3 + o
        h += self.make_H_bath(mul.outer([1, -1], eps),
                              mul.outer(sz + 0.2 * sx, V),
                              bath_mkind)

        params = parse_hamiltonian(
            h,
            self.fops_imp_up, self.fops_imp_dn,
            self.fops_bath_n_up, self.fops_bath_n_dn
        )

        self.assertEqual(params.ed_mode, "nonsu2")
        assert_allclose(params.Hloc, mul.outer(s0, h_loc))
        b = params.bath
        self.assertEqual(b.nbath, 2)
        self.assertEqual(b.name, "normal")
        self.assertEqual(b.eps.shape, (2, 3, 2))
        self.assertFalse(hasattr(b, 'Delta'))
        self.assertEqual(b.V.shape, (2, 3, 2))
        self.assertEqual(b.U.shape, (2, 3, 2))
        for s in range(2):
            eps_s = eps * (1 - 2 * s)
            V_s = V * (1 - 2 * s)
            # Check connected bath states
            for o in self.orbs:
                self.assertEqual(
                    set((e, v, u) for e, v, u
                        in zip(b.eps[s, o], b.V[s, o], b.U[s, o]) if v != 0),
                    set((e, v, u) for e, v, u
                        in zip(eps_s[o], V_s[o], 0.2 * V[o]) if v != 0)
                )
            # Check disconnected bath states
            self.assertEqual(
                set(e for e, v in zip(b.eps[s].flat, b.V[s].flat) if v == 0),
                set(e for e, v in zip(eps_s.flat, V_s.flat) if v == 0)
            )
        self.check_int_params(params)

    def test_parse_hamiltonian_superc_normal(self):
        h_loc = np.diag([1.5, 2.0, 2.5])
        h = self.make_H_loc(mul.outer(s0, h_loc)) + self.make_H_int()

        eps = np.array([[-0.1, 0.1],
                        [-0.2, 0.2],
                        [-0.3, 0.3]])
        V = np.array([[0.4, 0.7],
                      [0.0, 0.5],
                      [0.7, 0.0]])

        def bath_mkind(o, nu):
            return nu * 3 + o
        h += self.make_H_bath(mul.outer([1, 1], eps),
                              mul.outer(s0, V),
                              bath_mkind)

        Delta = np.array([[0.6, 0.7],
                          [0.8, 0.6],
                          [0.9, 0.7]])
        h += sum(Delta[o, nu] * (op.c_dag('B_dn', bath_mkind(o, nu))
                                 * op.c_dag('B_up', bath_mkind(o, nu))
                                 + op.c('B_up', bath_mkind(o, nu))
                                 * op.c('B_dn', bath_mkind(o, nu)))
                 for o, nu in product(self.orbs, range(2)))

        params = parse_hamiltonian(
            h,
            self.fops_imp_up, self.fops_imp_dn,
            self.fops_bath_n_up, self.fops_bath_n_dn
        )

        self.assertEqual(params.ed_mode, "superc")
        assert_allclose(params.Hloc, h_loc.reshape(1, 1, 3, 3))
        b = params.bath
        self.assertEqual(b.nbath, 2)
        self.assertEqual(b.name, "normal")
        self.assertEqual(b.eps.shape, (1, 3, 2))
        self.assertEqual(b.Delta.shape, (1, 3, 2))
        self.assertEqual(b.V.shape, (1, 3, 2))
        self.assertFalse(hasattr(b, 'U'))
        # Check connected bath states
        for o in self.orbs:
            self.assertEqual(
                set((e, d, v) for e, d, v
                    in zip(b.eps[0, o], b.Delta[0, o], b.V[0, o])
                    if v != 0),
                set((e, d, v) for e, d, v
                    in zip(eps[o], Delta[o], V[o])
                    if v != 0)
            )
        # Check disconnected bath states
        self.assertEqual(
            set((e, d) for e, d, v
                in zip(b.eps[0].flat, b.Delta[0].flat, b.V[0].flat)
                if v == 0),
            set((e, d) for e, d, v
                in zip(eps.flat, Delta.flat, V.flat)
                if v == 0)
        )
        self.check_int_params(params)

    def test_parse_hamiltonian_nspin1_hybrid(self):
        h_loc = np.diag([1.5, 2.0, 2.5])
        h = self.make_H_loc(mul.outer(s0, h_loc)) + self.make_H_int()

        eps = np.array([-0.1, -0.2, -0.3, -0.4])
        V = np.array([[0.4, 0.7, 0.1, 0.4],
                      [0.0, 0.5, 0.2, 0.5],
                      [0.7, 0.4, 0.3, 0.0]])

        def bath_mkind(o, nu):
            return nu
        h += self.make_H_bath(mul.outer([1, 1], eps.reshape(1, 4)),
                              mul.outer(s0, V),
                              bath_mkind)

        params = parse_hamiltonian(
            h,
            self.fops_imp_up, self.fops_imp_dn,
            self.fops_bath_h_up, self.fops_bath_h_dn
        )

        self.assertEqual(params.ed_mode, "normal")
        assert_allclose(params.Hloc, h_loc.reshape((1, 1, 3, 3)))
        self.assertEqual(params.bath.nbath, 4)
        self.assertEqual(params.bath.name, "hybrid")
        assert_allclose(params.bath.eps, eps.reshape(1, 4))
        self.assertFalse(hasattr(params.bath, 'Delta'))
        assert_allclose(params.bath.V, V.reshape(1, 3, 4))
        self.assertFalse(hasattr(params.bath, 'U'))
        self.check_int_params(params)

    def test_parse_hamiltonian_nspin2_hybrid(self):
        h_loc = np.diag([1.5, 2.0, 2.5])
        h = self.make_H_loc(mul.outer(sz, h_loc)) + self.make_H_int()

        eps = np.array([-0.1, -0.2, -0.3, -0.4])
        V = np.array([[0.4, 0.7, 0.1, 0.4],
                      [0.0, 0.5, 0.2, 0.5],
                      [0.7, 0.4, 0.3, 0.0]])

        def bath_mkind(o, nu):
            return nu
        h += self.make_H_bath(mul.outer([1, -1], eps.reshape(1, 4)),
                              mul.outer(sz, V),
                              bath_mkind)

        params = parse_hamiltonian(
            h,
            self.fops_imp_up, self.fops_imp_dn,
            self.fops_bath_h_up, self.fops_bath_h_dn
        )

        self.assertEqual(params.ed_mode, "normal")
        assert_allclose(params.Hloc, mul.outer(sz, h_loc))
        self.assertEqual(params.bath.nbath, 4)
        self.assertEqual(params.bath.name, "hybrid")
        assert_allclose(params.bath.eps, mul.outer([1, -1], eps.reshape(4)))
        self.assertFalse(hasattr(params.bath, 'Delta'))
        assert_allclose(params.bath.V, mul.outer([1, -1], V.reshape(3, 4)))
        self.assertFalse(hasattr(params.bath, 'U'))
        self.check_int_params(params)

    def test_parse_hamiltonian_nonsu2_hloc_hybrid(self):
        h_loc = np.diag([1.5, 2.0, 2.5])
        h = self.make_H_loc(mul.outer(sz + 0.2 * sx, h_loc)) + self.make_H_int()

        eps = np.array([-0.1, -0.2, -0.3, -0.4])
        V = np.array([[0.4, 0.7, 0.1, 0.4],
                      [0.0, 0.5, 0.2, 0.5],
                      [0.7, 0.4, 0.3, 0.0]])

        def bath_mkind(o, nu):
            return nu
        h += self.make_H_bath(mul.outer([1, 1], eps.reshape(1, 4)),
                              mul.outer(s0, V),
                              bath_mkind)

        params = parse_hamiltonian(
            h,
            self.fops_imp_up, self.fops_imp_dn,
            self.fops_bath_h_up, self.fops_bath_h_dn
        )

        self.assertEqual(params.ed_mode, "nonsu2")
        assert_allclose(params.Hloc, mul.outer(sz + 0.2 * sx, h_loc))
        self.assertEqual(params.bath.nbath, 4)
        self.assertEqual(params.bath.name, "hybrid")
        assert_allclose(params.bath.eps, mul.outer([1, 1], eps))
        self.assertFalse(hasattr(params.bath, 'Delta'))
        assert_allclose(params.bath.V, mul.outer([1, 1], V))
        assert_allclose(params.bath.U, np.zeros((2, 3, 4)))
        self.check_int_params(params)

    def test_parse_hamiltonian_nonsu2_bath_hybrid(self):
        h_loc = np.diag([1.5, 2.0, 2.5])
        h = self.make_H_loc(mul.outer(s0, h_loc)) + self.make_H_int()

        eps = np.array([-0.1, -0.2, -0.3, -0.4])
        V = np.array([[0.4, 0.7, 0.1, 0.4],
                      [0.0, 0.5, 0.2, 0.5],
                      [0.7, 0.4, 0.3, 0.0]])

        def bath_mkind(o, nu):
            return nu
        h += self.make_H_bath(mul.outer([1, -1], eps.reshape(1, 4)),
                              mul.outer(sz + 0.2 * sx, V),
                              bath_mkind)

        params = parse_hamiltonian(
            h,
            self.fops_imp_up, self.fops_imp_dn,
            self.fops_bath_h_up, self.fops_bath_h_dn
        )

        self.assertEqual(params.ed_mode, "nonsu2")
        assert_allclose(params.Hloc, mul.outer(s0, h_loc))
        self.assertEqual(params.bath.nbath, 4)
        self.assertEqual(params.bath.name, "hybrid")
        assert_allclose(params.bath.eps, mul.outer([1, -1], eps))
        self.assertFalse(hasattr(params.bath, 'Delta'))
        assert_allclose(params.bath.V, mul.outer([1, -1], V))
        assert_allclose(params.bath.U, mul.outer([0.2, 0.2], V))

    def test_parse_hamiltonian_superc_hybrid(self):
        h_loc = np.diag([1.5, 2.0, 2.5])
        h = self.make_H_loc(mul.outer(s0, h_loc)) + self.make_H_int()

        eps = np.array([-0.1, -0.2, -0.3, -0.4])
        V = np.array([[0.4, 0.7, 0.1, 0.4],
                      [0.0, 0.5, 0.2, 0.5],
                      [0.7, 0.4, 0.3, 0.0]])

        def bath_mkind(o, nu):
            return nu
        h += self.make_H_bath(mul.outer([1, 1], eps.reshape(1, 4)),
                              mul.outer(s0, V),
                              bath_mkind)

        Delta = np.array([0.6, 0.7, 0.8, 0.9])
        h += sum(Delta[nu] * (op.c_dag('B_dn', nu) * op.c_dag('B_up', nu)
                              + op.c('B_up', nu) * op.c('B_dn', nu))
                 for nu in range(4))

        params = parse_hamiltonian(
            h,
            self.fops_imp_up, self.fops_imp_dn,
            self.fops_bath_h_up, self.fops_bath_h_dn
        )

        self.assertEqual(params.ed_mode, "superc")
        assert_allclose(params.Hloc, h_loc.reshape(1, 1, 3, 3))
        self.assertEqual(params.bath.nbath, 4)
        self.assertEqual(params.bath.name, "hybrid")
        assert_allclose(params.bath.eps, eps.reshape(1, 4))
        assert_allclose(params.bath.Delta, Delta.reshape(1, 4))
        assert_allclose(params.bath.V, V.reshape(1, 3, 4))
        self.assertFalse(hasattr(params.bath, 'U'))
        self.check_int_params(params)

    # TODO
    # test_parse_hamiltonian_nspin1_replica()
    # test_parse_hamiltonian_nspin2_replica()
    # test_parse_hamiltonian_nonsu2_hloc_replica()
    # test_parse_hamiltonian_nonsu2_bath_replica()
    # test_parse_hamiltonian_superc_replica()


if __name__ == '__main__':
    unittest.main()

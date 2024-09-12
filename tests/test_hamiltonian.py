import unittest
from itertools import product

import numpy as np
from numpy.testing import assert_allclose
from numpy import multiply as mul

import triqs.operators as op
from triqs.operators.util.hamiltonians import h_int_kanamori

from edipack2triqs.hamiltonian import parse_hamiltonian


sz = np.array([[1, 0], [0, -1]])


class TestHamiltonian(unittest.TestCase):

    # Interaction parameters for make_H_int()
    spins = ('up', 'dn')
    fops_imp_up = [('up', o) for o in range(3)]
    fops_imp_dn = [('dn', o) for o in range(3)]

    Uloc = np.array([1.0, 2.0, 3.0, .0, .0])
    Ust = 0.6
    Jh = 0.15
    Jx = 0.01
    Jp = 0.03

    @classmethod
    def make_H_loc(cls, h_loc):
        orbs = range(h_loc.shape[1])
        return sum(h_loc[s, o1, o2] * op.c_dag(spin, o1) * op.c(spin, o2)
                   for (s, spin), o1, o2
                   in product(enumerate(cls.spins), orbs, orbs))

    @classmethod
    def make_H_int(cls):
        orbs = range(3)
        h_int = sum(cls.Uloc[o] * op.n('up', o) * op.n('dn', o) for o in orbs)
        h_int += cls.Ust * sum(int(o1 != o2) * op.n('up', o1) * op.n('dn', o2)
                               for o1, o2 in product(orbs, orbs))
        h_int += (cls.Ust - cls.Jh) * \
            sum(int(o1 < o2) * op.n(s, o1) * op.n(s, o2)
                for s, o1, o2 in product(cls.spins, orbs, orbs))
        h_int -= cls.Jx * sum(int(o1 != o2)
                              * op.c_dag('up', o1) * op.c('dn', o1)
                              * op.c_dag('dn', o2) * op.c('up', o2)
                              for o1, o2 in product(orbs, orbs))
        h_int += cls.Jp * sum(int(o1 != o2)
                              * op.c_dag('up', o1) * op.c_dag('dn', o1)
                              * op.c('dn', o2) * op.c('up', o2)
                              for o1, o2 in product(orbs, orbs))
        return h_int

    @classmethod
    def check_int_params(cls, params):
        assert_allclose(params.Uloc, cls.Uloc)
        assert_allclose(params.Ust, cls.Ust)
        assert_allclose(params.Jh, cls.Jh)
        assert_allclose(params.Jx, cls.Jx)
        assert_allclose(params.Jp, cls.Jp)

    @classmethod
    def make_bath_set(cls, eps, V):
        assert eps.shape == V.shape
        bath_set = set()
        for s, orb, nu in np.ndindex(*eps.shape):
            e, v = eps[s, orb, nu], V[s, orb, nu]
            if v != 0:
                bath_set.add((s, orb, e, v))
            else:  # Decoupled bath site
                bath_set.add((s, e, v))
        return bath_set

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
        orbs = range(3)

        h_loc = np.diag([1.5, 2.0, 2.5])
        h = self.make_H_loc(mul.outer([1, 1], h_loc)) + self.make_H_int()

        fops_bath_up = [('B_up', nu * 3 + o)
                        for nu, o in product(range(2), orbs)]
        fops_bath_dn = [('B_dn', nu * 3 + o)
                        for nu, o in product(range(2), orbs)]
        eps = np.array([[-0.1, 0.1],
                        [-0.2, 0.2],
                        [-0.3, 0.3]])
        V = np.array([[0.4, 0.7],
                      [0.0, 0.5],
                      [0.7, 0.0]])
        h += sum(eps[o, nu]
                 * op.c_dag("B_" + s, nu * 3 + o) * op.c("B_" + s, nu * 3 + o)
                 for nu, s, o in product(range(2), self.spins, orbs))
        h += sum(V[o, nu] * (
                 op.c_dag(s, o) * op.c("B_" + s, nu * 3 + o)
                 + op.c_dag("B_" + s, nu * 3 + o) * op.c(s, o))
                 for nu, s, o in product(range(2), self.spins, orbs))

        params = parse_hamiltonian(
            h, self.fops_imp_up, self.fops_imp_dn, fops_bath_up, fops_bath_dn
        )

        assert_allclose(params.Hloc, h_loc.reshape((1, 1, 3, 3)))
        self.assertEqual(params.bath.nbath, 2)
        self.assertEqual(params.bath.name, "normal")
        self.assertEqual(
            self.make_bath_set(params.bath.eps, params.bath.V),
            self.make_bath_set(eps.reshape(1, 3, 2), V.reshape(1, 3, 2))
        )
        self.check_int_params(params)

    def test_parse_hamiltonian_nspin2_normal(self):
        orbs = range(3)

        h_loc = np.diag([1.5, 2.0, 2.5])
        h = self.make_H_loc(mul.outer([1, -1], h_loc)) + self.make_H_int()

        fops_bath_up = [('B_up', nu * 3 + o)
                        for nu, o in product(range(2), orbs)]
        fops_bath_dn = [('B_dn', nu * 3 + o)
                        for nu, o in product(range(2), orbs)]
        eps = np.array([[-0.1, 0.1],
                        [-0.2, 0.2],
                        [-0.3, 0.3]])
        V = np.array([[0.4, 0.7],
                      [0.0, 0.5],
                      [0.7, 0.0]])
        h += sum((1 if s == 'up' else -1) * eps[o, nu]
                 * op.c_dag("B_" + s, nu * 3 + o) * op.c("B_" + s, nu * 3 + o)
                 for nu, s, o in product(range(2), self.spins, orbs))
        h += sum((1 if s == 'up' else -1) * V[o, nu] * (
                 op.c_dag(s, o) * op.c("B_" + s, nu * 3 + o)
                 + op.c_dag("B_" + s, nu * 3 + o) * op.c(s, o))
                 for nu, s, o in product(range(2), self.spins, orbs))

        params = parse_hamiltonian(
            h, self.fops_imp_up, self.fops_imp_dn, fops_bath_up, fops_bath_dn
        )

        assert_allclose(params.Hloc, mul.outer(sz, h_loc))
        self.assertEqual(params.bath.nbath, 2)
        self.assertEqual(params.bath.name, "normal")
        self.assertEqual(
            self.make_bath_set(params.bath.eps, params.bath.V),
            self.make_bath_set(mul.outer([1, -1], eps), mul.outer([1, -1], V))
        )
        self.check_int_params(params)

    def test_parse_hamiltonian_nspin1_hybrid(self):
        orbs = range(3)

        h_loc = np.diag([1.5, 2.0, 2.5])
        h = self.make_H_loc(mul.outer([1, 1], h_loc)) + self.make_H_int()

        fops_bath_up = [('B_up', nu) for nu in range(4)]
        fops_bath_dn = [('B_dn', nu) for nu in range(4)]
        eps = np.array([-0.1, -0.2, -0.3, -0.4])
        V = np.array([[0.4, 0.7, 0.1, 0.4],
                      [0.0, 0.5, 0.2, 0.5],
                      [0.7, 0.4, 0.3, 0.0]])
        h += sum(eps[nu] * op.c_dag("B_" + s, nu) * op.c("B_" + s, nu)
                 for nu, s in product(range(4), self.spins))
        h += sum(V[o, nu] * (
                 op.c_dag(s, o) * op.c("B_" + s, nu)
                 + op.c_dag("B_" + s, nu) * op.c(s, o))
                 for nu, s, o in product(range(4), self.spins, orbs))

        params = parse_hamiltonian(
            h, self.fops_imp_up, self.fops_imp_dn, fops_bath_up, fops_bath_dn
        )

        assert_allclose(params.Hloc, h_loc.reshape((1, 1, 3, 3)))
        self.assertEqual(params.bath.nbath, 4)
        self.assertEqual(params.bath.name, "hybrid")
        assert_allclose(params.bath.eps, eps.reshape(1, 4))
        assert_allclose(params.bath.V, V.reshape(1, 3, 4))
        self.check_int_params(params)

    def test_parse_hamiltonian_nspin2_hybrid(self):
        orbs = range(3)

        h_loc = np.diag([1.5, 2.0, 2.5])
        h = self.make_H_loc(mul.outer([1, -1], h_loc)) + self.make_H_int()

        fops_bath_up = [('B_up', nu) for nu in range(4)]
        fops_bath_dn = [('B_dn', nu) for nu in range(4)]
        eps = np.array([-0.1, -0.2, -0.3, -0.4])
        V = np.array([[0.4, 0.7, 0.1, 0.4],
                      [0.0, 0.5, 0.2, 0.5],
                      [0.7, 0.4, 0.3, 0.0]])
        h += sum((1 if s == 'up' else -1)
                 * eps[nu] * op.c_dag("B_" + s, nu) * op.c("B_" + s, nu)
                 for nu, s in product(range(4), self.spins))
        h += sum((1 if s == 'up' else -1) * V[o, nu]
                 * (op.c_dag(s, o) * op.c("B_" + s, nu)
                 + op.c_dag("B_" + s, nu) * op.c(s, o))
                 for nu, s, o in product(range(4), self.spins, orbs))

        params = parse_hamiltonian(
            h, self.fops_imp_up, self.fops_imp_dn, fops_bath_up, fops_bath_dn
        )

        assert_allclose(params.Hloc, mul.outer(sz, h_loc))
        self.assertEqual(params.bath.nbath, 4)
        self.assertEqual(params.bath.name, "hybrid")
        assert_allclose(params.bath.eps, mul.outer([1, -1], eps.reshape(4)))
        assert_allclose(params.bath.V, mul.outer([1, -1], V.reshape(3, 4)))
        self.check_int_params(params)

    # TODO
    # test_parse_hamiltonian_nspin1_replica()
    # test_parse_hamiltonian_nspin2_replica()


if __name__ == '__main__':
    unittest.main()

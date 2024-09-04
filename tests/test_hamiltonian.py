import unittest
import numpy as np

from itertools import product
from numpy.testing import assert_allclose

import triqs.operators as op
from triqs.operators.util.hamiltonians import h_int_kanamori

from edipack2triqs.hamiltonian import parse_hamiltonian


class TestHamiltonian(unittest.TestCase):

    # Interaction parameters for make_H_int()
    norb = 3
    spins = ('up', 'dn')
    fops_imp_up = [('up', o) for o in range(norb)]
    fops_imp_dn = [('dn', o) for o in range(norb)]

    Uloc = np.array([1.0, 2.0, 3.0, .0, .0])
    Ust = 0.6
    Jh = 0.15
    Jx = 0.01
    Jp = 0.03

    @classmethod
    def make_H_int(cls):
        orbs = range(cls.norb)
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
        assert_allclose(params.Uloc, cls.Uloc, atol=1e-10)
        assert_allclose(params.Ust, cls.Ust, atol=1e-10)
        assert_allclose(params.Jh, cls.Jh, atol=1e-10)
        assert_allclose(params.Jx, cls.Jx, atol=1e-10)
        assert_allclose(params.Jp, cls.Jp, atol=1e-10)

    def test_h_int_kanamori(self):
        Uloc = [1.0, 2.0, 3.0]
        Ust = 0.6
        J = 0.15

        Ud_mat = (Ust - J) * (
            np.ones((self.norb, self.norb)) - np.eye(self.norb)
        )
        Uod_mat = np.diag(Uloc) \
            + Ust * (np.ones((self.norb, self.norb)) - np.eye(self.norb))

        h = h_int_kanamori(self.spins,
                           self.norb,
                           Ud_mat, Uod_mat, J,
                           off_diag=True)

        params = parse_hamiltonian(h,
                                   self.fops_imp_up, self.fops_imp_dn, [], [])

        assert_allclose(params.Uloc[:3], Uloc, atol=1e-10)
        assert_allclose(params.Ust, Ust, atol=1e-10)
        assert_allclose(params.Jh, J, atol=1e-10)
        assert_allclose(params.Jx, J, atol=1e-10)
        assert_allclose(params.Jp, J, atol=1e-10)

    def test_parse_hamiltonian_nspin1_normal(self):
        orbs = range(self.norb)

        h_int = self.make_H_int()

        h_loc = np.array([[1.5, 0.0, 0.0],
                          [0.0, 2.0, 0.0],
                          [0.0, 0.0, 2.5]])
        h = h_int + sum(h_loc[o1, o2] * op.c_dag(s, o1) * op.c(s, o2)
                        for s, o1, o2 in product(self.spins, orbs, orbs))

        nbath = 2
        fops_bath_up = [('B_up', nu * self.norb + o)
                        for nu, o in product(range(nbath), orbs)]
        fops_bath_dn = [('B_dn', nu * self.norb + o)
                        for nu, o in product(range(nbath), orbs)]
        eps = [[-0.1, -0.2, -0.3], [0.1, 0.2, 0.3]]
        V = [[0.4, 0.0, 0.7], [0.7, 0.5, 0.4]]
        h += sum(eps[nu][o]
                 * op.c_dag("B_" + s, nu * self.norb + o)
                 * op.c("B_" + s, nu * self.norb + o)
                 for nu, s, o in product(range(nbath), self.spins, orbs))
        h += sum(V[nu][o] * (
                 op.c_dag(s, o) * op.c("B_" + s, nu * self.norb + o)
                 + op.c_dag("B_" + s, nu * self.norb + o) * op.c(s, o))
                 for nu, s, o in product(range(nbath), self.spins, orbs))

        params = parse_hamiltonian(h,
                                   self.fops_imp_up,
                                   self.fops_imp_dn,
                                   fops_bath_up,
                                   fops_bath_dn)

        assert_allclose(params.Hloc,
                        h_loc.reshape((1, 1, self.norb, self.norb)))
        self.assertEqual(params.Nbath, nbath)
        self.assertEqual(params.bath_type, "normal")
        # TODO: check params.bath
        self.check_int_params(params)

    def test_parse_hamiltonian_nspin2_normal(self):
        orbs = range(self.norb)

        h_int = self.make_H_int()

        h_loc = np.array([[1.5, 0.0, 0.0],
                          [0.0, 2.0, 0.0],
                          [0.0, 0.0, 2.5]])
        h = h_int + sum((1 if s == 'up' else -1)
                        * h_loc[o1, o2] * op.c_dag(s, o1) * op.c(s, o2)
                        for s, o1, o2 in product(self.spins, orbs, orbs))

        nbath = 2
        fops_bath_up = [('B_up', nu * self.norb + o)
                        for nu, o in product(range(nbath), orbs)]
        fops_bath_dn = [('B_dn', nu * self.norb + o)
                        for nu, o in product(range(nbath), orbs)]
        eps = [[-0.1, -0.2, -0.3], [0.1, 0.2, 0.3]]
        V = [[0.4, 0.0, 0.7], [0.7, 0.5, 0.4]]
        h += sum((1 if s == 'up' else -1) * eps[nu][o]
                 * op.c_dag("B_" + s, nu * self.norb + o)
                 * op.c("B_" + s, nu * self.norb + o)
                 for nu, s, o in product(range(nbath), self.spins, orbs))
        h += sum((1 if s == 'up' else -1) * V[nu][o] * (
                 op.c_dag(s, o) * op.c("B_" + s, nu * self.norb + o)
                 + op.c_dag("B_" + s, nu * self.norb + o) * op.c(s, o))
                 for nu, s, o in product(range(nbath), self.spins, orbs))

        params = parse_hamiltonian(h,
                                   self.fops_imp_up,
                                   self.fops_imp_dn,
                                   fops_bath_up,
                                   fops_bath_dn)

        assert_allclose(params.Hloc,
                        np.multiply.outer(np.array([[1, 0], [0, -1]]), h_loc))
        self.assertEqual(params.Nbath, nbath)
        self.assertEqual(params.bath_type, "normal")
        # TODO: check params.bath
        self.check_int_params(params)

    def test_parse_hamiltonian_nspin1_hybrid(self):
        orbs = range(self.norb)

        h_int = self.make_H_int()

        h_loc = np.array([[1.5, 0.0, 0.0],
                          [0.0, 2.0, 0.0],
                          [0.0, 0.0, 2.5]])
        h = h_int + sum(h_loc[o1, o2] * op.c_dag(s, o1) * op.c(s, o2)
                        for s, o1, o2 in product(self.spins, orbs, orbs))

        nbath = 4
        fops_bath_up = [('B_up', nu) for nu in range(nbath)]
        fops_bath_dn = [('B_dn', nu) for nu in range(nbath)]
        eps = [-0.1, -0.2, -0.3, -0.4]
        V = [[0.4, 0.0, 0.7],
             [0.7, 0.5, 0.4],
             [0.1, 0.2, 0.3],
             [0.4, 0.5, 0.0]]
        h += sum(eps[nu] * op.c_dag("B_" + s, nu) * op.c("B_" + s, nu)
                 for nu, s in product(range(nbath), self.spins))
        h += sum(V[nu][o] * (
                 op.c_dag(s, o) * op.c("B_" + s, nu)
                 + op.c_dag("B_" + s, nu) * op.c(s, o))
                 for nu, s, o in product(range(nbath), self.spins, orbs))

        params = parse_hamiltonian(h,
                                   self.fops_imp_up,
                                   self.fops_imp_dn,
                                   fops_bath_up,
                                   fops_bath_dn)

        assert_allclose(params.Hloc,
                        h_loc.reshape((1, 1, self.norb, self.norb)))
        self.assertEqual(params.Nbath, nbath)
        self.assertEqual(params.bath_type, "hybrid")
        # TODO: check params.bath
        self.check_int_params(params)

    def test_parse_hamiltonian_nspin2_hybrid(self):
        orbs = range(self.norb)

        h_int = self.make_H_int()

        h_loc = np.array([[1.5, 0.0, 0.0],
                          [0.0, 2.0, 0.0],
                          [0.0, 0.0, 2.5]])
        h = h_int + sum((1 if s == 'up' else -1)
                        * h_loc[o1, o2] * op.c_dag(s, o1) * op.c(s, o2)
                        for s, o1, o2 in product(self.spins, orbs, orbs))

        nbath = 4
        fops_bath_up = [('B_up', nu) for nu in range(nbath)]
        fops_bath_dn = [('B_dn', nu) for nu in range(nbath)]
        eps = [-0.1, -0.2, -0.3, -0.4]
        V = [[0.4, 0.0, 0.7],
             [0.7, 0.5, 0.4],
             [0.1, 0.2, 0.3],
             [0.4, 0.5, 0.0]]
        h += sum((1 if s == 'up' else -1)
                 * eps[nu] * op.c_dag("B_" + s, nu) * op.c("B_" + s, nu)
                 for nu, s in product(range(nbath), self.spins))
        h += sum((1 if s == 'up' else -1) * V[nu][o]
                 * (op.c_dag(s, o) * op.c("B_" + s, nu)
                 + op.c_dag("B_" + s, nu) * op.c(s, o))
                 for nu, s, o in product(range(nbath), self.spins, orbs))

        params = parse_hamiltonian(h,
                                   self.fops_imp_up,
                                   self.fops_imp_dn,
                                   fops_bath_up,
                                   fops_bath_dn)

        assert_allclose(params.Hloc,
                        np.multiply.outer(np.array([[1, 0], [0, -1]]), h_loc))
        self.assertEqual(params.Nbath, nbath)
        self.assertEqual(params.bath_type, "hybrid")
        # TODO: check params.bath
        self.check_int_params(params)

    # TODO
    # test_parse_hamiltonian_nspin1_replica()
    # test_parse_hamiltonian_nspin2_replica()

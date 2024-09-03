import unittest
import numpy as np

from itertools import product
from numpy.testing import assert_allclose

import triqs.operators as op
from triqs.operators.util.hamiltonians import h_int_kanamori

from edipack2triqs.hamiltonian import parse_hamiltonian


class TestHamiltonian(unittest.TestCase):

    def test_h_int_kanamori(self):
        norb = 3
        spins = ('up', 'dn')
        fops_imp_up = [['up', o] for o in range(norb)]
        fops_imp_dn = [['dn', o] for o in range(norb)]

        Uloc = [1.0, 2.0, 3.0]
        Ust = 0.6
        J = 0.15

        U_mat = (Ust - J) * (np.ones((norb, norb)) - np.eye(norb))
        Up_mat = np.diag(Uloc) + Ust * (np.ones((norb, norb)) - np.eye(norb))

        h = h_int_kanamori(spins, norb, U_mat, Up_mat, J, off_diag=True)

        params = parse_hamiltonian(h, fops_imp_up, fops_imp_dn)

        assert_allclose(params.Uloc[:3], Uloc, atol=1e-10)
        assert_allclose(params.Ust, Ust, atol=1e-10)
        assert_allclose(params.Jh, J, atol=1e-10)
        assert_allclose(params.Jx, J, atol=1e-10)
        assert_allclose(params.Jp, J, atol=1e-10)

    def test_parse_hamiltonian(self):
        norb = 3
        spins = ('up', 'dn')
        orbs = range(norb)
        fops_imp_up = [['up', o] for o in range(norb)]
        fops_imp_dn = [['dn', o] for o in range(norb)]

        Uloc = np.array([1.0, 2.0, 3.0, .0, .0])
        Ust = 0.6
        Jh = 0.15
        Jx = 0.01
        Jp = 0.03

        h = sum(Uloc[o] * op.n('up', o) * op.n('dn', o) for o in orbs)
        h += Ust * sum(int(o1 != o2) * op.n('up', o1) * op.n('dn', o2)
                       for o1, o2 in product(orbs, orbs))
        h += (Ust - Jh) * sum(int(o1 < o2) * op.n(s, o1) * op.n(s, o2)
                              for s, o1, o2 in product(spins, orbs, orbs))
        h -= Jx * sum(int(o1 != o2)
                      * op.c_dag('up', o1) * op.c('dn', o1)
                      * op.c_dag('dn', o2) * op.c('up', o2)
                      for o1, o2 in product(orbs, orbs))
        h += Jp * sum(int(o1 != o2)
                      * op.c_dag('up', o1) * op.c_dag('dn', o1)
                      * op.c('dn', o2) * op.c('up', o2)
                      for o1, o2 in product(orbs, orbs))

        params = parse_hamiltonian(h, fops_imp_up, fops_imp_dn)

        assert_allclose(params.Uloc, Uloc, atol=1e-10)
        assert_allclose(params.Ust, Ust, atol=1e-10)
        assert_allclose(params.Jh, Jh, atol=1e-10)
        assert_allclose(params.Jx, Jx, atol=1e-10)
        assert_allclose(params.Jp, Jp, atol=1e-10)

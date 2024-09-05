import unittest
from itertools import product

import numpy as np

import triqs.operators as op
from triqs.atom_diag import AtomDiag

from edipack2triqs.solver import EDIpackSolver


class TestEDIpackSolver(unittest.TestCase):

    # Interaction parameters for make_H_int()
    norb = 2
    spins = ('up', 'dn')
    fops_imp_up = [('up', o) for o in range(norb)]
    fops_imp_dn = [('dn', o) for o in range(norb)]

    Uloc = np.array([1.0, 2.0])
    Ust = 0.8
    Jh = 0.2
    Jx = 0.1
    Jp = 0.15

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

    def test_parse_nspin1_normal(self):
        orbs = range(self.norb)

        nbath = 2
        h_loc = np.array([[0.1, 0.0], [0.0, 0.2]])
        eps = [[-0.1, -0.2], [0.1, 0.2]]
        V = [[0.4, 0.6], [0.6, 0.4]]

        fops_bath_up = [('B_up', nu * self.norb + o)
                        for nu, o in product(range(nbath), orbs)]
        fops_bath_dn = [('B_dn', nu * self.norb + o)
                        for nu, o in product(range(nbath), orbs)]

        h_int = self.make_H_int()
        h = h_int + sum(h_loc[o1, o2] * op.c_dag(s, o1) * op.c(s, o2)
                        for s, o1, o2 in product(self.spins, orbs, orbs))
        h += sum(eps[nu][o]
                 * op.c_dag("B_" + s, nu * self.norb + o)
                 * op.c("B_" + s, nu * self.norb + o)
                 for nu, s, o in product(range(nbath), self.spins, orbs))
        h += sum(V[nu][o] * (
                 op.c_dag(s, o) * op.c("B_" + s, nu * self.norb + o)
                 + op.c_dag("B_" + s, nu * self.norb + o) * op.c(s, o))
                 for nu, s, o in product(range(nbath), self.spins, orbs))

        solver = EDIpackSolver(h,
                               self.fops_imp_up,
                               self.fops_imp_dn,
                               fops_bath_up,
                               fops_bath_dn)
        solver.solve()

        fops = self.fops_imp_up + self.fops_imp_dn + fops_bath_up + fops_bath_dn
        ad = AtomDiag(h, fops)
        # TODO: Compare results


if __name__ == '__main__':
    unittest.main()

import unittest
import gc
from itertools import product

import numpy as np

import triqs.operators as op

from edipack2triqs.solver import EDIpackSolver


s0 = np.eye(2)
sx = np.array([[0, 1], [1, 0]])
sz = np.array([[1, 0], [0, -1]])


class TestEDIpackSolverBathBasis(unittest.TestCase):

    spins = ('up', 'dn')
    orbs = range(2)

    def test_nspin1(self):
        fops_imp_up = [('up', o) for o in self.orbs]
        fops_imp_dn = [('dn', o) for o in self.orbs]
        fops_bath_up = [('B_up', nu * 2 + o)
                        for nu, o in product(range(2), self.orbs)]
        fops_bath_dn = [('B_dn', nu * 2 + o)
                        for nu, o in product(range(2), self.orbs)]

        h_loc = sum(op.n('up', o) + op.n('dn', o) for o in self.orbs)
        h_int = 3.0 * sum(op.n('up', o) * op.n('dn', o) for o in self.orbs)

        h = np.moveaxis(np.array([[[0.5, 0.1],
                                   [0.1, 0.5]],
                                  [[-0.5, 0.2],
                                   [0.2, -0.5]]]), 0, 2)
        V = np.array([[0.5, 0.6],
                      [0.7, 0.8]])
        h_bath = sum(h[o1, o2, nu]
                     * op.c_dag("B_" + spin, nu * 2 + o1)
                     * op.c("B_" + spin, nu * 2 + o2)
                     for spin, o1, o2, nu
                     in product(self.spins, self.orbs, self.orbs, range(2)))
        h_bath += sum(V[o, nu] * (
                      op.c_dag(spin, o)
                      * op.c("B_" + spin, nu * 2 + o)
                      + op.c_dag("B_" + spin, nu * 2 + o)
                      * op.c(spin, o))
                      for spin, o, nu
                      in product(self.spins, self.orbs, range(2)))

        bath_basis = [op.n('B_up', 0) + op.n('B_up', 0),
                      2 * (op.c_dag('B_up', 0) * op.c('B_up', 1)
                           + op.c_dag('B_up', 1) * op.c('B_up', 0))]

        h = h_loc + h_int + h_bath
        solver = EDIpackSolver(
            h,
            fops_imp_up, fops_imp_dn,
            fops_bath_up, fops_bath_dn,
            bath_basis=bath_basis,
            verbose=0
        )

        # TODO

    # FIXME
    #def test_norb(self):
    #    fops_imp_up = [('up', o) for o in self.orbs]
    #    fops_imp_dn = [('dn', o) for o in self.orbs]
    #    fops_bath_up = [('B_up', i) for i in range(5)]
    #    fops_bath_dn = [('B_dn', i) for i in range(5)]
    #    solver = EDIpackSolver(op.Operator(), fops_imp_up, fops_imp_dn,
    #        fops_bath_up, fops_bath_dn)


    def tearDown(self):
        # Make sure EDIpackSolver.__del__() is called
        gc.collect()


if __name__ == '__main__':
    unittest.main()

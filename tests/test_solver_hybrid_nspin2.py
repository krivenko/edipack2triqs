import unittest
from itertools import product

import numpy as np
from numpy.testing import assert_allclose
from numpy import multiply as mul

import triqs.operators as op
from triqs.atom_diag import AtomDiag, atomic_density_matrix, trace_rho_op

from edipack2triqs.solver import EDIpackSolver


class TestEDIpackSolver(unittest.TestCase):

    # Interaction parameters for make_H_int()
    spins = ('up', 'dn')
    orbs = range(2)
    fops_imp_up = [('up', o) for o in orbs]
    fops_imp_dn = [('dn', o) for o in orbs]

    def make_h_loc(self, h_loc):
        orbs = range(h_loc.shape[1])
        return sum(h_loc[s, o1, o2] * op.c_dag(spin, o1) * op.c(spin, o2)
                   for (s, spin), o1, o2
                   in product(enumerate(self.spins), orbs, orbs))

    def make_h_int(self, *, Uloc, Ust, Jh, Jx, Jp):
        h_int = sum(Uloc[o] * op.n('up', o) * op.n('dn', o) for o in self.orbs)
        h_int += Ust * sum(int(o1 != o2) * op.n('up', o1) * op.n('dn', o2)
                           for o1, o2 in product(self.orbs, self.orbs))
        h_int += (Ust - Jh) * \
            sum(int(o1 < o2) * op.n(s, o1) * op.n(s, o2)
                for s, o1, o2 in product(self.spins, self.orbs, self.orbs))
        h_int -= Jx * sum(int(o1 != o2)
                          * op.c_dag('up', o1) * op.c('dn', o1)
                          * op.c_dag('dn', o2) * op.c('up', o2)
                          for o1, o2 in product(self.orbs, self.orbs))
        h_int += Jp * sum(int(o1 != o2)
                          * op.c_dag('up', o1) * op.c_dag('dn', o1)
                          * op.c('dn', o2) * op.c('up', o2)
                          for o1, o2 in product(self.orbs, self.orbs))
        return h_int

    def make_h_bath(self, eps, V):
        h_bath = sum(eps[nu] * op.c_dag("B_" + s, nu) * op.c("B_" + s, nu)
                     for nu, s in product(range(3), self.spins))
        h_bath += sum(V[o, nu] * (
                      op.c_dag(s, o) * op.c("B_" + s, nu)
                      + op.c_dag("B_" + s, nu) * op.c(s, o))
                      for nu, s, o in product(range(3), self.spins, self.orbs))
        return h_bath

    def make_ref_results(self, h, fops, beta):
        ad = AtomDiag(h, fops)
        rho = atomic_density_matrix(ad, beta)

        densities = [
            trace_rho_op(rho, op.n('up', o) + op.n('dn', o), ad)
            for o in self.orbs]
        double_occ = [
            trace_rho_op(rho, op.n('up', o) * op.n('dn', o), ad)
            for o in self.orbs]
        magnetization = [
            trace_rho_op(rho, op.n('up', o) - op.n('dn', o), ad)
            for o in self.orbs]

        return densities, double_occ, magnetization

    def test_solve(self):
        h_loc = self.make_h_loc(mul.outer([0.8, 1.2], np.diag([0.5, 0.6])))
        h_int = self.make_h_int(Uloc=np.array([1.0, 2.0]),
                                Ust=0.8,
                                Jh=0.2,
                                Jx=0.1,
                                Jp=0.15)

        fops_bath_up = [('B_up', nu) for nu in range(3)]
        fops_bath_dn = [('B_dn', nu) for nu in range(3)]

        fops = self.fops_imp_up + self.fops_imp_dn + fops_bath_up + fops_bath_dn

        eps = np.array([-0.1, -0.2, -0.3])
        V = np.array([[0.4, 0.7, 0.1],
                      [0.3, 0.5, 0.2]])
        h_bath = self.make_h_bath(eps, V)

        h = h_loc + h_int + h_bath
        solver = EDIpackSolver(h,
                               self.fops_imp_up,
                               self.fops_imp_dn,
                               fops_bath_up,
                               fops_bath_dn,
                               verbose=0)

        self.assertEqual(solver.nspin, 2)
        self.assertEqual(solver.norb, 2)
        self.assertEqual(solver.bath().name, "hybrid")
        self.assertEqual(solver.bath().nbath, 3)

        # Part I
        solver.solve(beta=100.0)

        ## Reference solution
        densities_ref, double_occ_ref, magnetization_ref = \
            self.make_ref_results(h, fops, 100.0)

        assert_allclose(solver.densities(), densities_ref, atol=1e-8)
        assert_allclose(solver.double_occ(), double_occ_ref, atol=1e-8)
        assert_allclose(solver.magnetization(), magnetization_ref, atol=1e-8)
        # TODO: GF

        # Part II
        new_int_params = {'Uloc': np.array([2.0, 3.0]),
                          'Ust': 0.6,
                          'Jh': 0.1,
                          'Jx': 0.2,
                          'Jp': 0.0}
        solver.update_int_params(**new_int_params)
        solver.solve(beta=120.0)

        ## Reference solution
        h_int = self.make_h_int(**new_int_params)
        h = h_loc + h_int + h_bath
        densities_ref, double_occ_ref, magnetization_ref = \
            self.make_ref_results(h, fops, 120.0)

        assert_allclose(solver.densities(), densities_ref, atol=1e-8)
        assert_allclose(solver.double_occ(), double_occ_ref, atol=1e-8)
        assert_allclose(solver.magnetization(), magnetization_ref, atol=1e-8)
        # TODO: GF

        # Part III
        eps = np.array([-0.1, -0.2, -0.4])
        V = np.array([[0.4, 0.7, 0.1],
                      [0.3, 0.8, 0.2]])

        solver.bath().eps[:] = eps
        solver.bath().V[:] = V
        solver.solve(beta=100.0)

        ## Reference solution
        h_bath = self.make_h_bath(eps, V)
        h = h_loc + h_int + h_bath
        densities_ref, double_occ_ref, magnetization_ref = \
            self.make_ref_results(h, fops, 100.0)

        assert_allclose(solver.densities(), densities_ref, atol=1e-8)
        assert_allclose(solver.double_occ(), double_occ_ref, atol=1e-8)
        assert_allclose(solver.magnetization(), magnetization_ref, atol=1e-8)
        # TODO: GF


if __name__ == '__main__':
    unittest.main()

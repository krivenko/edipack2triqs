import unittest
import gc
from itertools import product

import numpy as np

import triqs.operators as op


class TestSolver(unittest.TestCase):

    #
    # Common parameters of the impurity
    #

    spins = ('up', 'dn')
    orbs = range(2)

    # Returns a mapping (spin, orb) -> (block name, inner index)

    @classmethod
    def mki_spin_blocks(cls, spin, orb):
        return (spin, orb)

    @classmethod
    def mki_spin_orb_block(cls, spin, orb):
        up, dn = cls.spins
        return (f"{up}_{dn}", len(cls.orbs) * cls.spins.index(spin) + orb)

    # TODO: Rethink
    @classmethod
    def make_fops_imp(cls, spin_blocks=True):
        if spin_blocks:
            return tuple([(spin, o) for o in cls.orbs] for spin in cls.spins)
        else:
            return ([('up_dn', so) for so in cls.orbs],
                    [('up_dn', so + len(cls.orbs)) for so in cls.orbs])

    #
    # Class methods returning various contributions to the Hamiltonian operator
    # object.
    #

    @classmethod
    def make_h_loc(cls, mat, spin_blocks=True):
        mki = cls.mki_spin_blocks if spin_blocks else cls.mki_spin_orb_block
        return sum(mat[s1, s2, o1, o2]
                   * op.c_dag(*mki(spin1, o1)) * op.c(*mki(spin2, o2))
                   for (s1, spin1), (s2, spin2), o1, o2
                   in product(enumerate(cls.spins), enumerate(cls.spins),
                              cls.orbs, cls.orbs))

    @classmethod
    def make_h_loc_an(cls, mat):
        return sum(mat[o1, o2] * op.c_dag('up', o1) * op.c_dag('dn', o2)
                   + np.conj(mat[o1, o2]) * op.c('dn', o2) * op.c('up', o1)
                   for o1, o2 in product(cls.orbs, cls.orbs))

    @classmethod
    def make_h_int(cls, *, Uloc, Ust, Jh, Jx, Jp, spin_blocks=True):
        mki = cls.mki_spin_blocks if spin_blocks else cls.mki_spin_orb_block
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
    def change_int_params(cls, U, new_int_params):
        Uloc = new_int_params["Uloc"]
        Ust = new_int_params["Ust"]
        Jh = new_int_params["Jh"]
        Jx = new_int_params["Jx"]
        Jp = new_int_params["Jp"]
        # Uloc
        for s, o in product(range(2), cls.orbs):
            U[o, s, o, 1 - s, o, s, o, 1 - s] = 0.5 * Uloc[o]
            U[o, s, o, 1 - s, o, 1 - s, o, s] = -0.5 * Uloc[o]
        for s, o1, o2 in product(range(2), cls.orbs, cls.orbs):
            if o1 == o2:
                continue
            # Ust
            U[o1, s, o2, 1 - s, o1, s, o2, 1 - s] = 0.5 * Ust
            U[o1, s, o2, 1 - s, o2, 1 - s, o1, s] = -0.5 * Ust
            # Ust - Jh
            U[o1, s, o2, s, o1, s, o2, s] = 0.5 * (Ust - Jh)
            U[o1, s, o2, s, o2, s, o1, s] = -0.5 * (Ust - Jh)
            # Jx
            U[o1, s, o2, 1 - s, o2, s, o1, 1 - s] = 0.5 * Jx
            U[o1, s, o2, 1 - s, o1, 1 - s, o2, s] = -0.5 * Jx
            # Jp
            U[o1, s, o1, 1 - s, o2, s, o2, 1 - s] = 0.5 * Jp
            U[o1, s, o1, 1 - s, o2, 1 - s, o2, s] = -0.5 * Jp

    #
    # Miscellaneous
    #

    # solve() parameters that enable computation of susceptibilities
    chi_params = {f"chi_{c}": True for c in ["spin", "dens", "pair", "exct"]}

    def tearDown(self):
        # Make sure EDIpackSolver.__del__() is called
        gc.collect()

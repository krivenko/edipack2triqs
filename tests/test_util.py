import unittest

import triqs.operators as op

from edipack2triqs.util import (canonical2op,
                                monomial2op,
                                validate_fops_up_dn,
                                normal_part,
                                spin_conjugate)


class TestUtil(unittest.TestCase):

    def test_canonical2op(self):
        self.assertEqual(canonical2op(True, ["up", 0]), op.c_dag("up", 0))
        self.assertEqual(canonical2op(False, ["up", 0]), op.c("up", 0))

    def test_monomial2op(self):
        self.assertEqual(monomial2op([(True, ["up", 0]), (False, ["dn", 1])]),
                         op.c_dag("up", 0) * op.c("dn", 1))

    def test_validate_fops_up_dn(self):
        fops_up = [("up", 0), ("up", 1)]
        fops_dn = [("dn", 0), ("dn", 1)]
        with self.assertRaises(AssertionError):
            validate_fops_up_dn([("up", 0), ("up", 0)], fops_dn, 'u', 'd')
        with self.assertRaises(AssertionError):
            validate_fops_up_dn(fops_up, [("dn", 0), ("dn", 0)], 'u', 'd')
        with self.assertRaises(AssertionError):
            validate_fops_up_dn([("up", 0)], fops_dn, 'u', 'd')
        with self.assertRaises(AssertionError):
            validate_fops_up_dn(fops_up, [("dn", 0), ("up", 1)], 'u', 'd')

    def test_normal_part(self):
        OP_n = 2 * op.n("up", 0) * op.n("dn", 1)
        OP = OP_n + op.c_dag("up", 2) * op.c_dag("dn", 2)
        self.assertEqual(normal_part(OP), OP_n)

    def test_spin_conjugate(self):
        OP = 2 * op.n("up", 0) * op.n("dn", 1)
        OP += 3 * op.c_dag("up", 2) * op.c("dn", 1)
        OP += 4 * op.c("up", 0) * op.c("up", 2)

        fops_up = [("up", 0), ("up", 1), ("up", 2)]
        fops_dn = [("dn", 0), ("dn", 1), ("dn", 2)]

        self.assertEqual(spin_conjugate(OP, fops_up, fops_dn),
                         2 * op.n("dn", 0) * op.n("up", 1)
                         + 3 * op.c_dag("dn", 2) * op.c("up", 1)
                         + 4 * op.c("dn", 0) * op.c("dn", 2)
                         )


if __name__ == '__main__':
    unittest.main()

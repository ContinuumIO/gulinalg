"""
Tests that are reproductions of bugs found, to avoid regressions.
"""

from __future__ import print_function
from unittest import TestCase
import numpy as np
from numpy.testing import run_module_suite, assert_allclose
import gulinalg


class TestBugs(TestCase):
    def test_uninit_gemm(self):
        a = np.zeros((298, 64))
        b = np.zeros((64, 298))
        c = np.zeros((298, 298), order='F')
        c.fill(np.nan)

        gulinalg.matrix_multiply(a, b, out=c)
        assert not np.isnan(c).any()

    def test_zero_dim_gemm(self):
        a = np.zeros((2,0))
        b = np.zeros((0,2))
        c = np.empty((2,2))
        c.fill(np.nan)
        # make sure that the result is set to zero
        gulinalg.matrix_multiply(a, b, out=c)
        assert_allclose(c, np.zeros((2,2)))

        # check that the correct shape is created as well...
        d = gulinalg.matrix_multiply(a,b)
        assert_allclose(d, np.zeros((2,2)))

        # check other border cases...
        e = gulinalg.matrix_multiply(np.zeros((0,2)), np.zeros((2,2)))
        assert_allclose(e, np.zeros((0,2)))

        f = gulinalg.matrix_multiply(np.zeros((2,2)), np.zeros((2,0)))
        assert_allclose(f, np.zeros((2,0)))


if __name__ == '__main__':
    run_module_suite()

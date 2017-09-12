"""
Tests that are reproductions of bugs found, to avoid regressions.
"""

from __future__ import print_function
from unittest import TestCase, skipIf
import numpy as np
from numpy.testing import run_module_suite, assert_allclose
from pkg_resources import parse_version
import gulinalg


class TestBugs(TestCase):
    def test_uninit_gemm(self):
        a = np.zeros((298, 64))
        b = np.zeros((64, 298))
        c = np.zeros((298, 298), order='F')
        c.fill(np.nan)

        gulinalg.matrix_multiply(a, b, out=c)
        assert not np.isnan(c).any()

    def test_zero_K_gemm(self):
        a = np.zeros((2,0))
        b = np.zeros((0,2))
        c = np.empty((2,2), order="C")
        c.fill(np.nan)
        # make sure that the result is set to zero
        gulinalg.matrix_multiply(a, b, out=c)
        assert_allclose(c, np.zeros((2,2)))

        # make sure that this works also when order is FORTRAN
        c = np.empty((2,2), order="F")
        c.fill(np.nan)
        gulinalg.matrix_multiply(a, b, out=c)
        assert_allclose(c, np.zeros((2,2)))

        # check that the correct shape is created as well...
        d = gulinalg.matrix_multiply(a,b)
        assert_allclose(d, np.zeros((2,2)))

    @skipIf(parse_version(np.__version__) < parse_version('1.13'),
            "prior to numpy 1.13 gufunc machinery raises an error on this code")
    def test_zero_MN_gemm(self):
        # check other border cases...
        e = gulinalg.matrix_multiply(np.zeros((0,2)), np.zeros((2,2)))
        assert_allclose(e, np.zeros((0,2)))

        f = gulinalg.matrix_multiply(np.zeros((2,2)), np.zeros((2,0)))
        assert_allclose(f, np.zeros((2,0)))


if __name__ == '__main__':
    run_module_suite()

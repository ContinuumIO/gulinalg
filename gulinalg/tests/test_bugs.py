"""
Tests that are reproductions of bugs found, to avoid regressions.
"""

from __future__ import print_function
from unittest import TestCase
import numpy as np
from numpy.testing import run_module_suite
import gulinalg


class TestBugs(TestCase):
    def test_uninit_gemm(self):
        a = np.zeros((298, 64))
        b = np.zeros((64, 298))
        c = np.zeros((298, 298), order='F')
        c.fill(np.nan)

        gulinalg.matrix_multiply(a, b, out=c)
        assert not np.isnan(c).any()


if __name__ == '__main__':
    run_module_suite()
        

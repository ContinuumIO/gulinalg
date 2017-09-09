"""
Tests on matrix multiply. As the underlying code is now playing with
the TRANSA TRANSB parameters to minimize copying, several tests are
needed to make sure that all cases are handled correctly as its logic
is rather complex.
"""

from __future__ import print_function
from unittest import TestCase, skipIf
import numpy as np
from numpy.testing import run_module_suite, assert_allclose
from pkg_resources import parse_version
import gulinalg

M = 75
N = 50
K = 100

class TestMatrixMultiply(TestCase):
    # no output specified (operation allocated) ================================
    def test_matrix_multiply_cc(self):
        """matrix multiply two C layout matrices"""
        a = np.ascontiguousarray(np.random.randn(M,N))
        b = np.ascontiguousarray(np.random.randn(N,K))
        res = gulinalg.matrix_multiply(a,b)
        ref = np.dot(a,b)
        assert_allclose(res, ref)

    def test_matrix_multiply_cf(self):
        """matrix multiply C layout by FORTRAN layout matrices"""
        a = np.ascontiguousarray(np.random.randn(M,N))
        b = np.asfortranarray(np.random.randn(N,K))
        res = gulinalg.matrix_multiply(a,b)
        ref = np.dot(a,b)
        assert_allclose(res, ref)

    def test_matrix_multiply_fc(self):
        """matrix multiply FORTRAN layout by C layout matrices"""
        a = np.asfortranarray(np.random.randn(M,N))
        b = np.ascontiguousarray(np.random.randn(N,K))
        res = gulinalg.matrix_multiply(a,b)
        ref = np.dot(a,b)
        assert_allclose(res, ref)

    def test_matrix_multiply_ff(self):
        """matrix multiply two FORTRAN layout matrices"""
        a = np.asfortranarray(np.random.randn(M,N))
        b = np.asfortranarray(np.random.randn(N,K))
        res = gulinalg.matrix_multiply(a,b)
        ref = np.dot(a,b)
        assert_allclose(res, ref)

    
    # C explicit outputs =======================================================
    def test_matrix_multiply_cc_c(self):
        """matrix multiply two C layout matrices, explicit C array output"""
        a = np.ascontiguousarray(np.random.randn(M,N))
        b = np.ascontiguousarray(np.random.randn(N,K))
        res = np.empty((M,K), order='C')
        gulinalg.matrix_multiply(a,b, out=res)
        ref = np.dot(a,b)
        assert_allclose(res, ref)

    def test_matrix_multiply_cf_c(self):
        """matrix multiply C layout by FORTRAN layout matrices, explicit C array output"""
        a = np.ascontiguousarray(np.random.randn(M,N))
        b = np.asfortranarray(np.random.randn(N,K))
        res = np.empty((M,K), order='C')
        gulinalg.matrix_multiply(a,b, out=res)
        ref = np.dot(a,b)
        assert_allclose(res, ref)

    def test_matrix_multiply_fc_c(self):
        """matrix multiply FORTRAN layout by C layout matrices, explicit C array output"""
        a = np.asfortranarray(np.random.randn(M,N))
        b = np.ascontiguousarray(np.random.randn(N,K))
        res = np.empty((M,K), order='C')
        gulinalg.matrix_multiply(a,b, out=res)
        ref = np.dot(a,b)
        assert_allclose(res, ref)

    def test_matrix_multiply_ff_c(self):
        """matrix multiply two FORTRAN layout matrices, explicit C array output"""
        a = np.asfortranarray(np.random.randn(M,N))
        b = np.asfortranarray(np.random.randn(N,K))
        res = np.empty((M,K), order='C')
        gulinalg.matrix_multiply(a,b, out=res)
        ref = np.dot(a,b)
        assert_allclose(res, ref)

    # FORTRAN explicit outputs =================================================
    def test_matrix_multiply_cc_f(self):
        """matrix multiply two C layout matrices, explicit FORTRAN array output"""
        a = np.ascontiguousarray(np.random.randn(M,N))
        b = np.ascontiguousarray(np.random.randn(N,K))
        res = np.empty((M,K), order='F')
        gulinalg.matrix_multiply(a,b, out=res)
        ref = np.dot(a,b)
        assert_allclose(res, ref)

    def test_matrix_multiply_cf_f(self):
        """matrix multiply C layout by FORTRAN layout matrices, explicit FORTRAN array output"""
        a = np.ascontiguousarray(np.random.randn(M,N))
        b = np.asfortranarray(np.random.randn(N,K))
        res = np.empty((M,K), order='F')
        gulinalg.matrix_multiply(a,b, out=res)
        ref = np.dot(a,b)
        assert_allclose(res, ref)

    def test_matrix_multiply_fc_f(self):
        """matrix multiply FORTRAN layout by C layout matrices, explicit FORTRAN array output"""
        a = np.asfortranarray(np.random.randn(M,N))
        b = np.ascontiguousarray(np.random.randn(N,K))
        res = np.empty((M,K), order='F')
        gulinalg.matrix_multiply(a,b, out=res)
        ref = np.dot(a,b)
        assert_allclose(res, ref)

    def test_matrix_multiply_ff_f(self):
        """matrix multiply two FORTRAN layout matrices, explicit FORTRAN array output"""
        a = np.asfortranarray(np.random.randn(M,N))
        b = np.asfortranarray(np.random.randn(N,K))
        res = np.empty((M,K), order='F')
        gulinalg.matrix_multiply(a,b, out=res)
        ref = np.dot(a,b)
        assert_allclose(res, ref)


if __name__ == '__main__':
    run_module_suite()

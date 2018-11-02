"""
Tests BLAS functions. Since it supports C as well as Fortran
matrix, that leads to various combinations of matrices to test.
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


class TestMatvecMultiplyNoCopy(TestCase):
    """
    Tests the cases that code can handle without copy-rearranging of any of
    the input/output arguments.
    """
    def test_matvec_multiply_c(self):
        """Multiply C layout matrix with vector"""
        a = np.ascontiguousarray(np.random.randn(M, N))
        b = np.random.randn(N)
        res = gulinalg.matvec_multiply(a, b)
        ref = np.dot(a, b)
        assert_allclose(res, ref)

    def test_matvec_multiply_f(self):
        """Multiply FORTRAN layout matrix with vector"""
        a = np.asfortranarray(np.random.randn(M, N))
        b = np.random.randn(N)
        res = gulinalg.matvec_multiply(a, b)
        ref = np.dot(a, b)
        assert_allclose(res, ref)

    def test_matvec_multiply_cv_c(self):
        """Test for explicit C array output for C layout input matrix"""
        a = np.ascontiguousarray(np.random.randn(M, N))
        b = np.ascontiguousarray(np.random.randn(N))
        res = np.zeros(M, order='C')
        gulinalg.matvec_multiply(a, b, out=res)
        ref = np.dot(a, b)
        assert_allclose(res, ref)

    def test_matvec_multiply_fv_c(self):
        """Test for explicit C array output for FORTRAN layout input matrix"""
        a = np.asfortranarray(np.random.randn(M, N))
        b = np.ascontiguousarray(np.random.randn(N))
        res = np.zeros(M, order='C')
        gulinalg.matvec_multiply(a, b, out=res)
        ref = np.dot(a, b)
        assert_allclose(res, ref)

    def test_matvec_multiply_cv_f(self):
        """Test for explicit FORTRAN array output for C layout input matrix"""
        a = np.ascontiguousarray(np.random.randn(M, N))
        b = np.ascontiguousarray(np.random.randn(N))
        res = np.zeros(M, order='F')
        gulinalg.matvec_multiply(a, b, out=res)
        ref = np.dot(a, b)
        assert_allclose(res, ref)

    def test_matvec_multiply_fv_f(self):
        """Test for explicit FORTRAN array output for F layout input matrix"""
        a = np.asfortranarray(np.random.randn(M, N))
        b = np.ascontiguousarray(np.random.randn(N))
        res = np.zeros(M, order='F')
        gulinalg.matvec_multiply(a, b, out=res)
        ref = np.dot(a, b)
        assert_allclose(res, ref)

    def test_matvec_multiply_for_complex_numbers(self):
        """Test for complex numbers input."""
        a = np.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + -8j]])
        b = np.array([1 - 2j, 4 + 5j])
        res = gulinalg.matvec_multiply(a, b)
        ref = np.dot(a, b)
        assert_allclose(res, ref)

    @skipIf(parse_version(np.__version__) < parse_version('1.13'),
            "Prior to 1.13, numpy low level iterators didn't support removing "
            "empty axis. So gufunc couldn't be called with empty inner loop")
    def test_matvec_size_zero_matrix(self):
        """Test matrix of size zero"""
        a = np.random.randn(0, 2)
        b = np.random.randn(2)
        res = gulinalg.matvec_multiply(a, b)
        ref = np.dot(a, b)
        assert_allclose(res, ref)

    @skipIf(parse_version(np.__version__) < parse_version('1.13'),
            "Prior to 1.13, numpy low level iterators didn't support removing "
            "empty axis. So gufunc couldn't be called with empty inner loop")
    def test_matvec_size_zero_vector(self):
        """Test vector of size zero"""
        a = np.random.randn(2, 0)
        b = np.random.randn(0)
        res = gulinalg.matvec_multiply(a, b)
        ref = np.dot(a, b)
        assert_allclose(res, ref)

    def test_matvec_size_one_vector(self):
        """Test vector of size one"""
        a = np.random.randn(1, 1)
        b = np.random.randn(1)
        res = gulinalg.matvec_multiply(a, b)
        ref = np.dot(a, b)
        assert_allclose(res, ref)


class TestMatvecMultiplyWithCopy(TestCase):
    """
    Test the cases where there is at least one operand/output that requires
    copy/rearranging.
    """
    def test_input_non_contiguous_1(self):
        """First input not contiguous"""
        a = np.ascontiguousarray(np.random.randn(M, N, 2))[:, :, 0]
        b = np.ascontiguousarray(np.random.randn(N))
        res = np.zeros(M, order='C')
        assert not a.flags.c_contiguous and not a.flags.f_contiguous
        gulinalg.matvec_multiply(a, b, out=res)
        ref = np.dot(a, b)
        assert_allclose(res, ref)

    def test_input_non_contiguous_2(self):
        """Second input not contiguous"""
        a = np.ascontiguousarray(np.random.randn(M, N))
        b = np.ascontiguousarray(np.random.randn(N, 2))[:, 0]
        res = np.zeros(M, order='C')
        assert not b.flags.c_contiguous and not b.flags.f_contiguous
        gulinalg.matvec_multiply(a, b, out=res)
        ref = np.dot(a, b)
        assert_allclose(res, ref)

    def test_input_non_contiguous_3(self):
        """Neither input contiguous"""
        a = np.ascontiguousarray(np.random.randn(M, N, 2))[:, :, 0]
        b = np.ascontiguousarray(np.random.randn(N, 2))[:, 0]
        res = np.zeros(M, order='C')
        assert not a.flags.c_contiguous and not a.flags.f_contiguous
        assert not b.flags.c_contiguous and not b.flags.f_contiguous
        gulinalg.matvec_multiply(a, b, out=res)
        ref = np.dot(a, b)
        assert_allclose(res, ref)

    def test_output_non_contiguous(self):
        """Output not contiguous"""
        a = np.ascontiguousarray(np.random.randn(M, N))
        b = np.ascontiguousarray(np.random.randn(N))
        res = np.zeros((M, 2), order='C')[:, 0]
        assert not res.flags.c_contiguous and not res.flags.f_contiguous
        gulinalg.matvec_multiply(a, b, out=res)
        ref = np.dot(a, b)
        assert_allclose(res, ref)

    def test_all_non_contiguous(self):
        """Neither input nor output contiguous"""
        a = np.ascontiguousarray(np.random.randn(M, N, 2))[:, :, 0]
        b = np.ascontiguousarray(np.random.randn(N, 2))[:, 0]
        res = np.zeros((M, 2), order='C')[:, 0]
        assert not a.flags.c_contiguous and not a.flags.f_contiguous
        assert not b.flags.c_contiguous and not b.flags.f_contiguous
        assert not res.flags.c_contiguous and not res.flags.f_contiguous
        gulinalg.matvec_multiply(a, b, out=res)
        ref = np.dot(a, b)
        assert_allclose(res, ref)

    def test_stride_tricks(self):
        """Test that matrices that are contiguous but have their dimension
        overlapped *copy*, as BLAS does not support them"""
        a = np.ascontiguousarray(np.random.randn(M + N))
        a = np.lib.stride_tricks.as_strided(a,
                                            shape=(M, N),
                                            strides=(a.itemsize, a.itemsize))
        b = np.ascontiguousarray(np.random.randn(N))
        res = gulinalg.matvec_multiply(a, b)
        ref = np.dot(a, b)
        assert_allclose(res, ref)


class TestMatvecMultiplyVector(TestCase):
    """Tests showing that the gufunc stuff works"""
    def test_vector(self):
        """test vectorized matrix multiply"""
        a = np.ascontiguousarray(np.random.randn(10, M, N))
        b = np.ascontiguousarray(np.random.randn(10, N))
        res = gulinalg.matvec_multiply(a, b)
        assert res.shape == (10, M)
        ref = np.stack([np.dot(a[i], b[i]) for i in range(len(a))])
        assert_allclose(res, ref)

    def test_broadcast(self):
        """test broadcast matrix multiply"""
        a = np.ascontiguousarray(np.random.randn(M, N))
        b = np.ascontiguousarray(np.random.randn(10, N))
        res = gulinalg.matvec_multiply(a, b)
        assert res.shape == (10, M)
        ref = np.stack([np.dot(a, b[i]) for i in range(len(b))])
        assert_allclose(res, ref)

    def test_nan_handling(self):
        """NaN in one output shouldn't contaminate remaining outputs"""
        a = np.eye(2)
        b = np.array([[1.0, 2.0], [np.nan, 1.0]])
        ref = np.array([[1., 2.], [np.nan, np.nan]])
        res = gulinalg.matvec_multiply(a, b)
        assert_allclose(res, ref)

    def test_infinity_handling(self):
        """Infinity in one output shouldn't contaminate remaining outputs"""
        a = np.eye(2)
        b = np.array([[1.0, 2.0], [np.inf, 1.0]])
        ref = np.array([[1., 2.], [np.inf, np.nan]])
        res = gulinalg.matvec_multiply(a, b)
        assert_allclose(res, ref)

    @skipIf(parse_version(np.__version__) < parse_version('1.13'),
            "Prior to 1.13, numpy low level iterators didn't support removing "
            "empty axis. So gufunc couldn't be called with empty inner loop")
    def test_size_zero_vector(self):
        """Test broadcasting for vector of size zero"""
        a = np.ascontiguousarray(np.random.randn(10, 2, 0))
        b = np.ascontiguousarray(np.random.randn(10, 0))
        res = gulinalg.matvec_multiply(a, b)
        assert res.shape == (10, 2)
        ref = np.stack([np.dot(a[i], b[i]) for i in range(len(a))])
        assert_allclose(res, ref)

    @skipIf(parse_version(np.__version__) < parse_version('1.13'),
            "Prior to 1.13, numpy low level iterators didn't support removing "
            "empty axis. So gufunc couldn't be called with empty inner loop")
    def test_size_zero_matrix(self):
        """Test broadcasting for matrix of size zero"""
        a = np.ascontiguousarray(np.random.randn(10, 0, 2))
        b = np.ascontiguousarray(np.random.randn(10, 2))
        res = gulinalg.matvec_multiply(a, b)
        assert res.shape == (10, 0)
        ref = np.stack([np.dot(a[i], b[i]) for i in range(len(a))])
        assert_allclose(res, ref)

    def test_size_one_vector(self):
        """Test broadcasting for vector of size one"""
        a = np.ascontiguousarray(np.random.randn(10, 1, 1))
        b = np.ascontiguousarray(np.random.randn(10, 1))
        res = gulinalg.matvec_multiply(a, b)
        assert res.shape == (10, 1)
        ref = np.stack([np.dot(a[i], b[i]) for i in range(len(a))])
        assert_allclose(res, ref)


class TestUpdateRank1Copy(TestCase):
    """
    Tests the cases that code can handle without copy-rearranging of any of
    the input/output arguments.
    """
    def test_update_rank1_c(self):
        """Rank update on C layout matrix"""
        a = np.random.randn(M)
        b = np.random.randn(N)
        c = np.ascontiguousarray(np.random.randn(M, N))
        res = gulinalg.update_rank1(a, b, c)
        ref = np.dot(a.reshape(M, 1), b.reshape(1, N)) + c
        assert_allclose(res, ref)

    def test_update_rank1_f(self):
        """Rank update on F layout matrix"""
        a = np.random.randn(M)
        b = np.random.randn(N)
        c = np.asfortranarray(np.random.randn(M, N))
        res = gulinalg.update_rank1(a, b, c)
        ref = np.dot(a.reshape(M, 1), b.reshape(1, N)) + c
        assert_allclose(res, ref)

    def test_update_rank1_for_complex_numbers(self):
        """Test for complex numbers"""
        a = np.array([1 + 3j, 3 - 4j])
        b = np.array([1 - 2j, 4 + 5j])
        c = np.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + -8j]])
        res = gulinalg.update_rank1(a, b, c)
        ref = np.dot(a.reshape(2, 1), b.conj().reshape(1, 2)) + c
        assert_allclose(res, ref)

    def test_update_rank1_for_complex_numbers_no_conjugate_transpose(self):
        """Test for complex numbers but no conjuage transpose"""
        a = np.array([1 + 3j, 3 - 4j])
        b = np.array([1 - 2j, 4 + 5j])
        c = np.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + -8j]])
        res = gulinalg.update_rank1(a, b, c, conjugate=False)
        ref = np.dot(a.reshape(2, 1), b.reshape(1, 2)) + c
        assert_allclose(res, ref)

    def test_update_rank1_c_c(self):
        """Rank1 update on C layout matrix, explicit C array output"""
        a = np.array([2, 3, 4])
        b = np.array([1, 3, 4, 5])
        c = np.arange(1, 13).reshape(3, 4)
        res = np.zeros((3, 4), order='C')
        gulinalg.update_rank1(a, b, c, out=res)
        ref = np.dot(a.reshape(3, 1), b.reshape(1, 4)) + c
        assert_allclose(res, ref)

    def test_update_rank1_f_c(self):
        """Rank1 update on F layout matrix, explicit C array output"""
        a = np.array([2, 3, 4])
        b = np.array([1, 3, 4, 5])
        c = np.asfortranarray(np.arange(1, 13).reshape(3, 4))
        res = np.zeros((3, 4), order='C')
        gulinalg.update_rank1(a, b, c, out=res)
        ref = np.dot(a.reshape(3, 1), b.reshape(1, 4)) + c
        assert_allclose(res, ref)

    def test_update_rank1_c_f(self):
        """Rank1 update on C layout matrix, explicit F array output"""
        a = np.array([2, 3, 4])
        b = np.array([1, 3, 4, 5])
        c = np.arange(1, 13).reshape(3, 4)
        res = np.zeros((3, 4), order='F')
        gulinalg.update_rank1(a, b, c, out=res)
        ref = np.dot(a.reshape(3, 1), b.reshape(1, 4)) + c
        assert_allclose(res, ref)

    def test_update_rank1_f_f(self):
        """Rank1 update on F layout matrix, explicit F array output"""
        a = np.array([2, 3, 4])
        b = np.array([1, 3, 4, 5])
        c = np.asfortranarray(np.arange(1, 13).reshape(3, 4))
        res = np.zeros((3, 4), order='F')
        gulinalg.update_rank1(a, b, c, out=res)
        ref = np.dot(a.reshape(3, 1), b.reshape(1, 4)) + c
        assert_allclose(res, ref)

    @skipIf(parse_version(np.__version__) < parse_version('1.13'),
            "Prior to 1.13, numpy low level iterators didn't support removing "
            "empty axis. So gufunc couldn't be called with empty inner loop")
    def test_size_zero_vector(self):
        """Test vector input of size zero"""
        a = np.zeros(1)
        b = np.zeros(0)
        c = np.ascontiguousarray(np.random.randn(1, 0))
        res = gulinalg.update_rank1(a, b, c)
        ref = np.dot(np.zeros((1, 0)), np.zeros((0, 0))) + c
        assert_allclose(res, ref)

    @skipIf(parse_version(np.__version__) < parse_version('1.13'),
            "Prior to 1.13, numpy low level iterators didn't support removing "
            "empty axis. So gufunc couldn't be called with empty inner loop")
    def test_size_zero_matrix(self):
        """Test matrix input of size zero"""
        a = np.zeros(0)
        b = np.zeros(2)
        c = np.full((0, 2), np.nan)
        res = gulinalg.update_rank1(a, b, c)
        ref = np.dot(np.zeros((0, 0)), np.zeros((0, 2))) + c
        assert_allclose(res, ref)

    def test_size_one_vector(self):
        """Test vector inputs of size one"""
        a = np.random.randn(1)
        b = np.random.randn(1)
        c = np.ascontiguousarray(np.random.randn(1, 1))
        res = gulinalg.update_rank1(a, b, c)
        ref = np.dot(a.reshape(1, 1), b.reshape(1, 1)) + c
        assert_allclose(res, ref)


class TestUpdateRank1WithCopy(TestCase):
    """
    Test the cases where there is at least one operand/output that requires
    copy/rearranging.
    """
    def test_input_non_contiguous_vectors(self):
        """Not contiguous vector inputs"""
        a = np.ascontiguousarray(np.random.randn(M, N, 2))[:, 0, 0]
        b = np.ascontiguousarray(np.random.randn(M, N, 2))[0, :, 0]
        c = np.ascontiguousarray(np.random.randn(M, N))
        assert not a.flags.c_contiguous and not a.flags.f_contiguous
        assert not b.flags.c_contiguous and not b.flags.f_contiguous
        res = gulinalg.update_rank1(a, b, c)
        ref = np.dot(a.reshape(M, 1), b.reshape(1, N)) + c
        assert_allclose(res, ref)

    def test_input_non_contiguous_matrix(self):
        """Non contiguous matrix input"""
        a = np.random.randn(M)
        b = np.random.randn(N)
        c = np.ascontiguousarray(np.random.randn(M, N, 2))[:, :, 0]
        assert not c.flags.c_contiguous and not c.flags.f_contiguous
        res = gulinalg.update_rank1(a, b, c)
        ref = np.dot(a.reshape(M, 1), b.reshape(1, N)) + c
        assert_allclose(res, ref)

    def test_output_non_contiguous(self):
        """Output not contiguous"""
        a = np.random.randn(M)
        b = np.random.randn(N)
        c = np.ascontiguousarray(np.random.randn(M, N))
        res = np.zeros((M, N, 2), order='C')[:, :, 0]
        gulinalg.update_rank1(a, b, c, out=res)
        ref = np.dot(a.reshape(M, 1), b.reshape(1, N)) + c
        assert_allclose(res, ref)

    def test_stride_tricks(self):
        """test that matrices that are contiguous but have their dimension
        overlapped *copy*, as BLAS does not support them"""
        a = np.random.randn(M)
        b = np.random.randn(N)
        c = np.ascontiguousarray(np.random.randn(M + N))
        c = np.lib.stride_tricks.as_strided(a,
                                            shape=(M, N),
                                            strides=(c.itemsize, c.itemsize))
        res = gulinalg.update_rank1(a, b, c)
        ref = np.dot(a.reshape(M, 1), b.reshape(1, N)) + c
        assert_allclose(res, ref)


class TestUpdateRank1Vector(TestCase):
    """Tests showing that the gufunc stuff works"""
    def test_vector(self):
        """test vectorized rank1 update"""
        a = np.ascontiguousarray(np.random.randn(10, M))
        b = np.ascontiguousarray(np.random.randn(10, N))
        c = np.ascontiguousarray(np.random.randn(10, M, N))
        res = gulinalg.update_rank1(a, b, c)
        assert res.shape == (10, M, N)
        ref = np.stack([np.dot(a[i].reshape(M, 1), b[i].reshape(1, N)) + c[i]
                        for i in range(len(c))])
        assert_allclose(res, ref)

    def test_broadcast(self):
        """test broadcast rank1 update"""
        a = np.ascontiguousarray(np.random.randn(10, M))
        b = np.ascontiguousarray(np.random.randn(10, N))
        c = np.ascontiguousarray(np.random.randn(M, N))
        res = gulinalg.update_rank1(a, b, c)
        assert res.shape == (10, M, N)
        ref = np.stack([np.dot(a[i].reshape(M, 1), b[i].reshape(1, N)) + c
                        for i in range(len(b))])
        assert_allclose(res, ref)

    def test_nan_handling(self):
        """NaN in one output shouldn't contaminate remaining outputs"""
        a = np.array([[1, 2], [1, np.nan]])
        b = np.array([3, 4])
        c = np.array([[1, 2], [3, 4]])
        ref = np.array([[[4, 6], [9, 12]],
                        [[4, 6], [np.nan, np.nan]]])
        res = gulinalg.update_rank1(a, b, c)
        assert_allclose(res, ref)

    def test_infinity_handling(self):
        """Infinity in one output shouldn't contaminate remaining outputs"""
        a = np.array([[1, 2], [1, np.inf]])
        b = np.array([3, 4])
        c = np.array([[1, 2], [3, 4]])
        ref = np.array([[[4, 6], [9, 12]],
                        [[4, 6], [np.inf, np.inf]]])
        res = gulinalg.update_rank1(a, b, c)
        assert_allclose(res, ref)

    @skipIf(parse_version(np.__version__) < parse_version('1.13'),
            "Prior to 1.13, numpy low level iterators didn't support removing "
            "empty axis. So gufunc couldn't be called with empty inner loop")
    def test_size_zero_vector(self):
        """Test broadcasting for matrix input of size zero"""
        a = np.ascontiguousarray(np.random.randn(10, 1))
        b = np.ascontiguousarray(np.random.randn(10, 0))
        c = np.ascontiguousarray(np.random.randn(10, 1, 0))
        res = gulinalg.update_rank1(a, b, c)
        assert res.shape == (10, 1, 0)
        ref = np.stack([np.dot(np.zeros((1, 0)), np.zeros((0, 0))) + c[i]
                        for i in range(len(c))])
        assert_allclose(res, ref)

    @skipIf(parse_version(np.__version__) < parse_version('1.13'),
            "Prior to 1.13, numpy low level iterators didn't support removing "
            "empty axis. So gufunc couldn't be called with empty inner loop")
    def test_size_zero_matrix(self):
        """Test broadcasting for matrix input of size zero"""
        a = np.ascontiguousarray(np.random.randn(10, 0))
        b = np.ascontiguousarray(np.random.randn(10, 2))
        c = np.ascontiguousarray(np.random.randn(10, 0, 2))
        res = gulinalg.update_rank1(a, b, c)
        assert res.shape == (10, 0, 2)
        ref = np.stack([np.dot(np.zeros((0, 0)), np.zeros((0, 2))) + c[i]
                        for i in range(len(c))])
        assert_allclose(res, ref)

    def test_size_one_vector(self):
        """Test broadcasting for vector inputs of size one"""
        a = np.ascontiguousarray(np.random.randn(10, 1))
        b = np.ascontiguousarray(np.random.randn(10, 1))
        c = np.ascontiguousarray(np.random.randn(10, 1, 1))
        res = gulinalg.update_rank1(a, b, c)
        assert res.shape == (10, 1, 1)
        ref = np.stack([np.dot(a[i].reshape(1, 1), b[i].reshape(1, 1)) + c[i]
                        for i in range(len(c))])
        assert_allclose(res, ref)


if __name__ == '__main__':
    run_module_suite()

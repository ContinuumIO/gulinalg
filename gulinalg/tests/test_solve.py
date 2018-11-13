"""
Tests different implementations of solve functions.
"""

from __future__ import print_function
from unittest import TestCase, skipIf
import numpy as np
from numpy.testing import run_module_suite, assert_allclose
from pkg_resources import parse_version
import gulinalg


class TestSolveTriangular(TestCase):
    """
    Test A * x = B and it's variants where A is a triangular matrix.

    Since names are abbreviated, here is what they mean:
      LO - A is a Lower triangular matrix.
      UP - A is a Upper diagonal matrix.
      TRANS N - No tranpose, T - Transpose, C - Conjuagte Transpose
      DIAG N - A is non-unit triangular, U - A is unit triangular
      B - By default B is a matrix, otherwise we specify it in test name.
    """
    def test_LO_TRANS_N_DIAG_N_B_VECTOR(self):
        """Test A * x = B where A is a lower triangular matrix"""
        a = np.array([[3, 0, 0, 0], [2, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1]])
        b = np.array([4, 2, 4, 2])
        x = gulinalg.solve_triangular(a, b)
        assert_allclose(np.dot(a, x), b)

    def test_UP_TRANS_N_DIAG_N(self):
        """Test A * x = B where A is a upper triangular matrix"""
        a = np.array([[1, 2, 3, 4], [0, 2, 3, 4], [0, 0, 3, 4], [0, 0, 0, 4]])
        b = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]])
        x = gulinalg.solve_triangular(a, b, UPLO='U')
        assert_allclose(np.dot(a, x), b)

    def test_UP_TRANS_T_DIAG_N(self):
        """Test A.T * x = B where A is a upper triangular matrix"""
        a = np.array([[1, 2, 3, 4], [0, 2, 3, 4], [0, 0, 3, 4], [0, 0, 0, 4]])
        b = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]])
        x = gulinalg.solve_triangular(a, b, UPLO='U', transpose_type='T')
        assert_allclose(np.dot(a.T, x), b)

    def test_UP_TRANS_C_DIAG_N(self):
        """Test A.H * x = B where A is a upper triangular matrix"""
        a = np.array([[1 + 2j, 2 + 2j], [0, 1 + 1j]])
        b = np.array([[1 + 0j, 0], [0, 1 + 0j]])
        ref = np.array([[0.2+0.4j, -0.0+0.j], [-0.4-0.8j, 0.5+0.5j]])
        x = gulinalg.solve_triangular(a, b, UPLO='U', transpose_type='C')
        assert_allclose(x, ref)

    def test_UP_TRANS_N_DIAG_U(self):
        """
        Test A * x = B where A is a upper triangular matrix and diagonal
        elements are considered unit diagonal.
        """
        a = np.array([[1, 2, 3, 4], [0, 2, 3, 4], [0, 0, 3, 4], [0, 0, 0, 4]])
        b = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]])
        res = gulinalg.solve_triangular(a, b, UPLO='U', unit_diagonal=True)
        # DIAG='U' assumes that diagonal elements are 1.
        a_unit_diag = np.array([[1, 2, 3, 4], [0, 1, 3, 4],
                                [0, 0, 1, 4], [0, 0, 0, 1]])
        ref = gulinalg.solve_triangular(a_unit_diag, b, UPLO='U')
        assert_allclose(res, ref)

    def test_UP_TRANS_T_DIAG_U(self):
        """
        Test A.T * x = B where A is a upper triangular matrix and diagonal
        elements are considered unit diagonal.
        """
        a = np.array([[1, 2, 3, 4], [0, 2, 3, 4], [0, 0, 3, 4], [0, 0, 0, 4]])
        b = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]])
        res = gulinalg.solve_triangular(
            a, b, UPLO='U', transpose_type='T', unit_diagonal=True)
        # DIAG='U' assumes that diagonal elements are 1.
        a_unit_diag = np.array([[1, 2, 3, 4], [0, 1, 3, 4],
                                [0, 0, 1, 4], [0, 0, 0, 1]])
        ref = gulinalg.solve_triangular(
            a_unit_diag, b, UPLO='U', transpose_type='T')
        assert_allclose(res, ref)

    def test_UP_TRANS_C_DIAG_U(self):
        """
        Test A.H * x = B where A is a upper triangular matrix and diagonal
        elements are considered unit diagonal.
        """
        a = np.array([[1 + 2j, 2 + 2j], [0, 1 + 1j]])
        b = np.array([[1, 0], [0, 1]])
        res = gulinalg.solve_triangular(
            a, b, UPLO='U', transpose_type='C', unit_diagonal=True)
        # DIAG='U' assumes that diagonal elements are 1.
        a_unit_diag = np.array([[1, 2 + 2j], [0, 1]])
        ref = gulinalg.solve_triangular(
            a_unit_diag, b, UPLO='U', transpose_type='C')
        assert_allclose(res, ref)

    def test_fortran_layout_matrix(self):
        """Input matrices have fortran layout"""
        a = np.asfortranarray([[1, 2, 3, 4], [0, 2, 3, 4],
                               [0, 0, 3, 4], [0, 0, 0, 4]])
        b = np.asfortranarray([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]])
        res = gulinalg.solve_triangular(
            a, b, UPLO='U', transpose_type='T', unit_diagonal=True)
        # DIAG='U' assumes that diagonal elements are 1.
        a_unit_diag = np.asfortranarray([[1, 2, 3, 4], [0, 1, 3, 4],
                                         [0, 0, 1, 4], [0, 0, 0, 1]])
        ref = gulinalg.solve_triangular(
            a_unit_diag, b, UPLO='U', transpose_type='T'
        )
        assert_allclose(res, ref)

    def test_input_matrix_non_contiguous(self):
        """Input matrix is not a contiguous matrix"""
        a = np.asfortranarray(
            [[[1, 2, 3, 4], [0, 2, 3, 4], [0, 0, 3, 4], [0, 0, 0, 4]],
             [[1, 2, 3, 4], [0, 2, 3, 4], [0, 0, 3, 4], [0, 0, 0, 4]]])[0]
        b = np.ascontiguousarray([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]])
        assert not a.flags.c_contiguous and not a.flags.f_contiguous
        x = gulinalg.solve_triangular(a, b, UPLO='U')
        assert_allclose(np.dot(a, x), b)

    @skipIf(parse_version(np.__version__) < parse_version('1.13'),
            "Prior to 1.13, numpy low level iterators didn't support removing "
            "empty axis. So gufunc couldn't be called with empty inner loop")
    def test_m_and_n_zero(self):
        """Corner case of solving where m = 0 and n = 0"""
        a = np.ascontiguousarray(np.random.randn(0, 0))
        b = np.ascontiguousarray(np.random.randn(0, 0))
        x = gulinalg.solve_triangular(a, b, UPLO='U')
        assert x.shape == (0, 0)
        assert_allclose(np.dot(a, x), b)

    @skipIf(parse_version(np.__version__) < parse_version('1.13'),
            "Prior to 1.13, numpy low level iterators didn't support removing "
            "empty axis. So gufunc couldn't be called with empty inner loop")
    def test_m_zero(self):
        """Corner case of solving where m = 0"""
        a = np.ascontiguousarray(np.random.randn(0, 0))
        b = np.ascontiguousarray(np.random.randn(0, 2))
        x = gulinalg.solve_triangular(a, b, UPLO='U')
        assert x.shape == (0, 2)
        assert_allclose(np.dot(a, x), b)

    @skipIf(parse_version(np.__version__) < parse_version('1.13'),
            "Prior to 1.13, numpy low level iterators didn't support removing "
            "empty axis. So gufunc couldn't be called with empty inner loop")
    def test_n_zero(self):
        """Corner case of solving where n = 0"""
        a = np.ascontiguousarray(np.random.randn(2, 2))
        b = np.ascontiguousarray(np.random.randn(2, 0))
        x = gulinalg.solve_triangular(a, b, UPLO='U')
        assert x.shape == (2, 0)
        assert_allclose(np.dot(a, x), b)

    def test_size_one_matrices(self):
        """Corner case of decomposing where m = 1 and n = 1"""
        a = np.ascontiguousarray(np.random.randn(1, 1))
        b = np.ascontiguousarray(np.random.randn(1, 1))
        x = gulinalg.solve_triangular(a, b, UPLO='U')
        assert x.shape == (1, 1)
        assert_allclose(np.dot(a, x), b)

    def test_vector(self):
        """test vectorized solve triangular"""
        e = np.array([[1, 2, 3, 4], [0, 2, 3, 4], [0, 0, 3, 4], [0, 0, 0, 4]])
        a = np.stack([e for _ in range(10)])
        b = np.stack([np.array([[1, 0, 0], [0, 1, 0],
                                [0, 0, 1], [0, 0, 0]]) for _ in range(10)])
        x = gulinalg.solve_triangular(a, b, UPLO='U')
        res = np.stack([np.dot(a[i], x[i]) for i in range(len(a))])
        assert_allclose(res, b)

    @skipIf(parse_version(np.__version__) < parse_version('1.13'),
            "Prior to 1.13, numpy low level iterators didn't support removing "
            "empty axis. So gufunc couldn't be called with empty inner loop")
    def test_vector_m_and_n_zero(self):
        """Corner case of solving where m = 0 and n = 0"""
        a = np.ascontiguousarray(np.random.randn(10, 0, 0))
        b = np.ascontiguousarray(np.random.randn(10, 0, 0))
        x = gulinalg.solve_triangular(a, b, UPLO='U')
        assert x.shape == (10, 0, 0)
        res = np.stack([np.dot(a[i], x[i]) for i in range(len(a))])
        assert_allclose(res, b)

    @skipIf(parse_version(np.__version__) < parse_version('1.13'),
            "Prior to 1.13, numpy low level iterators didn't support removing "
            "empty axis. So gufunc couldn't be called with empty inner loop")
    def test_vector_m_zero(self):
        """Corner case of solving where m = 0"""
        a = np.ascontiguousarray(np.random.randn(10, 0, 0))
        b = np.ascontiguousarray(np.random.randn(10, 0, 2))
        x = gulinalg.solve_triangular(a, b, UPLO='U')
        assert x.shape == (10, 0, 2)
        res = np.stack([np.dot(a[i], x[i]) for i in range(len(a))])
        assert_allclose(res, b)

    @skipIf(parse_version(np.__version__) < parse_version('1.13'),
            "Prior to 1.13, numpy low level iterators didn't support removing "
            "empty axis. So gufunc couldn't be called with empty inner loop")
    def test_vector_n_zero(self):
        """Corner case of solving where n = 0"""
        a = np.ascontiguousarray(np.random.randn(10, 2, 2))
        b = np.ascontiguousarray(np.random.randn(10, 2, 0))
        x = gulinalg.solve_triangular(a, b, UPLO='U')
        assert x.shape == (10, 2, 0)
        res = np.stack([np.dot(a[i], x[i]) for i in range(len(a))])
        assert_allclose(res, b)

    def test_vector_size_one_matrices(self):
        """Corner case of solving where m = 1 and n = 1"""
        a = np.ascontiguousarray(np.random.randn(10, 1, 1))
        b = np.ascontiguousarray(np.random.randn(10, 1, 1))
        x = gulinalg.solve_triangular(a, b, UPLO='U')
        assert x.shape == (10, 1, 1)
        res = np.stack([np.dot(a[i], x[i]) for i in range(len(a))])
        assert_allclose(res, b)

    def test_nan_handling(self):
        """NaN in one output shouldn't contaminate remaining outputs"""
        a = np.array([[[3, 0, 0], [np.nan, 1, 0], [1, 0, 1]],
                      [[3, 0, 0], [2, 1, 0], [1, 0, 1]]])
        b = np.array([[4, 2, 4], [4, 2, 4]])
        ref = np.array([[1.33333333, np.nan, np.nan],
                        [1.33333333, -0.66666667,  2.66666667]])
        res = gulinalg.solve_triangular(a, b)
        assert_allclose(res, ref)

    def test_infinity_handling(self):
        """Infinity in one output shouldn't contaminate remaining outputs"""
        a = np.array([[[3, 0, 0], [np.inf, 1, 0], [1, 0, 1]],
                      [[3, 0, 0], [2, 1, 0], [1, 0, 1]]])
        b = np.array([[4, 2, 4], [4, 2, 4]])
        ref = np.array([[1.33333333, -np.inf, np.nan],
                        [1.33333333, -0.66666667,  2.66666667]])
        res = gulinalg.solve_triangular(a, b)
        assert_allclose(res, ref)


if __name__ == '__main__':
    run_module_suite()

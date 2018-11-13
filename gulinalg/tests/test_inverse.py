"""
Tests different implementations of inverse functions.
"""

from __future__ import print_function
from unittest import TestCase, skipIf
import numpy as np
from numpy.testing import run_module_suite, assert_allclose
from pkg_resources import parse_version
import gulinalg


class TestInverseTriangular(TestCase):
    """
    Test A * A' = I (identity matrix) where A is a triangular matrix.
    """
    def test_lower_triangular_non_unit_diagonal(self):
        """
        Test A * A' = I where A is a lower triangular non unit diagonal matrix
        """
        a = np.array([[3, 0, 0, 0], [2, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1]])
        inva = gulinalg.inv_triangular(a)
        assert_allclose(np.dot(a, inva), np.identity(4))

    def test_lower_triangular_unit_diagonal(self):
        """
        Test inverse of non unit diagonal matrix against that of unit diagonal
        matrix.
        """
        a = np.array([[3, 0, 0, 0], [4, 2, 0, 0], [1, 0, 5, 0], [5, 6, 7, 6]])
        a_unit = np.array([[1, 0, 0, 0], [4, 1, 0, 0],
                           [1, 0, 1, 0], [5, 6, 7, 1]])
        inva = gulinalg.inv_triangular(a, unit_diagonal=True)
        inva_unit = gulinalg.inv_triangular(a_unit)
        # For a non-unit diagonal matrix, when unit_diagonal parameter is true
        # inv_triangular copies diagonal elements to output inverse matrix as
        # is. So change those diagonal elements to 1 before comparing against
        # inverse of a unit diagonal matrix.
        np.fill_diagonal(inva, 1)
        assert_allclose(inva, inva_unit)

    def test_upper_triangular_non_unit_diagonal(self):
        """
        Test A * A' = I where A is a upper triangular non unit diagonal matrix
        """
        a = np.array([[1, 2, 3, 4], [0, 2, 3, 4], [0, 0, 3, 4], [0, 0, 0, 4]])
        inva = gulinalg.inv_triangular(a, UPLO='U')
        assert_allclose(np.dot(a, inva), np.identity(4))

    def test_upper_triangular_unit_diagonal(self):
        """
        Test inverse of non unit diagonal matrix against that of unit diagonal
        matrix.
        """
        a = np.array([[5, 2, 3, 4], [0, 4, 3, 4], [0, 0, 2, 4], [0, 0, 0, 3]])
        a_unit = np.array([[1, 2, 3, 4], [0, 1, 3, 4],
                           [0, 0, 1, 4], [0, 0, 0, 1]])
        inva = gulinalg.inv_triangular(a, UPLO='U', unit_diagonal=True)
        inva_unit = gulinalg.inv_triangular(a_unit, UPLO='U')
        # For a non-unit diagonal matrix, when unit_diagonal parameter is true
        # inv_triangular copies diagonal elements to output inverse matrix as
        # is. So change those diagonal elements to 1 before comparing against
        # inverse of a unit diagonal matrix.
        np.fill_diagonal(inva, 1)
        assert_allclose(inva, inva_unit)

    def test_upper_for_complex_type(self):
        """Test A' where A's data type is complex"""
        a = np.array([[1 + 2j, 2 + 2j], [0, 1 + 1j]])
        inva = gulinalg.inv_triangular(a, UPLO='U')
        ref = np.array([[0.2-0.4j, -0.4+0.8j],
                        [0.0+0.j, 0.5-0.5j]])
        assert_allclose(inva, ref)

    def test_fortran_layout_matrix(self):
        """Input matrix is fortran layout matrix"""
        a = np.asfortranarray([[3, 0, 0, 0], [2, 1, 0, 0],
                               [1, 0, 1, 0], [1, 1, 1, 1]])
        inva = gulinalg.inv_triangular(a)
        assert_allclose(np.dot(a, inva), np.identity(4))

    def test_input_matrix_non_contiguous(self):
        """Input matrix is not a contiguous matrix"""
        a = np.asfortranarray(
            [[[3, 0, 0, 0], [2, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1]],
             [[3, 0, 0, 0], [2, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1]]])[0]
        assert not a.flags.c_contiguous and not a.flags.f_contiguous
        inva = gulinalg.inv_triangular(a)
        assert_allclose(np.dot(a, inva), np.identity(4))

    @skipIf(parse_version(np.__version__) < parse_version('1.13'),
            "Prior to 1.13, numpy low level iterators didn't support removing "
            "empty axis. So gufunc couldn't be called with empty inner loop")
    def test_m_zero(self):
        """Corner case of inverting where m = 0"""
        a = np.ascontiguousarray(np.random.randn(0, 0))
        inva = gulinalg.inv_triangular(a)
        assert inva.shape == (0, 0)
        assert_allclose(np.dot(a, inva), np.identity(0))

    def test_m_one(self):
        """Corner case of inverting where m = 1"""
        a = np.ascontiguousarray(np.random.randn(1, 1))
        inva = gulinalg.inv_triangular(a)
        assert inva.shape == (1, 1)
        assert_allclose(np.dot(a, inva), np.identity(1))

    def test_vector(self):
        """test vectorized inverse triangular"""
        e = np.array([[3, 0, 0, 0], [2, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1]])
        a = np.stack([e for _ in range(10)])
        ref = np.stack([np.identity(4) for _ in range(len(a))])
        inva = gulinalg.inv_triangular(a)
        res = np.stack([np.dot(a[i], inva[i]) for i in range(len(a))])
        assert_allclose(res, ref)

    @skipIf(parse_version(np.__version__) < parse_version('1.13'),
            "Prior to 1.13, numpy low level iterators didn't support removing "
            "empty axis. So gufunc couldn't be called with empty inner loop")
    def test_vector_m_zero(self):
        """Corner case of inverting matrices where m = 0"""
        a = np.ascontiguousarray(np.random.randn(10, 0, 0))
        ref = np.stack([np.identity(0) for _ in range(len(a))])
        inva = gulinalg.inv_triangular(a)
        assert inva.shape == (10, 0, 0)
        res = np.stack([np.dot(a[i], inva[i]) for i in range(len(a))])
        assert_allclose(res, ref)

    def test_vector_m_one(self):
        """Corner case of inverting matrices where m = 1"""
        a = np.ascontiguousarray(np.random.randn(10, 1, 1))
        ref = np.stack([np.identity(1) for _ in range(len(a))])
        inva = gulinalg.inv_triangular(a)
        assert inva.shape == (10, 1, 1)
        res = np.stack([np.dot(a[i], inva[i]) for i in range(len(a))])
        assert_allclose(res, ref)

    def test_nan_handling(self):
        """NaN in one output shouldn't contaminate remaining outputs"""
        a = np.array([[[3, 0, 0], [2, 1, 0], [1, 0, 1]],
                      [[3, 0, 0], [np.nan, 1, 0], [1, 0, 1]]])
        ref = np.array([[[0.33333333, 0., 0.],
                         [-0.66666667, 1., 0.],
                         [-0.33333333, -0., 1.]],
                        [[0.33333333, 0.,  0.],
                         [np.nan,     1.,  0.],
                         [np.nan,    -0.,  1.]]])
        res = gulinalg.inv_triangular(a)
        assert_allclose(res, ref)

    def test_infinity_handling(self):
        """Infinity in one output shouldn't contaminate remaining outputs"""
        a = np.array([[[3, 0, 0], [2, 1, 0], [1, 0, 1]],
                      [[3, 0, 0], [np.inf, 1, 0], [1, 0, 1]]])
        ref = np.array([[[0.33333333,   0., 0.],
                         [-0.66666667,  1., 0.],
                         [-0.33333333, -0., 1.]],
                        [[0.33333333, 0.,  0.],
                         [-np.inf,     1.,  0.],
                         [np.nan,    -0.,  1.]]])
        res = gulinalg.inv_triangular(a)
        assert_allclose(res, ref)


if __name__ == '__main__':
    run_module_suite()

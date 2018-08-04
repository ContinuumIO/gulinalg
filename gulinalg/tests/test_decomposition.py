"""
Tests different implementations of decomposition functions.
"""

from __future__ import print_function
from unittest import TestCase
import numpy as np
from numpy.testing import run_module_suite, assert_allclose
import gulinalg


class TestLU(TestCase):
    """
    Test LU decomposition.
    """
    def test_lu_m_gt_n(self):
        """LU decompose a matrix where dimension m > n"""
        a = np.ascontiguousarray(np.random.randn(75, 50))
        p, l, u = gulinalg.lu(a)
        assert_allclose(np.dot(np.dot(p, l), u), a)

    def test_lu_m_lt_n(self):
        """LU decompose a matrix where dimension m < n"""
        a = np.ascontiguousarray(np.random.randn(50, 75))
        p, l, u = gulinalg.lu(a)
        assert_allclose(np.dot(np.dot(p, l), u), a)

    def test_lu_size_1_matrix(self):
        """Corner case of decomposing size 1 matrix"""
        a = np.ascontiguousarray(np.random.randn(1, 1))
        p, l, u = gulinalg.lu(a)
        assert_allclose(np.dot(np.dot(p, l), u), a)

    def test_lu_permute_m_gt_n(self):
        """LU decompose a matrix where dimension m > n"""
        a = np.ascontiguousarray(np.random.randn(75, 50))
        pl, u = gulinalg.lu(a, permute_l=True)
        assert_allclose(np.dot(pl, u), a)

    def test_lu_permute_m_lt_n(self):
        """LU decompose a matrix where dimension m < n"""
        a = np.ascontiguousarray(np.random.randn(50, 75))
        pl, u = gulinalg.lu(a, permute_l=True)
        assert_allclose(np.dot(pl, u), a)

    def test_lu_permute_size_1_matrix(self):
        a = np.ascontiguousarray(np.random.randn(1, 1))
        pl, u = gulinalg.lu(a, permute_l=True)
        assert_allclose(np.dot(pl, u), a)

    def test_complex_numbers(self):
        """Test for complex numbers input."""
        a = np.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + -8j]])
        p, l, u = gulinalg.lu(a)
        assert_allclose(np.dot(np.dot(p, l), u), a)

    def test_fortran_layout_matrix(self):
        """Input matrix is fortran layout matrix"""
        a = np.asfortranarray(np.random.randn(75, 50))
        pl, u = gulinalg.lu(a, permute_l=True)
        assert_allclose(np.dot(pl, u), a)

    def test_input_matrix_non_contiguous(self):
        """Input matrix is not a contiguous matrix"""
        a = np.ascontiguousarray(np.random.randn(50, 75, 2))[:, :, 0]
        assert not a.flags.c_contiguous and not a.flags.f_contiguous
        p, l, u = gulinalg.lu(a)
        assert_allclose(np.dot(np.dot(p, l), u), a)

    def test_vector(self):
        """test vectorized LU decomposition"""
        a = np.ascontiguousarray(np.random.randn(10, 75, 50))
        p, l, u = gulinalg.lu(a)
        res = np.stack([np.dot(np.dot(p[i], l[i]), u[i])
                        for i in range(len(a))])
        assert_allclose(res, a)

    def test_nan_handling(self):
        """NaN in one output shouldn't contaminate remaining outputs"""
        a = np.array([[[3, 4, 5], [2, np.nan, 3]],
                      [[3, 4, 5], [2, 1, 3]]])
        pl, u = gulinalg.lu(a, permute_l=True)
        ref_pl = np.array([[[1., 0.], [0.66666667, 1.]],
                           [[1., 0.], [0.66666667, 1.]]])
        ref_u = np.array([[[3., 4., 5.], [0., np.nan, -0.33333333]],
                          [[3., 4., 5.], [0., -1.66666667, -0.33333333]]])
        assert_allclose(pl, ref_pl)
        assert_allclose(u, ref_u)

    def test_infinity_handling(self):
        """Infinity in one output shouldn't contaminate remaining outputs"""
        a = np.array([[[3, 4, 5], [2, np.inf, 3]],
                      [[3, 4, 5], [2, 1, 3]]])
        pl, u = gulinalg.lu(a, permute_l=True)
        ref_pl = np.array([[[1., 0.], [0.66666667, 1.]],
                           [[1., 0.], [0.66666667, 1.]]])
        ref_u = np.array([[[3., 4., 5.], [0., np.inf, -0.33333333]],
                          [[3., 4., 5.], [0., -1.66666667, -0.33333333]]])
        assert_allclose(pl, ref_pl)
        assert_allclose(u, ref_u)


class TestQR(TestCase):
    """Test QR decomposition"""
    def test_m_lt_n(self):
        """For M < N, return full MxM Q matrix and all M rows of R."""
        m = 50
        n = 75
        a = np.ascontiguousarray(np.random.randn(m, n))
        q, r = gulinalg.qr(a)
        assert_allclose(np.dot(q, q.T), np.identity(m), atol=1e-15)
        assert_allclose(np.dot(q, r), a)

    def test_m_gt_n(self):
        """For M > N, return full MxM Q matrix and all M rows of R."""
        m = 75
        n = 50
        a = np.ascontiguousarray(np.random.randn(m, n))
        q, r = gulinalg.qr(a)
        assert q.shape == (m, m)
        assert r.shape == (m, n)
        assert_allclose(np.dot(q, q.T), np.identity(m), atol=1e-15)
        assert_allclose(np.dot(q, r), a)

    def test_m_gt_n_economy(self):
        """
        If M > N, economy mode returns only N columns for Q and N rows for R.
        """
        m = 75
        n = 50
        a = np.ascontiguousarray(np.random.randn(m, n))
        q, r = gulinalg.qr(a, economy=True)
        assert q.shape == (m, n)
        assert r.shape == (n, n)
        assert_allclose(np.dot(q, r), a)

    def test_qr_size_1_matrix(self):
        """Corner case of decomposing size 1 matrix"""
        m = 1
        n = 1
        a = np.ascontiguousarray(np.random.randn(m, n))
        q, r = gulinalg.qr(a)
        assert q.shape == (m, n)
        assert r.shape == (n, n)
        assert_allclose(np.dot(q, r), a)

    def test_complex_numbers(self):
        """Test for complex numbers input."""
        a = np.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + -8j]])
        q, r = gulinalg.qr(a)
        assert_allclose(np.dot(q, r), a)

    def test_fortran_layout_matrix(self):
        """Input matrix is fortran layout matrix"""
        m = 75
        n = 50
        a = np.asfortranarray(np.random.randn(m, n))
        q, r = gulinalg.qr(a, economy=True)
        assert q.shape == (m, n)
        assert r.shape == (n, n)
        assert_allclose(np.dot(q, r), a)

    def test_input_matrix_non_contiguous(self):
        """Input matrix is not a contiguous matrix"""
        m = 75
        n = 50
        a = np.ascontiguousarray(np.random.randn(m, n, 2))[:, :, 0]
        assert not a.flags.c_contiguous and not a.flags.f_contiguous
        q, r = gulinalg.qr(a, economy=True)
        assert q.shape == (m, n)
        assert r.shape == (n, n)
        assert_allclose(np.dot(q, r), a)

    def test_vector(self):
        """test vectorized QR decomposition"""
        a = np.ascontiguousarray(np.random.randn(10, 75, 50))
        q, r = gulinalg.qr(a)
        res = np.stack([np.dot(q[i], r[i]) for i in range(len(a))])
        assert_allclose(res, a)

    def test_nan_handling(self):
        """NaN in one output shouldn't contaminate remaining outputs"""
        a = np.array([[[3, 4, 5], [np.nan, 1, 3]],
                      [[3, 4, 5], [2, 1, 3]]])
        q, r = gulinalg.qr(a)
        ref_q = np.array([[[np.nan, np.nan],
                           [np.nan, np.nan]],
                          [[-0.83205029, -0.5547002],
                           [-0.5547002, 0.83205029]]])
        ref_r = np.array([[[np.nan, np.nan, np.nan],
                           [0., np.nan, np.nan]],
                          [[-3.60555128, -3.88290137, -5.82435206],
                           [0., -1.38675049, -0.2773501]]])
        assert_allclose(q, ref_q)
        assert_allclose(r, ref_r)

    def test_infinity_handling(self):
        """Infinity in one output shouldn't contaminate remaining outputs"""
        a = np.array([[[3, 4, 5], [np.inf, 1, 3]],
                      [[3, 4, 5], [2, 1, 3]]])
        q, r = gulinalg.qr(a)
        ref_q = np.array([[[np.nan, np.nan], [np.nan, np.nan]],
                          [[-0.83205029, -0.5547002],
                           [-0.5547002, 0.83205029]]])
        ref_r = np.array([[[-np.inf, np.nan, np.nan],
                           [0., np.nan, np.nan]],
                          [[-3.60555128, -3.88290137, -5.82435206],
                           [0., -1.38675049, -0.2773501]]])
        assert_allclose(q, ref_q)
        assert_allclose(r, ref_r)


if __name__ == '__main__':
    run_module_suite()

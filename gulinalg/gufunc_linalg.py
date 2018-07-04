"""Linear Algebra functions implemented as gufuncs, broadcasting.

This file contains the wrappers for several linear algebra functions
as gufuncs. The underlying implementation is LAPACK based.

- det
- slogdet
- cholesky
- eig
- eigvals
- eigh
- eigvalsh
- solve
- svd
- chosolve
- inv
- poinv
- solve_triangular
- inv_triangular
- lu
- qr

"""

from __future__ import division, absolute_import, print_function


import numpy as np
from . import _impl
from .gufunc_general import matrix_multiply


def det(a, **kwargs):
    """
    Compute the determinant of arrays, with broadcasting.

    Parameters
    ----------
    a : (NDIMS, M, M) array
        Input array. Its inner dimensions must be those of a square 2-D array.

    Returns
    -------
    det : (NDIMS) array
        Determinants of `a`

    See Also
    --------
    slogdet : Another representation for the determinant, more suitable
        for large matrices where underflow/overflow may occur

    Notes
    -----
    Numpy broadcasting rules apply.

    The determinants are computed via LU factorization using the LAPACK
    routine _getrf.

    Implemented for single, double, csingle and cdouble. Numpy conversion
    rules apply.

    Examples
    --------
    The determinant of a 2-D array [[a, b], [c, d]] is ad - bc:

    >>> a = np.array([[1, 2], [3, 4]])
    >>> np.allclose(-2.0, det(a))
    True

    >>> a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]] ])
    >>> np.allclose(-2.0, det(a))
    True

    """
    return _impl.det(a, **kwargs)


def slogdet(a, **kwargs):
    """
    Compute the sign and (natural) logarithm of the determinant of an array,
    with broadcasting.

    If an array has a very small or very large determinant, then a call to
    `det` may overflow or underflow. This routine is more robust against such
    issues, because it computes the logarithm of the determinant rather than
    the determinant itself

    Parameters
    ----------
    a : (..., M, M) array
        Input array. Its inner dimensions must be those of a square 2-D array.

    Returns
    -------
    sign : (...) array
        An array of numbers representing the sign of the determinants. For real
        matrices, this is 1, 0, or -1. For complex matrices, this is a complex
        number with absolute value 1 (i.e., it is on the unit circle), or else
        0.
    logdet : (...) array
        The natural log of the absolute value of the determinant. This is always
        a real type.

    If the determinant is zero, then `sign` will be 0 and `logdet` will be -Inf.
    In all cases, the determinant is equal to ``sign * np.exp(logdet)``.

    See Also
    --------
    det

    Notes
    -----
    Numpy broadcasting rules apply.

    The determinants are computed via LU factorization using the LAPACK
    routine _getrf.

    Implemented for types single, double, csingle and cdouble. Numpy conversion
    rules apply. For complex versions `logdet` will be of the associated real
    type (single for csingle, double for cdouble).

    Examples
    --------
    The determinant of a 2-D array [[a, b], [c, d]] is ad - bc:

    >>> a = np.array([[1, 2], [3, 4]])
    >>> (sign, logdet) = slogdet(a)
    >>> sign.shape == ()
    True
    >>> logdet.shape == ()
    True
    >>> np.allclose(-2.0, sign * np.exp(logdet))
    True

    >>> a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]] ])
    >>> (sign, logdet) = slogdet(a)
    >>> sign.shape == (2,)
    True
    >>> logdet.shape == (2,)
    True
    >>> np.allclose(-2.0, sign * np.exp(logdet))
    True

    """
    return _impl.slogdet(a, **kwargs)


def inv(a, **kwargs):
    """
    Compute the (multiplicative) inverse of matrices, with broadcasting.

    Given a square matrix `a`, return the matrix `ainv` satisfying
    ``matrix_multiply(a, ainv) = matrix_multiply(ainv, a) = Identity matrix``

    Parameters
    ----------
    a : (..., M, M) array
        Matrices to be inverted

    Returns
    -------
    ainv : (..., M, M) array
        (Multiplicative) inverse of the `a` matrices.

    Notes
    -----
    Numpy broadcasting rules apply.

    Implemented for types single, double, csingle and cdouble. Numpy conversion
    rules apply.

    Singular matrices and thus, not invertible, result in an array of NaNs.

    See Also
    --------
    poinv : compute the multiplicative inverse of hermitian/symmetric matrices,
            using cholesky decomposition.

    Examples
    --------
    >>> a = np.array([[1, 2], [3, 4]])
    >>> ainv = inv(a)
    >>> np.allclose(matrix_multiply(a, ainv), np.eye(2))
    True
    >>> np.allclose(matrix_multiply(ainv, a), np.eye(2))
    True

    """
    return _impl.inv(a, **kwargs)


def inv_triangular(a, UPLO='L', unit_diagonal=False, **kwargs):
    """
    Computes the inverse of a real triangular matrix, with broadcasting.

    Given a triangular matrix `a`, return the matrix `ainv` satisfying
    ``matrix_multiply(a, ainv) = matrix_multiply(ainv, a) = Identity matrix``

    Parameters
    ----------
    a : (..., M, M) array
        Matrices to be inverted

    UPLO : {'U', 'L'}, optional
         Specifies whether the calculation is done with the lower
         triangular part of the elements in `a` ('L', default) or
         the upper triangular part ('U').

    unit_diagonal : bool, optional
         If True, the diagonal elements of 'a' are assumed to be 1 and are
         not referenced.

    Returns
    -------
    ainv : (..., M, M) array
        (Multiplicative) inverse of the `a` matrices.

    Notes
    -----
    Numpy broadcasting rules apply.

    The solutions are computed using LAPACK routine _trtri.

    Implemented for types single, double, csingle and cdouble. Numpy conversion
    rules apply.

    Singular matrices and thus, not invertible, result in an array of NaNs.

    See Also
    --------
    inv : compute the (multiplicative) inverse of matrices, with broadcasting.

    poinv : compute the multiplicative inverse of hermitian/symmetric matrices,
            using cholesky decomposition.

    Examples
    --------
    >>> a = np.array([[1, 2], [0, 3]])
    >>> ainv = inv_triangular(a, UPLO='U')
    >>> np.allclose(matrix_multiply(a, ainv), np.eye(2))
    True
    >>> np.allclose(matrix_multiply(ainv, a), np.eye(2))
    True

    """
    uplo_choices = ['U', 'L']
    if UPLO not in uplo_choices:
        raise ValueError("Invalid UPLO argument '%s', valid values are: %s" %
                         (UPLO, uplo_choices))

    if 'U' == UPLO:
        if unit_diagonal:
            gufunc = _impl.inv_upper_triangular_unit_diagonal
        else:
            gufunc = _impl.inv_upper_triangular_non_unit_diagonal
    else:
        if unit_diagonal:
            gufunc = _impl.inv_lower_triangular_unit_diagonal
        else:
            gufunc = _impl.inv_lower_triangular_non_unit_diagonal

    return gufunc(a, **kwargs)


def cholesky(a, UPLO='L', **kwargs):
    """
    Compute the cholesky decomposition of `a`, with broadcasting

    The Cholesky decomposition (or Cholesky triangle) is a decomposition of a
    Hermitian, positive-definite matrix into the product of a lower triangular
    matrix and its conjugate transpose.

    A = LL*

    where L* is the positive-definite matrix.

    Parameters
    ----------
    a : (..., M, M) array
        Matrices for which compute the cholesky decomposition

    Returns
    -------
    l : (..., M, M) array
        Matrices for each element where each entry is the lower triangular
        matrix with strictly positive diagonal entries such that a = ll* for
        all outer dimensions

    See Also
    --------
    chosolve : solve a system using cholesky decomposition
    poinv : compute the inverse of a matrix using cholesky decomposition

    Notes
    -----
    Numpy broadcasting rules apply.

    Implemented for types single, double, csingle and cdouble. Numpy conversion
    rules apply.

    Decomposition is performed using LAPACK routine _potrf.

    For elements where the LAPACK routine fails, the result will be set to NaNs.

    If an element of the source array is not a positive-definite matrix the
    result for that element is undefined.

    Examples
    --------
    >>> A = np.array([[1,0-2j],[0+2j,5]])
    >>> A
    array([[ 1.+0.j,  0.-2.j],
           [ 0.+2.j,  5.+0.j]])
    >>> L = cholesky(A)
    >>> L
    array([[ 1.+0.j,  0.+0.j],
           [ 0.+2.j,  1.+0.j]])

    """
    if 'L' == UPLO:
        gufunc = _impl.cholesky_lo
    else:
        gufunc = _impl.cholesky_up

    return gufunc(a, **kwargs)


def eig(a, **kwargs):
    """
    Compute the eigenvalues and right eigenvectors of square arrays,
    with broadcasting

    Parameters
    ----------
    a : (..., M, M) array
        Matrices for which the eigenvalues and right eigenvectors will
        be computed

    Returns
    -------
    w : (..., M) array
        The eigenvalues, each repeated according to its multiplicity.
        The eigenvalues are not necessarily ordered. The resulting
        array will be always be of complex type. When `a` is real
        the resulting eigenvalues will be real (0 imaginary part) or
        occur in conjugate pairs

    v : (..., M, M) array
        The normalized (unit "length") eigenvectors, such that the
        column ``v[:,i]`` is the eigenvector corresponding to the
        eigenvalue ``w[i]``.

    See Also
    --------
    eigvals : eigenvalues of general arrays.
    eigh : eigenvalues and right eigenvectors of symmetric/hermitian
        arrays.
    eigvalsh : eigenvalues of symmetric/hermitian arrays.

    Notes
    -----
    Numpy broadcasting rules apply.

    Implemented for types single, double, csingle and cdouble. Numpy
    conversion rules apply.

    Eigenvalues and eigenvectors for single and double versions will
    always be typed csingle and cdouble, even if all the results are
    real (imaginary part will be 0).

    This is implemented using the _geev LAPACK routines which compute
    the eigenvalues and eigenvectors of general square arrays.

    For elements where the LAPACK routine fails, the result will be set
    to NaNs.

    Examples
    --------
    First, a utility function to check if eigvals/eigvectors are correct.
    This checks the definition of eigenvectors. For each eigenvector v
    with associated eigenvalue w of a matrix M the following equality must
    hold: Mv == wv

    >>> def check_eigen(M, w, v):
    ...     '''vectorial check of Mv==wv'''
    ...     lhs = matrix_multiply(M, v)
    ...     rhs = w*v
    ...     return np.allclose(lhs, rhs)

    (Almost) Trivial example with real e-values and e-vectors. Note
    the complex types of the results

    >>> M = np.diag((1,2,3)).astype(float)
    >>> w, v = eig(M)
    >>> check_eigen(M, w, v)
    True

    Real matrix possessing complex e-values and e-vectors; note that the
    e-values are complex conjugates of each other.

    >>> M = np.array([[1, -1], [1, 1]])
    >>> w, v = eig(M)
    >>> check_eigen(M, w, v)
    True

    Complex-valued matrix with real e-values (but complex-valued e-vectors);
    note that a.conj().T = a, i.e., a is Hermitian.

    >>> M = np.array([[1, 1j], [-1j, 1]])
    >>> w, v = eig(M)
    >>> check_eigen(M, w, v)
    True

    """
    return _impl.eig(a, **kwargs)


def eigvals(a, **kwargs):
    """
    Compute the eigenvalues of general matrices, with broadcasting.

    Main difference between `eigvals` and `eig`: the eigenvectors aren't
    returned.

    Parameters
    ----------
    a : (..., M, M) array
        Matrices whose eigenvalues will be computed

    Returns
    -------
    w : (..., M) array
        The eigenvalues, each repeated according to its multiplicity.
        The eigenvalues are not necessarily ordered. The resulting
        array will be always be of complex type. When `a` is real
        the resulting eigenvalues will be real (0 imaginary part) or
        occur in conjugate pairs

    See Also
    --------
    eig : eigenvalues and right eigenvectors of general arrays.
    eigh : eigenvalues and right eigenvectors of symmetric/hermitian
        arrays.
    eigvalsh : eigenvalues of symmetric/hermitian arrays.

    Notes
    -----
    Numpy broadcasting rules apply.

    Implemented for types single, double, csingle and cdouble. Numpy
    conversion rules apply.

    Eigenvalues for single and double versions will always be typed
    csingle and cdouble, even if all the results are real (imaginary
    part will be 0).

    This is implemented using the _geev LAPACK routines which compute
    the eigenvalues and eigenvectors of general square arrays.

    For elements where the LAPACK routine fails, the result will be set
    to NaNs.

    Examples
    --------

    Eigenvalues for a diagonal matrix are its diagonal elements

    >>> D = np.diag((-1,1))
    >>> np.sort(eigvals(D))
    array([-1.+0.j,  1.+0.j])

    Multiplying on the left by an orthogonal matrix, `Q`, and on the
    right by `Q.T` (the transpose of `Q` preserves the eigenvalues of
    the original matrix

    >>> x = np.random.random()
    >>> Q = np.array([[np.cos(x), -np.sin(x)], [np.sin(x), np.cos(x)]])
    >>> A = matrix_multiply(Q, D)
    >>> A = matrix_multiply(A, Q.T)
    >>> np.sort(eigvals(A))
    array([-1.+0.j,  1.+0.j])

    """
    return _impl.eigvals(a, **kwargs)


def eigh(A, UPLO='L', **kw_args):
    """
    Computes the eigenvalues and eigenvectors for the square matrices
    in the inner dimensions of A, being those matrices
    symmetric/hermitian.

    Parameters
    ----------
    A : (..., M, M) array
         Hermitian/Symmetric matrices whose eigenvalues and
         eigenvectors are to be computed.
    UPLO : {'L', 'U'}, optional
         Specifies whether the calculation is done with the lower
         triangular part of the elements in `A` ('L', default) or
         the upper triangular part ('U').

    Returns
    -------
    w : (..., M) array
        The eigenvalues, not necessarily ordered.
    v : (..., M, M) array
        The inner dimensions contain matrices with the normalized
        eigenvectors as columns. The column-numbers are coherent with
        the associated eigenvalue.

    Notes
    -----
    Numpy broadcasting rules apply.

    The eigenvalues/eigenvectors are computed using LAPACK routines _ssyevd,
    _heevd

    For elements where the LAPACK routine fails, the result will be set
    to NaNs.

    Implemented for single, double, csingle and cdouble. Numpy conversion
    rules apply.

    Unlike eig, the results for single and double will always be single
    and doubles. It is not possible for symmetrical real matrices to result
    in complex eigenvalues/eigenvectors

    See Also
    --------
    eigvalsh : eigenvalues of symmetric/hermitian arrays.
    eig : eigenvalues and right eigenvectors for general matrices.
    eigvals : eigenvalues for general matrices.

    Examples
    --------
    First, a utility function to check if eigvals/eigvectors are correct.
    This checks the definition of eigenvectors. For each eigenvector v
    with associated eigenvalue w of a matrix M the following equality must
    hold: Mv == wv

    >>> def check_eigen(M, w, v):
    ...     '''vectorial check of Mv==wv'''
    ...     lhs = matrix_multiply(M, v)
    ...     rhs = w*v
    ...     return np.allclose(lhs, rhs)

    A simple example that computes eigenvectors and eigenvalues of
    a hermitian matrix and checks that A*v = v*w for both pairs of
    eignvalues(w) and eigenvectors(v)

    >>> M = np.array([[1, -2j], [2j, 1]])
    >>> w, v = eigh(M)
    >>> check_eigen(M, w, v)
    True

    """
    if 'L' == UPLO:
        gufunc = _impl.eigh_lo
    else:
        gufunc = _impl.eigh_up

    return gufunc(A, **kw_args)


def eigvalsh(A, UPLO='L', **kw_args):
    """
    Computes the eigenvalues for the square matrices in the inner
    dimensions of A, being those matrices symmetric/hermitian.

    Parameters
    ----------
    A : (..., M, M) array
         Hermitian/Symmetric matrices whose eigenvalues and
         eigenvectors are to be computed.
    UPLO : {'L', 'U'}, optional
         Specifies whether the calculation is done with the lower
         triangular part of the elements in `A` ('L', default) or
         the upper triangular part ('U').

    Returns
    -------
    w : (..., M) array
        The eigenvalues, not necessarily ordered.

    Notes
    -----
    Numpy broadcasting rules apply.

    The eigenvalues are computed using LAPACK routines _ssyevd, _heevd

    For elements where the LAPACK routine fails, the result will be set
    to NaNs.

    Implemented for single, double, csingle and cdouble. Numpy conversion
    rules apply.

    Unlike eigvals, the results for single and double will always be single
    and doubles. It is not possible for symmetrical real matrices to result
    in complex eigenvalues.

    See Also
    --------
    eigh : eigenvalues of symmetric/hermitian arrays.
    eig : eigenvalues and right eigenvectors for general matrices.
    eigvals : eigenvalues for general matrices.

    Examples
    --------
    eigvalsh results should be the same as the eigenvalues returned by eigh

    >>> a = np.array([[1, -2j], [2j, 5]])
    >>> w, v = eigh(a)
    >>> np.allclose(w, eigvalsh(a))
    True

    eigvalsh on an identity matrix is all ones
    >>> eigvalsh(np.eye(6))
    array([ 1.,  1.,  1.,  1.,  1.,  1.])

    """
    if ('L' == UPLO):
        gufunc = _impl.eigvalsh_lo
    else:
        gufunc = _impl.eigvalsh_up

    return gufunc(A,**kw_args)


def solve(A,B,**kw_args):
    """
    Solve the linear matrix equations on the inner dimensions.

    Computes the "exact" solution, `x`. of the well-determined,
    i.e., full rank, linear matrix equations `ax = b`.

    Parameters
    ----------
    A : (..., M, M) array
        Coefficient matrices.
    B : (..., M, N) array
        Ordinate or "dependent variable" values.

    Returns
    -------
    X : (..., M, N) array
        Solutions to the system A X = B for all the outer dimensions

    Notes
    -----
    Numpy broadcasting rules apply.

    The solutions are computed using LAPACK routine _gesv

    For elements where the LAPACK routine fails, the result will be set
    to NaNs.

    Implemented for single, double, csingle and cdouble. Numpy conversion
    rules apply.

    See Also
    --------
    chosolve : solve a system using cholesky decomposition (for equations
               having symmetric/hermitian coefficient matrices)

    solve_triangular : Solve the linear matrix equation A * X = B
               where A is a triangular matrix.

    Examples
    --------
    Solve the system of equations ``3 * x0 + x1 = 9`` and ``x0 + 2 * x1 = 8``:

    >>> a = np.array([[3,1], [1,2]])
    >>> b = np.array([9,8])
    >>> x = solve(a, b)
    >>> x
    array([ 2.,  3.])

    Check that the solution is correct:

    >>> np.allclose(np.dot(a, x), b)
    True

    """
    if len(B.shape) == (len(A.shape) - 1):
        gufunc = _impl.solve1
    else:
        gufunc = _impl.solve

    return gufunc(A,B,**kw_args)


def lu(a, permute_l=False, **kw_args):
    """
    LU decomposition on the inner dimensions.

    Computes LU decomposition for a general M x N matrix using partial
    pivoting with row interchanges. It decomposes A to:

    A = P * L * U

    where P is a permutation matrix, L is lower triangular with unit diagonal
    elements and U is upper triangular matrix.

    Parameters
    ----------
    a : (..., M, N) array
        Matrices for which compute the lu decomposition
    permute_l : bool
        Instead of returning P, L and U (default), return P*L and U.

    Returns
    -------
    ** if permute_l == False **

    p : (..., M, M) array
        Permutation matrix
    l : (..., M, K) array
        Lower triangular matrix with unit diagonal where K = min(M, N)
    u : (..., K, N) array
        Upper triangular matrix

    ** if permute_l == True **

    pl : (..., M, K) array
        Permuted lower triangular matrix
    u : (..., K, N) array
        Upper triangular matrix

    See Also
    --------
    cholesky : Computes cholesky decomposition

    ldl : Computes ldl decomposition

    qr : Computes QR decomposition.

    Notes
    -----
    Numpy broadcasting rules apply.

    LU decomposition is performed using LAPACK routine _getrf.

    For elements where the LAPACK routine fails, the result will be set
    to NaNs.

    Implemented for types single, double, csingle and cdouble. Numpy conversion
    rules apply.

    Examples
    --------
    >>> a = np.random.randn(3, 4) + 1j*np.random.randn(3, 4)
    >>> p, l, u = lu(a)
    >>> (p.shape, l.shape, u.shape) == ((3, 3), (3, 3), (3, 4))
    True
    >>> np.allclose(a, np.dot(np.dot(p, l), u))
    True

    >>> pl, u = lu(a, permute_l=True)
    >>> (pl.shape, u.shape) == ((3, 3), (3, 4))
    True
    >>> np.allclose(a, np.dot(pl, u))
    True

    """
    m = a.shape[-2]
    n = a.shape[-1]

    if permute_l:
        if m < n:
            gufunc = _impl.lu_permute_m
        else:
            gufunc = _impl.lu_permute_n
    else:
        if m < n:
            gufunc = _impl.lu_m
        else:
            gufunc = _impl.lu_n

    return gufunc(a, **kw_args)


def qr(a, economy=False, **kw_args):
    """
    QR decomposition of general matrices on the inner dimensions.

    Computes QR decomposition for a general M x N matrix i.e.

    A = Q * R

    where Q is a unitary matrix and R is a upper triangular matrix.

    Parameters
    ----------
    a : (..., M, N) array
        Matrices for which compute the qr decomposition
    economy : bool
        If True and M > N, return only first N columns of Q and first N
        rows of R.

    Returns
    -------
    Q : (..., M, M) for economy False (default)
        (..., M, K) for economy True where K is min(M, N).

    R : (..., M, N) for economy False (default)
        (..., K, N) for economy True where K is min(M, N).

    See Also
    --------
    cholesky : Computes cholesky decomposition

    ldl      : Computes ldl decomposition

    lu       : Compute LU decomposition using partial pivoting and
               row interchange.

    Notes
    -----
    Numpy broadcasting rules apply.

    QR decomposition is performed using LAPACK routine _geqrf.

    For elements where the LAPACK routine fails, the result will be set
    to NaNs.

    Implemented for types single, double, csingle and cdouble. Numpy conversion
    rules apply.

    Examples
    --------
    >>> a = np.random.randn(3, 4) + 1j*np.random.randn(3, 4)
    >>> q, r = qr(a)
    >>> (q.shape, r.shape) == ((3, 3), (3, 4))
    True
    >>> np.allclose(a, np.dot(q, r))
    True

    """
    m = a.shape[-2]
    n = a.shape[-1]

    if m > n and economy:
        gufunc = _impl.qr_n
    else:
        gufunc = _impl.qr_m

    return gufunc(a, **kw_args)


def svd(a, full_matrices=1, compute_uv=1 ,**kw_args):
    """
    Singular Value Decomposition on the inner dimensions.

    Factors the matrices in `a` as ``u * np.diag(s) * v``, where `u`
    and `v` are unitary and `s` is a 1-d array of `a`'s singular
    values.

    Parameters
    ----------
    a : (..., M, N) array
        The array of matrices to decompose.
    full_matrices : bool, optional
        If True (default), `u` and `v` have the shapes (`M`, `M`) and
        (`N`, `N`), respectively. Otherwise, the shapes are (`M`, `K`)
        and (`K`, `N`), respectively, where `K` = min(`M`, `N`).
    compute_uv : bool, optional
        Whether or not to compute `u` and `v` in addition to `s`. True
        by default.

    Returns
    -------
    u : { (..., M, M), (..., M, K) } array
        Unitary matrices. The actual shape depends on the value of
        ``full_matrices``. Only returned when ``compute_uv`` is True.
    s : (..., K) array
        The singular values for every matrix, sorted in descending order.
    v : { (..., N, N), (..., K, N) } array
        Unitary matrices. The actual shape depends on the value of
        ``full_matrices``. Only returned when ``compute_uv`` is True.

    Notes
    -----
    Numpy broadcasting rules apply.

    Singular Value Decomposition is performed using LAPACK routine _gesdd

    For elements where the LAPACK routine fails, the result will be set
    to NaNs.

    Implemented for types single, double, csingle and cdouble. Numpy conversion
    rules apply.

    Examples
    --------
    >>> a = np.random.randn(9, 6) + 1j*np.random.randn(9, 6)

    Reconstruction based on full SVD:

    >>> U, s, V = svd(a, full_matrices=True)
    >>> (U.shape, V.shape, s.shape) == ((9, 9), (6, 6), (6,))
    True
    >>> S = np.zeros((9, 6), dtype=complex)
    >>> S[:6, :6] = np.diag(s)
    >>> np.allclose(a, np.dot(U, np.dot(S, V)))
    True

    Reconstruction based on reduced SVD:

    >>> U, s, V = svd(a, full_matrices=False)
    >>> (U.shape, V.shape, s.shape) == ((9, 6), (6, 6), (6,))
    True
    >>> S = np.diag(s)
    >>> np.allclose(a, np.dot(U, np.dot(S, V)))
    True

    """
    m = a.shape[-2]
    n = a.shape[-1]
    if 1 == compute_uv:
        if 1 == full_matrices:
            if m < n:
                gufunc = _impl.svd_m_f
            else:
                gufunc = _impl.svd_n_f
        else:
            if m < n:
                gufunc = _impl.svd_m_s
            else:
                gufunc = _impl.svd_n_s
    else:
        if m < n:
            gufunc = _impl.svd_m
        else:
            gufunc = _impl.svd_n
    return gufunc(a, **kw_args)


def chosolve(A, B, UPLO='L', **kw_args):
    """
    Solve the linear matrix equations on the inner dimensions, using
    cholesky decomposition.

    Computes the "exact" solution, `x`. of the well-determined,
    i.e., full rank, linear matrix equations `ax = b`, where a is
    a symmetric/hermitian positive-definite matrix.

    Parameters
    ----------
    A : (..., M, M) array
        Coefficient symmetric/hermitian positive-definite matrices.
    B : (..., M, N) array
        Ordinate or "dependent variable" values.
    UPLO : {'L', 'U'}, optional
         Specifies whether the calculation is done with the lower
         triangular part of the elements in `A` ('L', default) or
         the upper triangular part ('U').

    Returns
    -------
    X : (..., M, N) array
        Solutions to the system A X = B for all elements in the outer
        dimensions

    Notes
    -----
    Numpy broadcasting rules apply.

    The solutions are computed using LAPACK routines _potrf, _potrs

    For elements where the LAPACK routine fails, the result will be set
    to NaNs.

    Implemented for single, double, csingle and cdouble. Numpy conversion
    rules apply.

    See Also
    --------
    solve : solve a system using cholesky decomposition (for equations
            having symmetric/hermitian coefficient matrices)

    solve_triangular : Solve the linear matrix equation A * X = B
                       where A is a triangular matrix.

    Examples
    --------
    Solve the system of equations ``3 * x0 + x1 = 9`` and ``x0 + 2 * x1 = 8``:
    (note the matrix is symmetric in this system)

    >>> a = np.array([[3,1], [1,2]])
    >>> b = np.array([9,8])
    >>> x = solve(a, b)
    >>> x
    array([ 2.,  3.])

    Check that the solution is correct:

    >>> np.allclose(np.dot(a, x), b)
    True

    """
    if len(B.shape) == (len(A.shape) - 1):
        if 'L' == UPLO:
            gufunc = _impl.chosolve1_lo
        else:
            gufunc = _impl.chosolve1_up
    else:
        if 'L' == UPLO:
            gufunc = _impl.chosolve_lo
        else:
            gufunc = _impl.chosolve_up

    return gufunc(A, B, **kw_args)


def solve_triangular(A, B, UPLO='L',
                     transpose_type='N', unit_diagonal=False, **kw_args):
    """
    Solve the linear matrix equation A * X = B on the inner dimensions
    where A is a triangular matrix.

    Parameters
    ----------
    A : (..., M, M) array
        A triangular matrix.

    B : (..., M, N) or (..., M,) array
        Right hand side matrix in A * X = B.

    UPLO : {'U', 'L'}, optional
         Specifies whether the calculation is done with the lower
         triangular part of the elements in `A` ('L', default) or
         the upper triangular part ('U').

    transpose_type : {'N', 'T', 'C'}, optional
         Transpose type which decides equation to be solved.
         N => No transpose i.e. solve A * X = B
         T => Transpose i.e. solve A^T * X = B
         C => Conjugate transpose i.e. solve A^H * X = B

     unit_diagonal : bool, optional
         If True, the diagonal elements of 'a' are assumed to be 1 and are
         not referenced.

    Returns
    -------
    X : (..., M, N) or (..., M,) array
        Solutions to the system A * X = B for all elements in the outer
        dimensions

    Notes
    -----
    Numpy broadcasting rules apply.

    The solutions are computed using LAPACK routine _trtrs.

    For elements where the LAPACK routine fails, the result will be set
    to NaNs.

    Implemented for single, double, csingle and cdouble. Numpy conversion
    rules apply.

    See Also
    --------
    solve : Solves a systems using LU decomposition.

    chosolve : solve a system using cholesky decomposition (for equations
            having symmetric/hermitian coefficient matrices)

    Examples
    --------
    Solve the system of equations ``3 * x0 + x1 = 9`` and ``x0 + 2 * x1 = 8``:

    >>> a = np.array([[3,0], [1,2]])
    >>> b = np.array([6,8])
    >>> x = solve_triangular(a, b)
    >>> x
    array([ 2.,  3.])

    Check that the solution is correct:

    >>> np.allclose(np.dot(a, x), b)
    True

    """
    uplo_choices = ['U', 'L']
    transpose_choices = ['N', 'T', 'C']
    diagonal_type = 'U' if unit_diagonal else 'N'

    if UPLO not in uplo_choices:
        raise ValueError("Invalid UPLO argument '%s', valid values are: %s" %
                         (UPLO, uplo_choices))

    if transpose_type not in transpose_choices:
        raise ValueError("'Invalid TRANS argument '%s', valid values are: %s" %
                         (transpose_type, transpose_choices))

    # Keyword arguments are not passed to the extension code i.e. you can't
    # use it to pass UPLO = 'U' to the extension code. As a workaround,
    # invoke variant of  solve_triangular for each combination of params.
    # Extension methods are named
    # solve_triangular_{NDIM}{UPLO}{transpose_type}{diagonal_type} i.e. for
    # UPLO='N', transpose_type='N' and diagonal_type='N', it has functions
    # solve_triangular_0LNN or solve_triangular_1LNN depending on shape of B.
    ext_method_prefix = 1
    if len(B.shape) == (len(A.shape) - 1):
        ext_method_prefix = "solve_triangular_0"
    else:
        ext_method_prefix = "solve_triangular_1"

    gufunc = getattr(_impl,
                     ext_method_prefix + UPLO + transpose_type + diagonal_type)

    return gufunc(A, B, **kw_args)


def poinv(A, UPLO='L', **kw_args):
    """
    Compute the (multiplicative) inverse of symmetric/hermitian positive
    definite matrices, with broadcasting.

    Given a square symmetic/hermitian positive-definite matrix `a`, return
    the matrix `ainv` satisfying ``matrix_multiply(a, ainv) =
    matrix_multiply(ainv, a) = Identity matrix``.

    Parameters
    ----------
    a : (..., M, M) array
        Symmetric/hermitian postive definite matrices to be inverted.

    Returns
    -------
    ainv : (..., M, M) array
        (Multiplicative) inverse of the `a` matrices.

    Notes
    -----
    Numpy broadcasting rules apply.

    The inverse is computed using LAPACK routines _potrf, _potri

    For elements where the LAPACK routine fails, the result will be set
    to NaNs.

    Implemented for types single, double, csingle and cdouble. Numpy conversion
    rules apply.

    See Also
    --------
    inv : compute the multiplicative inverse of general matrices.

    Examples
    --------
    >>> a = np.array([[5, 3], [3, 5]])
    >>> ainv = poinv(a)
    >>> np.allclose(matrix_multiply(a, ainv), np.eye(2))
    True
    >>> np.allclose(matrix_multiply(ainv, a), np.eye(2))
    True

    """
    if 'L' == UPLO:
        gufunc = _impl.poinv_lo
    else:
        gufunc = _impl.poinv_up

    return gufunc(A, **kw_args)


def ldl(A, **kwargs):
    """
    LDL decomposition.

    Compute the LDL decomposition `L * D * L.H` of the square matrix `a`
    using the Bunch-Kaufman diagonal pivoting method.  `L` is a product of
    permutation and unit lower-triangular matrices, D is Hermitian and block
    diagonal with 1-by-1 and 2-by-2 diagonal blocks, and .H is the conjugate
    transpose operator. `a` must be Hermitian, but need not be positive
    definite.  When `a` is positive definite, D is strictly diagonal (without
    2-by-2 blocks) and `L` is a true lower-triangular matrix.

    Parameters
    ----------
    a : (..., M, M) array_like
        Hermitian (symmetric if all elements are real) input matrix.

    Returns
    -------
    L : (..., M, M) array_like
        Permuted lower-triangular factor of `a`.  Returns a true unit lower-
        triangular matrix if `a` is positive definite.  Returns a matrix object
        if `a` is a matrix object.
    D : (..., M, M) array_like
        Hermitian block diagonal factor of `a`, with 1-by-1 and 2-by-2 blocks.
        Returns a true diagonal matrix if `a` is positive definite.  Returns a
        matrix object if `a` is a matrix object.

    Raises
    ------
    LinAlgError
       If the decomposition fails, for example, if `a` is singular or not
        square.

    Notes
    -----
    Broadcasting rules apply.

    This is an interface to the _sytrf and _hetrf LAPACK routines, which use
    the Bunch-Kaufman diagonal pivoting method [1]_ to factor the matrix `a`.

    This function reconstructs the full `L` and `D` matrices from the combined
    representation produced by the _sytrf and _hetrf functions.

    For elements where the LAPACK routine fails, the result will be set
    to NaNs.

    Implemented for types single, double, csingle and cdouble. Numpy conversion
    rules apply.

    Examples
    --------
    >>> a = np.array([[4, 1], [1, 2]])
    >>> l,d = ldl(a)
    >>> np.allclose(matrix_multiply(l, matrix_multiply(d, l.T)), a)
    True

    References
    ----------
    .. [1] G. H. Golub and C. F. Van Loan, *Matrix Computations*, 4th Ed.
           Baltimore, MD, Johns Hopkins University Press, 2013, pg. 192
    """

    return _impl.ldl(A, **kwargs)


if __name__ == "__main__":
    import doctest
    doctest.testmod()

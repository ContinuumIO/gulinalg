"""Basic Linear Algebra utility functions implemented as gufuncs,
broadcasting

This file contains the wrappers for several basic linear algebra
functions as gufuncs. The underlying implementation is BLAS based.

- inner1d: inner product (dot product) over the inner dimension,
  broadcasting

- dotc1d: inner product by conjugate (dot product) over the inner
  dimension, broadcasting

- innerwt: weighted inner product over the inner dimension,
  broadcasting

- matrix_multiply: matrix_multiply over the 2 inner dimensions,
  broadcasting

- matvec_multiply: matvec_multiply over the 2 inner dimensions,
  broadcasting

- quadratic form: quadratic form uQv over the inner dimensions,
  broadcasting

- update_rank1: rank1 update over the inner dimensions,
  broadcasting

"""

from __future__ import division, absolute_import, print_function


import numpy as np
from . import _impl


def inner1d(a, b, **kwargs):
    """
    Compute the dot product of vectors over the inner dimension, with
    broadcasting.

    Parameters
    ----------
    a : (..., N) array
        Input array
    b : (..., N) array
        Input array

    Returns
    -------
    inner : (...) array
        dot product over the inner dimension.

    Notes
    -----
    Numpy broadcasting rules apply when matching dimensions.

    Implemented for types single, double, csingle and cdouble. Numpy
    conversion rules apply.

    For single and double types this is equivalent to dotc1d.

    Maps to BLAS functions sdot, ddot, cdotu and zdotu.

    See Also
    --------
    dotc1d : dot product conjugating first vector.
    innerwt : weighted (i.e. triple) inner product.

    Examples
    --------
    >>> a = np.arange(1,5).reshape(2,2)
    >>> b = np.arange(1,8,2).reshape(2,2)
    >>> res = inner1d(a,b)
    >>> res.shape == (2,)
    True
    >>> print (res)
    [  7.  43.]

    """
    return _impl.inner1d(a, b, **kwargs)


def dotc1d(a, b, **kwargs):
    """
    Compute the dot product of vectors over the inner dimension, conjugating
    the first vector, with broadcasting

    Parameters
    ----------
    a : (..., N) array
        Input array
    b : (..., N) array
        Input array

    Returns
    -------
    dotc : (...) array
        dot product conjugating the first vector over the inner
        dimension.

    Notes
    -----
    Numpy broadcasting rules apply when matching dimensions.

    Implemented for types single, double, csingle and cdouble. Numpy
    conversion rules apply.

    For single and double types this is equivalent to inner1d.

    Maps to BLAS functions sdot, ddot, cdotc and zdotc.

    See Also
    --------
    inner1d : dot product
    innerwt : weighted (i.e. triple) inner product.

    Examples
    --------
    >>> a = np.arange(1,5).reshape(2,2)
    >>> b = np.arange(1,8,2).reshape(2,2)
    >>> res = inner1d(a,b)
    >>> res.shape == (2,)
    True
    >>> print (res)
    [  7.  43.]

    """
    return _impl.dotc1d(a, b, **kwargs)


def innerwt(a, b, c, **kwargs):
    """
    Compute the weighted (i.e. triple) inner product, with
    broadcasting.

    Parameters
    ----------
    a, b, c : (..., N) array
        Input arrays

    Returns
    -------
    inner : (...) array
        The weighted (i.e. triple) inner product.

    Notes
    -----
    Numpy broadcasting rules apply.

    Implemented for types single, double, csingle and cdouble. Numpy
    conversion rules apply.

    See Also
    --------
    inner1d : inner product.
    dotc1d : dot product conjugating first vector.

    Examples
    --------
    >>> a = np.arange(1,5).reshape(2,2)
    >>> b = np.arange(1,8,2).reshape(2,2)
    >>> c = np.arange(0.25,1.20,0.25).reshape(2,2)
    >>> res = innerwt(a,b,c)
    >>> res.shape == (2,)
    True
    >>> res
    array([  3.25,  39.25])

    """
    return _impl.innerwt(a, b, c, **kwargs)


def matrix_multiply(a,b,**kwargs):
    """
    Compute matrix multiplication, with broadcasting

    Parameters
    ----------
    a : (..., M, N) array
        Input array.
    b : (..., N, P) array
        Input array.

    Returns
    -------
    r : (..., M, P) array matrix multiplication of a and b over any number of
        outer dimensions

    Notes
    -----
    Numpy broadcasting rules apply.

    Matrix multiplication is computed using BLAS _gemm functions.

    Implemented for single, double, csingle and cdouble. Numpy conversion
    rules apply.

    Examples
    --------
    >>> a = np.arange(1,17).reshape(2,2,4)
    >>> b = np.arange(1,25).reshape(2,4,3)
    >>> res = matrix_multiply(a,b)
    >>> res.shape == (2, 2, 3)
    True
    >>> res
    array([[[   70.,    80.,    90.],
            [  158.,   184.,   210.]],
    <BLANKLINE>
           [[  750.,   792.,   834.],
            [ 1030.,  1088.,  1146.]]])

    """
    return _impl.matrix_multiply(a,b,**kwargs)


def matvec_multiply(a, b, **kwargs):
    """
    Compute matrix vector multiplication, with broadcasting

    Parameters
    ----------
    a : (..., M, N) array
        Input array.
    b : (..., N) array
        Input array

    Returns
    -------
    r : (..., M) matrix vector multiplication of a and b over any number of
        outer dimensions

    Notes
    -----
    Numpy broadcasting rules apply.

    Matrix vector multiplication is computed using BLAS _gemv functions.

    Implemented for single, double, csingle and cdouble. Numpy conversion
    rules apply.

    Examples
    --------
    >>> a = np.arange(1,17).reshape(2,2,4)
    >>> b = np.arange(1,9).reshape(2,4)
    >>> res = matvec_multiply(a,b)
    >>> res.shape == (2,2)
    True
    >>> res
    array([[  30.,   70.],
           [ 278.,  382.]])

    """
    return _impl.matvec_multiply(a, b, **kwargs)


def quadratic_form(u,Q,v, **kwargs):
    """
    Compute the quadratic form uQv, with broadcasting

    Parameters
    ----------
    u : (..., M) array
        The u vectors of the quadratic form uQv

    Q : (..., M, N) array
        The Q matrices of the quadratic form uQv

    v : (..., N) array
        The v vectors of the quadratic form uQv

    Returns
    -------
    qf : (...) array
        The result of the quadratic forms

    Notes
    -----
    Numpy broadcasting rules apply.

    Implemented for types single, double, csingle and cdouble. Numpy
    conversion rules apply.

    This is similar to PDL inner2

    Examples
    --------

    The result in absence of broadcasting is just as np.dot(np.dot(u,Q),v)
    or np.dot(u, np.dot(Q,v))

    >>> u = np.array([2., 3.])
    >>> Q = np.array([[1.,1.], [0.,1.]])
    >>> v = np.array([1.,2.])
    >>> quadratic_form(u,Q,v)
    12.0

    >>> np.dot(np.dot(u,Q),v)
    12.0

    >>> np.dot(u, np.dot(Q,v))
    12.0

    """
    return _impl.quadratic_form(u, Q, v, **kwargs)


def update_rank1(a, b, c, conjugate=True, **kwargs):
    """
    Compute rank1 update, with broadcasting

    Parameters
    ----------
    a : (..., M) array
        Input array.

    b : (..., N) array
        Input array

    c : (..., M, N) array
        Input array.

    conjugate : bool (default True)
        For complex numbers, use conjugate transpose of b instead of normal
        transpose. If false, use normal transpose.

    Returns
    -------
    r : (..., M, N) rank1 update of a, b and c over any number of
        outer dimensions

    Notes
    -----
    Numpy broadcasting rules apply.

    Rank1 update is computed using BLAS _ger functions for real numbers.
    For complex number, uses _gerc and _geru for conjuate and normal transpose
    variants respectively.

    Implemented for single, double, csingle and cdouble. Numpy conversion
    rules apply.

    Examples
    --------
    >>> a = np.array([1, 2])
    >>> b = np.array([3, 4])
    >>> c = np.array([[1, 2], [3, 4]])
    >>> res = update_rank1(a, b, c)
    >>> res.shape == (2, 2)
    True
    >>> res
    array([[  4.,   6.],
           [  9.,  12.]])

    """
    if conjugate:
        gufunc = _impl.update_rank1_conjugate
    else:
        gufunc = _impl.update_rank1

    return gufunc(a, b, c, **kwargs)

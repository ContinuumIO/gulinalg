"""A set of extra ufuncs inspired from PDL: Fused operations.

- add3
- multiply3
- multiply3_add
- multiply_add
- multiply_add2
- multiply4
- multiply4_add

Note: for many use-cases, numba may provide a better solution
"""

from __future__ import division, absolute_import, print_function


import numpy as np
from . import _impl


def add3(a, b, c, **kwargs):
    """
    Element-wise addition of 3 arrays: a + b + c.

    Parameters
    ----------
    a, b, c : (...) array
        arrays with the addends

    Returns
    -------
    add3 : (...) array
        resulting element-wise addition.

    Notes
    -----
    Numpy broadcasting rules apply.

    Implemented for types single, double, csingle and cdouble. Numpy
    conversion rules apply.

    See Also
    --------
    multiply3 : element-wise three-way multiplication.
    multiply3_add : element-wise three-way multiplication and addition.
    multiply_add : element-wise multiply-add.
    multiply_add2 : element-wise multiplication with two additions.
    multiply4 : element-wise four-way multiplication
    multiply4_add : element-wise four-way multiplication and addition,

    Examples
    --------
    >>> a = np.linspace(1.0, 30.0, 30)
    >>> add3(a[0::3], a[1::3], a[2::3])
    array([  6.,  15.,  24.,  33.,  42.,  51.,  60.,  69.,  78.,  87.])

    """
    return _impl.add3(a, b, c, **kwargs)


def multiply3(a, b, c, **kwargs):
    """
    Element-wise multiplication of 3 arrays: a*b*c.

    Parameters
    ----------
    a, b, c : (...) array
        arrays with the factors

    Returns
    -------
    m3 : (...) array
        resulting element-wise product

    Notes
    -----
    Numpy broadcasting rules apply.

    Implemented for types single, double, csingle and cdouble. Numpy
    conversion rules apply.

    See Also
    --------
    add3 : element-wise three-way addition
    multiply3_add : element-wise three-way multiplication and addition.
    multiply_add : element-wise multiply-add.
    multiply_add2 : element-wise multiplication with two additions.
    multiply4 : element-wise four-way multiplication
    multiply4_add : element-wise four-way multiplication and addition,

    Examples
    --------
    >>> a = np.linspace(1.0, 10.0, 10)
    >>> multiply3(a, 1.01, a)
    array([   1.01,    4.04,    9.09,   16.16,   25.25,   36.36,   49.49,
             64.64,   81.81,  101.  ])

    """
    return _impl.multiply3(a, b, c, **kwargs)


def multiply3_add(a, b, c, d, **kwargs):
    """
    Element-wise multiplication of 3 arrays adding a 4th array to the
    result: a*b*c + d

    Parameters
    ----------
    a, b, c : (...) array
        arrays with the factors

    d : (...) array
        array with the addend

    Returns
    -------
    m3a : (...) array
        element-wise result (a*b*c + d)

    Notes
    -----
    Numpy broadcasting rules apply.

    Implemented for types single, double, csingle and cdouble. Numpy
    conversion rules apply.

    See Also
    --------
    add3 : element-wise three-way addition
    multiply3 : element-wise three-way multiplication.
    multiply_add : element-wise multiply-add.
    multiply_add2 : element-wise multiplication with two additions.
    multiply4 : element-wise four-way multiplication
    multiply4_add : element-wise four-way multiplication and addition,

    Examples
    --------
    >>> a = np.linspace(1.0, 10.0, 10)
    >>> multiply3_add(a, 1.01, a, 42e-4)
    array([   1.0142,    4.0442,    9.0942,   16.1642,   25.2542,   36.3642,
             49.4942,   64.6442,   81.8142,  101.0042])

    """
    return _impl.multiply3_add(a, b, c, d, **kwargs)


def multiply_add(a, b, c, **kwargs):
    """
    Element-wise multiplication of 2 arrays, adding a 3rd array to the
    result: a*b + c

    Parameters
    ----------
    a, b : (...) array
        arrays with the factors

    c : (...) array
        array with the addend

    Returns
    -------
    madd : (...) array
        element-wise result (a*b + c)

    Notes
    -----
    Numpy broadcasting rules apply.

    Implemented for types single, double, csingle and cdouble. Numpy
    conversion rules apply.

    See Also
    --------
    add3 : element-wise three-way addition
    multiply3 : element-wise three-way multiplication.
    multiply3_add : element-wise three-way multiplication and addition.
    multiply_add2 : element-wise multiplication with two additions.
    multiply4 : element-wise four-way multiplication
    multiply4_add : element-wise four-way multiplication and addition,

    Examples
    --------
    >>> a = np.linspace(1.0, 10.0, 10)
    >>> multiply_add(a, a, 42e-4)
    array([   1.0042,    4.0042,    9.0042,   16.0042,   25.0042,   36.0042,
             49.0042,   64.0042,   81.0042,  100.0042])

    """
    return _impl.multiply_add(a, b, c, **kwargs)


def multiply_add2(a, b, c, d, **kwargs):
    """
    Element-wise multiplication of 2 arrays, adding a 3rd and a 4th
    array to the result: a*b + c + d

    Parameters
    ----------
    a, b : (...) array
        arrays with the factors

    c, d : (...) array
        arrays with the addends

    Returns
    -------
    mult_add2 : (...) array
        element-wise result (a*b + c + d)

    Notes
    -----
    Numpy broadcasting rules apply.

    Implemented for types single, double, csingle and cdouble. Numpy
    conversion rules apply.

    See Also
    --------
    add3 : element-wise three-way addition
    multiply3 : element-wise three-way multiplication.
    multiply3_add : element-wise three-way multiplication and addition.
    multiply_add : element-wise multiply-add.
    multiply4 : element-wise four-way multiplication
    multiply4_add : element-wise four-way multiplication and addition,

    Examples
    --------
    >>> a = np.linspace(1.0, 10.0, 10)
    >>> multiply_add2(a, a, a, 42e-4)
    array([   2.0042,    6.0042,   12.0042,   20.0042,   30.0042,   42.0042,
             56.0042,   72.0042,   90.0042,  110.0042])

    """
    return _impl.multiply_add2(a, b, c, d, **kwargs)


def multiply4(a, b, c, d, **kwargs):
    """
    Element-wise multiplication of 4 arrays: a*b*c*d

    Parameters
    ----------
    a, b, c, d : (...) array
        arrays with the factors

    Returns
    -------
    m4 : (...) array
        element-wise result (a*b*c*d)

    Notes
    -----
    Numpy broadcasting rules apply.

    Implemented for types single, double, csingle and cdouble. Numpy
    conversion rules apply.

    See Also
    --------
    add3 : element-wise three-way addition
    multiply3 : element-wise three-way multiplication.
    multiply3_add : element-wise three-way multiplication and addition.
    multiply_add : element-wise multiply-add.
    multiply_add2 : element-wise multiplication with two additions.
    multiply4_add : element-wise four-way multiplication and addition,

    Examples
    --------
    >>> a = np.linspace(1.0, 10.0, 10)
    >>> multiply4(a, a, a[::-1], 1.0001)
    array([  10.001 ,   36.0036,   72.0072,  112.0112,  150.015 ,  180.018 ,
            196.0196,  192.0192,  162.0162,  100.01  ])

    """
    return _impl.multiply4(a, b, c, d, **kwargs)


def multiply4_add(a, b, c, d, e, **kwargs):
    """
    Element-wise multiplication of 4 arrays, adding a 5th array to the
    result: a*b*c*d + e

    Parameters
    ----------
    a, b, c, d : (...) array
        arrays with the factors

    e : (...) array
        array with the addend

    Returns
    -------
    add3 : (...) array
        element-wise result (a*b*c*d + e)

    Notes
    -----
    Numpy broadcasting rules apply.

    Implemented for types single, double, csingle and cdouble. Numpy
    conversion rules apply.

    See Also
    --------
    add3 : element-wise three-way addition
    multiply3 : element-wise three-way multiplication.
    multiply3_add : element-wise three-way multiplication and addition.
    multiply_add : element-wise multiply-add.
    multiply_add2 : element-wise multiplication with two additions.
    multiply4 : element-wise four-way multiplication

    Examples
    --------
    >>> a = np.linspace(1.0, 10.0, 10)
    >>> multiply4_add(a, a, a[::-1], 1.01, 42e-4)
    array([  10.1042,   36.3642,   72.7242,  113.1242,  151.5042,  181.8042,
            197.9642,  193.9242,  163.6242,  101.0042])

    """
    return _impl.multiply4_add(a, b, c, d, e,**kwargs)



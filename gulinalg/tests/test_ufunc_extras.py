"""
Test ufunc extras in a more exhaustive way than the doctests

testing with:
 - all supported types (real and complex)
 - with different dimensions (scalar, 1, 2, and 3 dimensions)
 - not using the same operand

all tests will be compared against the equivalent numpy expressions.
"""
from __future__ import print_function

from unittest import TestCase

try:
    from itertools import izip
except ImportError:
    # in python 3* map izip to zip
    izip = zip

import numpy as np
from numpy.testing import TestCase, assert_allclose, run_module_suite
import gulinalg
from gulinalg.testing import assert_allclose_with_nans


if gulinalg.STRICT_FP:
    # with strict fp it should be able to handle the ranges with results
    # comparable to those of numpy
    _float_values = [ 0.0, 1.0, -1.0, 42.3, -42.3, 1.2e-30, -1.2e-30, 1.666e+30,
                      -1.666e+30, 1.45e-300, -1.45e-300, 1.0000000001,
                      -1.0000000001, float('nan'), float('inf'), -float('inf')]

    _complex_values = [ 0.0 + 0.0j, 1.0 + 0.0j, -1.0 + 0.0j, 0.0 + 1.0j,
                        -0.0 - 1.0j, 42.0 + 10.5j, -42.0 + 10.5j, 42.0 - 10.5j,
                        -42.0 - 10.5j, 1.6e+300 + 1.3e-300, 1.6e+300 - 1.3e-300,
                        -1.6e+300 + 1.3e-300, -1.6e+300 - 1.3e-300,
                        float('nan')*(1.0+1.0j), float('inf')*(1.0+1.0j),
                        float('inf')*(-1.0-1.0j) ]
else:
    # if not strict fp, this means the compiler is using an intermediate
    # precision that is higher than indicated (this will happen for x87 FPU
    # code, for example). This means that the random tests relying in these
    # values could 'fail' because of precision difference (including not
    # overflowing to inf). In this case, test without big exponents to avoid
    # the problems.
    _float_values = [ 0.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 1.5, -1.5,
                      2.5, -2.5, 1.125, -1.125, float('nan'), float('inf'),
                      -float('inf')]

    _complex_values = [ 0.0 + 0.0j, 1.0 + 0.0j, -1.0 + 0.0j, 0.0 + 1.0j,
                        -0.0 - 1.0j, 42.0 + 10.5j, -42.0 + 10.5j, 42.0 - 10.5j,
                        -42.0 - 10.5j, 1.6e+3 + 1.3e-3, 1.6e+3 - 1.3e-3,
                        -1.6e+3 + 1.3e-3, -1.6e+3 - 1.3e-3,
                        float('nan')*(1.0+1.0j), float('inf')*(1.0+1.0j),
                        float('inf')*(-1.0-1.0j) ]

    
assert(len(_float_values) == 16)
assert(len(_complex_values) == 16)

_perm_idx = np.arange(0, 16)
_perm_idx = np.concatenate((_perm_idx, _perm_idx^1))
_perm_idx = np.concatenate((_perm_idx, _perm_idx^2))
_perm_idx = np.concatenate((_perm_idx, _perm_idx^4))
_perm_idx = np.concatenate((_perm_idx, _perm_idx^8))


class UfuncTestCase(object):
    """
    wants a _test attribute containing a pair of functions:
    _test[0] = the function to test
    _test[1] = reference implementation using numpy based lambda
    """

    def setUp(self):
        self._old_err = np.seterr(all='ignore')

    def tearDown(self):
        # restore old seterr
        np.seterr(**self._old_err)

    def _check(self, args):
        ufunc_result = np.atleast_1d(self._test[0](*args))
        reference_result = np.atleast_1d(self._test[1](*args))
        assert_allclose_with_nans(reference_result, ufunc_result)

    def _mk_args(self, base_args, dtype, dims=1):
        nargs = self._test[1].__code__.co_argcount
        assert dims > 0 and dims < 4
        base_args = np.array(base_args, dtype=dtype)
        shape = [None, (256,), (16,16), (2,8,16)][dims]
        return [np.roll(base_args, arg)[_perm_idx].reshape(shape) for arg in range(nargs)]

    def test_scalar_single(self):
        args = self._mk_args(_float_values, np.single)
        for tup in izip(*args):
            self._check(tup)

    def test_scalar_double(self):
        args = self._mk_args(_float_values, np.double)
        for tup in izip(*args):
            self._check(tup)

    def test_scalar_csingle(self):
        args = self._mk_args(_complex_values, np.csingle)
        for tup in izip(*args):
            self._check(tup)

    def test_scalar_cdouble(self):
        args = self._mk_args(_complex_values, np.cdouble)
        for tup in izip(*args):
            self._check(tup)

    def test_1d_single(self):
        self._check(self._mk_args(_float_values, np.single))

    def test_1d_double(self):
        self._check(self._mk_args(_float_values, np.double))

    def test_1d_csingle(self):
        self._check(self._mk_args(_complex_values, np.csingle))

    def test_1d_cdouble(self):
        self._check(self._mk_args(_complex_values, np.cdouble))

    def test_2d_single(self):
        self._check(self._mk_args(_float_values, np.single, 2))

    def test_2d_double(self):
        self._check(self._mk_args(_float_values, np.double, 2))

    def test_2d_csingle(self):
        self._check(self._mk_args(_complex_values, np.csingle, 2))

    def test_2d_cdouble(self):
        self._check(self._mk_args(_complex_values, np.cdouble, 2))

    def test_3d_single(self):
        self._check(self._mk_args(_float_values, np.single, 3))

    def test_3d_double(self):
        self._check(self._mk_args(_float_values, np.double, 3))

    def test_3d_csingle(self):
        self._check(self._mk_args(_complex_values, np.csingle, 3))

    def test_3d_cdouble(self):
        self._check(self._mk_args(_complex_values, np.cdouble, 3))


class TestAdd3(UfuncTestCase, TestCase):
    _test = [gulinalg.add3, lambda x,y,z: x+y+z]

class TestMultiply3(UfuncTestCase, TestCase):
    _test = [gulinalg.multiply3, lambda x,y,z: x*y*z]

class TestMultiply3Add(UfuncTestCase, TestCase):
    _test = [gulinalg.multiply3_add, lambda x,y,z,u: x*y*z+u]

class TestMultiplyAdd(UfuncTestCase, TestCase):
    _test = [gulinalg.multiply_add, lambda x,y,z: x*y+z]

class TestMultiplyAdd2(UfuncTestCase, TestCase):
    _test = [gulinalg.multiply_add2, lambda x,y,z,u: (x*y)+(z+u)]

class TestMultiply4(UfuncTestCase, TestCase):
    _test = [gulinalg.multiply4, lambda x,y,z,u: (x*y)*(z*u)]

class TestMultiply4Add(UfuncTestCase, TestCase):
    _test = [gulinalg.multiply4_add, lambda x,y,z,u,v: (x*y)*(z*u)+v]

if __name__ == '__main__':
    run_module_suite()

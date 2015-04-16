

from __future__ import print_function, division, absolute_import
import sys

import numpy as np

if sys.version_info[:2] <= (2, 6):
    # fix compatibility issues with unittest
    import unittest2 as unittest
else:
    import unittest


def banner(title):
    print((' '+title+' ').center(72, '='))

def assert_allclose_with_nans(lho, rho):
    """like numpy's allclose but allowing nans/infinities"""
    mask1 = np.isfinite(lho)
    mask2 = np.isfinite(rho)

    #infinities in the same place
    assert np.all(mask1 == mask2)

    # all finite elements are "close"
    np.testing.assert_allclose(lho[mask1], rho[mask2])

    # all non finite elements are equal
    np.testing.assert_array_equal(lho[~mask1], rho[~mask2])


def run_doctests(verbosity=1):
    from gulinalg import gufunc_linalg, gufunc_general, ufunc_extras
    import gulinalg
    import doctest

    banner('Running doctests')

    total_fail, total_test = 0, 0
    for mod in (gufunc_general, gufunc_linalg, ufunc_extras):
        if verbosity > 0:
            print ('testing {0}:'.format(mod.__name__))
        failed, total = doctest.testmod(mod, verbose=verbosity>2)
        if verbosity > 0:
            print ('{0}({1} test(s), {2} failed)'.format(mod.__name__, total, failed))
        total_fail += failed
        total_test += total

    return total_fail == 0


def discover_unittests(startdir):
    loader = unittest.TestLoader()
    suite = loader.discover(startdir)
    return suite

def run_unittests(verbosity=1):
    banner("Running unittests")
    loader = unittest.TestLoader()
    suite = loader.discover('gulinalg.tests')
    result = unittest.TextTestRunner(verbosity=verbosity).run(suite)
    return result.wasSuccessful()


def test(verbosity=1):
    """
    Run all the tests
    -----------------

    Right now only doctests are present.
    """
    return run_unittests(verbosity=verbosity) and run_doctests(verbosity=verbosity)

if __name__ == "__main__":
    sys.exit(0 if test() else 1)

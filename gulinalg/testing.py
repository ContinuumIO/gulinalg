

from __future__ import print_function, division, absolute_import
import sys

import numpy as np

def assert_allclose_with_nans(lho, rho):
    """like numpy's allclose but allowing nans/infinities"""
    mask1 = np.isfinite(lho)
    mask2 = np.isfinite(rho)

    assert np.all(mask1 == mask2) #infinities in the same place

    # all finite elements are "close"
    np.testing.assert_allclose(lho[mask1], rho[mask2])

    # all non finite elements are equal
    np.testing.assert_array_equal(lho[~mask1], rho[~mask2])


def run_doctests():
    from gulinalg import gufunc_linalg, gufunc_general, ufunc_extras
    import gulinalg
    import doctest

    print(' Running doctests '.center(72, '='))

    total_fail, total_test = 0, 0
    for mod in (gufunc_general, gufunc_linalg, ufunc_extras):
        failed, total = doctest.testmod(mod)
        total_fail += failed
        total_test += total

    return total_fail == 0

def test():
    """
    Run all the tests
    -----------------

    Right now only doctests are present.
    """
    return run_doctests()

if __name__ == "__main__":
    sys.exit(0 if test() else 1)

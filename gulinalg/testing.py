

from __future__ import print_function, division, absolute_import
import sys

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

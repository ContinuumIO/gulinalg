"""Linear Algebra toolbox implemented as gufuncs


========
gulinalg
========

gulinalg provides a toolbox of linear algebra utility gufuncs. Some
extra utility unfuncs are also thrown in (fused operations ufuncs).

Most of this functions are already present in numpy.linalg, where
they weren't implemented as gufuncs (although in the latest versions
of NumPy most will broadcast, as the are actually based on the same
codebase as this).

Most of the functionality is implemented as a C submodule that
implements the different gufuncs/ufuncs. This Python module provides
wrappers to provide a sane interface to some of the functionality.  It
also serves as means of documentation, each of the public functions
providing an extensive docstring with examples tested using doctest.


========
Contents
========

The package is structured in 3 sections:

Basic Linear Algebra
--------------------

A set of linear algebra functions working as gufuncs that are
implemented using BLAS:

- inner1d
- doc1d
- innerwt
- matrix_multiply
- matvec_multiply
- update_rank1
- quadratic_form

Linear Algebra
--------------

A set of linear algebra functions working as gufuncs that are
implemented using LAPACK:

- det
- slogdet
- cholesky
- lu
- qr
- eig
- eigvals
- eigh
- eigvalsh
- solve
- solve_triangular
- svd
- chosolve
- inv
- inv_triangular
- poinv

Extra Ufuncs (Fused Operations)
-------------------------------

- add3
- multiply3
- multiply3_add
- multiply_add
- multiply_add2
- multiply4
- multiply4_add


================
 Error Handling
================

Unlike the numpy.linalg module, this module does not use exceptions to
notify errors in the execution of the kernels. As these functions are
thougth to be used in a vector way it didn't seem appropriate to raise
exceptions on failure of an element. So instead, when an error
computing an element occurs its associated result will be set to an
invalid value (all NaNs).

Exceptions can occur if the arguments fail to map properly to the
underlying gufunc (due to signature mismatch, for example).

Note that the generation of NaNs may raise exceptions depending on how
the error handling currently active in NumPy (see numpy.seterr /
numpy.geterr)

============
Requirements
============

Due to previous bugs in NumPy gufunc machinery, this module requires
NumPy >= 1.8

"""

from __future__ import print_function, division, absolute_import


from .gufunc_general import (inner1d, dotc1d, innerwt, matrix_multiply,
                             matvec_multiply, update_rank1, quadratic_form)
from .gufunc_linalg import (det, slogdet, cholesky, lu, qr, eig, eigvals,
                            eigh, eigvalsh, solve, solve_triangular, svd,
                            chosolve, inv, inv_triangular, poinv, ldl)
from .ufunc_extras import (add3, multiply3, multiply3_add, multiply_add,
                           multiply_add2, multiply4, multiply4_add)
from ._impl import STRICT_FP

from .testing import test

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

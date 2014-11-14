from __future__ import division, print_function, absolute_import

# note that this package depends on NumPy's distutils. It also
# piggy-backs on NumPy configuration in order to link agains the
# appropriate BLAS/LAPACK implementation.
from numpy.distutils.core import setup, Extension
from numpy.distutils import system_info as np_sys_info
from numpy.distutils import misc_util as np_misc_util
import os, sys, copy

import versioneer


BASE_PATH = 'gulinalg'
VERSION_FILE_PATH = os.path.join(BASE_PATH, '_version.py')
C_SRC_PATH = os.path.join(BASE_PATH, 'src')
LAPACK_LITE_PATH = os.path.join(C_SRC_PATH, 'lapack_lite')

versioneer.versionfile_source = VERSION_FILE_PATH
versioneer.versionfile_build = VERSION_FILE_PATH
versioneer.tag_prefix = ''
versioneer.parentdir_prefix = 'gulinalg-'

# Use information about the LAPACK library used in NumPy.
# if not present, fallback to using the included lapack-lite


MODULE_SOURCES = [os.path.join(C_SRC_PATH, 'gulinalg.c.src')]
MODULE_DEPENDENCIES = copy.copy(MODULE_SOURCES)

lapack_info = np_sys_info.get_info('lapack_opt', 0)
lapack_lite_files = [os.path.join(LAPACK_LITE_PATH, f)
                     for f in ['python_xerbla.c', 'zlapack_lite.c',
                               'dlapack_lite.c', 'blas_lite.c',
                               'dlamch.c', 'f2c_lite.c', 'f2c.h']]

if not lapack_info:
    # No LAPACK in NumPy
    print('### Warning: Using unoptimized blas/lapack @@@')
    MODULE_SOURCES.extend(lapack_lite_files[:-1]) # all but f2c.h
    MODULE_DEPENDENCIES.extend(lapack_lite_files)
else:
    if sys.platform == 'win32':
        print('### Warning: python.xerbla.c is disabled ###')
    else:
        MODULE_SOURCES.extend(lapack_lite_files[:1]) # python_xerbla.c
        MODULE_DEPENDENCIES.extend(lapack_lite_files[:1])



npymath_info = np_misc_util.get_info('npymath')
extra_opts = copy.deepcopy(lapack_info)

for key, val in npymath_info.items():
    if extra_opts.get(key):
        extra_opts[key].extend(val)
    else:
        extra_opts[key] = copy.deepcopy(val)


gufunc_module = Extension('gulinalg._impl',
                          sources = MODULE_SOURCES,
                          depends = MODULE_DEPENDENCIES,
                          **extra_opts)

packages = [
    'gulinalg',
    'gulinalg.tests',
]

ext_modules = [
    gufunc_module,
]

setup(name='gulinalg',
      version=versioneer.get_version(),
      description='gufuncs for linear algebra',
      author='Continuum Analytics, Inc.',
      ext_modules=ext_modules,
      packages=packages,
      license='BSD',
      long_description=open('README.md').read(),
      cmdclass=versioneer.get_cmdclass()
)

from distutils.core import setup, Extension
import versioneer

versioneer.versionfile_source = 'gulinalg/_version.py'
versioneer.versionfile_build = 'gulinalg/_version.py'
versioneer.tag_prefix = ''
versioneer.parentdir_prefix = 'gulinalg-'


packages = [
    'gulinalg',
]

ext_modules = [
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

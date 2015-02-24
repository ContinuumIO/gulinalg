REM This is the jenkins build script for building/testing the gulinalg
REM module.
REM
REM In order to work, Jenkins should be configured such as:
REM  - Anaconda is installed in C:\Anaconda
REM  - Jenkins build matrix is used for multiple platforms/python
REM    versions
REM  - Use XShell plugin to launch this script (on Windows, the
REM    .bat version will be executed instead)
REM  - Call the script from the root workspace directory as
REM    buildscripts/jenkins-build
REM


REM Require a version of python to be selected
if "%PYTHON_VERSION%" == "" exit /b 1

REM Require a version of NumPy to be selected
if "%NUMPY_VERSION%" == "" exit /b 1

REM use conda-build to build/test the package.
call C:\Anaconda\Scripts\conda build buildscripts/condarecipe --python %PYTHON_VERSION% --numpy %NUMPY_VERSION%

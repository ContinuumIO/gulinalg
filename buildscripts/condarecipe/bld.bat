%PYTHON% setup.py install
if errorlevel 1 exit 1

if "%PY3K%" (
   rd /s /q 
)
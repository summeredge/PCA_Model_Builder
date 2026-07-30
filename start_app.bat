@echo off
setlocal
cd /d "%~dp0"

set "APP_URL=http://127.0.0.1:8775/"
set "PYTHON_CMD="

if exist "C:\Users\shaoy\AppData\Local\Programs\Python\Python311\python.exe" (
  set "PYTHON_CMD=C:\Users\shaoy\AppData\Local\Programs\Python\Python311\python.exe"
)

if not defined PYTHON_CMD (
  where python >nul 2>nul
  if %errorlevel%==0 set "PYTHON_CMD=python"
)

if not defined PYTHON_CMD (
  echo Python was not found.
  echo Please install Python 3.11, then run this file again.
  pause
  exit /b 1
)

echo Checking Python dependencies...
"%PYTHON_CMD%" -c "import numpy, pandas, scipy, sklearn" >nul 2>nul
if errorlevel 1 (
  echo Required Python packages are missing.
  echo Please run: "%PYTHON_CMD%" -m pip install -e .
  pause
  exit /b 1
)

echo Starting PCA Model Builder at %APP_URL%
start "" powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -Command "$url='%APP_URL%'; for($i=0; $i -lt 120; $i++){ try { $response=Invoke-WebRequest -UseBasicParsing -Uri $url -TimeoutSec 1; if($response.StatusCode -ge 200){ Start-Process $url; exit 0 } } catch {}; Start-Sleep -Milliseconds 250 }; Start-Process $url"

pushd "%~dp0src"
"%PYTHON_CMD%" -m pca_model_builder.cli serve --host 127.0.0.1 --port 8775 --no-open
popd

pause

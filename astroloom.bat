@echo off
cd /d "%~dp0"

if exist ".venv\Scripts\pythonw.exe" (
    start "" ".venv\Scripts\pythonw.exe" "run_app.py"
    exit /b 0
)

if exist "venv\Scripts\pythonw.exe" (
    start "" "venv\Scripts\pythonw.exe" "run_app.py"
    exit /b 0
)

where pythonw >nul 2>&1
if %ERRORLEVEL% equ 0 (
    start "" pythonw "run_app.py"
    exit /b 0
)

where python >nul 2>&1
if %ERRORLEVEL% equ 0 (
    start "" python "run_app.py"
    exit /b 0
)

echo [ERROR] No se encontro Python ni pythonw en el sistema.
echo Asegurate de tener Python instalado y anadido al PATH.
pause
exit /b 1

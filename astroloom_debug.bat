@echo off
cd /d "%~dp0"
title AstroLoom - Modo Depuracion

echo =======================================================
echo Iniciando AstroLoom en modo depuracion (consola visible)
echo =======================================================
echo Registros en vivo:
echo   - Eventos de la aplicacion: logs\astroloom.log
echo   - Volcado de caidas:        logs\crash.log
echo =======================================================
echo.

set PYTHON_BIN=
if exist ".venv\Scripts\python.exe" (
    set "PYTHON_BIN=.venv\Scripts\python.exe"
) else if exist "venv\Scripts\python.exe" (
    set "PYTHON_BIN=venv\Scripts\python.exe"
) else (
    where python >nul 2>&1
    if %ERRORLEVEL% equ 0 (
        set "PYTHON_BIN=python"
    )
)

if "%PYTHON_BIN%"=="" (
    echo [ERROR] No se encontro python.exe en el entorno virtual ni en el PATH.
    pause
    exit /b 1
)

"%PYTHON_BIN%" "run_app.py"
set EXIT_CODE=%ERRORLEVEL%

echo.
echo =======================================================
echo AstroLoom ha finalizado con codigo de salida: %EXIT_CODE%
echo =======================================================
if not %EXIT_CODE%==0 (
    echo [ALERTA] La aplicacion se cerro inesperadamente (codigo %EXIT_CODE%).
    echo Revisa el archivo logs\crash.log y logs\astroloom.log para ver los detalles.
)
pause
exit /b %EXIT_CODE%

@echo off
setlocal
cd /d "%~dp0"

echo [1/4] Verificando entorno virtual...
if not exist ".venv\Scripts\activate.bat" (
    echo Creando entorno virtual .venv...
    python -m venv .venv
)
call .venv\Scripts\activate.bat

echo [2/4] Instalando dependencias...
python -m pip install --upgrade pip
pip install -r requirements.txt maturin pyinstaller

echo [3/4] Compilando motor Rust (astroloom_rs)...
cd astroloom_rs
maturin develop --release
cd ..

echo [4/4] Empaquetando AstroLoom con PyInstaller...
pyinstaller --clean Astroloom.spec

echo.
echo Compilacion completada. Ejecutable disponible en dist/AstroLoom.exe
pause
@echo off
rem Interactive demo: double-click this file, then type the search keyword.
rem Optional: run_demo.bat "query"   -> non-interactive single query.
cd /d "%~dp0"
set "PYEXE=%~dp0.venv\Scripts\python.exe"
if not exist "%PYEXE%" (
    echo [ERROR] Python venv not found: %PYEXE%
    echo Please run: uv venv --python 3.12 .venv
    pause
    exit /b 1
)
if "%~1"=="" (
    "%PYEXE%" "%~dp0code\demo_search.py" --images 3000 --topk 5
) else (
    "%PYEXE%" "%~dp0code\demo_search.py" --query %* --images 3000 --topk 5
)
echo.
echo ============================================================
echo  Result images: %~dp0results\demo
echo ============================================================
pause

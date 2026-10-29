@echo off
setlocal enabledelayedexpansion
title Generative AI Synthesis Studio - Dependency Installer
echo ========================================================================
echo   Generative AI Image Synthesis Studio - Dependency Installer
echo ========================================================================
echo.
cd /d "%~dp0"

set "PYTHON_EXE="

echo [1/3] Searching for Python / Anaconda environment...

:: 1. Search for gan-image-project conda environment directly
for %%P in (
    "%USERPROFILE%\anaconda3\envs\gan-image-project\python.exe"
    "%USERPROFILE%\miniconda3\envs\gan-image-project\python.exe"
    "%USERPROFILE%\.conda\envs\gan-image-project\python.exe"
    "%ProgramData%\anaconda3\envs\gan-image-project\python.exe"
    "%ProgramData%\miniconda3\envs\gan-image-project\python.exe"
    "%LOCALAPPDATA%\anaconda3\envs\gan-image-project\python.exe"
    "%LOCALAPPDATA%\miniconda3\envs\gan-image-project\python.exe"
    "C:\anaconda3\envs\gan-image-project\python.exe"
    "C:\miniconda3\envs\gan-image-project\python.exe"
    "D:\anaconda3\envs\gan-image-project\python.exe"
    "D:\miniconda3\envs\gan-image-project\python.exe"
) do (
    if exist %%P (
        set "PYTHON_EXE=%%~P"
        echo [OK] Found Conda environment 'gan-image-project':
        echo      %%~P
        goto python_ready
    )
)

:: 2. Try activating conda via standard conda scripts
if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" (
    call "%USERPROFILE%\anaconda3\Scripts\activate.bat" gan-image-project >nul 2>&1
) else if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" (
    call "%USERPROFILE%\miniconda3\Scripts\activate.bat" gan-image-project >nul 2>&1
) else if exist "%ProgramData%\anaconda3\Scripts\activate.bat" (
    call "%ProgramData%\anaconda3\Scripts\activate.bat" gan-image-project >nul 2>&1
)

:: 3. Search base Anaconda or Miniconda
for %%P in (
    "%USERPROFILE%\anaconda3\python.exe"
    "%USERPROFILE%\miniconda3\python.exe"
    "%ProgramData%\anaconda3\python.exe"
    "%ProgramData%\miniconda3\python.exe"
    "C:\anaconda3\python.exe"
    "C:\miniconda3\python.exe"
) do (
    if exist %%P (
        set "PYTHON_EXE=%%~P"
        echo [OK] Found Anaconda Python:
        echo      %%~P
        goto python_ready
    )
)

:: 4. Search system PATH (excluding WindowsApps stub)
for /f "delims=" %%i in ('where python 2^>nul') do (
    echo %%i | findstr /i "WindowsApps" >nul
    if errorlevel 1 (
        if not defined PYTHON_EXE (
            set "PYTHON_EXE=%%i"
            echo [OK] Found System Python:
            echo      %%i
            goto python_ready
        )
    )
)

:: 5. Check py launcher
where py >nul 2>&1
if %ERRORLEVEL% equ 0 (
    set "PYTHON_EXE=py"
    echo [OK] Found Python Launcher (py)
    goto python_ready
)

:python_ready
if not defined PYTHON_EXE (
    echo.
    echo ========================================================================
    echo  [ERROR] No Python or Anaconda installation was found!
    echo ========================================================================
    echo Please open Anaconda Prompt, activate 'gan-image-project', and run:
    echo   pip install torch torchvision
    echo ========================================================================
    echo.
    pause
    exit /b 1
)

echo.
echo [2/3] Installing PyTorch & Torchvision (CPU optimized)...
"%PYTHON_EXE%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo.
echo [3/3] Installing project dependencies from requirements.txt...
"%PYTHON_EXE%" -m pip install -r requirements.txt

echo.
echo ========================================================================
echo   Verification Check
echo ========================================================================
"%PYTHON_EXE%" -c "import torch; import torchvision; import streamlit; print('SUCCESS: PyTorch', torch.__version__, 'and Streamlit', streamlit.__version__, 'are working!')"

echo.
echo Setup finished! Press any key to exit.
pause

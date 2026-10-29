@echo off
setlocal enabledelayedexpansion
title Generative AI Synthesis Studio - Web Application
echo ========================================================================
echo   Generative AI Image Synthesis Studio (Web Application)
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
    echo Please launch this project using Anaconda Prompt:
    echo   1. Open 'Anaconda Prompt' from your Windows Start Menu.
    echo   2. Navigate to this folder:
    echo      cd /d "%~dp0"
    echo   3. Run:
    echo      conda activate gan-image-project
    echo      streamlit run src/app.py
    echo ========================================================================
    echo.
    pause
    exit /b 1
)

echo.
echo [2/3] Checking PyTorch and Streamlit dependencies...
"%PYTHON_EXE%" -c "import torch; import torchvision; import streamlit" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo.
    echo [SETUP] PyTorch or Streamlit is missing in this Python environment.
    echo Installing required packages now via pip...
    echo (This might take a minute on the first run, please wait...)
    echo.
    "%PYTHON_EXE%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    "%PYTHON_EXE%" -m pip install -r requirements.txt
    echo.
)

echo.
echo [3/3] Launching Web Application...
echo Opening in your browser at http://localhost:8501
echo (Keep this terminal window open while using the application)
echo.

"%PYTHON_EXE%" -m streamlit run src\app.py

:: Automatically close terminal when Streamlit exits
exit

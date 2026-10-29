@echo off
setlocal enabledelayedexpansion
title Generative AI Synthesis Studio - Window Mode
echo ========================================================================
echo   Generative AI Image Synthesis Studio (Windowed App Mode)
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
    echo   streamlit run src/app.py
    echo ========================================================================
    echo.
    pause
    exit /b 1
)

echo.
echo [2/3] Checking dependencies...
"%PYTHON_EXE%" -c "import torch; import torchvision; import streamlit" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Installing required packages via pip...
    "%PYTHON_EXE%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    "%PYTHON_EXE%" -m pip install -r requirements.txt
)

echo.
echo [3/3] Starting Studio in a dedicated desktop window...

:: Ensure port 8501 is clean before starting
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8501" ^| findstr "LISTENING" 2^>nul') do (
    taskkill /f /pid %%a >nul 2>&1
)

:: Start Streamlit server in background
start /b "" "%PYTHON_EXE%" -m streamlit run src\app.py --server.headless=true

:: Wait 2 seconds for server to initialize
timeout /t 2 /nobreak >nul

:: Launch dedicated application window
if exist "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" (
    start "" "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" --app=http://localhost:8501 --window-size=1240,860
) else if exist "C:\Program Files\Google\Chrome\Application\chrome.exe" (
    start "" "C:\Program Files\Google\Chrome\Application\chrome.exe" --app=http://localhost:8501 --window-size=1240,860
) else (
    start http://localhost:8501
)

:: The backend auto-shuts down in Python as soon as this window is closed
exit

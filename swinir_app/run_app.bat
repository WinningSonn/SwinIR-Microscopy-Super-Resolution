@echo off
setlocal enabledelayedexpansion

:: Ensure we are running in the script's directory
cd /d "%~dp0"

title SwinIR Microscopy Super-Resolution Setup & Launcher

echo ========================================================
echo   SwinIR Microscopy Super-Resolution - Auto Launcher
echo ========================================================
echo.

:: 1. Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not found! 
    echo Please install Python 3.11 - 3.12 as per the README requirements.
    pause
    exit /b
)

:: 2. Check/Create Virtual Environment
if exist "venv" (
    echo [INFO] Virtual environment 'venv' already exists.
) else (
    echo [INFO] Virtual environment not found. Creating 'venv'...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b
    )
    echo [SUCCESS] Virtual environment created.
)

:: 3. Activate Virtual Environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

:: 4. Install Dependencies
echo [INFO] Checking and installing dependencies...
python -m pip install --upgrade pip
if exist "requirements.txt" (
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to install dependencies. Check your internet connection.
        pause
        exit /b
    )
) else (
    echo [WARNING] requirements.txt not found. Skipping dependency installation.
)

:: 5. Check for Model Weights
if not exist "models\x2\sfsr_bicubic.pth" (
    echo.
    echo [ATTENTION] Model weights are missing!
    echo.
    echo Per the README, you must manually download the models from Kaggle:
    echo https://www.kaggle.com/models/milsonfeliciano/swinir-microscopy-models
    echo.
    echo Please extract them so your folder structure looks like:
    echo   models\x2\sfsr_bicubic.pth
    echo   models\x4\sfsr_bicubic.pth
    echo.
    pause
)

:: 6. Run the Application
echo.
echo [INFO] Starting Streamlit App...
echo.

"%~dp0venv\Scripts\python.exe" -m streamlit run app.py

:: Keep window open if app crashes
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] The application crashed or closed unexpectedly.
    pause
)
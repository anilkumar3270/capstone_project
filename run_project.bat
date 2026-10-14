@echo off
title Manufacturing Output Predictor

echo.
echo ==============================================
echo     MANUFACTURING OUTPUT PREDICTOR
echo ==============================================
echo.

echo Checking Python installation...
python --version >nul 2>&1

if errorlevel 1 (
    echo.
    echo ERROR: Python was not found.
    echo.
    echo Please install Python 3.10.x and enable
    echo "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

echo Python found.
echo.
echo Installing required Python libraries...
echo.

python -m pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo ERROR: Failed to install one or more requirements.
    echo Check your internet connection and try again.
    echo.
    pause
    exit /b 1
)

echo.
echo ==============================================
echo     Starting Streamlit Application
echo ==============================================
echo.
echo The application will open in your browser.
echo Press Ctrl+C in this window to stop the application.
echo.

python -m streamlit run app.py

pause

@echo off
REM AIMS Quick Start Script for Windows

echo Starting AIMS...

REM Check if virtual environment exists
if not exist "venv" (
    echo Error: Virtual environment not found. Please run setup.bat first.
    pause
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Check if .env exists
if not exist ".env" (
    echo .env file not found. Creating from template...
    python fix_startup_issues.py
    echo.
    echo Please edit .env and add your ANTHROPIC_API_KEY, then run this script again.
    pause
    exit /b 1
)

REM Check if API key is set
findstr "ANTHROPIC_API_KEY=your_anthropic_api_key_here" .env >nul
if %errorlevel% equ 0 (
    echo Error: ANTHROPIC_API_KEY not set in .env file
    echo Please edit .env and add your API key
    pause
    exit /b 1
)

REM Clear screen for clean start
cls

REM Run AIMS
echo Starting AIMS...
python src/main.py
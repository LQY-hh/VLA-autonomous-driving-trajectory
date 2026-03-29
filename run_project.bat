@echo off

REM Set Python executable path
set PYTHON_EXECUTABLE=C:\Users\11414\python-sdk\python3.9.21\python.exe

REM Check if Python exists
if not exist "%PYTHON_EXECUTABLE%" (
    echo Error: Python executable not found at %PYTHON_EXECUTABLE%
    pause
    exit /b 1
)

REM Run the project
echo Running autonomous driving trajectory prediction system...
"%PYTHON_EXECUTABLE%" src/main.py --input data/images/sequence_001 --output output

REM Wait for user input
echo Press any key to exit...
pause

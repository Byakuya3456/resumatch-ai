@echo off
title AI Job Matcher v2 Launcher
color 0A

echo ============================================================
echo   AI Job Matcher  v2.0  ^|  Modular + Upgraded Parsers
echo ============================================================
echo.

REM ── [0/3] Dependencies ─────────────────────────────────────
echo [0/3] Activating virtual environment and installing dependencies...
call .\venv\Scripts\activate
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate virtual environment.
    echo         Ensure 'venv' exists: python -m venv venv
    pause
    exit /b 1
)

pip install -r requirements.txt --quiet
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install Python dependencies.
    echo         Check requirements.txt and your internet connection.
    pause
    exit /b 1
)
echo [OK] Dependencies installed.
echo.

REM ── [1/3] MongoDB ───────────────────────────────────────────
echo [1/3] Checking for MongoDB (port 27017)...
netstat -ano | findstr "LISTENING" | findstr ":27017" >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] MongoDB not found. Attempting to start the 'MongoDB' service...
    net start MongoDB >nul 2>&1
    timeout /t 3 >nul
    netstat -ano | findstr "LISTENING" | findstr ":27017" >nul 2>&1
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to start MongoDB on port 27017.
        echo         Please start MongoDB manually:  mongod --port 27017
        pause
        exit /b 1
    )
)
echo [OK] MongoDB is running.
echo.

REM ── [2/3] Ollama ────────────────────────────────────────────
echo [2/3] Checking for Ollama (port 11434)...
netstat -ano | findstr "LISTENING" | findstr ":11434" >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Ollama not found. Starting 'ollama serve' in background...
    start "Ollama Background Server" /B ollama serve
    echo [INFO] Waiting for Ollama to initialise...
    timeout /t 5 >nul
    netstat -ano | findstr "LISTENING" | findstr ":11434" >nul 2>&1
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to start Ollama on port 11434.
        echo         Please run in a separate terminal:  ollama serve
        pause
        exit /b 1
    )
)
echo [OK] Ollama is running.
echo.

REM ── [3/3] Launch ────────────────────────────────────────────
echo [3/3] All checks passed. Starting server...
echo.
echo  Embedding model : %EMBEDDING_MODEL%
echo  LLM model       : %OLLAMA_MODEL%
echo  API endpoint    : http://127.0.0.1:5000
echo.

echo Launching browser at http://127.0.0.1:5000
start http://127.0.0.1:5000

echo Starting Flask app (modular src/ layout)...
echo Press Ctrl+C to stop.
echo.

python app.py

echo.
echo [INFO] Server stopped.
pause

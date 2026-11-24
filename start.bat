@echo off
title AI Job Matcher v2 Launcher
color 0A

echo ============================================================
echo   AI Job Matcher  v2.0  ^|  Modular + Upgraded Parsers
echo ============================================================
echo.

REM ── [0/3] Virtual Environment & Dependencies ────────────────
echo [0/3] Checking environment & installing dependencies...

where uv >nul 2>&1
if %errorlevel% equ 0 (
    echo [INFO] Astral 'uv' detected. Using ultra-fast package installer.
    if not exist ".\venv\Scripts\activate.bat" (
        echo [INFO] Creating virtual environment with uv...
        uv venv venv
    )
    call .\venv\Scripts\activate
    uv pip install -r requirements.txt --quiet
) else (
    echo [INFO] 'uv' not found. Falling back to standard Python venv and pip.
    if not exist ".\venv\Scripts\activate.bat" (
        echo [INFO] Creating virtual environment with python...
        python -m venv venv
    )
    call .\venv\Scripts\activate
    pip install -r requirements.txt --quiet
)

if %errorlevel% neq 0 (
    echo [ERROR] Failed to install Python dependencies.
    echo         Check requirements.txt and your internet connection.
    pause
    exit /b 1
)
echo [OK] Dependencies verified.
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

REM ── [2/3] LLM Inference Check ──────────────────────────────
REM Read LLM_PROVIDER from .env if present
set "ACTIVE_PROVIDER=ollama"
if exist ".env" (
    for /f "tokens=1,2 delims==" %%A in (.env) do (
        if "%%A"=="LLM_PROVIDER" set "ACTIVE_PROVIDER=%%B"
    )
)

if /i "%ACTIVE_PROVIDER%"=="ollama" (
    echo [2/3] Checking for local Ollama (port 11434)...
    netstat -ano | findstr "LISTENING" | findstr ":11434" >nul 2>&1
    if %errorlevel% neq 0 (
        echo [INFO] Ollama not found. Starting 'ollama serve' in background...
        start "Ollama Background Server" /B ollama serve
        echo [INFO] Waiting for Ollama to initialise...
        timeout /t 5 >nul
        netstat -ano | findstr "LISTENING" | findstr ":11434" >nul 2>&1
        if %errorlevel% neq 0 (
            echo [WARNING] Ollama port 11434 not responding.
            echo           Matches will use Hybrid RAG fallback unless Ollama is started.
        ) else (
            echo [OK] Ollama is running.
        )
    ) else (
        echo [OK] Ollama is running.
    )
) else (
    echo [2/3] Cloud LLM provider configured: %ACTIVE_PROVIDER% (skipping local Ollama requirement)
    echo [OK] Cloud LLM provider ready.
)
echo.

REM ── [3/3] Launch ────────────────────────────────────────────
echo [3/3] All checks passed. Starting server...
echo.
echo  Embedding model : %EMBEDDING_MODEL%
echo  LLM model       : %OLLAMA_MODEL%
echo  API endpoint    : http://127.0.0.1:5000
echo.

echo  Architecture  : Pure Headless REST API Gateway
echo  Root Catalog  : http://127.0.0.1:5000/
echo  Health Check  : http://127.0.0.1:5000/health
echo  Upload API    : http://127.0.0.1:5000/upload (POST)
echo  Generate API  : http://127.0.0.1:5000/generate (POST)
echo.

echo Starting Flask headless API service...
echo Press Ctrl+C to stop.
echo.

python app.py

echo.
echo [INFO] Server stopped.
pause

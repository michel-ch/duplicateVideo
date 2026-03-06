@echo off
echo ============================================
echo  Duplicate Video Detector — Starting App
echo ============================================

REM Start backend
echo [1/2] Starting backend on http://localhost:9000 ...
start "Backend" cmd /k "cd /d %~dp0backend && python -m uvicorn main:app --host 0.0.0.0 --port 9000 --reload"

REM Brief pause for backend to initialize
ping -n 4 127.0.0.1 > NUL

REM Start frontend dev server
echo [2/2] Starting frontend on http://localhost:3000 ...
start "Frontend" cmd /k "cd /d %~dp0frontend && npm run dev -- --port 3000"

echo.
echo Both servers started. Open http://localhost:3000 in your browser.
echo.
pause

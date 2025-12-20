@echo off
setlocal

REM Root folder of this repo
set "ROOT=%~dp0"

echo.
echo ============================================
echo  AI Research - Dev Runner (Frontend+Backend)
echo ============================================
echo.

REM Start Backend (FastAPI)
echo [1/2] Starting backend on http://127.0.0.1:8000 ...
start "backend-8000" cmd /k "cd /d "%ROOT%multi_agent_search" && if exist "%ROOT%venv\Scripts\activate.bat" (call "%ROOT%venv\Scripts\activate.bat") && "%ROOT%venv\Scripts\python.exe" -m uvicorn src.simple_copilot_backend:app --reload --host 127.0.0.1 --port 8000"

REM Start Frontend (Next.js)
echo [2/2] Starting frontend on http://localhost:3000 ...
start "frontend-3000" cmd /k "cd /d "%ROOT%multi_agent_search\copilotkit-ui" && npm run dev -- --port 3000"

echo.
echo Done. Two new terminal windows opened.
echo - Frontend: http://localhost:3000
echo - Backend:  http://127.0.0.1:8000
echo.
endlocal


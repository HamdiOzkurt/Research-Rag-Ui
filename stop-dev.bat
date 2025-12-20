@echo off
setlocal

echo.
echo ============================================
echo  AI Research - Stop Dev (ports 3000/8000)
echo ============================================
echo.

REM Kill by port (safer than killing all node/python)
for /f "tokens=5" %%P in ('netstat -ano ^| findstr :3000 ^| findstr LISTENING') do (
  echo Stopping process on :3000 (PID %%P)
  taskkill /F /PID %%P >nul 2>nul
)

for /f "tokens=5" %%P in ('netstat -ano ^| findstr :8000 ^| findstr LISTENING') do (
  echo Stopping process on :8000 (PID %%P)
  taskkill /F /PID %%P >nul 2>nul
)

echo.
echo Optional: stop Ollama too? (GPU usage)
echo - To stop Ollama: taskkill /F /IM ollama.exe
echo.

endlocal


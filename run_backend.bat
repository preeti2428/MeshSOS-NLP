@echo off
REM Start the MeshSOS backend and dashboard for a phone test.
REM
REM --host 0.0.0.0 is the part that matters. The default binds to
REM localhost only, which the laptop can reach and the phones cannot.

echo Installing dependencies if needed...
python -m pip install -q fastapi "uvicorn[standard]" pydantic

echo.
echo ================================================================
echo   Your laptop's address on this network:
echo ================================================================
ipconfig | findstr /C:"IPv4 Address"
echo.
echo   Dashboard on this laptop:  http://localhost:8000
echo   Put this in the app CONFIG: http://THAT-IPv4-ADDRESS:8000
echo   (not localhost - the phone cannot reach its own localhost)
echo ================================================================
echo.

python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000

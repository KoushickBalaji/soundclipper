@echo off
cd /d "%~dp0"
echo Quick checks...
.venv\Scripts\python.exe -m unittest test_evaluate_slakh || goto :fail
echo.
echo Scoring the 151 Slakh TEST songs (about 15 minutes)...
.venv\Scripts\python.exe evaluate_slakh.py --split test || goto :fail
echo.
echo Scoring the 270 Slakh VALIDATION songs (about 25 minutes)...
.venv\Scripts\python.exe evaluate_slakh.py --split validation || goto :fail
echo.
echo All done. You can close this window.
pause
exit /b 0
:fail
echo.
echo Something went wrong. Leave this window open and tell Claude.
pause
exit /b 1

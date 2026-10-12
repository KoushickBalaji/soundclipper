@echo off
cd /d "%~dp0"
echo Checks...
.venv\Scripts\python.exe -m unittest test_make_listening_clips test_ensemble_separator || goto :fail
echo.
echo Building the blind listening test (about 6 minutes)...
.venv\Scripts\python.exe make_listening_clips.py > runs\listening_test_log.txt 2>&1 || goto :fail
type runs\listening_test_log.txt
echo.
echo Opening the test in your browser...
start "" "runs\listening_test\index.html"
echo Answer at your own pace, then press Show results and paste them into the chat.
timeout /t 60
exit /b 0
:fail
if exist runs\listening_test_log.txt type runs\listening_test_log.txt
echo Something went wrong. Leave this window open and tell Claude.
pause
exit /b 1

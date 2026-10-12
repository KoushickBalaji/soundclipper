@echo off
cd /d "%~dp0"
echo Where did the guitar go? (about 3 minutes)
.venv\Scripts\python.exe -m unittest test_diagnose_guitar_buckets || goto :fail
.venv\Scripts\python.exe diagnose_guitar_buckets.py --split validation --worst 12 --best 3 > runs\diagnose_log.txt 2>&1 || goto :fail
type runs\diagnose_log.txt
echo Done. This window closes in 60 seconds.
timeout /t 60
exit /b 0
:fail
type runs\diagnose_log.txt
echo Something went wrong. Leave this window open and tell Claude.
pause
exit /b 1

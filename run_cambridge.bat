@echo off
cd /d "%~dp0"
echo Checks...
.venv\Scripts\python.exe -m unittest test_cambridge_mt test_metrics || goto :fail
echo.
echo Step 1 of 2: labelling tracks in the Cambridge-MT zips
.venv\Scripts\python.exe cambridge_mt.py --labels-only > runs\cambridge_labels_log.txt 2>&1 || goto :fail
type runs\cambridge_labels_log.txt
echo.
echo Step 2 of 2: scoring the songs that are ready (unpack, score, delete, one at a time)
.venv\Scripts\python.exe cambridge_mt.py > runs\cambridge_eval_log.txt 2>&1 || goto :fail
type runs\cambridge_eval_log.txt
echo.
echo Done. Closing in 90 seconds.
timeout /t 90
exit /b 0
:fail
if exist runs\cambridge_labels_log.txt type runs\cambridge_labels_log.txt
if exist runs\cambridge_eval_log.txt type runs\cambridge_eval_log.txt
echo Something went wrong. Leave this window open and tell Claude.
pause
exit /b 1

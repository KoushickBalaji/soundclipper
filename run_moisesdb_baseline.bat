@echo off
cd /d "%~dp0"
echo Checks...
.venv\Scripts\python.exe -m unittest test_metrics test_index_moisesdb test_make_moisesdb_split test_evaluate_moisesdb || goto :fail
echo.
echo Experiment A: htdemucs_6s on MoisesDB validation + test (about 25 minutes)
.venv\Scripts\python.exe evaluate_moisesdb.py --splits validation test > runs\moisesdb_eval_log.txt 2>&1 || goto :fail
type runs\moisesdb_eval_log.txt
echo Done. Closing in 60 seconds.
timeout /t 60
exit /b 0
:fail
if exist runs\moisesdb_eval_log.txt type runs\moisesdb_eval_log.txt
echo Something went wrong. Leave this window open and tell Claude.
pause
exit /b 1

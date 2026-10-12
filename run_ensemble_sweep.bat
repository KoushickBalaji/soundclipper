@echo off
cd /d "%~dp0"
echo Checks...
.venv\Scripts\python.exe -m unittest test_ensemble_separator test_separators test_compare_runs || goto :fail
echo.
echo Blend sweep on the 35 MoisesDB VALIDATION songs only (the test split stays untouched).
echo Roughly 35 minutes: every song goes through both models, three times.
echo.
echo --- blend with 25 percent community model
.venv\Scripts\python.exe evaluate_moisesdb.py --model ensemble --weight 0.25 --splits validation --skip-probe --clips 0 --out runs\moisesdb_ensemble_w25 > runs\ensemble_w25_log.txt 2>&1 || goto :fail
.venv\Scripts\python.exe compare_runs.py runs\moisesdb_htdemucs_6s runs\moisesdb_ensemble_w25 || goto :fail
echo.
echo --- blend with 50 percent community model
.venv\Scripts\python.exe evaluate_moisesdb.py --model ensemble --weight 0.50 --splits validation --skip-probe --clips 0 --out runs\moisesdb_ensemble_w50 > runs\ensemble_w50_log.txt 2>&1 || goto :fail
.venv\Scripts\python.exe compare_runs.py runs\moisesdb_htdemucs_6s runs\moisesdb_ensemble_w50 || goto :fail
echo.
echo --- blend with 75 percent community model
.venv\Scripts\python.exe evaluate_moisesdb.py --model ensemble --weight 0.75 --splits validation --skip-probe --clips 0 --out runs\moisesdb_ensemble_w75 > runs\ensemble_w75_log.txt 2>&1 || goto :fail
.venv\Scripts\python.exe compare_runs.py runs\moisesdb_htdemucs_6s runs\moisesdb_ensemble_w75 || goto :fail
echo.
echo Done. Closing in 120 seconds.
timeout /t 120
exit /b 0
:fail
if exist runs\ensemble_w25_log.txt type runs\ensemble_w25_log.txt
if exist runs\ensemble_w50_log.txt type runs\ensemble_w50_log.txt
if exist runs\ensemble_w75_log.txt type runs\ensemble_w75_log.txt
echo Something went wrong. Leave this window open and tell Claude.
pause
exit /b 1

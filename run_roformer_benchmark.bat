@echo off
cd /d "%~dp0"
echo Step 1: two small packages the community model needs
.venv\Scripts\python.exe -m pip install beartype rotary_embedding_torch || goto :fail
echo.
echo Step 2: download the guitar model (45 MB) and its code
.venv\Scripts\python.exe fetch_community_model.py || goto :fail
echo.
echo Step 3: offline checks
.venv\Scripts\python.exe -m unittest test_roformer_guitar test_separators test_compare_runs test_metrics || goto :fail
echo.
echo Step 4: smoke test on one 20-second clip
.venv\Scripts\python.exe roformer_guitar.py runs\moisesdb_htdemucs_6s\clips\\11845abc\\mixture.wav --out runs\roformer_smoke > runs\roformer_smoke_log.txt 2>&1 || goto :fail
type runs\roformer_smoke_log.txt
echo.
echo Step 5: score MoisesDB validation + test with the community model
.venv\Scripts\python.exe evaluate_moisesdb.py --model roformer --splits validation test > runs\moisesdb_roformer_log.txt 2>&1 || goto :fail
type runs\moisesdb_roformer_log.txt
echo.
echo Step 6: score the Cambridge sessions with the community model
.venv\Scripts\python.exe cambridge_mt.py --model roformer > runs\cambridge_roformer_log.txt 2>&1 || goto :fail
echo.
echo Step 7: paired comparisons
.venv\Scripts\python.exe compare_runs.py runs\moisesdb_htdemucs_6s runs\moisesdb_roformer_guitar
.venv\Scripts\python.exe compare_runs.py runs\cambridge_htdemucs_6s runs\cambridge_roformer_guitar
echo.
echo Done. Closing in 120 seconds.
timeout /t 120
exit /b 0
:fail
echo.
echo Something went wrong. Leave this window open and tell Claude.
pause
exit /b 1

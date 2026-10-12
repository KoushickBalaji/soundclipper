@echo off
cd /d "%~dp0"
setlocal
set PY=.venv\Scripts\python.exe
echo === 1/8 label + score remaining Cambridge songs with htdemucs
%PY% cambridge_mt.py > runs\cambridge_eval_log.txt 2>&1 || goto :fail
findstr /C:"si-sdr" runs\cambridge_eval_log.txt | find /C "si-sdr"
echo === 2/8 packages for the community model
%PY% -m pip install --quiet beartype rotary_embedding_torch || goto :fail
echo === 3/8 download the community model
%PY% fetch_community_model.py || goto :fail
echo === 4/8 offline checks
%PY% -m unittest test_roformer_guitar test_separators test_compare_runs test_metrics || goto :fail
echo === 5/8 smoke test on one 20-second clip
%PY% roformer_guitar.py runs\moisesdb_htdemucs_6s\clips\11845abc\mixture.wav --out runs\roformer_smoke || goto :fail
echo === 6/8 MoisesDB with the community model
%PY% evaluate_moisesdb.py --model roformer --splits validation test > runs\moisesdb_roformer_log.txt 2>&1 || goto :fail
echo === 7/8 Cambridge sessions with the community model
%PY% cambridge_mt.py --model roformer > runs\cambridge_roformer_log.txt 2>&1 || goto :fail
echo === 8/8 paired comparisons
%PY% compare_runs.py runs\moisesdb_htdemucs_6s runs\moisesdb_roformer_guitar || goto :fail
%PY% compare_runs.py runs\cambridge_htdemucs_6s runs\cambridge_roformer_guitar || goto :fail
echo.
echo ALL DONE. This window closes in 3 minutes.
timeout /t 180
exit /b 0
:fail
echo.
echo FAILED at the step above. Leave this window open and tell Claude.
pause
exit /b 1

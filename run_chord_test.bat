@echo off
cd /d "%~dp0"
echo Checks...
.venv\Scripts\python.exe -m unittest test_chords test_midi_notes || goto :fail
echo.
echo Chord reading test: full song vs separated guitar vs true stem (about 15 minutes)
.venv\Scripts\python.exe evaluate_chords.py --songs 20 > runs\chords_log.txt 2>&1 || goto :fail
type runs\chords_log.txt
echo.
echo Done. Closing in 120 seconds.
timeout /t 120
exit /b 0
:fail
if exist runs\chords_log.txt type runs\chords_log.txt
echo Something went wrong. Leave this window open and tell Claude.
pause
exit /b 1

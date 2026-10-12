@echo off
cd /d "%~dp0"
echo Collecting any Cambridge-MT zips from your Downloads folder...
.venv\Scripts\python.exe collect_downloads.py
echo.
echo Fetching up to 2 songs from urls.txt (if the site allows it)...
.venv\Scripts\python.exe fetch_cambridge.py --max 2
echo.
pause

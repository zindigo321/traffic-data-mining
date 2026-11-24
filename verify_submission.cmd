@echo off
cd /d "%~dp0"
call conda run --no-capture-output -n niulaoshi python -B -X utf8 verify_submission.py --run
if errorlevel 1 exit /b 1
pause

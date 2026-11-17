@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
if defined NIULAOSHI_PYTHON goto explicit_python
if /i "%CONDA_DEFAULT_ENV%"=="niulaoshi" goto active_env
where conda >nul 2>nul
if errorlevel 1 goto fallback
call conda run --no-capture-output -n niulaoshi python -B -X utf8 "%~dp0run.py" %*
exit /b %errorlevel%
:active_env
python -B -X utf8 "%~dp0run.py" %*
exit /b %errorlevel%
:explicit_python
"%NIULAOSHI_PYTHON%" -B -X utf8 "%~dp0run.py" %*
exit /b %errorlevel%
:fallback
if exist "D:\conda_envs\niulaoshi\python.exe" goto known_python
echo Please open Anaconda Prompt, run conda activate niulaoshi, then python run.py
exit /b 2
:known_python
"D:\conda_envs\niulaoshi\python.exe" -B -X utf8 "%~dp0run.py" %*
exit /b %errorlevel%

@echo off
call "%~dp0run_niulaoshi.cmd" train %*
exit /b %errorlevel%

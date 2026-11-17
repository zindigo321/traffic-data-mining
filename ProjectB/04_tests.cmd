@echo off
call "%~dp0run_niulaoshi.cmd" tests %*
exit /b %errorlevel%

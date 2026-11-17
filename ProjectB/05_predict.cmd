@echo off
call "%~dp0run_niulaoshi.cmd" predict %*
exit /b %errorlevel%

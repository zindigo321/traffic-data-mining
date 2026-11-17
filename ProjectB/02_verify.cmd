@echo off
call "%~dp0run_niulaoshi.cmd" verify %*
exit /b %errorlevel%

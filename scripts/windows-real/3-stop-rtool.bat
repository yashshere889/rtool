@echo off
setlocal
cd /d "%~dp0"
if not defined RTOOL_HOST set "RTOOL_HOST=barklalogin1.liv.ac.uk"
where ssh >nul 2>nul
if errorlevel 1 (
  echo This needs the OpenSSH client that comes with Windows, and it was not found.
  echo Turn it on under Settings, Optional features, OpenSSH Client, then try again.
  pause
  exit /b 1
)
if exist barkla-user.txt set /p BARKLA_USER=<barkla-user.txt
if not defined BARKLA_USER set /p "BARKLA_USER=Your Barkla username: "
if not defined BARKLA_USER exit /b 1
>barkla-user.txt echo %BARKLA_USER%
echo If you are off campus, connect the University VPN first.
echo Barkla will ask for your password. Nothing shows while you type it.
echo.
ssh -t "%BARKLA_USER%@%RTOOL_HOST%" "bash -l ~/'Research tool'/scripts/barkla_rtool.sh ui --stop"
echo.
echo The web page's job has stopped, if the line above says so. Runs keep going on Barkla without it.
pause

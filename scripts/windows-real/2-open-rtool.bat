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
if not defined RTOOL_PORT set "RTOOL_PORT=8780"
ssh -t -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -L "127.0.0.1:%RTOOL_PORT%:/mnt/scratch/users/%BARKLA_USER%/rtool/ui/tunnel.sock" "%BARKLA_USER%@%RTOOL_HOST%" "bash -l ~/'Research tool'/scripts/rtool_ui_session.sh %RTOOL_PORT%"
echo.
echo The connection has closed. Your runs keep going on Barkla.
echo "Could not resolve hostname" usually means the VPN is off.
pause

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
echo Copying the code to Barkla (first password)...
scp rtool-code.tar.gz "%BARKLA_USER%@%RTOOL_HOST%:rtool-code.tar.gz"
if errorlevel 1 goto failed
echo Setting up on Barkla (second password)...
ssh -t "%BARKLA_USER%@%RTOOL_HOST%" "mkdir -p ~/'Research tool' && tar -xzf ~/rtool-code.tar.gz -C ~/'Research tool' && rm ~/rtool-code.tar.gz && bash -l ~/'Research tool'/scripts/first_setup_barkla.sh"
if errorlevel 1 goto failed
echo.
echo Done. Next, double-click 2-open-rtool.bat. Real runs need the model download to finish first.
pause
exit /b 0
:failed
echo.
echo That did not finish. "Could not resolve hostname" usually means the VPN is off.
echo Otherwise, please send the text in this window to whoever gave you this.
pause
exit /b 1

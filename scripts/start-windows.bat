@echo off
rem Starts the Research Tool demo on this computer.
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
where uvx >nul 2>nul
if not errorlevel 1 goto start
echo Installing uv, a small Python tool...
set "PSModulePath="
powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
set "PATH=%USERPROFILE%\.local\bin;%PATH%"
where uvx >nul 2>nul
if not errorlevel 1 goto start
echo.
echo Could not install uv. This computer may block it. Please send this text to whoever gave you this.
goto end
:start
echo Starting. The first time takes a minute or two.
uvx --python 3.12 --from "./research_tool-0.1.0-py3-none-any.whl[web]" rtool ui --mode local --demo --open
echo.
echo The tool has stopped. If that was not you, please send this text to whoever gave you this.
:end
pause

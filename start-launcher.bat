@echo off
setlocal
rem NInfer Launcher: local WebUI to configure and supervise ninfer-serve.exe.
rem
rem Usage:
rem   start-launcher.bat [launcher args...]
rem     --port 8100        UI port (default 8100)
rem     --no-browser       do not open the browser automatically
rem
rem Python: tools\launcher\bootstrap.ps1 finds a usable Python 3.10+ (NINFER_LAUNCHER_PYTHON,
rem py launcher, PATH, common install dirs, conda). When none exists it downloads a portable
rem Python once into %LOCALAPPDATA%\NInfer-Launcher\runtime and installs the launcher
rem dependencies there, so the system Python is never touched.
rem
rem The launcher only listens on 127.0.0.1 and never modifies the engine.

set "BIN=%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%BIN%tools\launcher\bootstrap.ps1" %*
exit /b %ERRORLEVEL%
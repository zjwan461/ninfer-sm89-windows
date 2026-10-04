@echo off
setlocal
rem NInfer: OpenAI + Anthropic compatible server for Qwen3.8-27B (int8-prefill artifact).
rem
rem Usage:
rem   start-qwen38-server.bat [MODEL] [extra ninfer-serve args...]
rem     MODEL  path to a .ninfer file, or to a directory containing one
rem            (omit it to use the resolution order below)
rem
rem Model resolution, first hit wins:
rem   1. the first command-line argument
rem   2. NINFER_MODEL      full path to a .ninfer file
rem   3. NINFER_MODEL_DIR  directory holding qwen3_8_27b_a8.ninfer
rem   4. models\qwen3_8_27b_a8.ninfer beside this script
rem
rem Binary: set NINFER_SERVER to use a ninfer-serve.exe somewhere else.
rem Extra flags can also come from NINFER_SERVE_ARGS.
rem Download the artifact first:
rem   hf download jgamboa/Qwen3.8-27B-NInfer-4090 qwen3_8_27b_a8.ninfer --local-dir models
rem API http://127.0.0.1:8080/v1   Monitor http://127.0.0.1:8080/monitor

set "BIN=%~dp0"
set "SCRIPT_NAME=%~nx0"
set "DEFAULT_NAME=qwen3_8_27b_a8.ninfer"
if not defined NINFER_SERVER set "NINFER_SERVER=%BIN%ninfer-serve.exe"

set "MODEL=%~1"
if not defined MODEL goto :model_from_environment
shift
goto :resolve_model

:model_from_environment
if defined NINFER_MODEL set "MODEL=%NINFER_MODEL%"
if not defined MODEL if defined NINFER_MODEL_DIR set "MODEL=%NINFER_MODEL_DIR%\%DEFAULT_NAME%"
if not defined MODEL set "MODEL=%BIN%models\%DEFAULT_NAME%"

:resolve_model
rem Everything after MODEL is forwarded to ninfer-serve. %* is unusable here: cmd's shift does
rem not drop the first argument from %*, so the remaining arguments are collected explicitly.
set "EXTRA="
:collect_extra
if "%~1"=="" goto :expand_directory
set "EXTRA=%EXTRA% "%~1""
shift
goto :collect_extra

:expand_directory
rem A directory is accepted as well: expand it to the .ninfer file inside it.
if not exist "%MODEL%\" goto :check_model
set "MODEL_DIR=%MODEL%"
set "MODEL="
for %%F in ("%MODEL_DIR%\*.ninfer") do if not defined MODEL set "MODEL=%%~fF"
if defined MODEL goto :check_model
echo No .ninfer file found in directory: %MODEL_DIR%
goto :usage

:check_model
if not defined MODEL goto :usage
if not exist "%MODEL%" goto :usage
if not exist "%NINFER_SERVER%" (
  echo Missing %NINFER_SERVER%
  echo Put this launcher beside ninfer-serve.exe, or set NINFER_SERVER.
  pause
  exit /b 1
)

echo Starting Qwen3.8-27B at http://127.0.0.1:8080/v1
echo Model: %MODEL%
"%NINFER_SERVER%" "%MODEL%" ^
  --host 127.0.0.1 --port 8080 --model-id qwen3.8-27b ^
  --max-context 100000 --kv-capacity 100000 --kv-dtype rk4v4-e8 --max-concurrency 3 ^
  --max-pending-requests 10 --pending-timeout-ms 600000 --prefill-chunk 1408 ^
  --spec mtp --draft-tokens 3 --lm-head-draft --ngram chain --preserve-thinking ^
  --device-state-slots 3 --host-state-slots 4 --host-kv-mib 4096 %EXTRA% %NINFER_SERVE_ARGS%
set "SERVE_EXIT=%ERRORLEVEL%"
pause
exit /b %SERVE_EXIT%

:usage
echo Model not found.
echo Usage: %SCRIPT_NAME% [MODEL] [extra ninfer-serve args...]
echo   MODEL  path to a .ninfer file, or to a directory containing one
echo Resolution order: argument, NINFER_MODEL, NINFER_MODEL_DIR, %BIN%models\%DEFAULT_NAME%
echo Download it with:
echo   hf download jgamboa/Qwen3.8-27B-NInfer-4090 %DEFAULT_NAME% --local-dir models
pause
exit /b 1

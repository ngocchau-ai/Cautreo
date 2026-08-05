@echo off
REM ============================================================
REM run_deepseek.bat - Launch CAUTREO server with DeepSeek-V4-Flash-0731 MXFP4
REM ============================================================

set MODEL_DIR=E:\models\DeepSeek-V4-Flash\DeepSeek-V4-Flash-0731-MXFP4
set PART1=%MODEL_DIR%\DeepSeek-V4-Flash-0731-MXFP4-00001-of-00004.gguf
set PART2=%MODEL_DIR%\DeepSeek-V4-Flash-0731-MXFP4-00002-of-00004.gguf
set PART3=%MODEL_DIR%\DeepSeek-V4-Flash-0731-MXFP4-00003-of-00004.gguf
set PART4=%MODEL_DIR%\DeepSeek-V4-Flash-0731-MXFP4-00004-of-00004.gguf

echo =============================================
echo  CAUTREO x DeepSeek-V4-Flash-0731 MXFP4
echo =============================================
echo  Model: 4 parts ~145 GB
echo  SSD streaming: ON (8 GB expert cache)
echo  Device: CPU, Threads: 8, Port: 8080
echo =============================================

if not exist "%PART1%" (echo [ERROR] Part 1 not found & exit /b 1)
if not exist "%PART4%" (echo [ERROR] Part 4 not found & exit /b 1)

echo [OK] All model parts found. Starting server...

.\build\cautreo-server.exe ^
    --model-parts "%PART1%" ^
    --model-parts "%PART2%" ^
    --model-parts "%PART3%" ^
    --model-parts "%PART4%" ^
    --ssd-streaming ^
    --ssd-cache-gb 8 ^
    --port 8080 ^
    --threads 8 ^
    --ctx-size 4096

@echo off
setlocal

set CC=gcc
set CFLAGS=-O2 -std=c11 -Wall -Wextra -I src -I src/core
set CORE_LIB=build\libcautreo_core.a
set ENGINE_LIB=build\libcautreo_engine.a
set LIBS=-lm

set INT_DIR=build\integration
if not exist %INT_DIR% mkdir %INT_DIR%

set FAIL=0

echo === CAUTREO Integration Tests ===

rem Build agent object separately (needed by agent_e2e_test)
if not exist build\agent mkdir build\agent
%CC% %CFLAGS% -c src/agent/agent.c -o build\agent\agent.o
if errorlevel 1 (
    echo [BUILD FAIL] agent.c
    set FAIL=1
)

rem --- streaming_engine_test ---
echo --- streaming_engine_test ---
%CC% %CFLAGS% tests\integration\streaming_engine_test.c %CORE_LIB% %ENGINE_LIB% %LIBS% -o %INT_DIR%\streaming_engine_test.exe
if errorlevel 1 (
    echo   [BUILD FAIL] streaming_engine_test
    set FAIL=1
) else (
    %INT_DIR%\streaming_engine_test.exe
    if errorlevel 1 set FAIL=1
)

rem --- core_engine_test ---
echo --- core_engine_test ---
%CC% %CFLAGS% tests\integration\core_engine_test.c %CORE_LIB% %ENGINE_LIB% %LIBS% -o %INT_DIR%\core_engine_test.exe
if errorlevel 1 (
    echo   [BUILD FAIL] core_engine_test
    set FAIL=1
) else (
    %INT_DIR%\core_engine_test.exe
    if errorlevel 1 set FAIL=1
)

rem --- agent_e2e_test (needs agent.o) ---
echo --- agent_e2e_test ---
%CC% %CFLAGS% tests\integration\agent_e2e_test.c build\agent\agent.o %CORE_LIB% %ENGINE_LIB% %LIBS% -o %INT_DIR%\agent_e2e_test.exe
if errorlevel 1 (
    echo   [BUILD FAIL] agent_e2e_test
    set FAIL=1
) else (
    %INT_DIR%\agent_e2e_test.exe
    if errorlevel 1 set FAIL=1
)

if %FAIL%==0 (
    echo.
    echo ALL INTEGRATION TESTS PASS
    exit /b 0
) else (
    echo.
    echo SOME INTEGRATION TESTS FAILED
    exit /b 1
)

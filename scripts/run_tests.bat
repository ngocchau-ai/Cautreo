@echo off
setlocal

set CC=gcc
set CFLAGS=-O2 -std=c11 -Wall -Wextra -I src -I src/core
set CORE_LIB=build\libcautreo_core.a
set ENGINE_LIB=build\libcautreo_engine.a
set LIBS=-lm

set TEST_DIR=build\tests
if not exist %TEST_DIR% mkdir %TEST_DIR%

set FAIL=0
set PASS=0

echo === CAUTREO Unit Tests ===

for %%F in (tests\unit\*_test.c) do (
    set NAME=%%~nF
    echo --- %%~nF ---
    %CC% %CFLAGS% %%F %CORE_LIB% %ENGINE_LIB% %LIBS% -o %TEST_DIR%\%%~nF.exe
    if errorlevel 1 (
        echo   [BUILD FAIL] %%~nF
        set FAIL=1
    ) else (
        %TEST_DIR%\%%~nF.exe
        if errorlevel 1 (
            echo   [FAIL] %%~nF
            set FAIL=1
        )
    )
)

if %FAIL%==0 (
    echo.
    echo ALL UNIT TESTS PASS
    exit /b 0
) else (
    echo.
    echo SOME TESTS FAILED
    exit /b 1
)

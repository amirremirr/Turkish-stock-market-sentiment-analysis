@echo off
REM run.bat - Convenience launcher for the BIST 100 sentiment pipeline.
REM
REM Python resolution order:
REM   1. .venv\Scripts\python.exe  (standard venv in the project directory)
REM   2. venv\Scripts\python.exe   (alternative venv name)
REM   3. python  (system Python / whatever is on PATH)
REM
REM Usage:
REM   run.bat run                          full pipeline (scrape + score + aggregate + prices + plot)
REM   run.bat scrape                       fetch latest headlines only
REM   run.bat score                        score unscored headlines with the sentiment model
REM   run.bat aggregate                    recompute daily sentiment averages
REM   run.bat recategorize                 re-classify all headlines with keyword rules + re-aggregate
REM   run.bat recategorize --llm           re-classify with the LLM + delete irrelevant headlines
REM   run.bat relabel                      recompute sentiment labels from stored probs (after threshold change)
REM   run.bat prices                       download BIST 100 price history
REM   run.bat fx-rates                     download USD/TRY FX rates (Alpha Vantage)
REM   run.bat plot                         generate the visualisation
REM   run.bat status                       show DB statistics
REM   run.bat dashboard                    generate + open the HTML dashboard
REM   run.bat clean                        remove off-topic headlines from DB
REM   run.bat clean --dry-run              preview how many would be removed
REM   run.bat export-labels                export CSV for model validation (150 headlines)
REM   run.bat export-labels --n 300        export 300 headlines for validation
REM   run.bat validate-labels <csv>        validate model against a labeled CSV
REM   run.bat validate-labels --tracker    show label collection progress
REM   run.bat test                         run the pytest test suite
REM   run.bat --days 60 run               override lookback window to 60 days

chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8

REM ── Resolve Python ────────────────────────────────────────────────────────────
set "PYTHON="
if exist "%~dp0.venv\Scripts\python.exe" set "PYTHON=%~dp0.venv\Scripts\python.exe"
if not defined PYTHON (
    if exist "%~dp0venv\Scripts\python.exe" set "PYTHON=%~dp0venv\Scripts\python.exe"
)
if not defined PYTHON set "PYTHON=python"

REM ── Special alias: "run.bat test" -> pytest ───────────────────────────────────
if /i "%~1"=="test" (
    "%PYTHON%" -m pytest tests\ -v
    exit /b %ERRORLEVEL%
)

"%PYTHON%" main.py %*

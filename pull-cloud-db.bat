@echo off
REM Pull the latest cloud-run database from the `data` branch into the working
REM directory, so you can inspect it locally (status, dashboard, evaluate).
REM The cloud is the single source of truth once the GitHub Actions job is live.
cd /d "%~dp0"
echo Fetching latest cloud DB from origin/data ...
git fetch origin data || goto :err
git checkout origin/data -- finance_sentiment.db || goto :err
echo.
echo Done. Latest cloud DB is now in finance_sentiment.db
echo Inspect with:  run.bat status   ^|   run.bat dashboard
goto :eof
:err
echo.
echo Could not pull the data branch. Has the cloud job run at least once yet?
exit /b 1

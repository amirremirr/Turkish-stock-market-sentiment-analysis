# Cloud daily run (GitHub Actions)

The daily pipeline runs in GitHub Actions instead of (or as well as) the local
Windows Task Scheduler. Free for public repos; no laptop needed.

## How state works

The SQLite DB is the project's memory but is gitignored on `main`. It lives on
an **orphan `data` branch**: the workflow restores it at the start of each run
and force-pushes the updated DB (single snapshot, no history bloat) at the end.
`sentiment_vs_bist100.png` rides along so the latest chart is viewable on the
data branch too.

**Single writer rule:** once the cloud job is confirmed working, **disable the
local Task Scheduler task** or the laptop and the cloud will both mutate the DB
and diverge:

```powershell
Disable-ScheduledTask -TaskName "BIST100-Sentiment"   # run as admin
```

The local task stays as a manual fallback (`run.bat run`) if Actions ever fails.

## One-time setup

1. **Rotate the OpenAI key** (the old one was shared in chat) at
   platform.openai.com.
2. Add it as a repo secret: GitHub → repo **Settings → Secrets and variables →
   Actions → New repository secret**, name `OPENAI_API_KEY`, value = new key.
   (Later, when KAP goes live: add `MKK_API_KEY` / `MKK_API_SECRET` too.)
3. The `data` branch is seeded with the current DB at setup time.
4. Test: Actions tab → **daily-pipeline → Run workflow**. Watch it go green.
5. Confirm it worked: `pull-cloud-db.bat` then `run.bat status` locally.
6. Disable the local task (command above).

## Schedule

`30 6 * * 1-5` = 06:30 UTC ≈ 09:30 Istanbul, Mon–Fri (pre-open). GitHub cron can
run a few minutes late under load — fine for a next-day signal.

## Inspecting cloud data locally

```bat
pull-cloud-db.bat      :: fetch the latest cloud DB into the working dir
run.bat status         :: or  run.bat dashboard
```

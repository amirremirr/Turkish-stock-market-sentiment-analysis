# run_scheduled.py - Daily pipeline runner.
#
# Intended usage — call with the Python from your virtual environment:
#
#   Windows Task Scheduler example:
#     Program:  C:\path\to\your-venv\Scripts\python.exe
#     Args:     run_scheduled.py
#     Start in: C:\path\to\this\project
#
#   Or from any shell in the project directory:
#     python run_scheduled.py
#
# Crash recovery:
#   If "main.py run" crashes (OOM, network error, etc.), the script
#   automatically retries each step individually in a fresh subprocess.
#   Each step gets a clean memory space, which fixes OOM crashes that
#   happen when scraping + model loading compete for the same RAM.

import os
import sys
import datetime
import subprocess
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Setup ─────────────────────────────────────────────────────────────────────

HERE = Path(__file__).resolve().parent
os.chdir(HERE)

# Only force HuggingFace offline if the model is already cached.
# If the cache is missing (fresh install, model update), allow download so the
# pipeline doesn't fail silently with a cryptic "file not found" error.
_HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"
_MODEL_SLUG = "models--cardiffnlp--twitter-xlm-roberta-base-sentiment"
if (_HF_CACHE / _MODEL_SLUG).exists():
    os.environ["HF_HUB_OFFLINE"]       = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
# else: leave unset — HuggingFace will download on first run (may be slow)

os.environ["PYTHONIOENCODING"] = "utf-8"

# Use the same Python interpreter that launched this script.
# When called via Task Scheduler with your venv Python, this is automatically correct.
PYTHON = sys.executable
TODAY  = datetime.date.today()

# ── Log file ──────────────────────────────────────────────────────────────────

log_dir = HERE / "logs"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"{TODAY}.log"


def log(msg: str = "") -> None:
    line = msg + "\n"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(line)
    print(line, end="")


# ── Weekend + BIST holiday check ──────────────────────────────────────────────

if TODAY.weekday() >= 5:
    log(f"{TODAY}  Weekend — skipping.")
    sys.exit(0)

try:
    sys.path.insert(0, str(HERE))
    from config import BIST_HOLIDAYS
    if TODAY.isoformat() in BIST_HOLIDAYS:
        log(f"{TODAY}  BIST holiday — skipping.")
        sys.exit(0)
except Exception:
    pass  # config unavailable — continue anyway

# ── Fix stale pipeline_runs records from previous crashes ─────────────────────
# If a previous run crashed mid-way, it left status='running' in the DB.
# Fix it before starting so the audit layer shows accurate history.

try:
    sys.path.insert(0, str(HERE))
    import database as db
    with db._conn() as con:
        stale = con.execute(
            "UPDATE pipeline_runs SET status='crashed', finished_at=datetime('now') "
            "WHERE status='running'"
        ).rowcount
    if stale:
        log(f"  Fixed {stale} stale 'running' pipeline record(s) from previous crash.")
except Exception as exc:
    log(f"  Warning: could not fix stale records: {exc}")

# ── Header ────────────────────────────────────────────────────────────────────

log("=" * 62)
log(f"  BIST100 Sentiment Pipeline — {TODAY}  {datetime.datetime.now().strftime('%H:%M:%S')}")
log(f"  HuggingFace offline: ON  |  batch size: 1  |  auto-recovery: ON")
log("=" * 62)
log()

# ── Power management: prevent sleep while pipeline is running ─────────────────
# SetThreadExecutionState tells Windows this thread needs the system awake.
# ES_CONTINUOUS | ES_SYSTEM_REQUIRED prevents sleep until we release it.
# This stops the "process killed at 92%" problem when the laptop closes mid-run.

import ctypes

ES_CONTINUOUS       = 0x80000000
ES_SYSTEM_REQUIRED  = 0x00000001

def _prevent_sleep() -> None:
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        log("  Power: sleep prevented for duration of pipeline run.")
    except Exception:
        pass

def _allow_sleep() -> None:
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
    except Exception:
        pass


# ── Helper: run a subprocess, stream output to log ────────────────────────────

def run(cmd: list, label: str) -> int:
    log(f"  [{label}] starting ...")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(HERE),
        env=os.environ.copy(),
    )
    for line in proc.stdout:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(line)
        sys.stdout.write(line)
    proc.wait()
    status = "OK" if proc.returncode == 0 else f"FAILED (exit={proc.returncode})"
    log(f"  [{label}] {status}")
    log()
    return proc.returncode


# ── Step 1: try the full pipeline in one go ───────────────────────────────────

_prevent_sleep()
pipeline_exit = run([PYTHON, "main.py", "run", "--no-show"], "1/3 pipeline")

# ── Auto-recovery: if full pipeline failed, retry each step separately ────────
# Each step is a fresh subprocess = fresh memory allocation.
# This is the main defence against OOM crashes: scraping used RAM is released
# before the model tries to allocate its batch tensors.

if pipeline_exit != 0:
    log("  Full pipeline failed — retrying step by step (fresh memory per step) ...")
    log()

    exits = {}
    exits["scrape"]    = run([PYTHON, "main.py", "scrape"],           "recovery scrape")
    exits["score"]     = run([PYTHON, "main.py", "score"],            "recovery score")
    exits["aggregate"] = run([PYTHON, "main.py", "aggregate"],        "recovery aggregate")
    exits["prices"]    = run([PYTHON, "main.py", "prices"],           "recovery prices")
    exits["fx-rates"]  = run([PYTHON, "main.py", "fx-rates"],        "recovery fx-rates")
    exits["plot"]      = run([PYTHON, "main.py", "plot", "--no-show"],"recovery plot")

    failed = [k for k, v in exits.items() if v != 0]
    if not failed:
        log("  Recovery: all steps OK.")
        pipeline_exit = 0
        # Record the recovery in pipeline_runs — without this, a successfully
        # recovered day stays in the audit trail as a crashed/error run.
        try:
            rec_id = db.log_run_start(model_name=None)
            db.log_run_end(rec_id, status="recovered")
        except Exception as exc:
            log(f"  Warning: could not record recovery run: {exc}")
    else:
        log(f"  Recovery: these steps still failed: {failed}")

# ── Step 2: quality audit ─────────────────────────────────────────────────────

_allow_sleep()   # pipeline done — system can sleep normally again
log("-" * 62)
eval_exit = run([PYTHON, "evaluate.py"], "2/3 evaluate")

# ── Step 3: refresh the HTML dashboard ────────────────────────────────────────

run([PYTHON, "dashboard.py"], "3/3 dashboard")

# ── Rolling DB backup (keep last 7 daily copies) ──────────────────────────────

try:
    import shutil
    backup_dir = HERE / "backups"
    backup_dir.mkdir(exist_ok=True)
    dst = backup_dir / f"daily_{TODAY}.db"
    shutil.copy2(HERE / "finance_sentiment.db", dst)
    for old_file in sorted(backup_dir.glob("daily_*.db"))[:-7]:
        old_file.unlink()
    log(f"  Backup: {dst.name} (rolling 7)")
except Exception as exc:
    log(f"  Warning: backup failed: {exc}")

# ── Footer ────────────────────────────────────────────────────────────────────

log("=" * 62)
log(f"  Finished: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log(f"  Pipeline: {'OK' if pipeline_exit == 0 else 'FAILED'}    "
    f"Eval: {'OK' if eval_exit == 0 else 'FAILED'}")
log("=" * 62)

sys.exit(pipeline_exit)

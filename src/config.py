import os
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(override=True)

# ---- DB env vars (support BOTH naming styles) ----
# Preferred: PGHOST/PGDATABASE/PGUSER/PGPASSWORD/PGPORT (Neon default)
# Also supports: DB_HOST/DB_NAME/DB_USER/DB_PASSWORD/DB_PORT (legacy)

DB_HOST = os.getenv("PGHOST") or os.getenv("DB_HOST", "")
DB_PORT = int(os.getenv("PGPORT") or os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("PGDATABASE") or os.getenv("DB_NAME", "")
DB_USER = os.getenv("PGUSER") or os.getenv("DB_USER", "")
DB_PASSWORD = os.getenv("PGPASSWORD") or os.getenv("DB_PASSWORD", "")

# Backward/compat variables (so src.db can import PGHOST etc if needed)
PGHOST = DB_HOST
PGPORT = DB_PORT
PGDATABASE = DB_NAME
PGUSER = DB_USER
PGPASSWORD = DB_PASSWORD

# Expected columns (adjust if your DB uses different names)
TIME_COL = "time_s"
AXIS_COLS = [f"axis_{i}" for i in range(1, 9)]  # axis_1 ... axis_8

RAW_TABLE = os.getenv("RAW_TABLE", "robot_currents_raw")
EVENTS_TABLE = os.getenv("EVENTS_TABLE", "pm_events")

# Optional: a separate table used for streaming synthetic TEST data (recommended for the rubric)
STREAM_TABLE = os.getenv("STREAM_TABLE", "robot_currents_stream_test")

# Where discovered thresholds are stored (created by the notebook)
THRESHOLDS_PATH = os.getenv("THRESHOLDS_PATH", "outputs/models/thresholds.json")

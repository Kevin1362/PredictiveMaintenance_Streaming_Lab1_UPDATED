import psycopg2
import pandas as pd
from psycopg2.extras import execute_values

from .config import (
    PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD,
    RAW_TABLE, STREAM_TABLE, EVENTS_TABLE
)


def get_conn():
    return psycopg2.connect(
        host=PGHOST,
        port=PGPORT,
        dbname=PGDATABASE,
        user=PGUSER,
        password=PGPASSWORD,
        sslmode="require",
    )


def ensure_readings_table(table: str, time_col: str, axis_cols: list[str]):
    """Create one readings table (same schema as RAW/STREAM tables)."""
    with get_conn() as conn, conn.cursor() as cur:
        cols_sql = ", ".join([f"{c} DOUBLE PRECISION" for c in axis_cols])
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {table} (
                id BIGSERIAL PRIMARY KEY,
                {time_col} DOUBLE PRECISION,
                {cols_sql},
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        conn.commit()


def ensure_tables(time_col: str, axis_cols: list[str]):
    """
    Creates:
    - RAW_TABLE: training readings (uploaded/streamed)
    - STREAM_TABLE: synthetic TEST readings streamed in time (recommended for the rubric)
    - EVENTS_TABLE: detected ALERT/ERROR events
    """
    with get_conn() as conn, conn.cursor() as cur:
        cols_sql = ", ".join([f"{c} DOUBLE PRECISION" for c in axis_cols])

        for table in (RAW_TABLE, STREAM_TABLE):
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    id BIGSERIAL PRIMARY KEY,
                    {time_col} DOUBLE PRECISION,
                    {cols_sql},
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)

        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {EVENTS_TABLE} (
                id BIGSERIAL PRIMARY KEY,
                axis_name TEXT NOT NULL,
                event_type TEXT NOT NULL,          -- ALERT or ERROR
                start_time DOUBLE PRECISION NOT NULL,
                end_time DOUBLE PRECISION NOT NULL,
                duration_s DOUBLE PRECISION NOT NULL,
                threshold DOUBLE PRECISION NOT NULL,
                max_deviation DOUBLE PRECISION NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)

        conn.commit()


def insert_rows(df: pd.DataFrame, table: str, time_col: str, axis_cols: list[str]):
    """
    Insert many rows into a given table.

    IMPORTANT FIX:
    Convert NumPy scalars (np.float64 etc.) to Python float,
    otherwise psycopg2 may generate SQL like np.float64(...)
    and Postgres throws: schema "np" does not exist.
    """
    cols = [time_col] + axis_cols

    # Ensure correct cols + order
    df2 = df[cols].copy()

    # Force numeric + convert numpy types -> python floats
    for c in cols:
        df2[c] = pd.to_numeric(df2[c], errors="coerce")

    df2 = df2.fillna(0.0)

    # Build safe tuples (pure python floats)
    values = [tuple(float(x) for x in row) for row in df2.to_numpy()]

    with get_conn() as conn, conn.cursor() as cur:
        execute_values(
            cur,
            f"INSERT INTO {table} ({', '.join(cols)}) VALUES %s",
            values,
            page_size=1000,
        )
        conn.commit()


def insert_raw_rows(df: pd.DataFrame, time_col: str, axis_cols: list[str]):
    """Backward-compatible helper: inserts into RAW_TABLE."""
    insert_rows(df, RAW_TABLE, time_col, axis_cols)


def read_training_data(limit: int | None = None) -> pd.DataFrame:
    q = f"SELECT * FROM {RAW_TABLE} ORDER BY id"
    if limit:
        q += f" LIMIT {limit}"
    with get_conn() as conn:
        return pd.read_sql(q, conn)


def read_stream_data(limit: int | None = None) -> pd.DataFrame:
    q = f"SELECT * FROM {STREAM_TABLE} ORDER BY id"
    if limit:
        q += f" LIMIT {limit}"
    with get_conn() as conn:
        return pd.read_sql(q, conn)


def insert_events(events: list[dict]):
    if not events:
        return

    cols = [
        "axis_name",
        "event_type",
        "start_time",
        "end_time",
        "duration_s",
        "threshold",
        "max_deviation"
    ]

    # Convert to safe python floats/strings
    values = []
    for e in events:
        values.append((
            str(e["axis_name"]),
            str(e["event_type"]),
            float(e["start_time"]),
            float(e["end_time"]),
            float(e["duration_s"]),
            float(e["threshold"]),
            float(e["max_deviation"]),
        ))

    with get_conn() as conn, conn.cursor() as cur:
        execute_values(
            cur,
            f"INSERT INTO {EVENTS_TABLE} ({', '.join(cols)}) VALUES %s",
            values,
            page_size=1000,
        )
        conn.commit()

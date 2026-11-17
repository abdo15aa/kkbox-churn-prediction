"""Stage 1 – Build kkbox_cohorts_staging.db.

This script creates a SEPARATE staging database at data/kkbox_cohorts_staging.db.
It does NOT touch data/kkbox.db.

Design decisions encoded here
──────────────────────────────
1.  Two cohorts, two label tables, never merged.
2.  Transaction merge rule: "v2 wins on (msno, transaction_date)".
    The 361 187 v2 rows whose transaction_date <= 20170228 are treated as
    corrections to the historical file.  Every (msno, transaction_date) pair
    present anywhere in transactions_v2 causes ALL matching rows in
    transactions_v1 to be skipped.  This is explicitly documented as a
    provisional policy – see the validation report.
3.  Listening logs: union-all, zero date overlap confirmed in audit
    (v1 ends 20170228, v2 starts 20170301).
4.  Point-in-time enforcement:  ALL feature queries MUST filter by
    transaction_date <= cutoff  and  date <= cutoff.
    The staging DB stores the raw rows; per-cohort filtering happens at
    feature-engineering time.  source_file column is added so we can
    always trace which file a row came from.
5.  Correction-set provenance: rows in transactions_v2 that superseded v1
    rows (transaction_date <= 20170228) are tagged source_file='v2_correction'
    so they are always distinguishable from genuinely new March rows.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("stage1")

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
ROOT        = SCRIPT_DIR.parent
RAW         = ROOT / "data" / "raw"
DATA        = ROOT / "data"
STAGING_DB  = DATA / "kkbox_cohorts_staging.db"
ORIGINAL_DB = DATA / "kkbox.db"          # must never be touched
REPORT_DIR  = ROOT / "reports"

# Source CSV files
TRAIN_V1_CSV   = RAW / "train.csv"
TRAIN_V2_CSV   = RAW / "train_v2.csv"
MEMBERS_CSV    = RAW / "members_v3.csv"
TX_V1_CSV      = RAW / "transactions.csv"
TX_V2_CSV      = RAW / "transactions_v2.csv"
LOGS_V1_CSV    = RAW / "user_logs.csv"
LOGS_V2_CSV    = RAW / "user_logs_v2.csv"

# Sampling parameters – must match future feature-engineering scripts
SEED        = 42
SAMPLE_SIZE = 50_000
CHUNK       = 200_000          # rows per read-csv iteration

# Cutoff dates as integers (YYYYMMDD) matching the column format
CUTOFF_A = 20170228   # Cohort A: training
CUTOFF_B = 20170331   # Cohort B: evaluation


# ── Helpers ───────────────────────────────────────────────────────────────────

def configure_sqlite(conn: sqlite3.Connection) -> None:
    """Performance pragmas for bulk loading (identical to existing 01_load_sql.py)."""
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous  = NORMAL;")
    conn.execute("PRAGMA temp_store   = MEMORY;")
    conn.execute("PRAGMA cache_size   = -64000;")   # 64 MB


def sha256_first_mb(path: Path) -> str:
    """Lightweight content fingerprint: SHA-256 of the first 1 MB."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(1024 * 1024))
    return h.hexdigest()[:16]


def count_lines(path: Path) -> int:
    """Count data rows (excluding header) by streaming the file as text."""
    n = 0
    with open(path, encoding="utf-8") as f:
        next(f)        # skip header
        for _ in f:
            n += 1
    return n


def two_pass_sample(csv_path: Path, size: int, seed: int, chunk_size: int) -> pd.DataFrame:
    """
    Reproduce the exact sampling logic from 01_load_sql.py.

    Pass 1 – count total rows (text-stream, no CSV parsing).
    Pass 2 – read CSV in chunks, keep only the sampled row indices.

    Why two passes?  We need to know N before we can draw a uniform sample
    without replacement using np.random.default_rng.  A single-pass reservoir
    sample would work too but this approach matches the existing codebase.
    """
    log.info("Pass 1 – counting rows in %s …", csv_path.name)
    total = count_lines(csv_path)
    log.info("  %d data rows found.", total)
    if total < size:
        raise ValueError(f"{csv_path.name}: only {total} rows, need {size}")

    rng = np.random.default_rng(seed)
    chosen = set(int(i) for i in rng.choice(total, size=size, replace=False))

    log.info("Pass 2 – extracting %d sampled rows from %s …", size, csv_path.name)
    parts: list[pd.DataFrame] = []
    cursor = 0
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
        length = len(chunk)
        # Build a boolean mask for rows whose global index is in `chosen`
        idx = np.arange(cursor, cursor + length)
        mask = np.isin(idx, list(chosen))   # vectorised – faster than list comp
        if mask.any():
            parts.append(chunk[mask])
        cursor += length

    df = pd.concat(parts, ignore_index=True)
    log.info("  Sampled %d rows (%d unique msno).", len(df), df["msno"].nunique())
    return df


# ── Step 1 – Verify original DB is untouched ─────────────────────────────────

def verify_original_db() -> dict:
    info = {}
    if ORIGINAL_DB.exists():
        stat = ORIGINAL_DB.stat()
        info["size_bytes"]     = stat.st_size
        info["mtime"]          = stat.st_mtime
        info["mtime_readable"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime))
        info["sha256_1mb"]     = sha256_first_mb(ORIGINAL_DB)
        log.info("Original DB fingerprint: size=%d, mtime=%s, sha256_prefix=%s",
                 info["size_bytes"], info["mtime_readable"], info["sha256_1mb"])
    else:
        info["exists"] = False
        log.warning("Original DB not found at %s", ORIGINAL_DB)
    return info


# ── Step 2 – Sample cohort labels ─────────────────────────────────────────────

def build_cohort_labels(conn: sqlite3.Connection) -> tuple[set[str], set[str]]:
    """Sample 50k users from each train file and persist label tables."""

    for tbl in ("cohort_a_labels", "cohort_b_labels"):
        conn.execute(f"DROP TABLE IF EXISTS {tbl};")

    # Cohort A
    df_a = two_pass_sample(TRAIN_V1_CSV, SAMPLE_SIZE, SEED, CHUNK)
    df_a.to_sql("cohort_a_labels", conn, if_exists="replace", index=False)
    msno_a = set(df_a["msno"])
    log.info("Cohort A: %d labels, churn rate %.2f%%",
             len(df_a), df_a["is_churn"].mean() * 100)

    # Cohort B
    df_b = two_pass_sample(TRAIN_V2_CSV, SAMPLE_SIZE, SEED, CHUNK)
    df_b.to_sql("cohort_b_labels", conn, if_exists="replace", index=False)
    msno_b = set(df_b["msno"])
    log.info("Cohort B: %d labels, churn rate %.2f%%",
             len(df_b), df_b["is_churn"].mean() * 100)

    overlap = msno_a & msno_b
    log.info("User overlap between cohorts: %d users", len(overlap))

    return msno_a, msno_b


# ── Step 3 – Members ──────────────────────────────────────────────────────────

def load_members(conn: sqlite3.Connection, all_msnos: set[str]) -> int:
    """Stream members_v3.csv and keep rows for our ~97k unique users."""
    conn.execute("DROP TABLE IF EXISTS members;")
    total = 0
    for chunk in pd.read_csv(MEMBERS_CSV, chunksize=CHUNK, low_memory=False):
        sub = chunk[chunk["msno"].isin(all_msnos)]
        if not sub.empty:
            sub.to_sql("members", conn, if_exists="append", index=False)
            total += len(sub)
    log.info("members: %d rows inserted.", total)
    return total


# ── Step 4 – Transactions ─────────────────────────────────────────────────────

def build_v2_correction_key_set() -> set[str]:
    """
    Single pass through transactions_v2.csv to collect
    (msno, transaction_date) pairs whose date <= CUTOFF_A.
    These are the correction keys – any matching row in v1 will be skipped.

    Returns a set of "msno|transaction_date" strings rather than tuples so
    that load_transactions() can use a fast vectorized pandas .isin() check
    instead of a slow row-by-row Python apply().

    Why single pass here?  transactions_v2 is only ~110 MB; we can afford it.
    """
    log.info("Building v2 correction key set (tx_date <= %d) …", CUTOFF_A)
    parts: list[pd.Series] = []
    for chunk in pd.read_csv(TX_V2_CSV, chunksize=CHUNK,
                             usecols=["msno", "transaction_date"]):
        hist = chunk[chunk["transaction_date"] <= CUTOFF_A]
        if not hist.empty:
            # Build "msno|date" string key — fast vectorized string concatenation
            key_series = hist["msno"] + "|" + hist["transaction_date"].astype(str)
            parts.append(key_series)
    correction_str_set: set[str] = set(pd.concat(parts)) if parts else set()
    log.info("  %d unique (msno, transaction_date) correction keys found.", len(correction_str_set))
    return correction_str_set


def load_transactions(conn: sqlite3.Connection, all_msnos: set[str]) -> dict:
    """
    Merge transactions.csv + transactions_v2.csv applying the v2-wins rule.

    Schema added:  source_file  TEXT  ('v1', 'v2_correction', 'v2_new')

    Provisional policy (documented):
      For any (msno, transaction_date) present in transactions_v2 with
      transaction_date <= 20170228, ALL matching rows in transactions.csv
      are skipped.  All rows from transactions_v2 are kept as-is.

    Uncertainty:  We cannot confirm that each v2 correction was genuinely
    available at the Feb-28 cutoff.  The source_file tag lets feature
    engineering queries exclude v2_correction rows for Cohort A if needed.
    """
    conn.execute("DROP TABLE IF EXISTS transactions;")
    # correction_str_set is a set of "msno|transaction_date" strings.
    # Using strings (not tuples) lets us check membership with a fast
    # vectorized pandas .isin() call rather than a slow apply(axis=1) loop.
    correction_str_set = build_v2_correction_key_set()
    stats = {"v1_read": 0, "v1_skipped_correction": 0, "v1_inserted": 0,
             "v2_inserted": 0, "v2_correction_inserted": 0, "v2_new_inserted": 0}

    # ── Pass A: stream transactions.csv, skip correction-keyed rows ──────────
    log.info("Streaming transactions.csv (21.5 M rows) – this takes a few minutes …")
    for chunk in pd.read_csv(TX_V1_CSV, chunksize=CHUNK, low_memory=False):
        stats["v1_read"] += len(chunk)
        # Filter to our users first (cheap hash-set lookup)
        sub = chunk[chunk["msno"].isin(all_msnos)].copy()
        if sub.empty:
            continue
        # Build "msno|date" key column for vectorized membership check.
        # This is equivalent to the old apply(lambda r: (r.msno, r.tx_date) in set)
        # but runs at C speed instead of Python-loop speed — 50-100x faster.
        key_col = sub["msno"] + "|" + sub["transaction_date"].astype(str)
        mask_skip = key_col.isin(correction_str_set)
        skipped  = sub[mask_skip]
        kept     = sub[~mask_skip]
        stats["v1_skipped_correction"] += len(skipped)
        if not kept.empty:
            kept = kept.copy()
            kept["source_file"] = "v1"
            kept.to_sql("transactions", conn, if_exists="append", index=False)
            stats["v1_inserted"] += len(kept)

    log.info("  v1: %d read, %d kept, %d skipped (correction keys).",
             stats["v1_read"], stats["v1_inserted"], stats["v1_skipped_correction"])

    # ── Pass B: stream transactions_v2.csv, keep all rows for our users ──────
    log.info("Streaming transactions_v2.csv …")
    for chunk in pd.read_csv(TX_V2_CSV, chunksize=CHUNK, low_memory=False):
        sub = chunk[chunk["msno"].isin(all_msnos)].copy()
        if sub.empty:
            continue
        # Tag each row with its provenance
        is_correction = sub["transaction_date"] <= CUTOFF_A
        sub["source_file"] = "v2_new"
        sub.loc[is_correction, "source_file"] = "v2_correction"
        stats["v2_correction_inserted"] += int(is_correction.sum())
        stats["v2_new_inserted"]        += int((~is_correction).sum())
        stats["v2_inserted"]            += len(sub)
        sub.to_sql("transactions", conn, if_exists="append", index=False)

    log.info("  v2: %d rows inserted (%d corrections + %d new March rows).",
             stats["v2_inserted"], stats["v2_correction_inserted"], stats["v2_new_inserted"])
    return stats


# ── Step 5 – Listening logs ───────────────────────────────────────────────────

def load_user_logs(conn: sqlite3.Connection, all_msnos: set[str]) -> dict:
    """
    Combine user_logs.csv (2015-01-01 to 2017-02-28) and
    user_logs_v2.csv (2017-03-01 to 2017-03-31).

    No deduplication needed: date ranges are strictly disjoint (audited).
    source_file column added for traceability.

    user_logs.csv is ~28 GB; we read it with usecols to cut memory by ~30%.
    """
    conn.execute("DROP TABLE IF EXISTS user_logs;")
    cols = ["msno", "date", "num_25", "num_50", "num_75",
            "num_985", "num_100", "num_unq", "total_secs"]
    stats = {"v1_read": 0, "v1_inserted": 0, "v2_read": 0, "v2_inserted": 0,
             "min_date_v1": 99999999, "max_date_v1": 0,
             "min_date_v2": 99999999, "max_date_v2": 0}

    log.info("Streaming user_logs.csv (~28 GB) – the longest step …")
    for chunk in pd.read_csv(LOGS_V1_CSV, chunksize=CHUNK,
                             usecols=cols, low_memory=False):
        stats["v1_read"] += len(chunk)
        sub = chunk[chunk["msno"].isin(all_msnos)].copy()
        if not sub.empty:
            stats["min_date_v1"] = min(stats["min_date_v1"], int(sub["date"].min()))
            stats["max_date_v1"] = max(stats["max_date_v1"], int(sub["date"].max()))
            sub["source_file"] = "v1"
            sub.to_sql("user_logs", conn, if_exists="append", index=False)
            stats["v1_inserted"] += len(sub)

    log.info("  user_logs.csv: %d rows read, %d inserted.", stats["v1_read"], stats["v1_inserted"])

    log.info("Streaming user_logs_v2.csv …")
    for chunk in pd.read_csv(LOGS_V2_CSV, chunksize=CHUNK,
                             usecols=cols, low_memory=False):
        stats["v2_read"] += len(chunk)
        sub = chunk[chunk["msno"].isin(all_msnos)].copy()
        if not sub.empty:
            stats["min_date_v2"] = min(stats["min_date_v2"], int(sub["date"].min()))
            stats["max_date_v2"] = max(stats["max_date_v2"], int(sub["date"].max()))
            sub["source_file"] = "v2"
            sub.to_sql("user_logs", conn, if_exists="append", index=False)
            stats["v2_inserted"] += len(sub)

    log.info("  user_logs_v2.csv: %d rows read, %d inserted.", stats["v2_read"], stats["v2_inserted"])
    return stats


# ── Step 6 – Create indexes ───────────────────────────────────────────────────

def create_indexes(conn: sqlite3.Connection) -> None:
    """Create msno indexes AFTER all inserts (standard bulk-load best practice)."""
    log.info("Creating indexes …")
    index_specs = [
        ("idx_ca_labels_msno",  "cohort_a_labels",  "msno"),
        ("idx_cb_labels_msno",  "cohort_b_labels",  "msno"),
        ("idx_members_msno",    "members",           "msno"),
        ("idx_tx_msno",         "transactions",      "msno"),
        ("idx_tx_date",         "transactions",      "transaction_date"),
        ("idx_logs_msno",       "user_logs",         "msno"),
        ("idx_logs_date",       "user_logs",         "date"),
    ]
    for idx_name, table, col in index_specs:
        t0 = time.time()
        conn.execute(f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table}({col});")
        log.info("  %s created in %.1f s", idx_name, time.time() - t0)


# ── Step 7 – Validation queries ───────────────────────────────────────────────

def run_validation(conn: sqlite3.Connection,
                   msno_a: set[str],
                   msno_b: set[str],
                   tx_stats: dict,
                   log_stats: dict) -> dict:
    """Run all validation checks and return a structured results dict."""
    cur = conn.cursor()

    def q(sql: str):
        return cur.execute(sql).fetchone()[0]

    def qall(sql: str):
        return cur.execute(sql).fetchall()

    results = {}

    # ── Labels ────────────────────────────────────────────────────────────────
    results["cohort_a"] = {
        "rows":          q("SELECT COUNT(*)          FROM cohort_a_labels"),
        "distinct_msno": q("SELECT COUNT(DISTINCT msno) FROM cohort_a_labels"),
        "churners":      q("SELECT SUM(is_churn)     FROM cohort_a_labels"),
        "churn_rate_pct":round(q("SELECT AVG(is_churn)*100 FROM cohort_a_labels"), 4),
        "null_msno":     q("SELECT COUNT(*) FROM cohort_a_labels WHERE msno IS NULL"),
        "null_is_churn": q("SELECT COUNT(*) FROM cohort_a_labels WHERE is_churn IS NULL"),
    }
    results["cohort_b"] = {
        "rows":          q("SELECT COUNT(*)          FROM cohort_b_labels"),
        "distinct_msno": q("SELECT COUNT(DISTINCT msno) FROM cohort_b_labels"),
        "churners":      q("SELECT SUM(is_churn)     FROM cohort_b_labels"),
        "churn_rate_pct":round(q("SELECT AVG(is_churn)*100 FROM cohort_b_labels"), 4),
        "null_msno":     q("SELECT COUNT(*) FROM cohort_b_labels WHERE msno IS NULL"),
        "null_is_churn": q("SELECT COUNT(*) FROM cohort_b_labels WHERE is_churn IS NULL"),
    }
    results["cohort_overlap"] = {
        "shared_users":  len(msno_a & msno_b),
        "only_in_a":     len(msno_a - msno_b),
        "only_in_b":     len(msno_b - msno_a),
    }

    # ── Members ───────────────────────────────────────────────────────────────
    results["members"] = {
        "rows":          q("SELECT COUNT(*) FROM members"),
        "distinct_msno": q("SELECT COUNT(DISTINCT msno) FROM members"),
    }

    # ── Transactions ──────────────────────────────────────────────────────────
    results["transactions"] = {
        "total_rows":    q("SELECT COUNT(*) FROM transactions"),
        "distinct_msno": q("SELECT COUNT(DISTINCT msno) FROM transactions"),
        "v1_rows":       q("SELECT COUNT(*) FROM transactions WHERE source_file='v1'"),
        "v2_correction_rows": q("SELECT COUNT(*) FROM transactions WHERE source_file='v2_correction'"),
        "v2_new_rows":   q("SELECT COUNT(*) FROM transactions WHERE source_file='v2_new'"),
        # Duplicate (msno, transaction_date) keys — expected and preserved
        "dup_keys_total": q("""
            SELECT COUNT(*) - COUNT(DISTINCT msno || '|' || transaction_date)
            FROM transactions
        """),
        "min_tx_date_v1":   q("SELECT MIN(transaction_date) FROM transactions WHERE source_file='v1'"),
        "max_tx_date_v1":   q("SELECT MAX(transaction_date) FROM transactions WHERE source_file='v1'"),
        "min_tx_date_v2c":  q("SELECT MIN(transaction_date) FROM transactions WHERE source_file='v2_correction'"),
        "max_tx_date_v2c":  q("SELECT MAX(transaction_date) FROM transactions WHERE source_file='v2_correction'"),
        "min_tx_date_v2n":  q("SELECT MIN(transaction_date) FROM transactions WHERE source_file='v2_new'"),
        "max_tx_date_v2n":  q("SELECT MAX(transaction_date) FROM transactions WHERE source_file='v2_new'"),
        # Suspicious expiration dates (epoch sentinel)
        "epoch_expire_rows": q("SELECT COUNT(*) FROM transactions WHERE membership_expire_date < 19900101"),
        # v1 pipeline stats
        "v1_read_total":     tx_stats.get("v1_read", "n/a"),
        "v1_skipped_corrections": tx_stats.get("v1_skipped_correction", "n/a"),
    }

    # ── Listening logs ─────────────────────────────────────────────────────────
    results["user_logs"] = {
        "total_rows":    q("SELECT COUNT(*) FROM user_logs"),
        "distinct_msno": q("SELECT COUNT(DISTINCT msno) FROM user_logs"),
        "v1_rows":       q("SELECT COUNT(*) FROM user_logs WHERE source_file='v1'"),
        "v2_rows":       q("SELECT COUNT(*) FROM user_logs WHERE source_file='v2'"),
        "min_date_v1":   q("SELECT MIN(date) FROM user_logs WHERE source_file='v1'"),
        "max_date_v1":   q("SELECT MAX(date) FROM user_logs WHERE source_file='v1'"),
        "min_date_v2":   q("SELECT MIN(date) FROM user_logs WHERE source_file='v2'"),
        "max_date_v2":   q("SELECT MAX(date) FROM user_logs WHERE source_file='v2'"),
        # Check for any overlap (should be 0)
        "date_overlap_rows": q("""
            SELECT COUNT(*) FROM user_logs
            WHERE source_file='v1' AND date >= 20170301
        """),
        "missing_date":  q("SELECT COUNT(*) FROM user_logs WHERE date IS NULL"),
    }

    # ── Coverage: users with no transactions or logs ───────────────────────────
    results["coverage"] = {
        "a_users_with_no_tx": q("""
            SELECT COUNT(*) FROM cohort_a_labels l
            LEFT JOIN transactions t ON l.msno=t.msno
            WHERE t.msno IS NULL
        """),
        "b_users_with_no_tx": q("""
            SELECT COUNT(*) FROM cohort_b_labels l
            LEFT JOIN transactions t ON l.msno=t.msno
            WHERE t.msno IS NULL
        """),
        "a_users_with_no_logs": q("""
            SELECT COUNT(*) FROM cohort_a_labels l
            LEFT JOIN user_logs ul ON l.msno=ul.msno
            WHERE ul.msno IS NULL
        """),
        "b_users_with_no_logs": q("""
            SELECT COUNT(*) FROM cohort_b_labels l
            LEFT JOIN user_logs ul ON l.msno=ul.msno
            WHERE ul.msno IS NULL
        """),
    }

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    total_start = time.time()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Guard: never touch the original DB ────────────────────────────────────
    if STAGING_DB == ORIGINAL_DB:
        raise RuntimeError("STAGING_DB path equals ORIGINAL_DB – aborting.")

    log.info("=== Stage 1: Build kkbox_cohorts_staging.db ===")
    log.info("Staging path : %s", STAGING_DB)
    log.info("Original DB  : %s (will NOT be touched)", ORIGINAL_DB)

    # Capture original DB fingerprint
    original_fingerprint_before = verify_original_db()

    # Remove stale staging DB if it exists (safe: not the original)
    if STAGING_DB.exists():
        log.info("Removing stale staging DB …")
        STAGING_DB.unlink()

    conn = sqlite3.connect(STAGING_DB)
    configure_sqlite(conn)

    try:
        # Step 2: Labels
        log.info("── Step 2: Sampling cohort labels ──")
        msno_a, msno_b = build_cohort_labels(conn)
        all_msnos = msno_a | msno_b
        log.info("Combined user universe: %d unique msno values.", len(all_msnos))

        # Step 3: Members
        log.info("── Step 3: Loading members ──")
        load_members(conn, all_msnos)

        # Step 4: Transactions
        log.info("── Step 4: Loading transactions (long step) ──")
        tx_stats = load_transactions(conn, all_msnos)

        # Step 5: Listening logs
        log.info("── Step 5: Loading listening logs (longest step) ──")
        log_stats = load_user_logs(conn, all_msnos)

        # Step 6: Indexes
        log.info("── Step 6: Creating indexes ──")
        create_indexes(conn)

        # Step 7: Validation
        log.info("── Step 7: Running validation queries ──")
        results = run_validation(conn, msno_a, msno_b, tx_stats, log_stats)

        # Verify DB integrity
        integrity = conn.execute("PRAGMA integrity_check;").fetchone()[0]
        results["sqlite_integrity"] = integrity
        log.info("SQLite integrity_check: %s", integrity)

        conn.commit()

    finally:
        conn.close()

    # ── Verify original DB unchanged ──────────────────────────────────────────
    original_fingerprint_after = verify_original_db()
    original_unchanged = (
        original_fingerprint_before.get("size_bytes") ==
        original_fingerprint_after.get("size_bytes") and
        original_fingerprint_before.get("sha256_1mb") ==
        original_fingerprint_after.get("sha256_1mb")
    )
    results["original_db_unchanged"] = original_unchanged
    log.info("Original DB unchanged: %s", original_unchanged)

    # ── Staging DB size ───────────────────────────────────────────────────────
    staging_size = STAGING_DB.stat().st_size
    results["staging_db_size_bytes"] = staging_size
    log.info("Staging DB size: %d bytes (%.1f MB)", staging_size, staging_size / 1e6)

    # ── Persist results JSON ──────────────────────────────────────────────────
    json_path = REPORT_DIR / "stage1_validation_raw.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("Raw validation results saved to %s", json_path)

    elapsed = time.time() - total_start
    log.info("=== Stage 1 complete in %.1f s (%.1f min) ===", elapsed, elapsed / 60)
    print("\n=== VALIDATION SUMMARY ===")
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()


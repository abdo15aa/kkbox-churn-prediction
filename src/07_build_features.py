"""Stage 2 – Feature Engineering for KKBox Churn Prediction.

Reads from data/kkbox_cohorts_staging.db (built by 06_build_staging_db.py).
Writes features to data/features/cohort_a_features.csv and cohort_b_features.csv.
Does NOT touch data/kkbox.db or any raw CSV file.

═══════════════════════════════════════════════════════════════════════════════
WHAT IS FEATURE ENGINEERING?
═══════════════════════════════════════════════════════════════════════════════
Raw data (transactions, listening logs, membership info) is stored one row per
event. A machine-learning model needs ONE row per user with NUMBER columns.
Feature engineering transforms those many event rows into that one summary row.

Example:
  Raw data          → user A has 47 transaction rows, 312 listening log rows
  After engineering → user A has one row: {tx_count: 47, listening_days: 89, …}

═══════════════════════════════════════════════════════════════════════════════
PREDICTION CUTOFFS AND LEAKAGE PREVENTION
═══════════════════════════════════════════════════════════════════════════════
Cohort A cutoff: 2017-02-28  (as integer: 20170228)
Cohort B cutoff: 2017-03-31  (as integer: 20170331)

"Leakage" means accidentally using information that would NOT have been known
at prediction time. For example:
  ✗ Using a March transaction for Cohort A (cutoff was February 28)
  ✗ Using the churn label as a feature
  ✗ Using future expiration dates that were not yet set at the cutoff

Every SQL query in this script filters: transaction_date <= CUTOFF
                                         date            <= CUTOFF

═══════════════════════════════════════════════════════════════════════════════
DOCUMENTED LIMITATIONS (known uncertainties, NOT hidden)
═══════════════════════════════════════════════════════════════════════════════
1. v2_correction rows: rows tagged source_file='v2_correction' in the
   transactions table were present in transactions_v2.csv with dates before
   2017-02-28. They are treated as corrections to historical records, but we
   cannot prove they were available before the Feb-28 cutoff. For Cohort A
   features, we EXCLUDE these rows and document this as a conservative choice.
   For Cohort B, all transactions <= 20170331 are used (v2_correction rows ARE
   legitimately available since the v2 file was released after Feb-28).

2. membership_expire_date: used to calculate days_until_expiry. If a v2
   correction changed the expiry date, the value in our DB may differ from
   what was observable at Feb-28. This is flagged per the limitation above.

3. Missing members: ~13% of users have no row in the members table. We treat
   them as missing (NaN), not as zero or dropped users.
"""

from __future__ import annotations

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
log = logging.getLogger("stage2")

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
ROOT        = SCRIPT_DIR.parent
DATA        = ROOT / "data"
STAGING_DB  = DATA / "kkbox_cohorts_staging.db"
FEATURES_DIR = DATA / "features"
REPORT_DIR  = ROOT / "reports"

# ── Cutoff dates (YYYYMMDD integers — same format used in the DB) ─────────────
CUTOFF_A = 20170228   # Cohort A: February 28, 2017
CUTOFF_B = 20170331   # Cohort B: March 31, 2017

# As Python date integers for arithmetic (days since epoch not needed;
# we'll convert to pandas Timestamps only where needed for subtraction).
CUTOFF_A_DATE = pd.Timestamp("2017-02-28")
CUTOFF_B_DATE = pd.Timestamp("2017-03-31")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 – Helpers
# ─────────────────────────────────────────────────────────────────────────────

def open_staging_db_readonly() -> sqlite3.Connection:
    """Open the staging database in read-only mode.

    The URI ?mode=ro prevents any accidental writes.
    We still set WAL and cache pragmas for read performance.
    """
    if not STAGING_DB.exists():
        raise FileNotFoundError(
            f"Staging database not found at {STAGING_DB}.\n"
            "Run 06_build_staging_db.py first (Stage 1)."
        )
    uri = f"file:{STAGING_DB}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA cache_size   = -64000;")   # 64 MB read cache
    conn.row_factory = sqlite3.Row   # access columns by name, not position
    return conn


def verify_staging_db(conn: sqlite3.Connection) -> None:
    """Assert that all required tables exist and have expected row counts.

    Stops the script with a clear error message if Stage 1 is incomplete.
    This is a 'pre-flight check' — always run before expensive queries.
    """
    log.info("Verifying staging database tables …")
    cursor = conn.cursor()

    required = {
        "cohort_a_labels": 50_000,
        "cohort_b_labels": 50_000,
        "members":         1,       # at least 1 row
        "transactions":    1,
        "user_logs":       1,
    }

    for table, min_rows in required.items():
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
        )
        if cursor.fetchone() is None:
            raise RuntimeError(
                f"Table '{table}' is missing from the staging database.\n"
                "Stage 1 may not have completed successfully."
            )
        count = cursor.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        if count < min_rows:
            raise RuntimeError(
                f"Table '{table}' has only {count} rows (expected >= {min_rows}).\n"
                "Stage 1 may have completed with errors."
            )
        log.info("  %-20s  %d rows  ✓", table, count)

    # Verify source_file column exists in transactions (needed for leakage guard)
    cols = [r[1] for r in cursor.execute("PRAGMA table_info(transactions)").fetchall()]
    if "source_file" not in cols:
        raise RuntimeError(
            "Column 'source_file' missing from transactions table.\n"
            "Re-run 06_build_staging_db.py to rebuild the staging database."
        )
    log.info("  source_file column present in transactions  ✓")

    # Verify source_file column exists in user_logs
    cols = [r[1] for r in cursor.execute("PRAGMA table_info(user_logs)").fetchall()]
    if "source_file" not in cols:
        raise RuntimeError(
            "Column 'source_file' missing from user_logs table.\n"
            "Re-run 06_build_staging_db.py to rebuild the staging database."
        )
    log.info("  source_file column present in user_logs  ✓")


def inspect_schema(conn: sqlite3.Connection) -> dict[str, list[str]]:
    """Return a dict of {table_name: [column_names]} for all tables.

    Logged at INFO level so we always know what we're working with.
    """
    cursor = conn.cursor()
    tables = [
        r[0] for r in
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    ]
    schema: dict[str, list[str]] = {}
    for tbl in tables:
        cols = [r[1] for r in cursor.execute(f"PRAGMA table_info({tbl})").fetchall()]
        schema[tbl] = cols
        log.info("  %s: %s", tbl, cols)
    return schema


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 – Label loading
# ─────────────────────────────────────────────────────────────────────────────

def load_labels(conn: sqlite3.Connection, cohort: str) -> pd.DataFrame:
    """Load cohort labels (msno, is_churn) from the staging DB.

    Args:
        conn:   Read-only connection to staging DB.
        cohort: 'A' or 'B'.

    Returns:
        DataFrame with columns [msno, is_churn], one row per user.
    """
    table = f"cohort_{cohort.lower()}_labels"
    df = pd.read_sql(f"SELECT msno, is_churn FROM {table}", conn)
    assert df["msno"].nunique() == len(df), f"Duplicate msno in {table}!"
    log.info("Cohort %s: %d users loaded from %s", cohort, len(df), table)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 – Transaction features
# ─────────────────────────────────────────────────────────────────────────────

# WHY SQL FOR TRANSACTIONS INSTEAD OF PANDAS?
# ─────────────────────────────────────────────
# The transactions table may contain hundreds of thousands of rows for our
# 97k users. SQLite can aggregate them in a single GROUP BY pass, which is far
# more memory-efficient than loading all rows into pandas first.
# We use pandas only for the final per-user result (50k rows × ~15 columns).

def build_transaction_features(
    conn: sqlite3.Connection,
    cohort_table: str,
    cutoff_int: int,
    cutoff_date: pd.Timestamp,
    exclude_v2_correction: bool,
) -> pd.DataFrame:
    """Aggregate transaction-level rows into per-user features.

    Args:
        conn:                  Read-only staging DB connection.
        cohort_table:          Name of label table (e.g. 'cohort_a_labels').
        cutoff_int:            Cutoff date as YYYYMMDD integer.
        cutoff_date:           Cutoff date as Timestamp (for day arithmetic).
        exclude_v2_correction: If True, rows tagged source_file='v2_correction'
                               are excluded. Set True for Cohort A (leakage guard).
                               Set False for Cohort B (corrections are legitimate).

    Returns:
        DataFrame indexed by msno with transaction feature columns.

    Leakage prevention:
        WHERE transaction_date <= cutoff_int  ensures no future transactions.
        exclude_v2_correction removes rows whose point-in-time availability
        cannot be confirmed for Cohort A.

    Feature definitions:
        transaction_count          COUNT of eligible rows per user
        days_since_last_tx         cutoff - MAX(transaction_date)
        avg_plan_list_price        AVG(plan_list_price)
        avg_actual_amount_paid     AVG(actual_amount_paid)
        avg_discount               AVG(plan_list_price - actual_amount_paid)
                                   Note: can be negative (overpayment possible)
        auto_renew_rate            SUM(is_auto_renew) / COUNT(*)
        cancel_rate                SUM(is_cancel)     / COUNT(*)
        unique_payment_methods     COUNT(DISTINCT payment_method_id)
        unique_payment_plans       COUNT(DISTINCT payment_plan_days)
    """
    log.info(
        "Building transaction features (cutoff=%d, exclude_v2_correction=%s) …",
        cutoff_int, exclude_v2_correction
    )

    # Build the source_file exclusion clause
    # When exclude_v2_correction=True, rows tagged 'v2_correction' are skipped.
    # This is the leakage guard for Cohort A.
    source_filter = ""
    if exclude_v2_correction:
        source_filter = "AND source_file != 'v2_correction'"

    sql = f"""
    SELECT
        msno,
        COUNT(*)                                        AS transaction_count,
        MAX(transaction_date)                           AS last_tx_date_int,
        AVG(plan_list_price)                            AS avg_plan_list_price,
        AVG(actual_amount_paid)                         AS avg_actual_amount_paid,
        AVG(plan_list_price - actual_amount_paid)       AS avg_discount,
        CAST(SUM(is_auto_renew) AS REAL) / COUNT(*)     AS auto_renew_rate,
        CAST(SUM(is_cancel) AS REAL) / COUNT(*)         AS cancel_rate,
        COUNT(DISTINCT payment_method_id)               AS unique_payment_methods,
        COUNT(DISTINCT payment_plan_days)               AS unique_payment_plans
    FROM transactions
    WHERE transaction_date <= {cutoff_int}
      AND msno IN (SELECT msno FROM {cohort_table})
      {source_filter}
    GROUP BY msno
    """

    df = pd.read_sql(sql, conn)

    # Convert last transaction date (integer YYYYMMDD) to days before cutoff.
    df["last_tx_date"] = pd.to_datetime(df["last_tx_date_int"], format="%Y%m%d")
    df["days_since_last_tx"] = (cutoff_date - df["last_tx_date"]).dt.days
    df = df.drop(columns=["last_tx_date", "last_tx_date_int"])

    log.info("  Transaction features computed for %d users.", len(df))
    return df.set_index("msno")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 – Listening-log features
# ─────────────────────────────────────────────────────────────────────────────

def build_log_features(
    conn: sqlite3.Connection,
    cohort_table: str,
    cutoff_int: int,
    cutoff_date: pd.Timestamp,
) -> pd.DataFrame:
    """Aggregate listening-log rows into per-user features in a single SQL pass.

    Uses covering index idx_logs_msno and subquery on cohort_table for fast lookup.

    Leakage prevention:
        WHERE date <= cutoff_int  ensures no future listening activity is used.
        For Cohort A: date <= 20170228 (v2 log rows start 20170301 → auto-excluded)
        For Cohort B: date <= 20170331 (includes March logs)

    Feature definitions:
        total_listening_days    COUNT(DISTINCT date) with activity
        total_log_rows          Total rows
        total_num_25/50/75/985/100  Cumulative play-depth counts
        total_num_unq           Total unique songs played
        total_secs              Total seconds listened
        avg_secs_per_day        total_secs / total_listening_days
        days_since_last_listen  cutoff - MAX(date)
        active_days_last_7d     Distinct days with activity in last 7 days
        active_days_last_14d    Distinct days with activity in last 14 days
        active_days_last_30d    Distinct days with activity in last 30 days
        secs_last_7d            total_secs in last 7 days
        secs_last_30d           total_secs in last 30 days
        listen_trend            secs_last_7d / (secs_last_30d / 4)
                                = ratio of last-week activity vs weekly average
                                > 1 means increasing, < 1 means declining
    """
    log.info("Building listening-log features (cutoff=%d, table=%s) …", cutoff_int, cohort_table)

    # Precompute window boundaries (as YYYYMMDD integers)
    d7  = int((cutoff_date - pd.Timedelta(days=7)).strftime("%Y%m%d"))
    d14 = int((cutoff_date - pd.Timedelta(days=14)).strftime("%Y%m%d"))
    d30 = int((cutoff_date - pd.Timedelta(days=30)).strftime("%Y%m%d"))

    # Unified single-pass SQL query using covering index
    sql = f"""
    SELECT
        msno,
        COUNT(DISTINCT date)                                     AS total_listening_days,
        COUNT(*)                                                 AS total_log_rows,
        SUM(num_25)                                              AS total_num_25,
        SUM(num_50)                                              AS total_num_50,
        SUM(num_75)                                              AS total_num_75,
        SUM(num_985)                                             AS total_num_985,
        SUM(num_100)                                             AS total_num_100,
        SUM(num_unq)                                             AS total_num_unq,
        SUM(total_secs)                                          AS total_secs,
        MAX(date)                                                AS last_listen_date_int,
        COUNT(DISTINCT CASE WHEN date > {d7}  THEN date END)     AS active_days_last_7d,
        COUNT(DISTINCT CASE WHEN date > {d14} THEN date END)     AS active_days_last_14d,
        COUNT(DISTINCT CASE WHEN date > {d30} THEN date END)     AS active_days_last_30d,
        SUM(CASE WHEN date > {d7}  THEN total_secs ELSE 0 END)   AS secs_last_7d,
        SUM(CASE WHEN date > {d30} THEN total_secs ELSE 0 END)   AS secs_last_30d
    FROM user_logs
    WHERE date <= {cutoff_int}
      AND msno IN (SELECT msno FROM {cohort_table})
    GROUP BY msno
    """

    log.info("  Running unified log aggregation query …")
    df = pd.read_sql(sql, conn)

    # Convert last listen date to days-since
    df["last_listen_date"] = pd.to_datetime(df["last_listen_date_int"], format="%Y%m%d")
    df["days_since_last_listen"] = (cutoff_date - df["last_listen_date"]).dt.days
    df = df.drop(columns=["last_listen_date", "last_listen_date_int"])

    # Average seconds per listening day (avoid division by zero)
    df["avg_secs_per_day"] = np.where(
        df["total_listening_days"] > 0,
        df["total_secs"] / df["total_listening_days"],
        np.nan
    )

    # Listening trend: last-7d vs weekly average of last-30d
    df["listen_trend"] = np.where(
        df["secs_last_30d"] > 0,
        df["secs_last_7d"] / (df["secs_last_30d"] / 4.0),
        np.nan
    )

    log.info("  Log features computed for %d users.", len(df))
    return df.set_index("msno")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 – Membership (member profile) features
# ─────────────────────────────────────────────────────────────────────────────

# ABOUT REGISTRATION DATE:
# ─────────────────────────
# members.registration_init_time is stored as an integer YYYYMMDD.
# This date was set when the user first registered — it doesn't change.
# It was definitely known at the prediction cutoff.
# → days_since_registration is point-in-time SAFE.
#
# ABOUT GENDER:
# ──────────────
# ~65% of users have no gender in members_v3.csv (it was never collected).
# We encode: male=1, female=0, missing=NaN. Do not treat NaN as zero.

GENDER_MAP = {"male": 1, "female": 0}

def build_member_features(
    conn: sqlite3.Connection,
    cutoff_date: pd.Timestamp,
) -> pd.DataFrame:
    """Load and engineer features from the members table.

    Point-in-time safety:
        registration_init_time is a historical fact set at signup.
        city, bd (birth date/year), gender were self-reported at signup.
        All are known at any prediction cutoff. SAFE.

    Known issues:
        bd = 0 or bd < 1900 or bd > 2005 are implausible birth years.
        We treat implausible values as missing (NaN) rather than zero.
    """
    log.info("Building membership features (cutoff=%s) …", cutoff_date.date())

    sql = """
    SELECT
        msno,
        city,
        bd,
        gender,
        registered_via,
        registration_init_time
    FROM members
    """
    df = pd.read_sql(sql, conn)

    # ── Registration tenure ───────────────────────────────────────────────────
    # Convert YYYYMMDD integer to Timestamp, then compute days since registration
    df["reg_date"] = pd.to_datetime(
        df["registration_init_time"].astype("Int64"),  # handles NaN safely
        format="%Y%m%d",
        errors="coerce"   # invalid dates → NaT
    )
    df["days_since_registration"] = (cutoff_date - df["reg_date"]).dt.days
    # Negative tenure = registration after cutoff → implausible → NaN
    df.loc[df["days_since_registration"] < 0, "days_since_registration"] = np.nan
    df = df.drop(columns=["reg_date", "registration_init_time"])

    # ── Birth date / age ─────────────────────────────────────────────────────
    # Note on KKBox data semantics: 'bd' is self-reported age in years,
    # NOT birth year! Values of 0 represent unknown/unfilled profiles.
    # We keep reasonable ages (10 to 100) and map 0 and outliers to NaN.
    df["age"] = np.where(
        (df["bd"] >= 10) & (df["bd"] <= 100),
        df["bd"].astype(float),
        np.nan
    )
    df = df.drop(columns=["bd"])

    # ── Gender encoding ───────────────────────────────────────────────────────
    df["gender_encoded"] = df["gender"].map(GENDER_MAP)   # NaN for missing
    df = df.drop(columns=["gender"])

    log.info("  Member features computed for %d users.", len(df))
    return df.set_index("msno")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 – Cross-group (derived) features
# ─────────────────────────────────────────────────────────────────────────────

def build_cross_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features from combined transaction + log columns.

    These are computed AFTER joining all feature groups, so we can use
    columns from multiple tables.

    Features:
        tx_per_listening_month   transaction_count / (total_listening_days / 30)
                                 = how often the user buys per active month
        completion_ratio         total_num_100 / total_plays
                                 = fraction of songs listened to completion
                                 (uses num_25 as a proxy for "started playing")
    """
    log.info("Building cross-group derived features …")

    # Transactions per active listening month
    # total_listening_days / 30 ≈ number of active months
    listening_months = df["total_listening_days"] / 30.0
    df["tx_per_listening_month"] = np.where(
        listening_months > 0,
        df["transaction_count"] / listening_months,
        np.nan
    )

    # Completion ratio: finished songs / (all songs started at any depth)
    # num_25 represents songs where the user reached at least the 25% mark.
    # We treat (num_25 + num_50 + num_75 + num_985 + num_100) as total plays.
    # Note: this is an approximation — the columns are NOT mutually exclusive
    # in the raw data (each represents songs reaching THAT threshold that day).
    # A better denominator would be num_unq (unique songs), but sum-of-depths
    # is a reasonable proxy for engagement depth.
    total_plays = (
        df["total_num_25"].fillna(0)
        + df["total_num_50"].fillna(0)
        + df["total_num_75"].fillna(0)
        + df["total_num_985"].fillna(0)
        + df["total_num_100"].fillna(0)
    )
    df["completion_ratio"] = np.where(
        total_plays > 0,
        df["total_num_100"].fillna(0) / total_plays,
        np.nan
    )

    log.info("  Cross-group features added.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 – Assemble one cohort's full feature matrix
# ─────────────────────────────────────────────────────────────────────────────

def build_cohort_features(
    conn: sqlite3.Connection,
    cohort: str,
    cutoff_int: int,
    cutoff_date: pd.Timestamp,
    exclude_v2_correction: bool,
) -> pd.DataFrame:
    """Assemble the complete feature matrix for one cohort.

    Steps:
    1. Load labels (msno + is_churn)
    2. Build transaction features (SQL GROUP BY)
    3. Build listening-log features (SQL GROUP BY)
    4. Build membership features (members table)
    5. Left-join all feature groups onto the label frame
       (left join preserves ALL 50k users even if some have no transactions/logs)
    6. Add cross-group derived features
    7. Add a flag: is_missing_member (no row in members table)

    Returns:
        DataFrame with columns: msno, is_churn, [all features]
        Exactly 50,000 rows (one per sampled user).
    """
    log.info("═══ Building Cohort %s features (cutoff %s) ═══", cohort, cutoff_date.date())

    # Step 1 – Labels
    labels = load_labels(conn, cohort)
    cohort_table = f"cohort_{cohort.lower()}_labels"

    # Step 2 – Transaction features
    tx_feats = build_transaction_features(
        conn, cohort_table, cutoff_int, cutoff_date, exclude_v2_correction
    )

    # Step 3 – Listening log features
    log_feats = build_log_features(conn, cohort_table, cutoff_int, cutoff_date)

    # Step 4 – Membership features
    mem_feats = build_member_features(conn, cutoff_date)

    # Step 5 – Left-join everything onto the label frame
    # LEFT JOIN: every user from labels is preserved.
    # Users with no transactions get NaN in all tx columns.
    # Users with no listening logs get NaN in all log columns.
    # Users with no members row get NaN in all member columns.
    df = (
        labels
        .set_index("msno")
        .join(tx_feats,  how="left")
        .join(log_feats, how="left")
        .join(mem_feats, how="left")
        .reset_index()
    )

    # Step 6 – Cross-group derived features (needs both tx and log columns)
    df = build_cross_features(df)

    # Step 7 – Missing-member flag
    # 1 = user has no row in members table (information gap, not "no membership")
    df["is_missing_member"] = df["days_since_registration"].isna().astype(int)

    log.info(
        "Cohort %s: %d rows × %d columns assembled.",
        cohort, len(df), df.shape[1]
    )

    # Invariant check: must have exactly 50k rows, all unique msno
    assert len(df) == 50_000, f"Expected 50000 rows for Cohort {cohort}, got {len(df)}"
    assert df["msno"].nunique() == 50_000, "Duplicate msno found in feature matrix!"

    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 – Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_features(df: pd.DataFrame, cohort: str, cutoff_date: pd.Timestamp) -> dict:
    """Run automated validation checks on a feature matrix.

    Returns a dict of check results for inclusion in the report.
    Raises AssertionError for critical failures that must block the pipeline.
    """
    log.info("Validating Cohort %s feature matrix …", cohort)
    results = {}
    feature_cols = [c for c in df.columns if c not in ("msno", "is_churn")]

    # CHECK 1: Row and user counts
    results["row_count"]          = len(df)
    results["distinct_msno"]      = df["msno"].nunique()
    results["duplicate_msno"]     = len(df) - df["msno"].nunique()
    assert results["duplicate_msno"] == 0, "Duplicate msno in feature matrix!"

    # CHECK 2: Label integrity
    results["label_values"]       = sorted(df["is_churn"].unique().tolist())
    results["churn_count"]        = int(df["is_churn"].sum())
    results["churn_rate_pct"]     = round(df["is_churn"].mean() * 100, 4)
    assert set(df["is_churn"].unique()).issubset({0, 1}), "is_churn has non-binary values!"

    # CHECK 3: Missing values per feature
    missing = df[feature_cols].isna().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    results["missing_counts"]     = missing.to_dict()
    results["missing_pct"]        = missing_pct.to_dict()

    # CHECK 4: Infinite values
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    inf_counts = {c: int(np.isinf(df[c]).sum()) for c in numeric_cols}
    results["inf_counts"]         = {k: v for k, v in inf_counts.items() if v > 0}
    total_inf = sum(inf_counts.values())
    assert total_inf == 0, f"Infinite values found: {results['inf_counts']}"

    # CHECK 5: Negative impossible values
    # days_since_* must be >= 0 (we already set negatives to NaN above)
    for col in ["days_since_last_tx", "days_since_last_listen", "days_since_registration"]:
        if col in df.columns:
            neg = int((df[col] < 0).sum())
            results[f"negative_{col}"] = neg
            assert neg == 0, f"Negative values in {col}: {neg} rows"

    # CHECK 6: Constant or near-constant features (low variance → useless)
    low_var = {}
    for col in numeric_cols:
        nuniq = df[col].nunique()
        if nuniq <= 1:
            low_var[col] = nuniq
    results["constant_features"]  = low_var
    if low_var:
        log.warning("  Constant features (drop before modeling): %s", list(low_var.keys()))

    # CHECK 7: User-level distributions (key features)
    summary_cols = [
        "transaction_count", "days_since_last_tx",
        "total_listening_days", "days_since_last_listen",
        "days_since_registration",
    ]
    summary_cols = [c for c in summary_cols if c in df.columns]
    results["distributions"] = (
        df[summary_cols].describe().round(2).to_dict()
    )

    # CHECK 8: Users with no transactions (leakage check proxy)
    results["users_no_tx"]        = int(df["transaction_count"].isna().sum())
    results["users_no_logs"]      = int(df["total_listening_days"].isna().sum())
    results["users_missing_member"] = int(df["is_missing_member"].sum())

    # CHECK 9: Sample rows for sanity check (anonymized = just first 5)
    results["sample_rows"]        = df.head(5).to_dict(orient="records")

    log.info("  Cohort %s validation complete.", cohort)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 – Report writing
# ─────────────────────────────────────────────────────────────────────────────

def write_validation_report(
    results_a: dict,
    results_b: dict,
    overlap_check: dict,
    elapsed: float,
) -> Path:
    """Write reports/stage2_feature_validation.md."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "stage2_feature_validation.md"

    def fmt_dict(d: dict, indent: int = 0) -> str:
        pad = "  " * indent
        lines = []
        for k, v in d.items():
            if isinstance(v, dict):
                lines.append(f"{pad}- **{k}**:")
                lines.append(fmt_dict(v, indent + 1))
            else:
                lines.append(f"{pad}- `{k}`: {v}")
        return "\n".join(lines)

    def check_line(label: str, passed: bool, detail: str = "") -> str:
        icon = "✅ PASS" if passed else "❌ FAIL"
        return f"| {label} | {icon} | {detail} |"

    lines = [
        "# Stage 2 — Feature Engineering Validation Report",
        "",
        f"> Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"> Elapsed: {elapsed:.1f} s  ",
        f"> Cutoff A: 2017-02-28 | Cutoff B: 2017-03-31",
        "",
        "---",
        "",
        "## 1. Critical Invariants",
        "",
        "| Check | Result | Detail |",
        "|---|---|---|",
        check_line(
            "Cohort A — 50k rows",
            results_a["row_count"] == 50_000,
            str(results_a["row_count"])
        ),
        check_line(
            "Cohort A — no duplicate msno",
            results_a["duplicate_msno"] == 0,
            str(results_a["duplicate_msno"])
        ),
        check_line(
            "Cohort A — binary labels",
            results_a["label_values"] in ([0, 1], [0], [1]),
            str(results_a["label_values"])
        ),
        check_line(
            "Cohort B — 50k rows",
            results_b["row_count"] == 50_000,
            str(results_b["row_count"])
        ),
        check_line(
            "Cohort B — no duplicate msno",
            results_b["duplicate_msno"] == 0,
            str(results_b["duplicate_msno"])
        ),
        check_line(
            "Cohort B — binary labels",
            results_b["label_values"] in ([0, 1], [0], [1]),
            str(results_b["label_values"])
        ),
        check_line(
            "No infinite values — Cohort A",
            len(results_a.get("inf_counts", {})) == 0,
            str(results_a.get("inf_counts", {}))
        ),
        check_line(
            "No infinite values — Cohort B",
            len(results_b.get("inf_counts", {})) == 0,
            str(results_b.get("inf_counts", {}))
        ),
        check_line(
            "No negative days columns — Cohort A",
            all(
                results_a.get(f"negative_{c}", 0) == 0
                for c in ["days_since_last_tx", "days_since_last_listen", "days_since_registration"]
            ),
            "all zero"
        ),
        check_line(
            "No negative days columns — Cohort B",
            all(
                results_b.get(f"negative_{c}", 0) == 0
                for c in ["days_since_last_tx", "days_since_last_listen", "days_since_registration"]
            ),
            "all zero"
        ),
        "",
        "---",
        "",
        "## 2. Cohort A Summary (cutoff 2017-02-28)",
        "",
        f"- **Users**: {results_a['row_count']:,}",
        f"- **Churners**: {results_a['churn_count']:,} ({results_a['churn_rate_pct']}%)",
        f"- **Users with no transactions**: {results_a['users_no_tx']:,}",
        f"- **Users with no listening logs**: {results_a['users_no_logs']:,}",
        f"- **Users missing from members table**: {results_a['users_missing_member']:,}",
        "",
        "### Missing values (Cohort A)",
        "",
        "| Feature | Missing N | Missing % |",
        "|---|---|---|",
    ]
    for col in sorted(results_a["missing_counts"].keys()):
        n = results_a["missing_counts"][col]
        pct = results_a["missing_pct"][col]
        if n > 0:
            lines.append(f"| `{col}` | {n:,} | {pct}% |")

    lines += [
        "",
        "---",
        "",
        "## 3. Cohort B Summary (cutoff 2017-03-31)",
        "",
        f"- **Users**: {results_b['row_count']:,}",
        f"- **Churners**: {results_b['churn_count']:,} ({results_b['churn_rate_pct']}%)",
        f"- **Users with no transactions**: {results_b['users_no_tx']:,}",
        f"- **Users with no listening logs**: {results_b['users_no_logs']:,}",
        f"- **Users missing from members table**: {results_b['users_missing_member']:,}",
        "",
        "### Missing values (Cohort B)",
        "",
        "| Feature | Missing N | Missing % |",
        "|---|---|---|",
    ]
    for col in sorted(results_b["missing_counts"].keys()):
        n = results_b["missing_counts"][col]
        pct = results_b["missing_pct"][col]
        if n > 0:
            lines.append(f"| `{col}` | {n:,} | {pct}% |")

    lines += [
        "",
        "---",
        "",
        "## 4. Overlapping Users (appear in both cohorts)",
        "",
        f"- **Count**: {overlap_check['overlap_count']:,}",
        f"- **Features independently computed**: {overlap_check['features_independent']}",
        f"- **Label agreement (same outcome both months)**: {overlap_check['same_label']:,}",
        f"- **Label disagreement (different outcome)**: {overlap_check['diff_label']:,}",
        "",
        "Overlapping users have **independent** feature rows in each cohort — "
        "Cohort A uses only data up to Feb-28, Cohort B uses data up to Mar-31. "
        "Same user, different observation windows.",
        "",
        "---",
        "",
        "## 5. Known Limitations and Leakage Risks",
        "",
        "> [!WARNING]",
        "> The following are documented limitations, not confirmed errors.",
        "",
        "### 5.1 v2_correction transaction rows (Cohort A only)",
        "",
        "- For Cohort A, rows tagged `source_file='v2_correction'` are **excluded** from "
        "transaction features. These are rows from `transactions_v2.csv` with dates before "
        "2017-02-28 that superseded v1 records.",
        "- We cannot confirm these corrections were available before the Feb-28 cutoff.",
        "- **Impact**: Cohort A transaction features use only `source_file='v1'` rows. "
        "Some users may have fewer transactions in Cohort A than their actual history. "
        "This is the conservative, leakage-safe choice.",
        "- **Cohort B** includes `v2_correction` rows (they were released in the v2 file, "
        "which was available after Feb-28 and before Mar-31 cutoff).",
        "",
        "### 5.2 membership_expire_date (NOT used as a feature)",
        "",
        "- The expiration date from the most recent transaction is NOT computed as a feature "
        "because v2_correction rows may have changed it. Its point-in-time value at Feb-28 "
        "cannot be guaranteed for Cohort A.",
        "- `days_since_last_tx` is safe because it uses `transaction_date` (when the "
        "transaction was made), not the expiration date.",
        "",
        "### 5.3 Missing historical context",
        "",
        "- Some users registered before 2015 but have no listening logs before their cutoff. "
        "These are not errors — they may be low-activity or lapsed users.",
        "",
        "---",
        "",
        "## 6. Feature Descriptions",
        "",
        "| Feature | Source | Leakage Safe? | Notes |",
        "|---|---|---|---|",
        "| `transaction_count` | transactions | ✅ Yes | COUNT of rows ≤ cutoff |",
        "| `days_since_last_tx` | transactions | ✅ Yes | cutoff − MAX(transaction_date ≤ cutoff) |",
        "| `avg_plan_list_price` | transactions | ✅ Yes | AVG over eligible rows |",
        "| `avg_actual_amount_paid` | transactions | ✅ Yes | AVG over eligible rows |",
        "| `avg_discount` | transactions | ✅ Yes | AVG(list_price − actual_paid) |",
        "| `auto_renew_rate` | transactions | ✅ Yes | fraction of rows with auto_renew=1 |",
        "| `cancel_rate` | transactions | ✅ Yes | fraction of rows with is_cancel=1 |",
        "| `unique_payment_methods` | transactions | ✅ Yes | COUNT DISTINCT payment_method_id |",
        "| `unique_payment_plans` | transactions | ✅ Yes | COUNT DISTINCT payment_plan_days |",
        "| `total_listening_days` | user_logs | ✅ Yes | COUNT DISTINCT date ≤ cutoff |",
        "| `total_log_rows` | user_logs | ✅ Yes | raw row count |",
        "| `total_num_25/50/75/985/100` | user_logs | ✅ Yes | SUM of play-depth columns |",
        "| `total_num_unq` | user_logs | ✅ Yes | SUM of unique songs per day |",
        "| `total_secs` | user_logs | ✅ Yes | SUM of total_secs |",
        "| `avg_secs_per_day` | user_logs | ✅ Yes | derived from above |",
        "| `days_since_last_listen` | user_logs | ✅ Yes | cutoff − MAX(date ≤ cutoff) |",
        "| `active_days_last_7/14/30d` | user_logs | ✅ Yes | windowed count |",
        "| `secs_last_7d`, `secs_last_30d` | user_logs | ✅ Yes | windowed sum |",
        "| `listen_trend` | user_logs | ✅ Yes | secs_last_7d / (secs_last_30d/4) |",
        "| `days_since_registration` | members | ✅ Yes | cutoff − registration_init_time |",
        "| `age` | members | ✅ Yes | cutoff_year − bd (birth year) |",
        "| `gender_encoded` | members | ✅ Yes | 1=male, 0=female, NaN=missing |",
        "| `city` | members | ✅ Yes | categorical city code |",
        "| `registered_via` | members | ✅ Yes | registration channel code |",
        "| `is_missing_member` | members | ✅ Yes | 1 if no row in members table |",
        "| `tx_per_listening_month` | derived | ✅ Yes | tx_count / (listening_days/30) |",
        "| `completion_ratio` | derived | ✅ Yes | total_num_100 / total_plays |",
        "",
        "---",
        "",
        f"*Report generated in {elapsed:.1f} seconds.*",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Validation report written to %s", report_path)
    return report_path


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 – Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    total_start = time.time()
    log.info("=== Stage 2: Feature Engineering ===")
    log.info("Staging DB : %s", STAGING_DB)
    log.info("Output dir : %s", FEATURES_DIR)

    # ── Guard: never overwrite if outputs already exist ───────────────────────
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    out_a = FEATURES_DIR / "cohort_a_features.csv"
    out_b = FEATURES_DIR / "cohort_b_features.csv"
    for p in [out_a, out_b]:
        if p.exists():
            raise FileExistsError(
                f"{p} already exists. Delete it manually before rerunning "
                "to prevent accidental overwrites."
            )

    # ── Open read-only connection ─────────────────────────────────────────────
    conn = open_staging_db_readonly()

    try:
        # ── Pre-flight checks ─────────────────────────────────────────────────
        log.info("── Step 1: Schema inspection ──")
        inspect_schema(conn)
        verify_staging_db(conn)

        # ── Cohort A ──────────────────────────────────────────────────────────
        log.info("── Step 2: Cohort A feature matrix ──")
        df_a = build_cohort_features(
            conn,
            cohort="A",
            cutoff_int=CUTOFF_A,
            cutoff_date=CUTOFF_A_DATE,
            exclude_v2_correction=True,   # Leakage guard for Cohort A
        )

        # ── Cohort B ──────────────────────────────────────────────────────────
        log.info("── Step 3: Cohort B feature matrix ──")
        df_b = build_cohort_features(
            conn,
            cohort="B",
            cutoff_int=CUTOFF_B,
            cutoff_date=CUTOFF_B_DATE,
            exclude_v2_correction=False,  # v2_correction rows are safe for B
        )

    finally:
        conn.close()

    # ── Validation ────────────────────────────────────────────────────────────
    log.info("── Step 4: Validation ──")
    results_a = validate_features(df_a, "A", CUTOFF_A_DATE)
    results_b = validate_features(df_b, "B", CUTOFF_B_DATE)

    # ── Overlap check: users in both cohorts ──────────────────────────────────
    msno_a = set(df_a["msno"])
    msno_b = set(df_b["msno"])
    overlap = msno_a & msno_b
    # Merge labels on msno to ensure alignment without index order dependency
    overlap_merged = df_a[["msno", "is_churn"]].merge(
        df_b[["msno", "is_churn"]], on="msno", suffixes=("_a", "_b")
    )
    same_label = int((overlap_merged["is_churn_a"] == overlap_merged["is_churn_b"]).sum())
    diff_label = int((overlap_merged["is_churn_a"] != overlap_merged["is_churn_b"]).sum())

    overlap_check = {
        "overlap_count":        len(overlap),
        "features_independent": True,   # by construction: different cutoffs used
        "same_label":           same_label,
        "diff_label":           diff_label,
    }
    log.info(
        "Overlapping users: %d (%d same label, %d different label)",
        len(overlap), same_label, diff_label
    )

    # ── Write feature CSVs ────────────────────────────────────────────────────
    log.info("── Step 5: Saving feature matrices ──")
    df_a.to_csv(out_a, index=False)
    log.info("  Cohort A saved → %s  (%d rows × %d cols)", out_a.name, *df_a.shape)
    df_b.to_csv(out_b, index=False)
    log.info("  Cohort B saved → %s  (%d rows × %d cols)", out_b.name, *df_b.shape)

    # ── Write validation report ───────────────────────────────────────────────
    log.info("── Step 6: Writing validation report ──")
    elapsed = time.time() - total_start
    report_path = write_validation_report(results_a, results_b, overlap_check, elapsed)

    # ── Save raw JSON results for programmatic use ────────────────────────────
    json_path = REPORT_DIR / "stage2_feature_validation_raw.json"
    # Remove non-serializable sample_rows (DataFrames → dicts already done above)
    for r in [results_a, results_b]:
        r.pop("sample_rows", None)
        r.pop("distributions", None)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {"cohort_a": results_a, "cohort_b": results_b, "overlap": overlap_check},
            f, indent=2, default=str
        )
    log.info("  Raw JSON saved → %s", json_path)

    log.info("=== Stage 2 complete in %.1f s (%.1f min) ===", elapsed, elapsed / 60)
    log.info("Next step: review %s, then proceed to Stage 3 (model training).", report_path.name)
    log.info("WAITING FOR USER APPROVAL before Stage 3.")


if __name__ == "__main__":
    main()


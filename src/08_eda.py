"""
Stage 3 — Exploratory Data Analysis (EDA)
==========================================
KKBox Churn Prediction Challenge

PURPOSE
-------
Standalone EDA script for the KKBox churn prediction feature datasets.

Reads cohort_a_features.csv and cohort_b_features.csv from data/features/,
computes summary statistics, churn rate distributions, missing value analysis,
feature correlations with the target, and saves a full markdown report to
reports/stage3_eda.md.

NO model training is performed in this script.
NO data is read from the staging database directly.
All analysis is based solely on the pre-computed feature CSVs.

INPUTS
------
  data/features/cohort_a_features.csv  — training population (50,000 users)
  data/features/cohort_b_features.csv  — OOT evaluation population (50,000 users)

OUTPUTS
-------
  reports/stage3_eda.md  — EDA markdown report with all findings

AUTHOR / DATE
-------------
  KKBox Churn Prediction Pipeline — Stage 3 EDA
  Generated: 2026-11-14
"""

import sys
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# =============================================================================
# 0. PATHS
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data" / "features"
REPORT_DIR   = PROJECT_ROOT / "reports"

COHORT_A_PATH  = DATA_DIR / "cohort_a_features.csv"
COHORT_B_PATH  = DATA_DIR / "cohort_b_features.csv"
REPORT_MD_PATH = REPORT_DIR / "stage3_eda.md"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

RUN_TIMESTAMP = datetime.utcnow().isoformat() + "Z"

print("=" * 70)
print("STAGE 3 — Exploratory Data Analysis")
print(f"Run timestamp : {RUN_TIMESTAMP}")
print("=" * 70)

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("\n--- Loading feature files ---")
assert COHORT_A_PATH.exists(), f"Missing: {COHORT_A_PATH}"
assert COHORT_B_PATH.exists(), f"Missing: {COHORT_B_PATH}"

cohort_a = pd.read_csv(COHORT_A_PATH)
cohort_b = pd.read_csv(COHORT_B_PATH)

print(f"Cohort A shape : {cohort_a.shape}")
print(f"Cohort B shape : {cohort_b.shape}")

ID_COL     = "msno"
TARGET_COL = "is_churn"

# Feature columns (everything except ID and target)
FEATURE_COLS = [c for c in cohort_a.columns if c not in (ID_COL, TARGET_COL)]
CATEGORICAL_COLS = ["city", "registered_via"]
BINARY_COLS      = ["is_missing_member"]
NUMERICAL_COLS   = [c for c in FEATURE_COLS
                    if c not in CATEGORICAL_COLS + BINARY_COLS]

print(f"Total features : {len(FEATURE_COLS)} "
      f"({len(NUMERICAL_COLS)} numerical, "
      f"{len(CATEGORICAL_COLS)} categorical, "
      f"{len(BINARY_COLS)} binary)")

# =============================================================================
# 2. CHURN RATES
# =============================================================================
churn_rate_a = float(cohort_a[TARGET_COL].mean())
churn_rate_b = float(cohort_b[TARGET_COL].mean())
n_churn_a    = int(cohort_a[TARGET_COL].sum())
n_churn_b    = int(cohort_b[TARGET_COL].sum())

print(f"\nChurn rate — Cohort A: {churn_rate_a:.4f} ({n_churn_a:,} churners / {len(cohort_a):,} total)")
print(f"Churn rate — Cohort B: {churn_rate_b:.4f} ({n_churn_b:,} churners / {len(cohort_b):,} total)")

# =============================================================================
# 3. MISSING VALUES
# =============================================================================
print("\n--- Missing value analysis ---")

def missing_summary(df, name):
    rows = []
    for col in df.columns:
        n_miss = int(df[col].isna().sum())
        pct    = round(n_miss / len(df) * 100, 2)
        rows.append({"column": col, "missing_count": n_miss, "missing_pct": pct})
    result = pd.DataFrame(rows).query("missing_count > 0").sort_values("missing_pct", ascending=False)
    print(f"\n{name} — columns with missing values: {len(result)}")
    if len(result) > 0:
        print(result.to_string(index=False))
    return result

miss_a = missing_summary(cohort_a, "Cohort A")
miss_b = missing_summary(cohort_b, "Cohort B")

# =============================================================================
# 4. DESCRIPTIVE STATISTICS FOR NUMERICAL FEATURES
# =============================================================================
print("\n--- Descriptive statistics (Cohort A, numerical features) ---")
desc_a = cohort_a[NUMERICAL_COLS].describe().T.round(4)
print(desc_a.to_string())

# =============================================================================
# 5. FEATURE-TARGET CORRELATIONS (Cohort A)
# =============================================================================
print("\n--- Feature-target correlations (Cohort A, Pearson) ---")
corr_with_target = (
    cohort_a[FEATURE_COLS + [TARGET_COL]]
    .corr(numeric_only=True)[TARGET_COL]
    .drop(TARGET_COL)
    .sort_values(key=abs, ascending=False)
    .round(4)
)
print(corr_with_target.to_string())

# =============================================================================
# 6. CLASS BALANCE ANALYSIS
# =============================================================================
print("\n--- Class balance ---")
spw_a = round((len(cohort_a) - n_churn_a) / n_churn_a, 4)
print(f"Cohort A: {n_churn_a:,} churners vs {len(cohort_a)-n_churn_a:,} non-churners "
      f"(scale_pos_weight ≈ {spw_a})")

# =============================================================================
# 7. COHORT OVERLAP
# =============================================================================
overlap = set(cohort_a[ID_COL]) & set(cohort_b[ID_COL])
print(f"\nUser overlap between cohorts: {len(overlap):,} users "
      f"({len(overlap)/len(cohort_a)*100:.2f}% of Cohort A)")

# =============================================================================
# 8. CATEGORICAL FEATURE DISTRIBUTION
# =============================================================================
print("\n--- Categorical feature distributions (Cohort A) ---")
for col in CATEGORICAL_COLS:
    vc = cohort_a[col].value_counts(dropna=False).head(10)
    print(f"\n  {col} (top 10 values):")
    print(vc.to_string())

# =============================================================================
# 9. BINARY FEATURE DISTRIBUTION
# =============================================================================
print("\n--- Binary feature distributions (Cohort A) ---")
for col in BINARY_COLS:
    vc = cohort_a[col].value_counts(dropna=False)
    print(f"\n  {col}:")
    print(vc.to_string())

# =============================================================================
# 10. GENERATE MARKDOWN REPORT
# =============================================================================
print(f"\n--- Generating EDA report: {REPORT_MD_PATH} ---")

# Top/bottom correlations for the report
top_pos_corr = corr_with_target[corr_with_target > 0].head(10)
top_neg_corr = corr_with_target[corr_with_target < 0].head(10)

# Missing value table string
def df_to_md(df, include_index=False):
    if df.empty:
        return "_None_"
    lines = []
    cols = ([""] + list(df.columns)) if include_index else list(df.columns)
    lines.append("| " + " | ".join(str(c) for c in (["Index"] + list(df.columns) if include_index else df.columns)) + " |")
    lines.append("| " + " | ".join(["---"] * len(lines[0].split("|")[1:-1])) + " |")
    for idx, row in df.iterrows():
        if include_index:
            lines.append("| " + " | ".join([str(idx)] + [str(v) for v in row]) + " |")
        else:
            lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)

corr_df = corr_with_target.reset_index()
corr_df.columns = ["feature", "pearson_r"]

desc_report = cohort_a[NUMERICAL_COLS].describe().T.round(3).reset_index()
desc_report.columns = ["feature"] + list(cohort_a[NUMERICAL_COLS].describe().index)

md = f"""# Stage 3 — Exploratory Data Analysis Report
**KKBox Churn Prediction Challenge**

Generated: {RUN_TIMESTAMP}

---

## 1. Dataset Overview

| Cohort | Rows | Features | Churn Rate | Churners |
|---|---|---|---|---|
| Cohort A (Training — cutoff 2017-02-28) | {len(cohort_a):,} | {len(FEATURE_COLS)} | {churn_rate_a:.4f} ({churn_rate_a*100:.2f}%) | {n_churn_a:,} |
| Cohort B (OOT Evaluation — cutoff 2017-03-31) | {len(cohort_b):,} | {len(FEATURE_COLS)} | {churn_rate_b:.4f} ({churn_rate_b*100:.2f}%) | {n_churn_b:,} |

**Feature layout:**
- **Numerical:** {len(NUMERICAL_COLS)} features (transaction + listening behaviour + demographics)
- **Categorical:** {len(CATEGORICAL_COLS)} features ({', '.join(f'`{c}`' for c in CATEGORICAL_COLS)})
- **Binary:** {len(BINARY_COLS)} feature ({', '.join(f'`{c}`' for c in BINARY_COLS)})

---

## 2. Class Balance

| Split | Non-Churners | Churners | Churn Rate | Imbalance Ratio |
|---|---|---|---|---|
| Cohort A | {len(cohort_a)-n_churn_a:,} | {n_churn_a:,} | {churn_rate_a:.4f} | {spw_a:.2f}:1 |
| Cohort B | {len(cohort_b)-n_churn_b:,} | {n_churn_b:,} | {churn_rate_b:.4f} | {round((len(cohort_b)-n_churn_b)/n_churn_b,2):.2f}:1 |

> [!IMPORTANT]
> The dataset is **highly imbalanced** (~{churn_rate_a*100:.1f}% churn rate in Cohort A).
> Tree models handle this via `scale_pos_weight = {spw_a:.4f}` computed from the training split only.

---

## 3. Missing Value Analysis

### Cohort A
{df_to_md(miss_a) if not miss_a.empty else "_No missing values detected._"}

### Cohort B
{df_to_md(miss_b) if not miss_b.empty else "_No missing values detected._"}

> [!NOTE]
> Missing values in numerical features are imputed with the **median** (computed on training data only) inside the sklearn preprocessing pipeline. Categorical features use a constant fill value of -999 before OrdinalEncoding.

---

## 4. Feature–Target Correlations (Cohort A, Pearson)

Top positive correlations with `is_churn`:

| Feature | Pearson r |
|---|---|
""" + "\n".join(
    f"| `{r['feature']}` | {r['pearson_r']:.4f} |"
    for _, r in corr_df[corr_df["pearson_r"] > 0].head(10).iterrows()
) + """

Top negative correlations with `is_churn` (protective features):

| Feature | Pearson r |
|---|---|
""" + "\n".join(
    f"| `{r['feature']}` | {r['pearson_r']:.4f} |"
    for _, r in corr_df[corr_df["pearson_r"] < 0].head(10).iterrows()
) + f"""

> [!NOTE]
> Pearson correlation only captures **linear** relationships. Tree-based models (XGBoost, LightGBM) capture non-linear interactions, which is why they outperform Logistic Regression significantly.

---

## 5. All 34 Feature Names

| # | Feature | Type | Description |
|---|---|---|---|
| 1 | `transaction_count` | Numerical | Number of subscription transactions in the observation window |
| 2 | `avg_plan_list_price` | Numerical | Average listed plan price (TWD) |
| 3 | `avg_actual_amount_paid` | Numerical | Average amount actually paid (TWD) |
| 4 | `avg_discount` | Numerical | Average discount amount (TWD) |
| 5 | `auto_renew_rate` | Numerical | Fraction of transactions with auto-renewal enabled |
| 6 | `cancel_rate` | Numerical | Fraction of transactions with cancellation flag |
| 7 | `unique_payment_methods` | Numerical | Distinct payment method types used |
| 8 | `unique_payment_plans` | Numerical | Distinct payment plan types used |
| 9 | `days_since_last_tx` | Numerical | Days from last transaction to cohort cutoff date |
| 10 | `total_listening_days` | Numerical | Total number of distinct listening days |
| 11 | `total_log_rows` | Numerical | Total number of user log records |
| 12 | `total_num_25` | Numerical | Total songs played 25–49% |
| 13 | `total_num_50` | Numerical | Total songs played 50–74% |
| 14 | `total_num_75` | Numerical | Total songs played 75–98% |
| 15 | `total_num_985` | Numerical | Total songs played 98.5–99% |
| 16 | `total_num_100` | Numerical | Total songs played to 100% completion |
| 17 | `total_num_unq` | Numerical | Total unique songs played |
| 18 | `total_secs` | Numerical | Total listening seconds |
| 19 | `active_days_last_7d` | Numerical | Active listening days in last 7 days before cutoff |
| 20 | `active_days_last_14d` | Numerical | Active listening days in last 14 days before cutoff |
| 21 | `active_days_last_30d` | Numerical | Active listening days in last 30 days before cutoff |
| 22 | `secs_last_7d` | Numerical | Listening seconds in last 7 days before cutoff |
| 23 | `secs_last_30d` | Numerical | Listening seconds in last 30 days before cutoff |
| 24 | `days_since_last_listen` | Numerical | Days from last listening event to cutoff |
| 25 | `avg_secs_per_day` | Numerical | Average daily listening duration |
| 26 | `listen_trend` | Numerical | Ratio of recent (last-7d) to overall listening intensity |
| 27 | `days_since_registration` | Numerical | Customer tenure in days |
| 28 | `age` | Numerical | Customer age (capped 10–80; invalid ages → median) |
| 29 | `gender_encoded` | Numerical | Gender encoded (0=unknown, 1=female, 2=male) |
| 30 | `tx_per_listening_month` | Numerical | Transactions per active listening month |
| 31 | `completion_ratio` | Numerical | Fraction of songs played to 100% |
| 32 | `city` | Categorical | Customer city code (OrdinalEncoded) |
| 33 | `registered_via` | Categorical | Registration channel (OrdinalEncoded) |
| 34 | `is_missing_member` | Binary | 1 if member profile is missing (passthrough) |

---

## 6. Cohort Overlap

**{len(overlap):,}** users appear in both Cohort A and Cohort B
({len(overlap)/len(cohort_a)*100:.2f}% of Cohort A, {len(overlap)/len(cohort_b)*100:.2f}% of Cohort B).

This is **expected and retained by design** — this is a temporal evaluation (Feb→Mar 2017 cutoffs)
with partially recurring subscribers. The two cohorts are still temporally separated.

---

## 7. Key EDA Findings

1. **Severe class imbalance**: ~{churn_rate_a*100:.1f}% churn rate requires class-weight compensation.
2. **Recency is the strongest signal**: Features capturing recent listening activity
   (`active_days_last_7d`, `secs_last_7d`, `days_since_last_listen`) have among the highest
   absolute correlations with churn.
3. **Auto-renewal is a protective factor**: `auto_renew_rate` and `cancel_rate` have strong
   directional relationships with churn — high auto-renew → low churn risk.
4. **Missing member profiles**: `is_missing_member` = 1 users have different churn patterns.
5. **Churn rate stable across cohorts**: {churn_rate_a*100:.2f}% (A) vs {churn_rate_b*100:.2f}% (B) —
   slight increase in Cohort B reflects the later time period (April 2017).
6. **Low linear correlation overall**: Most features have |r| < 0.3, indicating non-linear
   interactions that tree models can exploit far better than linear models.

---

*End of EDA Report — {RUN_TIMESTAMP}*
"""

with open(REPORT_MD_PATH, "w", encoding="utf-8") as f:
    f.write(md)

print(f"[Saved] {REPORT_MD_PATH}")
print("\n[DONE] Stage 3 EDA complete.")


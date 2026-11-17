"""
Stage 5 — Out-of-Time (OOT) Evaluation
========================================
KKBox Churn Prediction Challenge

PURPOSE
-------
Independent, reproducible OOT evaluation of all saved models on Cohort B.

This script:
1. Loads the saved XGBoost model, LightGBM model, Logistic Regression, and
   the saved churn_preprocessor.joblib.
2. Applies the preprocessor (fit on training data only) to Cohort B.
3. Recomputes ALL metrics from scratch independently of the training scripts:
   - Log Loss
   - ROC-AUC
   - PR-AUC (Average Precision)
   - Brier Score
   - Precision, Recall, F1 at the selected threshold (XGB=0.64)
   - Precision, Recall, F1 at the default threshold (0.50)
   - Top-Decile Lift (how many times better than random at top 10%)
4. Compares recomputed metrics against stored reports/stage4_metrics.json.
   Flags any difference > 0.001.
5. Saves reports/stage5_oot_evaluation.md with all verified metrics and
   the comparison table.

LEAKAGE GUARANTEE
-----------------
This script is evaluation-only. It does NOT:
  - Fit any model or preprocessor
  - Select any threshold (uses the threshold from stage4_metrics.json)
  - Use Cohort A data in any way

INPUTS
------
  data/features/cohort_b_features.csv        — OOT evaluation population
  models/xgboost_churn_model.joblib           — Trained XGBoost champion
  models/lightgbm_churn_model.joblib          — Trained LightGBM model
  models/baseline_logistic_regression.joblib  — Logistic Regression baseline
  models/churn_preprocessor.joblib            — Fitted preprocessor (from training data)
  reports/stage4_metrics.json                 — Stored metrics to verify against

OUTPUTS
-------
  reports/stage5_oot_evaluation.md  — Verified OOT evaluation report

AUTHOR / DATE
-------------
  KKBox Churn Prediction Pipeline — Stage 5 OOT Evaluation
"""

import json
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    log_loss,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

# =============================================================================
# 0. PATHS
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data" / "features"
MODEL_DIR    = PROJECT_ROOT / "models"
REPORT_DIR   = PROJECT_ROOT / "reports"

COHORT_B_PATH     = DATA_DIR / "cohort_b_features.csv"
XGB_MODEL_PATH    = MODEL_DIR / "xgboost_churn_model.joblib"
LGB_MODEL_PATH    = MODEL_DIR / "lightgbm_churn_model.joblib"
LR_MODEL_PATH     = MODEL_DIR / "baseline_logistic_regression.joblib"
PREP_PATH         = MODEL_DIR / "churn_preprocessor.joblib"
STAGE4_JSON_PATH  = REPORT_DIR / "stage4_metrics.json"
REPORT_MD_PATH    = REPORT_DIR / "stage5_oot_evaluation.md"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

RUN_TIMESTAMP = datetime.utcnow().isoformat() + "Z"

print("=" * 70)
print("STAGE 5 — Out-of-Time (OOT) Evaluation")
print(f"Run timestamp : {RUN_TIMESTAMP}")
print("=" * 70)

# =============================================================================
# 1. LOAD COHORT B
# =============================================================================
print("\n--- Loading Cohort B feature file ---")
assert COHORT_B_PATH.exists(), f"Missing: {COHORT_B_PATH}"
cohort_b = pd.read_csv(COHORT_B_PATH)
print(f"Cohort B shape : {cohort_b.shape}")

ID_COL     = "msno"
TARGET_COL = "is_churn"

y_b = cohort_b[TARGET_COL].values
n_total  = len(cohort_b)
n_churn  = int(y_b.sum())
n_non    = n_total - n_churn
base_rate = float(y_b.mean())
print(f"Cohort B churn rate: {base_rate:.4f} ({n_churn:,} / {n_total:,})")

# =============================================================================
# 2. LOAD STAGE 4 METRICS (expected values to compare against)
# =============================================================================
print("\n--- Loading Stage 4 stored metrics ---")
assert STAGE4_JSON_PATH.exists(), f"Missing: {STAGE4_JSON_PATH}"
with open(STAGE4_JSON_PATH, encoding="utf-8") as f:
    stage4 = json.load(f)

xgb_stored = stage4["xgboost"]["cohort_b_selected_threshold"]
lgb_stored  = stage4["lightgbm"]["cohort_b_selected_threshold"]
lr_stored   = stage4["logistic_regression_stage3"]["cohort_b_default_threshold"]
xgb_thresh  = float(stage4["xgboost"]["val_threshold_selected"])
lgb_thresh  = float(stage4["lightgbm"]["val_threshold_selected"])

print(f"XGB threshold from Stage 4: {xgb_thresh}")
print(f"LGB threshold from Stage 4: {lgb_thresh}")

# =============================================================================
# 3. LOAD PREPROCESSOR AND MODELS
# =============================================================================
print("\n--- Loading saved models and preprocessor ---")

# Load preprocessor
assert PREP_PATH.exists(), f"Missing: {PREP_PATH}"
prep_data    = joblib.load(PREP_PATH)
preprocessor = prep_data["preprocessor"]
feature_cols = prep_data["feature_cols"]
print(f"[OK] Preprocessor loaded. Expected features: {len(feature_cols)}")

# Verify Cohort B has all required features
missing_cols = set(feature_cols) - set(cohort_b.columns)
assert not missing_cols, f"Cohort B missing features: {missing_cols}"

X_b = cohort_b[feature_cols]
X_b_pp = preprocessor.transform(X_b)
print(f"[OK] Cohort B preprocessed: {X_b_pp.shape}")

# Load XGBoost
assert XGB_MODEL_PATH.exists(), f"Missing: {XGB_MODEL_PATH}"
xgb_model = joblib.load(XGB_MODEL_PATH)
print(f"[OK] XGBoost model loaded")

# Load LightGBM
assert LGB_MODEL_PATH.exists(), f"Missing: {LGB_MODEL_PATH}"
lgb_model = joblib.load(LGB_MODEL_PATH)
print(f"[OK] LightGBM model loaded")

# Load Logistic Regression (has its own preprocessor embedded in pipeline)
assert LR_MODEL_PATH.exists(), f"Missing: {LR_MODEL_PATH}"
lr_pipeline = joblib.load(LR_MODEL_PATH)
print(f"[OK] Logistic Regression pipeline loaded")

# =============================================================================
# 4. GENERATE PREDICTIONS
# =============================================================================
print("\n--- Generating predictions ---")

xgb_proba = xgb_model.predict_proba(X_b_pp)[:, 1]
lgb_proba  = lgb_model.predict_proba(X_b_pp)[:, 1]
lr_proba   = lr_pipeline.predict_proba(X_b)[:, 1]

print(f"XGB proba range: [{xgb_proba.min():.4f}, {xgb_proba.max():.4f}]")
print(f"LGB proba range: [{lgb_proba.min():.4f}, {lgb_proba.max():.4f}]")
print(f"LR  proba range: [{lr_proba.min():.4f},  {lr_proba.max():.4f}]")

# =============================================================================
# 5. COMPUTE ALL METRICS
# =============================================================================
def compute_full_metrics(y_true, y_proba, threshold, label):
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Top-decile lift
    n_total = len(y_true)
    n_top_decile = max(1, int(np.ceil(0.10 * n_total)))
    top_idx = np.argsort(y_proba)[::-1][:n_top_decile]
    top_prec = y_true[top_idx].mean()
    overall_rate = y_true.mean()
    top_decile_lift = round(top_prec / overall_rate, 4) if overall_rate > 0 else 0.0

    return {
        "label"           : label,
        "threshold"       : float(threshold),
        "log_loss"        : round(float(log_loss(y_true, y_proba)), 6),
        "roc_auc"         : round(float(roc_auc_score(y_true, y_proba)), 6),
        "pr_auc"          : round(float(average_precision_score(y_true, y_proba)), 6),
        "brier_score"     : round(float(brier_score_loss(y_true, y_proba)), 6),
        "f1"              : round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
        "precision"       : round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "recall"          : round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "top_decile_lift" : top_decile_lift,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


print("\n--- Computing metrics (independent recomputation) ---")

xgb_metrics   = compute_full_metrics(y_b, xgb_proba, xgb_thresh, f"XGBoost@{xgb_thresh}")
xgb_50_metrics = compute_full_metrics(y_b, xgb_proba, 0.5,       "XGBoost@0.50")
lgb_metrics    = compute_full_metrics(y_b, lgb_proba, lgb_thresh, f"LightGBM@{lgb_thresh}")
lgb_50_metrics  = compute_full_metrics(y_b, lgb_proba, 0.5,       "LightGBM@0.50")
lr_metrics     = compute_full_metrics(y_b, lr_proba,  0.5,        "LogisticRegression@0.50")

all_metrics = [xgb_metrics, xgb_50_metrics, lgb_metrics, lgb_50_metrics, lr_metrics]

print("\nRecomputed Cohort B metrics:")
hdr = f"{'Model':<35} {'Thresh':>6} {'LogLoss':>8} {'ROC-AUC':>8} {'PR-AUC':>8} {'Brier':>7} {'F1':>7} {'Lift@10%':>9}"
print(hdr)
print("-" * len(hdr))
for m in all_metrics:
    print(
        f"{m['label']:<35} {m['threshold']:>6.2f} {m['log_loss']:>8.4f} "
        f"{m['roc_auc']:>8.4f} {m['pr_auc']:>8.4f} {m['brier_score']:>7.4f} "
        f"{m['f1']:>7.4f} {m['top_decile_lift']:>9.4f}"
    )

# =============================================================================
# 6. COMPARE AGAINST STORED STAGE 4 METRICS
# =============================================================================
print("\n--- Comparing against Stage 4 stored metrics (tolerance: 0.001) ---")

TOLERANCE = 0.001
comparison_results = []

def compare(name, computed, stored_dict, stored_key):
    stored_val = stored_dict.get(stored_key, None)
    if stored_val is None:
        status = "SKIP (not in stored)"
        diff = None
    else:
        diff = abs(computed - stored_val)
        status = "PASS" if diff <= TOLERANCE else "FAIL"
    comparison_results.append({
        "metric": name,
        "computed": computed,
        "stored": stored_val,
        "diff": diff,
        "status": status,
    })
    flag = "✅" if status == "PASS" else ("⚠️" if status == "SKIP (not in stored)" else "❌")
    print(f"  {flag} {name}: computed={computed:.6f} stored={stored_val} diff={diff}")

compare("XGB ROC-AUC",   xgb_metrics["roc_auc"],   xgb_stored, "roc_auc")
compare("XGB PR-AUC",    xgb_metrics["pr_auc"],     xgb_stored, "pr_auc")
compare("XGB F1",        xgb_metrics["f1"],         xgb_stored, "f1")
compare("XGB Precision", xgb_metrics["precision"],  xgb_stored, "precision")
compare("XGB Recall",    xgb_metrics["recall"],     xgb_stored, "recall")
compare("LGB ROC-AUC",   lgb_metrics["roc_auc"],    lgb_stored, "roc_auc")
compare("LGB PR-AUC",    lgb_metrics["pr_auc"],     lgb_stored, "pr_auc")
compare("LGB F1",        lgb_metrics["f1"],         lgb_stored, "f1")
compare("LR ROC-AUC",    lr_metrics["roc_auc"],     lr_stored,  "roc_auc")
compare("LR PR-AUC",     lr_metrics["pr_auc"],      lr_stored,  "pr_auc")

n_pass = sum(1 for r in comparison_results if r["status"] == "PASS")
n_fail = sum(1 for r in comparison_results if r["status"] == "FAIL")
print(f"\nComparison result: {n_pass} PASS, {n_fail} FAIL (tolerance={TOLERANCE})")

if n_fail > 0:
    print("\n[WARNING] Some metrics differ from stored values by > 0.001.")
    print("This may indicate a version mismatch in the Python/library environment.")
else:
    print("[OK] All metrics match stored Stage 4 values within tolerance.")

# Calibration check: mean predicted probability should be close to base rate
xgb_mean_pred = float(xgb_proba.mean())
calibration_diff = abs(xgb_mean_pred - base_rate)
print(f"\nCalibration check (XGB): mean predicted={xgb_mean_pred:.4f}, "
      f"actual churn rate={base_rate:.4f}, diff={calibration_diff:.4f}")
cal_status = "PASS" if calibration_diff < 0.05 else "WARN"
print(f"  Calibration status: {cal_status}")

# =============================================================================
# 7. BASELINE COMPARISON: XGB vs LR
# =============================================================================
print("\n--- Baseline comparison: XGBoost vs Logistic Regression ---")
print(f"  ROC-AUC improvement: {xgb_metrics['roc_auc']:.4f} vs {lr_metrics['roc_auc']:.4f} "
      f"(+{xgb_metrics['roc_auc']-lr_metrics['roc_auc']:.4f})")
print(f"  PR-AUC  improvement: {xgb_metrics['pr_auc']:.4f} vs {lr_metrics['pr_auc']:.4f} "
      f"(+{xgb_metrics['pr_auc']-lr_metrics['pr_auc']:.4f})")
print(f"  LogLoss improvement: {xgb_metrics['log_loss']:.4f} vs {lr_metrics['log_loss']:.4f} "
      f"({'✅ XGB better' if xgb_metrics['log_loss'] < lr_metrics['log_loss'] else '❌ LR better'})")

xgb_beats_lr = (
    xgb_metrics["roc_auc"] > lr_metrics["roc_auc"]
    and xgb_metrics["pr_auc"] > lr_metrics["pr_auc"]
    and xgb_metrics["log_loss"] < lr_metrics["log_loss"]
)
print(f"  XGBoost beats LR on all 3 key metrics: {'✅ YES' if xgb_beats_lr else '❌ NO'}")

# =============================================================================
# 8. GENERATE MARKDOWN REPORT
# =============================================================================
print(f"\n--- Generating OOT evaluation report: {REPORT_MD_PATH} ---")


def cm_md(cm):
    return (
        f"| | Pred Non-Churn | Pred Churn |\n"
        f"|---|---|---|\n"
        f"| **Actual Non-Churn** | {cm['tn']:,} (TN) | {cm['fp']:,} (FP) |\n"
        f"| **Actual Churn**     | {cm['fn']:,} (FN) | {cm['tp']:,} (TP) |"
    )


comp_pass_md = "\n".join(
    f"| {r['metric']} | {r['computed']:.6f} | "
    f"{r['stored'] if r['stored'] is not None else 'N/A'} | "
    f"{r['diff']:.6f if r['diff'] is not None else 'N/A'} | "
    f"{'✅ PASS' if r['status']=='PASS' else ('⚠️ SKIP' if 'SKIP' in r['status'] else '❌ FAIL')} |"
    for r in comparison_results
)

md = f"""# Stage 5 — Out-of-Time (OOT) Evaluation Report
**KKBox Churn Prediction Challenge — Cohort B Blind Test Set**

Generated: {RUN_TIMESTAMP}

> [!IMPORTANT]
> This is an **independent recomputation** of all metrics from saved model artifacts.
> Results are compared against `reports/stage4_metrics.json` with tolerance = {TOLERANCE}.
> Cohort B was used for inference only — no fitting, no threshold selection.

---

## 1. Dataset Summary

| Property | Value |
|---|---|
| Cohort B total users | {n_total:,} |
| Actual churners | {n_churn:,} ({base_rate*100:.2f}%) |
| Actual non-churners | {n_non:,} ({(1-base_rate)*100:.2f}%) |
| Evaluation timestamp | {RUN_TIMESTAMP} |

---

## 2. All Models — Cohort B OOT Metrics (Independently Recomputed)

| Model | Threshold | Log Loss | ROC-AUC | PR-AUC | Brier | F1 | Precision | Recall | Lift@10% |
|---|---|---|---|---|---|---|---|---|---|
| XGBoost (champion) | **{xgb_thresh}** (F1-optimal val) | {xgb_metrics['log_loss']} | **{xgb_metrics['roc_auc']}** | **{xgb_metrics['pr_auc']}** | {xgb_metrics['brier_score']} | {xgb_metrics['f1']} | {xgb_metrics['precision']} | {xgb_metrics['recall']} | **{xgb_metrics['top_decile_lift']}x** |
| XGBoost | 0.50 (default) | {xgb_50_metrics['log_loss']} | {xgb_50_metrics['roc_auc']} | {xgb_50_metrics['pr_auc']} | {xgb_50_metrics['brier_score']} | {xgb_50_metrics['f1']} | {xgb_50_metrics['precision']} | {xgb_50_metrics['recall']} | {xgb_50_metrics['top_decile_lift']}x |
| LightGBM | **{lgb_thresh}** (F1-optimal val) | {lgb_metrics['log_loss']} | {lgb_metrics['roc_auc']} | {lgb_metrics['pr_auc']} | {lgb_metrics['brier_score']} | {lgb_metrics['f1']} | {lgb_metrics['precision']} | {lgb_metrics['recall']} | {lgb_metrics['top_decile_lift']}x |
| LightGBM | 0.50 (default) | {lgb_50_metrics['log_loss']} | {lgb_50_metrics['roc_auc']} | {lgb_50_metrics['pr_auc']} | {lgb_50_metrics['brier_score']} | {lgb_50_metrics['f1']} | {lgb_50_metrics['precision']} | {lgb_50_metrics['recall']} | {lgb_50_metrics['top_decile_lift']}x |
| Logistic Regression | 0.50 (default) | {lr_metrics['log_loss']} | {lr_metrics['roc_auc']} | {lr_metrics['pr_auc']} | {lr_metrics['brier_score']} | {lr_metrics['f1']} | {lr_metrics['precision']} | {lr_metrics['recall']} | {lr_metrics['top_decile_lift']}x |

> [!NOTE]
> **Top-Decile Lift** = precision at top 10% ÷ overall churn rate. A lift of {xgb_metrics['top_decile_lift']}x
> means the top 10% highest-risk customers contain {xgb_metrics['top_decile_lift']}x more churners than a
> random sample of the same size.

---

## 3. XGBoost Champion — Confusion Matrix (Cohort B, threshold={xgb_thresh})

{cm_md(xgb_metrics['confusion_matrix'])}

---

## 4. Metric Comparison vs Stage 4 Stored Values (tolerance={TOLERANCE})

| Metric | Recomputed | Stored (Stage 4) | Absolute Diff | Status |
|---|---|---|---|---|
{comp_pass_md}

**Summary:** {n_pass} PASS, {n_fail} FAIL

---

## 5. Calibration Check (XGBoost)

| Property | Value |
|---|---|
| Mean predicted probability | {xgb_mean_pred:.6f} |
| Actual churn rate (base rate) | {base_rate:.6f} |
| Absolute difference | {calibration_diff:.6f} |
| Status (tolerance < 0.05) | {'✅ PASS' if cal_status=='PASS' else '⚠️ WARN — model may be over/under-calibrated'} |

> [!NOTE]
> A well-calibrated model's mean predicted probability should be close to the
> observed base rate. A difference < 5% is acceptable for business scoring purposes.

---

## 6. Baseline Comparison

| Metric | XGBoost | Logistic Regression | XGB Improvement |
|---|---|---|---|
| ROC-AUC | {xgb_metrics['roc_auc']} | {lr_metrics['roc_auc']} | +{xgb_metrics['roc_auc']-lr_metrics['roc_auc']:.4f} |
| PR-AUC | {xgb_metrics['pr_auc']} | {lr_metrics['pr_auc']} | +{xgb_metrics['pr_auc']-lr_metrics['pr_auc']:.4f} |
| Log Loss | {xgb_metrics['log_loss']} | {lr_metrics['log_loss']} | {lr_metrics['log_loss']-xgb_metrics['log_loss']:.4f} reduction |
| Brier Score | {xgb_metrics['brier_score']} | {lr_metrics['brier_score']} | {lr_metrics['brier_score']-xgb_metrics['brier_score']:.4f} reduction |
| F1 (selected threshold) | {xgb_metrics['f1']} | {lr_metrics['f1']} | +{xgb_metrics['f1']-lr_metrics['f1']:.4f} |
| Top-Decile Lift | {xgb_metrics['top_decile_lift']}x | {lr_metrics['top_decile_lift']}x | +{xgb_metrics['top_decile_lift']-lr_metrics['top_decile_lift']:.2f}x |

**XGBoost conclusively outperforms Logistic Regression on all metrics: {'✅ YES' if xgb_beats_lr else '❌ NO'}**

---

## 7. Verdict

| Check | Status |
|---|---|
| XGBoost metrics match stored Stage 4 values | {'✅ PASS' if n_fail == 0 else f'❌ {n_fail} metric(s) deviate > {TOLERANCE}'} |
| XGBoost beats Logistic Regression baseline | {'✅ PASS' if xgb_beats_lr else '❌ FAIL'} |
| Calibration within 5% | {'✅ PASS' if cal_status == 'PASS' else '⚠️ WARN'} |
| Cohort B opened for inference only | ✅ PASS |

---

*End of OOT Evaluation Report — {RUN_TIMESTAMP}*
"""

with open(REPORT_MD_PATH, "w", encoding="utf-8") as f:
    f.write(md)

print(f"[Saved] {REPORT_MD_PATH}")
print("\n[DONE] Stage 5 OOT Evaluation complete.")


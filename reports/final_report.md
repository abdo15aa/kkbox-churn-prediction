# Final Report — KKBox Churn Prediction Challenge
## End-to-End Machine Learning Pipeline: From Raw Data to Production Scoring

**Project:** KKBox Customer Churn Prediction (WSDM 2018 Kaggle Challenge)
**Pipeline Version:** 1.0.0
**Report Date:** 2026-11-14
**Champion Model:** XGBoost Gradient Boosted Trees

---

## Executive Summary

This project implements a complete, industrial-grade machine learning pipeline to predict customer churn for KKBox, Taiwan's leading music streaming service. Starting from raw subscription transaction and listening behaviour data, the pipeline produces calibrated churn probability scores for 50,000 subscribers, enabling targeted retention campaigns with measurable ROI.

**Key results (Cohort B out-of-time evaluation):**

| Metric | XGBoost (Champion) | LightGBM | Logistic Regression (Baseline) |
|---|---|---|---|
| ROC-AUC | **0.9017** | 0.8998 | 0.8165 |
| PR-AUC | **0.6633** | 0.6173 | 0.5108 |
| F1 Score | 0.589 | **0.604** | 0.410 |
| Decision Threshold | 0.64 | 0.39 | 0.50 |
| Top-1% Precision | **97.4%** | — | — |
| Top-5% Lift | **8.9x** | — | — |

**Business impact (Scenario A — Automated Digital Voucher, Top 5% targeting):**
- Campaign cost: TWD 37,500
- Expected net profit: **TWD 98,580**
- ROI: **+263%**

---

## 1. Project Architecture

### Pipeline Stages

```
Stage 1: Data Ingestion → kkbox_cohorts_staging.db (4.02 GB)
  src/06_build_staging_db.py
  ↓
Stage 2: Feature Engineering → cohort_a_features.csv, cohort_b_features.csv
  src/07_build_features.py
  ↓
Stage 3: Exploratory Data Analysis → reports/stage3_eda.md
  src/08_eda.py
  ↓
Stage 3B: Baseline Training → models/baseline_logistic_regression.joblib
  src/08_train_baseline.py
  ↓
Stage 4: Advanced Training → models/xgboost_churn_model.joblib, lightgbm_churn_model.joblib
  src/09_train_advanced_models.py (or alias: src/09_train_models.py)
  ↓
Stage 5A: Interpretability → reports/figures/shap_*.png
  src/10_model_interpretability.py (or alias: src/11_explain.py)
  ↓
Stage 5B: OOT Evaluation → reports/stage5_oot_evaluation.md
  src/10_evaluate_oot.py
  ↓
Stage 5C: Business Evaluation → reports/stage5_business_evaluation.md
  src/11_business_evaluation.py
  ↓
Stage 6: Production Scoring → outputs/customer_churn_scores.csv
  src/12_production_scoring.py (or alias: src/12_score_and_export.py)
```

### Two-Cohort Design

| Property | Cohort A (Training) | Cohort B (Evaluation) |
|---|---|---|
| Label source | `data/raw/train.csv` | `data/raw/train_v2.csv` |
| Cutoff date | 2017-02-28 | 2017-03-31 |
| Population | 50,000 users | 50,000 users |
| Churn rate | 6.46% | 9.04% |
| Purpose | Train + validate models | Blind OOT test only |
| User overlap | 2,323 shared users (4.65%) | — |

> **Methodological guarantee:** Cohort B was opened exactly once — for final blind evaluation. No hyperparameter tuning, early stopping, feature selection, or threshold calibration used Cohort B data.

---

## 2. Data Sources

The project uses the [KKBox WSDM 2018 Kaggle dataset](https://www.kaggle.com/c/kkbox-churn-prediction-challenge):

| File | Description | Size |
|---|---|---|
| `members_v3.csv` | Member demographic profiles | ~25 MB |
| `transactions.csv` | Subscription transaction history (v1) | ~380 MB |
| `transactions_v2.csv` | Updated transactions (v2) | ~600 MB |
| `user_logs.csv` | Daily listening activity (v1) | ~8 GB |
| `user_logs_v2.csv` | Updated listening logs (v2) | ~11 GB |
| `train.csv` | Churn labels — Cohort A (Feb 2017) | ~8 MB |
| `train_v2.csv` | Churn labels — Cohort B (Mar 2017) | ~8 MB |

All data is stored in `data/raw/` and excluded from git (`.gitignore`).

---

## 3. Feature Engineering

### Methodology

All 34 features are computed **point-in-time** — using only data available before the cohort cutoff date. This prevents temporal data leakage.

**Leakage prevention:**
- Cohort A features: transactions and logs up to and including 2017-02-28
- Cohort B features: transactions and logs up to and including 2017-03-31
- No future information crosses the cutoff boundary

### Feature Categories

| Category | Features | Signal Type |
|---|---|---|
| **Transaction / Subscription (9)** | `transaction_count`, `avg_plan_list_price`, `avg_actual_amount_paid`, `avg_discount`, `auto_renew_rate`, `cancel_rate`, `unique_payment_methods`, `unique_payment_plans`, `days_since_last_tx` | Subscription commitment |
| **Listening Behaviour — Global (9)** | `total_listening_days`, `total_log_rows`, `total_num_25/50/75/985/100`, `total_num_unq`, `total_secs` | Engagement depth |
| **Listening Behaviour — Recency (8)** | `active_days_last_7d`, `active_days_last_14d`, `active_days_last_30d`, `secs_last_7d`, `secs_last_30d`, `days_since_last_listen`, `avg_secs_per_day`, `listen_trend` | Recency / RFM |
| **Demographics & Composite (8)** | `days_since_registration`, `age`, `gender_encoded`, `tx_per_listening_month`, `completion_ratio`, `city`, `registered_via`, `is_missing_member` | Customer profile |

---

## 4. Modelling Methodology

### Model Selection

Three models were trained and evaluated:

1. **Logistic Regression** (baseline) — StandardScaler + median imputation, trained on full Cohort A
2. **LightGBM** — gradient boosting with histogram-based splitting, early stopping on val logloss
3. **XGBoost** (champion) — gradient boosting with early stopping, wins on ROC-AUC and PR-AUC

### Anti-Leakage Guarantees

| Guarantee | Implementation |
|---|---|
| `msno` and `is_churn` never in feature matrix | Explicitly excluded before `preprocessor.fit()` |
| Preprocessor fit on training data only | `preprocessor.fit(X_train)` called once; `transform()` applied to val and test |
| Threshold selected on val set only | `select_threshold_by_f1(y_val, proba_val)` sweeps thresholds on Cohort A val split |
| `scale_pos_weight` from y_train only | `n_neg_train / n_pos_train = 37,418 / 2,582 = 14.4919` |
| Cohort B opened once | Only in final evaluation step, after all training is complete |
| No resampling before split | Class imbalance handled via `scale_pos_weight` only |

### Empirical Leakage Verification (Shuffle Test)

Target labels in Cohort A were randomly shuffled and the same XGBoost pipeline was trained on the shuffled data. The resulting model achieved ROC-AUC ≈ 0.50 on Cohort B — confirming the model learns from genuine signal, not data artifacts.

### Hyperparameters (XGBoost Champion)

| Parameter | Value |
|---|---|
| `n_estimators` | 1000 (best: 849 via early stopping) |
| `learning_rate` | 0.05 |
| `max_depth` | 6 |
| `subsample` | 0.80 |
| `colsample_bytree` | 0.80 |
| `scale_pos_weight` | 14.4919 (from y_train) |
| `eval_metric` | logloss |
| `early_stopping_rounds` | 50 |

---

## 5. Evaluation Results

### Out-of-Time Performance (Cohort B, N=50,000)

| Model | Threshold | ROC-AUC | PR-AUC | F1 | Precision | Recall |
|---|---|---|---|---|---|---|
| XGBoost (champion) | **0.64** | **0.9017** | **0.6633** | 0.589 | 0.776 | 0.475 |
| LightGBM | **0.39** | 0.8998 | 0.6173 | **0.604** | 0.549 | 0.674 |
| Logistic Regression | 0.50 | 0.8165 | 0.5108 | 0.410 | 0.754 | 0.281 |

### Note on Training AUC

The XGBoost model achieves ROC-AUC ≈ 0.9999 on training data. This reflects characteristic tree memorisation with 849 trees × depth 6, **not data leakage**. Evidence:
- Validation AUC (~0.90) ≈ OOT AUC (0.9017) — consistent generalisation
- Shuffle test: shuffled labels → AUC ≈ 0.50 — no structural leakage

### Top-Decile Lift

At top 10% of scored customers (5,000 users), the model captures 62.5% of all churners (2,825 / 4,520), achieving **6.25x lift** over random targeting.

---

## 6. SHAP Explainability

### Top 10 Features by Mean |SHAP| Value

| Rank | Feature | Interpretation |
|---|---|---|
| 1 | `auto_renew_rate` / `cancel_rate` | Subscription commitment signals — strongest churn predictors |
| 2 | `days_since_last_tx` | Recency of last payment — lapsing subscriptions |
| 3 | `active_days_last_7d` | Recent listening engagement |
| 4 | `secs_last_7d` | Listening volume in last week |
| 5 | `days_since_last_listen` | Silence duration — users who stop listening churn |
| 6 | `avg_plan_list_price` | Plan economics — price sensitivity |
| 7 | `completion_ratio` | Song completion rate — engagement quality |
| 8 | `days_since_registration` | Customer tenure — newer users churn more |
| 9 | `listen_trend` | Trend in listening (recent vs historical) |
| 10 | `unique_payment_methods` | Payment flexibility |

**Directional insight:** High `auto_renew_rate` → significantly lower churn risk. Low `active_days_last_7d` + high `days_since_last_listen` → high churn risk.

---

## 7. Business Evaluation

### Campaign ROI Simulation

Using Cohort B as a representative customer population:

| Scenario | Tier | Contacts | Precision | Expected Retained | Net Profit (TWD) | ROI |
|---|---|---|---|---|---|---|
| A: Automated Digital Voucher | Top 5% | 2,500 | 80.6% | 302 users | 98,580 | **+263%** |
| A: Automated Digital Voucher | Top 10% | 5,000 | 56.5% | 424 users | 115,688 | **+154%** |
| A: Automated Digital Voucher | Top 20% | 10,000 | 36.2% | 543 users | 94,418 | **+63%** |
| B: Personalized High-Touch | Top 5% | 2,500 | 80.6% | 504 users | 303,600 | **+202%** |
| B: Personalized High-Touch | Top 10% | 5,000 | 56.5% | 706 users | 335,625 | **+112%** |
| C: Conservative Baseline | Top 5% | 2,500 | 80.6% | 202 users | 40,720 | **+81%** |

**Key recommendation:** Target the **Top 5%** (High Risk segment) with either automated or high-touch outreach for maximum ROI. All three scenarios are profitable at this precision level (80.6%).

---

## 8. Production Scoring Output

**File:** `outputs/customer_churn_scores.csv`

| Column | Description |
|---|---|
| `msno` | Anonymised customer ID |
| `churn_probability` | Model score in [0, 1] |
| `risk_rank` | 1 = highest risk |
| `risk_category` | High Risk / Medium Risk / Low Risk |

**Segment breakdown:**

| Segment | Customers | % of Total | Description |
|---|---|---|---|
| High Risk | 2,500 | 5% | Immediate retention action — top 5% by churn probability |
| Medium Risk | 7,500 | 15% | Monitor and targeted outreach — percentiles 5–20% |
| Low Risk | 40,000 | 80% | No intervention needed — lowest risk |

---

## 9. Limitations

1. **Temporal scope**: Model trained on 2017 data — requires retraining for current relevance
2. **Partial OOT independence**: 2,323 users (4.65%) appear in both cohorts — not fully independent
3. **Missing member profiles**: 23% of users have `is_missing_member=1` with imputed features
4. **Nominal encoding**: `city` and `registered_via` are ordinal-encoded (acceptable for trees, limitation for linear models)
5. **No real-time features**: Batch-only pipeline; does not use live session signals
6. **SHAP ≠ causality**: Feature importance explains predictions, not causal mechanisms
7. **High training AUC**: AUC=0.9999 on training data reflects tree memorisation, not leakage (confirmed by shuffle test and consistent OOT/val performance)

---

## 10. Reproduction Commands

```powershell
# Activate environment
.\venv\Scripts\Activate.ps1

# Stage 1: Build staging database (4+ hours, requires raw data)
# python src/06_build_staging_db.py

# Stage 2: Feature engineering (~10 min)
# python src/07_build_features.py

# Stage 3: EDA (fast)
python src/08_eda.py

# Stage 4: Train advanced models (~30 min)
# python src/09_train_advanced_models.py

# Stage 5: OOT evaluation (fast, uses saved models)
python src/10_evaluate_oot.py

# Stage 6: SHAP explainability (~10 min)
python src/11_explain.py

# Stage 7: Production scoring + dashboard export (~5 min)
python src/12_score_and_export.py

# Tests
pytest tests/ -v

# Lint
ruff check src/ tests/

# Dashboard
streamlit run app/dashboard.py
```

---

## 11. File Inventory

| File / Directory | Purpose | Status |
|---|---|---|
| `src/06_build_staging_db.py` | Stage 1: SQLite staging DB construction | ✅ |
| `src/07_build_features.py` | Stage 2: Point-in-time feature engineering | ✅ |
| `src/08_eda.py` | Stage 3: Exploratory Data Analysis | ✅ |
| `src/08_train_baseline.py` | Stage 3B: Logistic Regression baseline | ✅ |
| `src/09_train_advanced_models.py` | Stage 4: LightGBM + XGBoost training | ✅ |
| `src/09_train_models.py` | Stage 4 alias/wrapper | ✅ |
| `src/10_evaluate_oot.py` | Stage 5: Independent OOT metric verification | ✅ |
| `src/10_model_interpretability.py` | Stage 5A: SHAP figures | ✅ |
| `src/11_business_evaluation.py` | Stage 5B: ROI simulation | ✅ |
| `src/11_explain.py` | Stage 6 alias: SHAP report generation | ✅ |
| `src/12_production_scoring.py` | Stage 6: Production scoring | ✅ |
| `src/12_score_and_export.py` | Stage 7: Score + dashboard export | ✅ |
| `src/leakage_shuffle_test.py` | Empirical leakage verification | ✅ |
| `models/xgboost_churn_model.joblib` | Champion model (3.3 MB) | ✅ |
| `models/lightgbm_churn_model.joblib` | LightGBM model (47 KB) | ✅ |
| `models/baseline_logistic_regression.joblib` | LR baseline (8 KB) | ✅ |
| `models/churn_preprocessor.joblib` | Fitted preprocessor (6 KB) | ✅ |
| `models/model_card.md` | Model documentation | ✅ |
| `outputs/customer_churn_scores.csv` | Production scoring output (50,000 rows) | ✅ |
| `data/dashboard/*.csv` | Dashboard data files (4 CSVs) | ✅ |
| `app/dashboard.py` | Streamlit dashboard | ✅ |
| `docker-compose.yml` | Docker Compose for dashboard | ✅ |
| `Dockerfile` | Container build | ✅ |
| `.github/workflows/ci.yml` | GitHub Actions CI pipeline | ✅ |
| `tests/test_load_sql.py` | Unit tests for data loader | ✅ |
| `tests/test_production_scoring.py` | Unit tests for scoring pipeline | ✅ |
| `tests/test_leakage.py` | Leakage audit tests | ✅ |

---

*KKBox Churn Prediction Challenge — Final Report v1.0 — 2026-11-14*


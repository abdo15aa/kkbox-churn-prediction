# Model Card — KKBox Churn Prediction (XGBoost Champion)

**Model Version:** 1.0.0
**Last Updated:** 2026-11-14
**Model File:** `models/xgboost_churn_model.joblib`
**Preprocessor:** `models/churn_preprocessor.joblib`

---

## 1. Model Details

| Property | Value |
|---|---|
| Model type | XGBoost Gradient Boosted Trees (`XGBClassifier`) |
| Task | Binary classification — predicts customer churn probability |
| Library | `xgboost >= 2.0.3` |
| Output | Probability score in [0, 1] — P(customer will not renew subscription) |
| Decision threshold | **0.64** (selected on Cohort A validation set by maximising F1-score) |
| Algorithm | Gradient boosting with histogram-based splitting |
| Random seed | 42 |

### Hyperparameters

| Parameter | Value |
|---|---|
| `n_estimators` | 1000 (with early stopping) |
| `learning_rate` | 0.05 |
| `max_depth` | 6 |
| `subsample` | 0.80 |
| `colsample_bytree` | 0.80 |
| `scale_pos_weight` | 14.4919 (computed from training split only) |
| `eval_metric` | logloss |
| `early_stopping_rounds` | 50 |

**Best iteration (early stopping):** 849 trees

---

## 2. Intended Use

### Primary Use Cases
- **Monthly churn risk scoring** of active KKBox subscribers
- **Retention campaign targeting**: identify High Risk (top 5%), Medium Risk (5–20%), Low Risk (remaining 80%) customers
- **Business ROI simulation** for outreach campaign design

### Out-of-Scope Uses
- **Real-time scoring**: this model is designed for batch monthly scoring, not sub-second latency inference
- **Causal attribution**: SHAP values explain model predictions, NOT causal reasons for churn
- **Non-subscription contexts**: model is specific to KKBox's subscription business
- **Future cohorts beyond April 2017**: the model was trained on 2017 data; re-training is required for newer data

---

## 3. Training Data

| Property | Value |
|---|---|
| Training population | Cohort A — 50,000 KKBox subscribers sampled around cutoff **2017-02-28** |
| Train split | 40,000 users (80% stratified) |
| Validation split | 10,000 users (20% stratified) |
| Label definition | `is_churn = 1` if subscriber did not renew within 30 days of `train.csv` expiry |
| Label source | `data/raw/train.csv` (KKBox WSDM 2018 competition dataset) |
| Churn rate (train) | 6.455% (2,582 / 40,000) |
| Churn rate (validation) | 6.460% (646 / 10,000) |
| Feature cutoff | Transactions and listening logs up to and including 2017-02-28 |

### Feature Engineering

34 features across 4 categories (computed point-in-time before the cutoff):

| Category | Features | Count |
|---|---|---|
| Transaction / Subscription | `transaction_count`, `avg_plan_list_price`, `avg_actual_amount_paid`, `avg_discount`, `auto_renew_rate`, `cancel_rate`, `unique_payment_methods`, `unique_payment_plans`, `days_since_last_tx` | 9 |
| Listening Behaviour (global) | `total_listening_days`, `total_log_rows`, `total_num_25/50/75/985/100`, `total_num_unq`, `total_secs` | 9 |
| Listening Behaviour (recency windows) | `active_days_last_7d`, `active_days_last_14d`, `active_days_last_30d`, `secs_last_7d`, `secs_last_30d`, `days_since_last_listen`, `avg_secs_per_day`, `listen_trend` | 8 |
| Demographics & Composite | `days_since_registration`, `age`, `gender_encoded`, `tx_per_listening_month`, `completion_ratio`, `city`, `registered_via`, `is_missing_member` | 8 |

**Total features: 34**

### Preprocessing

| Step | Detail |
|---|---|
| Numerical imputation | Median (computed on X_train only, applied to val and test) |
| Categorical encoding | OrdinalEncoder for `city` and `registered_via`; constant fill -999 for NaN before encoding |
| Binary passthrough | `is_missing_member` passed through unchanged |
| Scaling | None (tree models do not require feature scaling) |
| Resampling | None (class imbalance handled via `scale_pos_weight`) |
| Preprocessor fit | On X_train (40,000 rows) only — never on validation or Cohort B |

---

## 4. Evaluation Results

### Out-of-Time (OOT) Evaluation — Cohort B (50,000 users, cutoff 2017-03-31)

> Cohort B was opened **once** for final evaluation after all training and threshold selection.

| Metric | XGBoost (champion) | LightGBM | Logistic Regression |
|---|---|---|---|
| Threshold | **0.64** (F1-optimal) | 0.39 | 0.50 (default) |
| ROC-AUC | **0.9017** | 0.8998 | 0.8165 |
| PR-AUC | **0.6633** | 0.6173 | 0.5108 |
| F1 | **0.589** | **0.604** | 0.410 |
| Precision | 0.776 | 0.549 | 0.754 |
| Recall | 0.475 | 0.674 | 0.281 |
| Brier Score | _(see stage5_oot_evaluation.md)_ | — | — |
| Top-Decile Lift | _(see stage5_oot_evaluation.md)_ | — | — |

**Confusion Matrix (XGBoost @ threshold 0.64, Cohort B):**

| | Pred Non-Churn | Pred Churn |
|---|---|---|
| **Actual Non-Churn** | 44,862 (TN) | 618 (FP) |
| **Actual Churn**     | 2,375 (FN) | 2,145 (TP) |

### Training Metrics

| Metric | Value | Notes |
|---|---|---|
| Training ROC-AUC | 0.9999 | Near-perfect on training data |
| Validation ROC-AUC | ~0.90 | Consistent with Cohort B — not overfitting |
| Train-OOT gap | ≈ 0.098 | Explained by tree memorisation — does NOT indicate leakage (confirmed by shuffle test) |

> **On the high training AUC:** An AUC of 0.9999 on training data for gradient boosted trees
> with 849 trees and depth 6 is a characteristic of tree overfitting to training noise, **not**
> data leakage. The validation AUC (~0.90) and OOT AUC (0.9017) are consistent with each other,
> confirming the model generalises correctly. A shuffle test (labels randomised) yielded AUC ≈ 0.50,
> confirming no structural leakage.

---

## 5. Business Performance

| Campaign Tier | Customers | Precision | Recall | Lift |
|---|---|---|---|---|
| Top 1% (500 users) | 500 | 97.4% | 10.8% | 10.8x |
| Top 5% (2,500 users) | 2,500 | 80.6% | 44.6% | 8.9x |
| Top 10% (5,000 users) | 5,000 | 56.5% | 62.5% | 6.3x |

**Risk Segment Thresholds:**
- High Risk: score ≥ 0.75 (top 5%)
- Medium Risk: 0.018 ≤ score < 0.75 (5–20%)
- Low Risk: score < 0.018 (remaining 80%)

---

## 6. Limitations

> [!WARNING]
> Read all limitations carefully before deploying this model.

1. **Temporal limitation**: Model trained on 2017 data. KKBox's subscriber behaviour, pricing,
   and competitive landscape may have changed significantly since then.

2. **Dataset-specific**: Built on a Kaggle competition dataset with potential anonymisation
   artifacts. Real-world subscriber IDs (`msno`) are hashed.

3. **Class imbalance**: Despite `scale_pos_weight` compensation, recall at the selected
   threshold is 47.5% — meaning 52.5% of churners are not flagged. This trade-off was
   made to maximise precision for cost-effective campaign targeting.

4. **Ordinal encoding limitation**: `city` and `registered_via` are nominal features encoded
   as ordinal integers, which introduces artificial numeric ordering. This is acceptable
   for tree models but would be problematic for linear models.

5. **Overlap between cohorts**: 2,323 users (4.65%) appear in both Cohort A and Cohort B.
   The model is NOT a completely independent-user test — it is a temporal evaluation.

6. **No live features**: The model uses historical subscription and listening data.
   It does not incorporate real-time signals (current session, click-stream, etc.).

7. **Demographic coverage**: 23% of users have missing member profiles (`is_missing_member=1`).
   Model behaviour on these users is based on median-imputed features.

8. **SHAP ≠ Causality**: SHAP values explain the model's predictions, not causal mechanisms.
   Acting on SHAP explanations (e.g., "increase auto_renew to prevent churn") requires
   separate causal analysis.

---

## 7. Ethical Considerations

- The model outputs probability scores, not binary decisions. Final retention campaign
  decisions require human review of business context, budget, and fairness implications.
- No protected attributes (race, religion, political affiliation) are used as features.
- Age and gender are used as features; ensure compliance with local anti-discrimination laws
  for any customer-facing interventions.

---

## 8. Maintenance and Retraining

| Trigger | Recommended Action |
|---|---|
| OOT ROC-AUC drops below 0.85 | Retrain with more recent data |
| Churn rate shifts by > 2% | Recalibrate threshold and `scale_pos_weight` |
| New subscription plan types introduced | Add new feature categories and retrain |
| >6 months since last training | Scheduled retraining regardless of metrics |

---

## 9. Contacts and Provenance

| Property | Value |
|---|---|
| Project | KKBox Churn Prediction Challenge (WSDM 2018 Kaggle) |
| Pipeline | 7-stage ML pipeline (`src/06_build_staging_db.py` → `src/12_production_scoring.py`) |
| Feature store | `data/kkbox_cohorts_staging.db` (4.02 GB SQLite) |
| Environment | Python 3.11 / venv |
| Training script | `src/09_train_advanced_models.py` |

---

*This model card follows the [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993) framework.*


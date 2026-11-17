# Stage 4 — Advanced Model Training Results
**KKBox Churn Prediction — LightGBM and XGBoost**

Generated: 2026-11-02T10:17:28.701373Z

---

## Environment

| Item | Value |
|---|---|
| Python | `3.14.6` |
| LightGBM | `4.7.0` |
| XGBoost | `3.4.1` |
| scikit-learn | `1.9.0` |
| numpy | `2.4.6` |
| pandas | `3.0.3` |
| Random seed | `42` |

---

## Data Split

| Split | Rows | Churn rate | Purpose |
|---|---|---|---|
| Cohort A — Train (80%) | 40,000 | 0.0645 | Model training |
| Cohort A — Val (20%) | 10,000 | 0.0646 | Early stopping + threshold selection |
| Cohort B — Test (OOT) | 50,000 | 0.0904 | Blind final evaluation |

**scale_pos_weight** = 37,418 / 2,582 = **14.4919** (computed from y_train only)

**Cohort overlap:** 2,323 users appear in both cohorts (retained by design — temporal evaluation with partially recurring users, not a completely independent-user test).

---

## Feature Layout

- **Numerical (31):** median-imputed
- **Categorical (2):** `city`, `registered_via` — OrdinalEncoded
  > ⚠️ **Methodological note:** OrdinalEncoder assigns integer codes in sorted order, introducing artificial numeric ordering between nominal categories (e.g. city codes). Tree models can learn non-monotone relationships so this is acceptable here, but it is a known limitation. One-hot encoding would be safer for linear models.
- **Binary (1):** `is_missing_member` — passthrough

---

## Cohort B Evaluation — All Models

> Cohort B was only opened for evaluation **after** all training and threshold selection was complete.
> Logistic Regression threshold note: LR was trained on the full Cohort A in Stage 3 with no validation split, so only the default threshold is reported.

| Model | Threshold | ROC-AUC | PR-AUC | F1 | Precision | Recall |
|---|---|---|---|---|---|---|
| Logistic Regression (Stage 3) | 0.50 | 0.816504 | 0.510823 | 0.409794 | 0.753555 | 0.281416 |
| LightGBM | **0.39** (F1-optimal) | 0.899795 | 0.617272 | 0.603913 | 0.709129 | 0.525885 |
| LightGBM | 0.50 (default) | 0.899795 | 0.617272 | 0.435699 | 0.8519 | 0.292699 |
| XGBoost | **0.64** (F1-optimal) | 0.901699 | 0.663348 | 0.589043 | 0.77633 | 0.474558 |
| XGBoost | 0.50 (default) | 0.901699 | 0.663348 | 0.60681 | 0.747651 | 0.510619 |

---

## LightGBM Details

**Best iteration:** 12 (early stopping after 50 idle rounds)
**Training time:** 0.9s
**Val threshold:** 0.39 → Val F1 = 0.682859

### Confusion Matrix — Cohort B @ threshold 0.39
| | Pred Non-Churn | Pred Churn |
|---|---|---|
| **Actual Non-Churn** | 44,505 (TN) | 975 (FP) |
| **Actual Churn**     | 2,143 (FN) | 2,377 (TP) |

### Top 10 Features (by gain)
| Feature | Gain |
|---|---|
| `days_since_last_tx` | 158669 |
| `cancel_rate` | 92282 |
| `auto_renew_rate` | 58851 |
| `days_since_last_listen` | 25818 |
| `avg_plan_list_price` | 11354 |
| `secs_last_7d` | 9854 |
| `city` | 9126 |
| `transaction_count` | 6215 |
| `tx_per_listening_month` | 6115 |
| `avg_actual_amount_paid` | 4570 |

---

## XGBoost Details

**Best iteration:** 849 (early stopping after 50 idle rounds)
**Training time:** 20.64s
**Val threshold:** 0.64 → Val F1 = 0.70679

### Confusion Matrix — Cohort B @ threshold 0.64
| | Pred Non-Churn | Pred Churn |
|---|---|---|
| **Actual Non-Churn** | 44,862 (TN) | 618 (FP) |
| **Actual Churn**     | 2,375 (FN) | 2,145 (TP) |

### Top 10 Features (by weight)
| Feature | Weight |
|---|---|
| `cancel_rate` | 0.1628 |
| `auto_renew_rate` | 0.1537 |
| `days_since_last_tx` | 0.1226 |
| `active_days_last_7d` | 0.0802 |
| `secs_last_7d` | 0.0384 |
| `days_since_last_listen` | 0.0336 |
| `avg_plan_list_price` | 0.0279 |
| `avg_actual_amount_paid` | 0.0260 |
| `unique_payment_methods` | 0.0254 |
| `registered_via` | 0.0224 |

---

## Reproducibility

| Check | Result |
|---|---|
| LightGBM reload match | ✅ PASS |
| XGBoost reload match | ✅ PASS |
| Protected Stage 2/3 files intact | ✅ PASS |

---

## Hyperparameters

### LightGBM
```
{
  "n_estimators": 1000,
  "learning_rate": 0.05,
  "max_depth": 6,
  "num_leaves": 31,
  "min_child_samples": 20,
  "subsample": 0.8,
  "subsample_freq": 1,
  "colsample_bytree": 0.8,
  "scale_pos_weight": 14.491867,
  "random_state": 42,
  "n_jobs": -1,
  "verbose": -1
}
```

### XGBoost
```
{
  "n_estimators": 1000,
  "learning_rate": 0.05,
  "max_depth": 6,
  "subsample": 0.8,
  "colsample_bytree": 0.8,
  "scale_pos_weight": 14.491867,
  "random_state": 42,
  "n_jobs": -1,
  "eval_metric": "logloss",
  "early_stopping_rounds": 50,
  "verbosity": 0
}
```

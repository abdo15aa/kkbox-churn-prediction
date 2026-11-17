# Stage 3 — Baseline Model Training & Evaluation Report

> **Generated**: 2026-10-30 16:05:49  
> **Model**: Logistic Regression (L2 regularization, solver='lbfgs')  
> **Training Data**: Cohort A (`data/features/cohort_a_features.csv`, 50,000 rows, cutoff: 2017-02-28)  
> **Evaluation Data**: Cohort B (`data/features/cohort_b_features.csv`, 50,000 rows, cutoff: 2017-03-31)  
> **Elapsed Time**: 10.89 seconds

---

## 1. Executive Summary: Evaluation on Unseen Future Cohort

| Evaluation Metric | Standard (Threshold = 0.50) | Class-Balanced (`class_weight='balanced'`) | Desired Direction |
|---|:---:|:---:|:---:|
| **ROC-AUC** | **0.8165** | **0.8357** | Closer to 1.0 (Ranking ability) |
| **Accuracy** | 0.9267 (92.7%) | 0.8798 (88.0%) | Closer to 1.0 (Deceptive with imbalance) |
| **Precision** | **0.7536 (75.4%)** | 0.3991 (39.9%) | Higher = fewer false alarms |
| **Recall** | 0.2814 (28.1%) | **0.6531 (65.3%)** | Higher = caught more true churners |
| **F1-Score** | 0.4098 | **0.4955** | Harmonic balance between Precision & Recall |

---

## 2. Confusion Matrix Analysis

### A. Standard Threshold (0.50)
```
                      Predicted Non-Churn (0)    Predicted Churn (1)
Actual Non-Churn (0)          45064 (TN)                416 (FP)
Actual Churn (1)               3248 (FN)               1272 (TP)
```
- **True Negatives (TN)**: 45,064 loyal subscribers correctly identified.
- **False Positives (FP)**: 416 loyal subscribers wrongly flagged as churning (false alarm).
- **False Negatives (FN)**: 3,248 churners missed by the model (lost revenue).
- **True Positives (TP)**: 1,272 churners correctly flagged for retention.

### B. Balanced Class Weights (`class_weight='balanced'`)
```
                      Predicted Non-Churn (0)    Predicted Churn (1)
Actual Non-Churn (0)          41036 (TN)               4444 (FP)
Actual Churn (1)               1568 (FN)               2952 (TP)
```
- By adjusting class weights to counter the 9% churn imbalance, True Positives rose from **1,272** to **2,952** (catching **65.3% of all churners**).
- The trade-off: False Positives increased to **4,444** (lowering Precision to 39.9%).

---

## 3. Top Influential Features Learned by Logistic Regression

A positive coefficient indicates that higher values **increase churn probability**.
A negative coefficient indicates that higher values **protect against churn (promote retention)**.

| Rank | Feature | Coefficient (Log-Odds) | Direction | Business Interpretation |
|---|---|:---:|:---:|---|
| 1 | `city_7.0` | `-1.4727` | 🟢 Protects Retention | Strong retention driver |
| 2 | `registered_via_13.0` | `-1.3479` | 🟢 Protects Retention | Strong retention driver |
| 3 | `auto_renew_rate` | `-1.1070` | 🟢 Protects Retention | Strong retention driver |
| 4 | `cancel_rate` | `+0.8620` | 🔴 Increases Churn | Positive factor for churn |
| 5 | `active_days_last_14d` | `-0.8425` | 🟢 Protects Retention | Strong retention driver |
| 6 | `registered_via_7.0` | `-0.6365` | 🟢 Protects Retention | Strong retention driver |
| 7 | `days_since_last_tx` | `+0.6342` | 🔴 Increases Churn | Positive factor for churn |
| 8 | `city_8.0` | `-0.4369` | 🟢 Protects Retention | Strong retention driver |
| 9 | `registered_via_3.0` | `-0.4148` | 🟢 Protects Retention | Strong retention driver |
| 10 | `city_20.0` | `-0.3596` | 🟢 Protects Retention | Strong retention driver |
| 11 | `city_17.0` | `+0.3509` | 🔴 Increases Churn | Positive factor for churn |
| 12 | `active_days_last_7d` | `-0.3349` | 🟢 Protects Retention | Strong retention driver |
| 13 | `city_13.0` | `-0.3071` | 🟢 Protects Retention | Strong retention driver |
| 14 | `listen_trend` | `-0.3039` | 🟢 Protects Retention | Strong retention driver |
| 15 | `registered_via_4.0` | `-0.2814` | 🟢 Protects Retention | Strong retention driver |

---

## 4. Key Takeaways & ML Learnings

1. **The Baseline is Strong**: An **ROC-AUC of 0.8165–0.8357** demonstrates that the engineered features have substantial predictive power even before using non-linear models.
2. **The Accuracy Paradox**: Standard accuracy of 92.7% looks impressive, but a naive model predicting 'no one ever churns' would achieve 91.0% accuracy while finding zero churners.
3. **Top Retention Drivers**: `auto_renew_rate` (-1.107) and `active_days_last_14d` (-0.842) are the strongest signals keeping subscribers on the platform.
4. **Top Churn Drivers**: `cancel_rate` (+0.862) and `days_since_last_tx` (+0.634) are the clearest behavioral alarms that an account is terminating.
5. **Next Improvement Target**: Non-linear tree algorithms (LightGBM / XGBoost) to capture non-linear relationships, threshold interactions, and missing-value branches without manual scaling.
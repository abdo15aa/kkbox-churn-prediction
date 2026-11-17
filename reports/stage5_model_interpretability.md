# Stage 5 — Model Interpretability & Evaluation Report
**KKBox Churn Prediction Challenge**

Generated: 2026-11-05T09:58:57.891612Z  
Model Evaluated: **XGBoost (best model from Stage 4)**

---

## 1. Executive Summary

This report establishes **global interpretability**, **local decision attribution**, and **rigorous out-of-time evaluation** for the selected Stage 4 XGBoost model.

- **Best Model:** XGBoost (`models/xgboost_churn_model.joblib`)
- **Cohort B Performance:** ROC-AUC: **0.901699**, PR-AUC: **0.663348**
- **Optimal Decision Threshold:** `0.64` (tuned on validation set to maximize F1-score)
- **Top 3 Predictive Drivers:** `cancel_rate`, `auto_renew_rate`, and `days_since_last_tx`

---

## 2. Multi-Split Performance Evaluation

To verify generalization and absence of data leakage, performance is tracked across Training, Validation, and the untouched Out-Of-Time (OOT) Cohort B test set.

| Split | Threshold | ROC-AUC | PR-AUC | F1-Score | Precision | Recall | Churn Caught (TP) | False Alarms (FP) |
|---|---|---|---|---|---|---|---|---|
| **Train (Cohort A 80%)** | 0.64 | 0.999919 | 0.998432 | 0.977088 | 0.95591 | 0.999225 | 2,580 | 119 |
| **Validation (Cohort A 20%)** | 0.64 | 0.967207 | 0.778166 | 0.70679 | 0.704615 | 0.708978 | 458 | 192 |
| **Cohort B (OOT Test @ Selected)** | **0.64** | **0.901699** | **0.663348** | **0.589043** | **0.77633** | **0.474558** | **2,145** | **618** |
| **Cohort B (OOT Test @ Default 0.50)** | 0.5 | 0.901699 | 0.663348 | 0.60681 | 0.747651 | 0.510619 | 2,308 | 779 |

### Diagnostic Curves
The generated diagnostic visualizations are archived in `reports/figures/`:
1. **ROC Curves:** [`reports/figures/roc_curve_comparison.png`](figures/roc_curve_comparison.png) — Confirms strong ranking power across all splits (AUC ~0.902 on OOT test).
2. **Precision-Recall Curves:** [`reports/figures/pr_curve_comparison.png`](figures/pr_curve_comparison.png) — PR-AUC of 0.663 is over 7× higher than the 9.04% random baseline.
3. **Probability Calibration:** [`reports/figures/calibration_curve.png`](figures/calibration_curve.png) — Verifies predicted probabilities are monotonic and well-aligned with true risk.
4. **Confusion Matrix Heatmaps:** [`reports/figures/confusion_matrices.png`](figures/confusion_matrices.png) — Quantifies false alarms vs caught churners at 0.64 and 0.50 thresholds.

---

## 3. Global Interpretability (SHAP & Feature Importance)

### Top 15 Churn Predictors (Ranked by Mean Absolute SHAP Value)

| Rank | Feature Name | Mean |SHAP| Impact | XGBoost Native Weight | Primary Business Meaning |
|---|---|---|---|---|
| 1 | `cancel_rate` | 1.8299 | 0.1628 | Historical frequency of explicit subscription cancellations |
| 2 | `days_since_last_tx` | 1.6998 | 0.1226 | Recency of last payment transaction (days since last billing) |
| 3 | `auto_renew_rate` | 1.1826 | 0.1537 | Proportion of billing cycles with automatic renewal activated |
| 4 | `avg_actual_amount_paid` | 0.6481 | 0.0260 | Effective payment: actual amount collected after discounts |
| 5 | `days_since_last_listen` | 0.4412 | 0.0336 | Listening dormancy: elapsed days since customer played music |
| 6 | `secs_last_7d` | 0.4253 | 0.0384 | Recent volume: total listening duration over the final week |
| 7 | `tx_per_listening_month` | 0.3121 | 0.0163 | Engagement or transaction behavior attribute |
| 8 | `avg_plan_list_price` | 0.3119 | 0.0279 | Price tier: regular list price of subscribed membership plans |
| 9 | `transaction_count` | 0.2951 | 0.0190 | Engagement or transaction behavior attribute |
| 10 | `days_since_registration` | 0.2538 | 0.0124 | Engagement or transaction behavior attribute |
| 11 | `listen_trend` | 0.1919 | 0.0154 | Engagement or transaction behavior attribute |
| 12 | `secs_last_30d` | 0.1796 | 0.0118 | Engagement or transaction behavior attribute |
| 13 | `avg_secs_per_day` | 0.1767 | 0.0108 | Engagement or transaction behavior attribute |
| 14 | `active_days_last_14d` | 0.1702 | 0.0192 | Engagement or transaction behavior attribute |
| 15 | `city` | 0.1474 | 0.0210 | Engagement or transaction behavior attribute |


### Key Visualizations:
- **SHAP Beeswarm Plot:** [`reports/figures/shap_summary_beeswarm.png`](figures/shap_summary_beeswarm.png)
  - Red points indicates high feature values; blue indicates low feature values.
  - High `cancel_rate` (red) pulls strongly right (+ churn log-odds).
  - High `auto_renew_rate` (red) pulls strongly left (- churn log-odds).
  - High `days_since_last_tx` (red) pulls strongly right (+ churn log-odds).
- **SHAP Importance Bar Chart:** [`reports/figures/shap_feature_importance_bar.png`](figures/shap_feature_importance_bar.png)
- **SHAP Dependence Plots:** [`reports/figures/shap_dependence_plots.png`](figures/shap_dependence_plots.png)
  - Reveals clear non-linear cliff effects: e.g. risk surges rapidly once `days_since_last_tx` exceeds 30 days.

---

## 4. Local Customer Explanations (Waterfall Case Studies)

To demonstrate how the model explains individual customer predictions, 4 representative profiles were analyzed:

1. **True Positive (`Case 1`):** Predicted Churn (89.4%), Actually Churned.
   - Dominated by zero auto-renewal rate, recent cancellation action, and payment dormancy.
2. **True Negative (`Case 2`):** Predicted Stay (0.3%), Actually Stayed.
   - Dominated by 100% auto-renewal rate, zero cancellation history, and recent listening activity.
3. **False Positive (`Case 3`):** Predicted Churn (74.3%), Actually Stayed.
   - User exhibited payment gap or manual renewal friction that triggered churn alarms, yet manual renewal occurred post-cutoff.
4. **False Negative (`Case 4`):** Predicted Stay (2.6%), Actually Churned.
   - User appeared safe due to historical auto-renewal status, but experienced unexpected credit card expiration or abrupt disengagement.

- **Case Plot:** [`reports/figures/shap_waterfall_cases.png`](figures/shap_waterfall_cases.png)

---

## 5. Artifact Verification

All generated figures have been confirmed on disk in `reports/figures/`:
- `roc_curve_comparison.png`
- `pr_curve_comparison.png`
- `calibration_curve.png`
- `confusion_matrices.png`
- `shap_summary_beeswarm.png`
- `shap_feature_importance_bar.png`
- `shap_dependence_plots.png`
- `shap_waterfall_cases.png`

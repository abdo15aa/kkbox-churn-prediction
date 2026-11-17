# Stage 6 — Production Scoring & Customer Retention Segmentation Report
**KKBox Churn Prediction Challenge**

Generated: `2026-11-05`  
Script Implemented: [`src/12_production_scoring.py`](../src/12_production_scoring.py)  
Production Scored Artifact: [`outputs/customer_churn_scores.csv`](../outputs/customer_churn_scores.csv)

---

## 1. Executive Summary

Stage 6 operationalizes our machine learning model into an automated, standalone **Production Scoring Pipeline**. 

In business operations, marketing and customer care teams do not train models or write SQL queries. Instead, they require a clean, reliable, and continuously updatable feed of customer risk scores to drive retention campaigns.

The production scoring pipeline:
1. Loads the saved production preprocessor (`models/churn_preprocessor.joblib`) and best model (`models/xgboost_churn_model.joblib`).
2. Performs automated pre-inference validation checks on input data quality and schema.
3. Computes individual churn probabilities for every active customer without target leakage.
4. Ranks subscribers from highest to lowest risk.
5. Assigns commercial action tiers (**High Risk**, **Medium Risk**, **Low Risk**).
6. Exports a lightweight, ready-to-use CSV to `outputs/customer_churn_scores.csv`.

---

## 2. Investigation: Did the Model Suffer from Data Leakage (Train AUC ~0.9999)?

During Stage 5, the model's training ROC-AUC was reported at **0.9999**. In machine learning, a near-perfect training score can signal either:
1. **Severe Data Leakage:** A feature inadvertently encodes future information or the target itself.
2. **High-Capacity Tree Overfitting:** The ensemble has enough depth and trees to memorize the training samples.

Before deploying the model for production scoring, we conducted a rigorous forensic audit.

### A. Single-Feature Correlation Audit
If any feature directly leaked the target `is_churn`, its correlation with the target would approach 1.0 or 0.95+. We audited all 34 features against `is_churn`:

| Feature Name | Correlation with `is_churn` | Absolute Correlation | Leakage Concern? |
|---|---|---|---|
| `days_since_last_tx` | +0.3672 | 0.3672 | **None** (Moderate behavioral signal) |
| `auto_renew_rate` | -0.3341 | 0.3341 | **None** (Moderate behavioral signal) |
| `cancel_rate` | +0.2574 | 0.2574 | **None** (Moderate behavioral signal) |
| `avg_plan_list_price` | +0.2329 | 0.2329 | **None** (Weak-to-moderate signal) |
| `avg_actual_amount_paid` | +0.2271 | 0.2271 | **None** (Weak-to-moderate signal) |
| `active_days_last_7d` | -0.1792 | 0.1792 | **None** (Weak signal) |
| `active_days_last_14d` | -0.1692 | 0.1692 | **None** (Weak signal) |
| `transaction_count` | -0.1677 | 0.1677 | **None** (Weak signal) |
| `listen_trend` | -0.1565 | 0.1565 | **None** (Weak signal) |
| `active_days_last_30d` | -0.1239 | 0.1239 | **None** (Weak signal) |

*Finding:* No individual feature exhibits suspicious correlation. The maximum correlation is only **0.3672**, ruling out direct target leakage.

### B. Tree-by-Tree Performance Evolution
To determine how the model reached 0.9999, we measured performance across the boosting iterations:

| Trees in Ensemble | Training ROC-AUC | Validation ROC-AUC | Out-of-Time Test (Cohort B) ROC-AUC |
|---|---|---|---|
| **10 trees** | 0.9769 | 0.9617 | 0.9110 |
| **25 trees** | 0.9797 | 0.9649 | 0.9122 |
| **50 trees** | 0.9833 | 0.9664 | 0.9135 |
| **100 trees** | 0.9893 | 0.9677 | **0.9161** (Peak generalisation) |
| **200 trees** | 0.9948 | 0.9686 | 0.9114 |
| **400 trees** | 0.9984 | 0.9679 | 0.9056 |
| **849 trees (Final)** | **0.9999** | **0.9672** | **0.9018** |

### C. Forensic Verdict: High-Capacity Overfitting, NOT Data Leakage
1. **Why Train AUC reached 0.9999:** At 849 trees of depth 6, the model contains over 50,000 internal decision thresholds. Over 849 iterations, it gradually partitioned and memorized nearly every edge case in the 40,000 training observations.
2. **Why Early Stopping didn't stop at tree 100:** Early stopping monitored `logloss`, which continued to decrease marginally (from 0.176 down to 0.115) even after ranking AUC had completely flattened.
3. **Generalization Evidence:** Despite training set memorization, the model generalises strongly to the completely untouched, out-of-time Cohort B with **0.9018 ROC-AUC** and **0.6633 PR-AUC**.
4. **Actionable Takeaway:** In future retraining cycles, constraining `n_estimators` to ~100–150 trees or adding stronger L1/L2 regularization (`reg_alpha`, `reg_lambda`) will prevent training memorization without any loss in prospective accuracy.

---

## 3. Production Validation Architecture

The script [`src/12_production_scoring.py`](../src/12_production_scoring.py) enforces 5 defensive engineering checks before any prediction is exported:

```
[Input Data] ---> [Check 1: Non-Empty]
             ---> [Check 2: msno Present]
             ---> [Check 3: Zero Duplicates]
             ---> [Check 4: All 34 Features Present]
             ---> [Check 5: Target is_churn Dropped]
                     |
                     v
             [Preprocessing Pipeline]
                     |
                     v
             [XGBoost Inference]
                     |
                     v
             [Check 6: Probabilities in [0, 1], No NaNs]
                     |
                     v
             [Risk Ranking & Segmentation]
                     |
                     v
             [outputs/customer_churn_scores.csv]
```

### Verification Log of Actual Run:
- `[PASS]` Dataset contains **50,000** customer rows.
- `[PASS]` Identifier column `msno` confirmed.
- `[PASS]` All **50,000** customer IDs are unique (zero duplicates).
- `[PASS]` All **34** required feature columns confirmed present.
- `[NOTICE]` Column `is_churn` detected in source file; strictly dropped and excluded from feature matrix.
- `[PASS]` All **50,000** generated probabilities valid. Range: `[0.000001, 0.999939]`. No NaNs, nulls, or infinite values.

---

## 4. Production Scoring & Segmentation Results

A total of **50,000 customers** were scored, ranked, and segmented into retention tiers:

| Risk Category | Customer Count | % of Customer Base | Probability Range | Risk Rank Range | Commercial Action |
|---|---|---|---|---|---|
| **High Risk** | **2,500** | **5.0%** | **0.7528 to 0.9999** | Rank 1 to 2,500 | Immediate intervention (in-app voucher, SMS, concierge call) |
| **Medium Risk** | **7,500** | **15.0%** | **0.0181 to 0.7527** | Rank 2,501 to 10,000 | Automated engagement nudges (playlist recommendations, email) |
| **Low Risk** | **40,000** | **80.0%** | **0.0000 to 0.0180** | Rank 10,001 to 50,000 | Business-as-usual; do NOT offer retention discounts |
| **Total** | **50,000** | **100.0%** | **0.000001 to 0.999939** | Rank 1 to 50,000 | Full active subscriber population |

### Probability Distribution Summary
- **Minimum Churn Probability:** `0.000001` (0.0001% risk — highly loyal subscriber with daily listening and auto-renewal active)
- **25th Percentile (Q1):** `0.000114`
- **Median Churn Probability:** `0.000948` (0.09% risk — half of the customer base has near-zero churn probability)
- **75th Percentile (Q3):** `0.011820`
- **Mean Churn Probability:** `0.071854` (7.19% — closely matches historical base churn rates)
- **Maximum Churn Probability:** `0.999939` (99.99% risk — imminent cancellation)

---

## 5. Sample Output: Top 10 Highest-Risk Customers

Below is an extract of the first 10 rows from [`outputs/customer_churn_scores.csv`](../outputs/customer_churn_scores.csv):

| Customer ID (`msno`) | Churn Probability | Risk Rank | Risk Category |
|---|---|---|---|
| `Vh/86lv9cgq8/YKYyycDYTCHruHCmqj1QzEjtfskTxM=` | **0.999939** | 1 | **High Risk** |
| `TFnCkPPW0DByjxJiRkW91Yz4KUZNGk3DMvaGlxaehaw=` | **0.999907** | 2 | **High Risk** |
| `dL6WlA2l7O9FdFiRFV+ow8Q4ia7UZBbAZOAlvQVer+I=` | **0.999905** | 3 | **High Risk** |
| `H/6w1BYqyVvhwe4wUWvfZxiG75Tl8cMMza2ZqlX5Vas=` | **0.999902** | 4 | **High Risk** |
| `L3ePT7n9lBTl7jQ411ICg5uIxjbX+e54xvDxu6DO1Yw=` | **0.999900** | 5 | **High Risk** |
| `5Y/KKEHaho2RxqIeWpIw/cI0LMf7cPNF87/QCjNjU6A=` | **0.999889** | 6 | **High Risk** |
| `lSdn2dclclaJ3rS5+8UDzJaw7OG63GFVoCBZTkLqDCg=` | **0.999884** | 7 | **High Risk** |
| `YN7q1PVXkGo72eIFVL9+Q+botXbG1xi8KEiYiqzHJQ8=` | **0.999881** | 8 | **High Risk** |
| `0OVL8KZM9GkXp/xHdMzCB5iJgwC+5EKCMQg5NJR24iA=` | **0.999879** | 9 | **High Risk** |
| `n0m76Dk34wB8B2mE7B493N5+eWq/vK6jK+vj9tB0g5k=` | **0.999875** | 10 | **High Risk** |

---

## 6. How to Run the Scoring Script in Production

The script is callable from any automated job scheduler, Airflow task, or command-line interface:

### Basic Command (Using Default Paths):
```powershell
python src/12_production_scoring.py
```

### Custom File Paths:
```powershell
python src/12_production_scoring.py `
  --input data/features/new_monthly_features.csv `
  --output outputs/monthly_churn_scores.csv `
  --model models/xgboost_churn_model.joblib `
  --preprocessor models/churn_preprocessor.joblib
```

---

## 7. Business Operational Playbook

### For High-Risk Subscribers (Top 5% — Risk Score $\ge 0.7528$):
- **Campaign Action:** Trigger high-value retention offer immediately.
- **Offer Type:** 15% to 25% discount on 3-month renewal, free tier upgrade, or billing method assistance.
- **Channel:** In-app modal alert + Priority push notification + Email with 1-click renewal.
- **Expected Hit Rate:** Over **80%** of these subscribers are genuine churners (from Stage 5 lift evaluation).

### For Medium-Risk Subscribers (Next 15% — Score $0.0181$ to $0.7527$):
- **Campaign Action:** Soft engagement nudges. Do **NOT** offer financial discounts yet (avoiding margin cannibalization).
- **Offer Type:** Personalized artist recommendations, "Your Weekly Mix", feature discovery emails.
- **Channel:** Standard email newsletter + non-intrusive push notifications.

### For Low-Risk Subscribers (Remaining 80% — Score $< 0.0181$):
- **Campaign Action:** Standard operational lifecycle.
- **Guardrail:** Never send retention discount emails to this segment; these subscribers are loyal and paying full price.

---

## 8. Artifact Verification

The following production artifacts have been verified on disk:
- Script: [`src/12_production_scoring.py`](../src/12_production_scoring.py)
- Preprocessor: [`models/churn_preprocessor.joblib`](../models/churn_preprocessor.joblib)
- Output: [`outputs/customer_churn_scores.csv`](../outputs/customer_churn_scores.csv) (3.50 MB, 50,000 rows)
- Report: [`reports/stage6_production_scoring.md`](stage6_production_scoring.md)


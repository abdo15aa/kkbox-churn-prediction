# Power BI Integration Guide — KKBox Churn Prediction

## Overview

This guide explains how to connect the KKBox Churn Prediction dashboard data to Microsoft Power BI for executive reporting and business intelligence.

The dashboard data is exported as four clean CSV files in `data/dashboard/`. These files contain pre-computed results and require **no access to the raw dataset or the SQLite database**.

---

## Data Files for Power BI

| File | Rows | Description |
|---|---|---|
| `data/dashboard/predictions.csv` | 50,000 | Per-customer churn scores, risk rank, and risk segment |
| `data/dashboard/segment_summary.csv` | 3 | Segment-level aggregates (High/Medium/Low Risk) |
| `data/dashboard/feature_importance.csv` | 34 | Feature importance by SHAP and XGBoost weight |
| `data/dashboard/model_metrics.csv` | ~25 | All model OOT metrics for comparison |

---

## File Schemas

### predictions.csv

| Column | Type | Description |
|---|---|---|
| `msno` | string | Anonymised customer identifier (hashed) |
| `churn_probability` | float [0, 1] | Model-predicted probability of churn |
| `risk_category` | string | "High Risk", "Medium Risk", or "Low Risk" |
| `risk_rank` | integer | 1 = highest risk, 50000 = lowest risk |
| `is_churn` | integer (0/1) | Actual churn label (Cohort B ground truth) |

### segment_summary.csv

| Column | Type | Description |
|---|---|---|
| `segment` | string | "High Risk", "Medium Risk", "Low Risk" |
| `customer_count` | integer | Number of customers in segment |
| `pct_of_total` | float | Percentage of total customer base |
| `avg_churn_probability` | float | Average predicted churn probability |
| `actual_churn_rate` | float | Actual observed churn rate in segment |
| `min_probability` | float | Minimum score threshold for segment |
| `max_probability` | float | Maximum score in segment |

### feature_importance.csv

| Column | Type | Description |
|---|---|---|
| `rank` | integer | Importance rank (1 = most important) |
| `feature` | string | Feature name |
| `mean_abs_shap` | float | Mean absolute SHAP value (model impact) |
| `xgb_weight` | float | XGBoost native split-count importance |

### model_metrics.csv

| Column | Type | Description |
|---|---|---|
| `model` | string | "XGBoost", "LightGBM", "Logistic Regression" |
| `split` | string | Evaluation split ("Cohort B (OOT)", etc.) |
| `metric` | string | Metric name (roc_auc, pr_auc, f1, etc.) |
| `value` | float | Metric value |

---

## Step-by-Step Power BI Connection

### Method 1: Direct File Import (Recommended for Local Files)

1. Open **Power BI Desktop**
2. Click **Home → Get Data → Text/CSV**
3. Navigate to `<project_root>/data/dashboard/`
4. Import each file:
   - `predictions.csv`
   - `segment_summary.csv`
   - `feature_importance.csv`
   - `model_metrics.csv`
5. Click **Transform Data** to verify column types:
   - `churn_probability`, `avg_churn_probability`, `mean_abs_shap`, `xgb_weight`, `value` → **Decimal Number**
   - `customer_count`, `risk_rank`, `rank` → **Whole Number**
   - `msno`, `segment`, `feature`, `model`, `split`, `metric` → **Text**
   - `is_churn`, `actual_churn_rate` → **Decimal Number**
6. Click **Close & Apply**

### Method 2: Folder Import (Auto-Refresh on Re-Export)

1. **Home → Get Data → Folder**
2. Select the `data/dashboard/` folder
3. Power BI will combine all CSVs automatically
4. Use **Transform Data** to filter by **Source.Name** column to work with individual files

---

## Recommended Power BI Report Pages

### Page 1: Executive Summary
- **Card visuals** for: ROC-AUC, PR-AUC, F1 Score (from `model_metrics.csv`)
- **Donut chart**: Risk segment distribution (from `segment_summary.csv`)
- **KPI cards**: Total customers scored, High Risk count, churn rate by segment

### Page 2: Customer Risk Scores
- **Histogram**: `churn_probability` distribution (from `predictions.csv`)
- **Table**: Top 100 highest-risk customers (filter `risk_rank <= 100`)
- **Slicer**: Filter by `risk_category`

### Page 3: Feature Drivers
- **Horizontal bar chart**: `mean_abs_shap` by `feature` (top 20 from `feature_importance.csv`)
- **Tooltip**: Add `xgb_weight` as secondary importance measure
- Sort by `rank` (ascending = most important at top)

### Page 4: Model Comparison
- **Clustered bar chart**: `value` by `model` grouped by `metric`
- **Filter**: Select metrics: `roc_auc`, `pr_auc`, `f1`, `precision`, `recall`
- **Split by**: `split` (OOT vs default threshold)

### Page 5: Business Campaign Planner
- **Gauge**: Precision for each risk tier (from `segment_summary.csv`)
- **Table**: Campaign simulation assumptions and expected ROI
- **Target**: Use `actual_churn_rate` as the KPI target line

---

## DAX Measures (Optional)

```dax
-- Total High Risk Customers
High Risk Count = 
    CALCULATE(
        SUM(segment_summary[customer_count]),
        segment_summary[segment] = "High Risk"
    )

-- Weighted Average Churn Probability
Weighted Avg Churn Prob = 
    AVERAGEX(
        predictions,
        predictions[churn_probability]
    )

-- Actual Churn Rate (Cohort B)
Actual Churn Rate = 
    DIVIDE(
        COUNTROWS(FILTER(predictions, predictions[is_churn] = 1)),
        COUNTROWS(predictions)
    )

-- Lift at Top Decile
Top Decile Lift = 
    VAR TopDecile = TOPN(5000, predictions, predictions[churn_probability], DESC)
    VAR TopDecileChurnRate = DIVIDE(
        CALCULATE(COUNTROWS(FILTER(TopDecile, [is_churn] = 1))),
        CALCULATE(COUNTROWS(TopDecile))
    )
    VAR BaseRate = [Actual Churn Rate]
    RETURN DIVIDE(TopDecileChurnRate, BaseRate)
```

---

## Scheduled Refresh

To keep the Power BI report up to date after each monthly scoring run:

1. Re-run `python src/12_score_and_export.py` to regenerate `data/dashboard/*.csv`
2. In Power BI Desktop: **Home → Refresh**
3. If published to Power BI Service: configure **Scheduled Refresh** via the dataset settings

> [!IMPORTANT]
> The `data/dashboard/` CSVs are the ONLY files needed for Power BI. Do NOT connect Power BI directly to `data/kkbox_cohorts_staging.db` (4 GB SQLite) or `data/features/*.csv` — these are too large for typical Power BI imports.

---

## Colour Coding Convention

For consistent visual design across all report pages:

| Segment | Hex Colour | Usage |
|---|---|---|
| High Risk | `#D32F2F` (Red) | High urgency — immediate retention action |
| Medium Risk | `#F57C00` (Orange) | Monitor — targeted outreach |
| Low Risk | `#388E3C` (Green) | Healthy — no intervention needed |
| XGBoost | `#1976D2` (Blue) | Champion model |
| LightGBM | `#7B1FA2` (Purple) | Comparison model |
| Logistic Regression | `#616161` (Grey) | Baseline |

---

*KKBox Churn Prediction — Power BI Integration Guide v1.0*
*Dashboard data generated by `src/12_score_and_export.py`*


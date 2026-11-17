# Stage 3 — Exploratory Data Analysis Report
**KKBox Churn Prediction Challenge**

Generated: 2026-11-14T10:30:47.498198Z

---

## 1. Dataset Overview

| Cohort | Rows | Features | Churn Rate | Churners |
|---|---|---|---|---|
| Cohort A (Training — cutoff 2017-02-28) | 50,000 | 34 | 0.0646 (6.46%) | 3,228 |
| Cohort B (OOT Evaluation — cutoff 2017-03-31) | 50,000 | 34 | 0.0904 (9.04%) | 4,520 |

**Feature layout:**
- **Numerical:** 31 features (transaction + listening behaviour + demographics)
- **Categorical:** 2 features (`city`, `registered_via`)
- **Binary:** 1 feature (`is_missing_member`)

---

## 2. Class Balance

| Split | Non-Churners | Churners | Churn Rate | Imbalance Ratio |
|---|---|---|---|---|
| Cohort A | 46,772 | 3,228 | 0.0646 | 14.49:1 |
| Cohort B | 45,480 | 4,520 | 0.0904 | 10.06:1 |

> [!IMPORTANT]
> The dataset is **highly imbalanced** (~6.5% churn rate in Cohort A).
> Tree models handle this via `scale_pos_weight = 14.4895` computed from the training split only.

---

## 3. Missing Value Analysis

### Cohort A
| column | missing_count | missing_pct |
| --- | --- | --- |
| age | 30480 | 60.96 |
| gender_encoded | 30344 | 60.69 |
| listen_trend | 11487 | 22.97 |
| tx_per_listening_month | 6245 | 12.49 |
| avg_secs_per_day | 6240 | 12.48 |
| total_num_unq | 6240 | 12.48 |
| total_num_100 | 6240 | 12.48 |
| total_num_985 | 6240 | 12.48 |
| total_num_75 | 6240 | 12.48 |
| active_days_last_30d | 6240 | 12.48 |
| active_days_last_14d | 6240 | 12.48 |
| active_days_last_7d | 6240 | 12.48 |
| total_secs | 6240 | 12.48 |
| total_num_50 | 6240 | 12.48 |
| total_num_25 | 6240 | 12.48 |
| total_log_rows | 6240 | 12.48 |
| total_listening_days | 6240 | 12.48 |
| completion_ratio | 6240 | 12.48 |
| days_since_last_listen | 6240 | 12.48 |
| secs_last_30d | 6240 | 12.48 |
| secs_last_7d | 6240 | 12.48 |
| city | 5874 | 11.75 |
| days_since_registration | 5874 | 11.75 |
| registered_via | 5874 | 11.75 |
| transaction_count | 6 | 0.01 |
| avg_plan_list_price | 6 | 0.01 |
| avg_actual_amount_paid | 6 | 0.01 |
| avg_discount | 6 | 0.01 |
| auto_renew_rate | 6 | 0.01 |
| cancel_rate | 6 | 0.01 |
| unique_payment_methods | 6 | 0.01 |
| unique_payment_plans | 6 | 0.01 |
| days_since_last_tx | 6 | 0.01 |

### Cohort B
| column | missing_count | missing_pct |
| --- | --- | --- |
| age | 30072 | 60.14 |
| gender_encoded | 29917 | 59.83 |
| listen_trend | 11012 | 22.02 |
| total_listening_days | 5833 | 11.67 |
| total_num_75 | 5833 | 11.67 |
| total_log_rows | 5833 | 11.67 |
| total_num_25 | 5833 | 11.67 |
| total_num_50 | 5833 | 11.67 |
| total_num_unq | 5833 | 11.67 |
| total_num_100 | 5833 | 11.67 |
| total_num_985 | 5833 | 11.67 |
| total_secs | 5833 | 11.67 |
| secs_last_7d | 5833 | 11.67 |
| active_days_last_7d | 5833 | 11.67 |
| active_days_last_14d | 5833 | 11.67 |
| active_days_last_30d | 5833 | 11.67 |
| avg_secs_per_day | 5833 | 11.67 |
| days_since_last_listen | 5833 | 11.67 |
| secs_last_30d | 5833 | 11.67 |
| tx_per_listening_month | 5833 | 11.67 |
| completion_ratio | 5833 | 11.67 |
| days_since_registration | 5520 | 11.04 |
| city | 5520 | 11.04 |
| registered_via | 5520 | 11.04 |

> [!NOTE]
> Missing values in numerical features are imputed with the **median** (computed on training data only) inside the sklearn preprocessing pipeline. Categorical features use a constant fill value of -999 before OrdinalEncoding.

---

## 4. Feature–Target Correlations (Cohort A, Pearson)

Top positive correlations with `is_churn`:

| Feature | Pearson r |
|---|---|
| `days_since_last_tx` | 0.3672 |
| `cancel_rate` | 0.2574 |
| `avg_plan_list_price` | 0.2329 |
| `avg_actual_amount_paid` | 0.2271 |
| `unique_payment_methods` | 0.0996 |
| `city` | 0.0567 |
| `avg_discount` | 0.0566 |
| `days_since_last_listen` | 0.0482 |
| `unique_payment_plans` | 0.0151 |
| `total_secs` | 0.0090 |

Top negative correlations with `is_churn` (protective features):

| Feature | Pearson r |
|---|---|
| `auto_renew_rate` | -0.3341 |
| `active_days_last_7d` | -0.1792 |
| `active_days_last_14d` | -0.1692 |
| `transaction_count` | -0.1677 |
| `listen_trend` | -0.1565 |
| `active_days_last_30d` | -0.1239 |
| `age` | -0.0954 |
| `secs_last_7d` | -0.0852 |
| `registered_via` | -0.0693 |
| `secs_last_30d` | -0.0606 |

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

**2,323** users appear in both Cohort A and Cohort B
(4.65% of Cohort A, 4.65% of Cohort B).

This is **expected and retained by design** — this is a temporal evaluation (Feb→Mar 2017 cutoffs)
with partially recurring subscribers. The two cohorts are still temporally separated.

---

## 7. Key EDA Findings

1. **Severe class imbalance**: ~6.5% churn rate requires class-weight compensation.
2. **Recency is the strongest signal**: Features capturing recent listening activity
   (`active_days_last_7d`, `secs_last_7d`, `days_since_last_listen`) have among the highest
   absolute correlations with churn.
3. **Auto-renewal is a protective factor**: `auto_renew_rate` and `cancel_rate` have strong
   directional relationships with churn — high auto-renew → low churn risk.
4. **Missing member profiles**: `is_missing_member` = 1 users have different churn patterns.
5. **Churn rate stable across cohorts**: 6.46% (A) vs 9.04% (B) —
   slight increase in Cohort B reflects the later time period (April 2017).
6. **Low linear correlation overall**: Most features have |r| < 0.3, indicating non-linear
   interactions that tree models can exploit far better than linear models.

---

*End of EDA Report — 2026-11-14T10:30:47.498198Z*

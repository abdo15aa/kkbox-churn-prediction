# Stage 2 — Feature Engineering Validation Report

> Generated: 2026-10-29 17:48:06  
> Elapsed: 360.6 s  
> Cutoff A: 2017-02-28 | Cutoff B: 2017-03-31

---

## 1. Critical Invariants

| Check | Result | Detail |
|---|---|---|
| Cohort A — 50k rows | ✅ PASS | 50000 |
| Cohort A — no duplicate msno | ✅ PASS | 0 |
| Cohort A — binary labels | ✅ PASS | [0, 1] |
| Cohort B — 50k rows | ✅ PASS | 50000 |
| Cohort B — no duplicate msno | ✅ PASS | 0 |
| Cohort B — binary labels | ✅ PASS | [0, 1] |
| No infinite values — Cohort A | ✅ PASS | {} |
| No infinite values — Cohort B | ✅ PASS | {} |
| No negative days columns — Cohort A | ✅ PASS | all zero |
| No negative days columns — Cohort B | ✅ PASS | all zero |

---

## 2. Cohort A Summary (cutoff 2017-02-28)

- **Users**: 50,000
- **Churners**: 3,228 (6.456%)
- **Users with no transactions**: 6
- **Users with no listening logs**: 6,240
- **Users missing from members table**: 5,874

### Missing values (Cohort A)

| Feature | Missing N | Missing % |
|---|---|---|
| `active_days_last_14d` | 6,240 | 12.48% |
| `active_days_last_30d` | 6,240 | 12.48% |
| `active_days_last_7d` | 6,240 | 12.48% |
| `age` | 30,480 | 60.96% |
| `auto_renew_rate` | 6 | 0.01% |
| `avg_actual_amount_paid` | 6 | 0.01% |
| `avg_discount` | 6 | 0.01% |
| `avg_plan_list_price` | 6 | 0.01% |
| `avg_secs_per_day` | 6,240 | 12.48% |
| `cancel_rate` | 6 | 0.01% |
| `city` | 5,874 | 11.75% |
| `completion_ratio` | 6,240 | 12.48% |
| `days_since_last_listen` | 6,240 | 12.48% |
| `days_since_last_tx` | 6 | 0.01% |
| `days_since_registration` | 5,874 | 11.75% |
| `gender_encoded` | 30,344 | 60.69% |
| `listen_trend` | 11,487 | 22.97% |
| `registered_via` | 5,874 | 11.75% |
| `secs_last_30d` | 6,240 | 12.48% |
| `secs_last_7d` | 6,240 | 12.48% |
| `total_listening_days` | 6,240 | 12.48% |
| `total_log_rows` | 6,240 | 12.48% |
| `total_num_100` | 6,240 | 12.48% |
| `total_num_25` | 6,240 | 12.48% |
| `total_num_50` | 6,240 | 12.48% |
| `total_num_75` | 6,240 | 12.48% |
| `total_num_985` | 6,240 | 12.48% |
| `total_num_unq` | 6,240 | 12.48% |
| `total_secs` | 6,240 | 12.48% |
| `transaction_count` | 6 | 0.01% |
| `tx_per_listening_month` | 6,245 | 12.49% |
| `unique_payment_methods` | 6 | 0.01% |
| `unique_payment_plans` | 6 | 0.01% |

---

## 3. Cohort B Summary (cutoff 2017-03-31)

- **Users**: 50,000
- **Churners**: 4,520 (9.04%)
- **Users with no transactions**: 0
- **Users with no listening logs**: 5,833
- **Users missing from members table**: 5,520

### Missing values (Cohort B)

| Feature | Missing N | Missing % |
|---|---|---|
| `active_days_last_14d` | 5,833 | 11.67% |
| `active_days_last_30d` | 5,833 | 11.67% |
| `active_days_last_7d` | 5,833 | 11.67% |
| `age` | 30,072 | 60.14% |
| `avg_secs_per_day` | 5,833 | 11.67% |
| `city` | 5,520 | 11.04% |
| `completion_ratio` | 5,833 | 11.67% |
| `days_since_last_listen` | 5,833 | 11.67% |
| `days_since_registration` | 5,520 | 11.04% |
| `gender_encoded` | 29,917 | 59.83% |
| `listen_trend` | 11,012 | 22.02% |
| `registered_via` | 5,520 | 11.04% |
| `secs_last_30d` | 5,833 | 11.67% |
| `secs_last_7d` | 5,833 | 11.67% |
| `total_listening_days` | 5,833 | 11.67% |
| `total_log_rows` | 5,833 | 11.67% |
| `total_num_100` | 5,833 | 11.67% |
| `total_num_25` | 5,833 | 11.67% |
| `total_num_50` | 5,833 | 11.67% |
| `total_num_75` | 5,833 | 11.67% |
| `total_num_985` | 5,833 | 11.67% |
| `total_num_unq` | 5,833 | 11.67% |
| `total_secs` | 5,833 | 11.67% |
| `tx_per_listening_month` | 5,833 | 11.67% |

---

## 4. Overlapping Users (appear in both cohorts)

- **Count**: 2,323
- **Features independently computed**: True
- **Label agreement (same outcome both months)**: 2,213
- **Label disagreement (different outcome)**: 110

Overlapping users have **independent** feature rows in each cohort — Cohort A uses only data up to Feb-28, Cohort B uses data up to Mar-31. Same user, different observation windows.

---

## 5. Known Limitations and Leakage Risks

> [!WARNING]
> The following are documented limitations, not confirmed errors.

### 5.1 v2_correction transaction rows (Cohort A only)

- For Cohort A, rows tagged `source_file='v2_correction'` are **excluded** from transaction features. These are rows from `transactions_v2.csv` with dates before 2017-02-28 that superseded v1 records.
- We cannot confirm these corrections were available before the Feb-28 cutoff.
- **Impact**: Cohort A transaction features use only `source_file='v1'` rows. Some users may have fewer transactions in Cohort A than their actual history. This is the conservative, leakage-safe choice.
- **Cohort B** includes `v2_correction` rows (they were released in the v2 file, which was available after Feb-28 and before Mar-31 cutoff).

### 5.2 membership_expire_date (NOT used as a feature)

- The expiration date from the most recent transaction is NOT computed as a feature because v2_correction rows may have changed it. Its point-in-time value at Feb-28 cannot be guaranteed for Cohort A.
- `days_since_last_tx` is safe because it uses `transaction_date` (when the transaction was made), not the expiration date.

### 5.3 Missing historical context

- Some users registered before 2015 but have no listening logs before their cutoff. These are not errors — they may be low-activity or lapsed users.

---

## 6. Feature Descriptions

| Feature | Source | Leakage Safe? | Notes |
|---|---|---|---|
| `transaction_count` | transactions | ✅ Yes | COUNT of rows ≤ cutoff |
| `days_since_last_tx` | transactions | ✅ Yes | cutoff − MAX(transaction_date ≤ cutoff) |
| `avg_plan_list_price` | transactions | ✅ Yes | AVG over eligible rows |
| `avg_actual_amount_paid` | transactions | ✅ Yes | AVG over eligible rows |
| `avg_discount` | transactions | ✅ Yes | AVG(list_price − actual_paid) |
| `auto_renew_rate` | transactions | ✅ Yes | fraction of rows with auto_renew=1 |
| `cancel_rate` | transactions | ✅ Yes | fraction of rows with is_cancel=1 |
| `unique_payment_methods` | transactions | ✅ Yes | COUNT DISTINCT payment_method_id |
| `unique_payment_plans` | transactions | ✅ Yes | COUNT DISTINCT payment_plan_days |
| `total_listening_days` | user_logs | ✅ Yes | COUNT DISTINCT date ≤ cutoff |
| `total_log_rows` | user_logs | ✅ Yes | raw row count |
| `total_num_25/50/75/985/100` | user_logs | ✅ Yes | SUM of play-depth columns |
| `total_num_unq` | user_logs | ✅ Yes | SUM of unique songs per day |
| `total_secs` | user_logs | ✅ Yes | SUM of total_secs |
| `avg_secs_per_day` | user_logs | ✅ Yes | derived from above |
| `days_since_last_listen` | user_logs | ✅ Yes | cutoff − MAX(date ≤ cutoff) |
| `active_days_last_7/14/30d` | user_logs | ✅ Yes | windowed count |
| `secs_last_7d`, `secs_last_30d` | user_logs | ✅ Yes | windowed sum |
| `listen_trend` | user_logs | ✅ Yes | secs_last_7d / (secs_last_30d/4) |
| `days_since_registration` | members | ✅ Yes | cutoff − registration_init_time |
| `age` | members | ✅ Yes | cutoff_year − bd (birth year) |
| `gender_encoded` | members | ✅ Yes | 1=male, 0=female, NaN=missing |
| `city` | members | ✅ Yes | categorical city code |
| `registered_via` | members | ✅ Yes | registration channel code |
| `is_missing_member` | members | ✅ Yes | 1 if no row in members table |
| `tx_per_listening_month` | derived | ✅ Yes | tx_count / (listening_days/30) |
| `completion_ratio` | derived | ✅ Yes | total_num_100 / total_plays |

---

*Report generated in 360.6 seconds.*
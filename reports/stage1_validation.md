# Stage 1 — Staging Database Validation Report

> **Generated**: 2026-10-29 17:09:36  
> **Source Database**: `data/kkbox_cohorts_staging.db` (4,015,792,128 bytes / 4.02 GB)  
> **Original Reference Database**: `data/kkbox.db` (108,187,648 bytes, SHA256: `fbb3b833bbe3b50d...`) — **UNTOUCHED**  
> **Status**: **PASS (ALL CHECKS)**

---

## 1. Executive Summary & Critical Invariants

| Check Item | Requirement | Observed Metric | Status |
|---|---|---|:---:|
| **Original DB Unchanged** | Size & hash match baseline | `size=108,187,648`, SHA256 prefix intact | ✅ **PASS** |
| **SQLite Integrity** | `PRAGMA integrity_check` | `ok` | ✅ **PASS** |
| **Cohort A Sample** | Exactly 50,000 distinct users | 50,000 rows, 50,000 unique `msno` | ✅ **PASS** |
| **Cohort B Sample** | Exactly 50,000 distinct users | 50,000 rows, 50,000 unique `msno` | ✅ **PASS** |
| **Cohort A Churn Rate** | Consistent with February labels | 3,228 churners (6.456%) | ✅ **PASS** |
| **Cohort B Churn Rate** | Consistent with March labels | 4,520 churners (9.040%) | ✅ **PASS** |
| **Label Nullity** | Zero nulls in labels/keys | 0 null `msno`, 0 null `is_churn` in both | ✅ **PASS** |
| **User Universe** | Combined distinct `msno` | 97,677 unique users (2,323 in both) | ✅ **PASS** |
| **Transaction Date Boundaries** | Cutoff alignment | v1: 20150101–20170228; v2_new: 20170301–20170331 | ✅ **PASS** |
| **Log Date Disjointness** | Zero date overlap v1 vs v2 | 0 rows overlap (v1 max: 20170228, v2 min: 20170301) | ✅ **PASS** |
| **Transaction Coverage** | Every cohort user has transaction data | 0 users missing transactions in either cohort | ✅ **PASS** |

---

## 2. Cohort Cohere & Overlap Analysis

* **Cohort A (February 2017 cutoff: `2017-02-28`)**:
  * Total sampled: 50,000
  * Non-churners (`is_churn = 0`): 46,772 (93.54%)
  * Churners (`is_churn = 1`): 3,228 (6.46%)
* **Cohort B (March 2017 cutoff: `2017-03-31`)**:
  * Total sampled: 50,000
  * Non-churners (`is_churn = 0`): 45,480 (90.96%)
  * Churners (`is_churn = 1`): 4,520 (9.04%)
* **Overlap Structure**:
  * Users in both Cohort A and Cohort B: **2,323**
  * Users strictly in Cohort A: **47,677**
  * Users strictly in Cohort B: **47,677**
  * Total unique members across both cohorts: **97,677**

> [!NOTE]
> The 2,323 overlapping users are preserved independently in each cohort's label table. They represent users observed at two distinct points in time. Their features will be computed strictly using their respective observation cutoff.

---

## 3. Transactions Architecture & Provenance

* **Total rows inserted**: 1,653,850
* **Unique members with transactions**: 97,677 (100.0% coverage across all sampled users)
* **Duplicate keys `(msno, transaction_date)`**: 20,615 (preserved as distinct billing actions)

### Provenance Breakdown:
| `source_file` | Row Count | Date Range (Min – Max) | Semantics & Role |
|---|---|---|---|
| `v1` | 1,541,659 | 2015-01-01 – 2017-02-28 | Historical transactions from `transactions.csv` |
| `v2_correction` | 17,551 | 2015-01-01 – 2017-02-28 | Historical adjustments from `transactions_v2.csv` |
| `v2_new` | 94,640 | 2017-03-01 – 2017-03-31 | New transactions occurring during March 2017 |

* **Historical deduplication rule applied**:
  * 338,868 keys `(msno, transaction_date)` with `date <= 20170228` were identified in `transactions_v2.csv`.
  * In `transactions.csv`, exactly **481 matching rows** for our sampled cohort were superseded and skipped.
  * **17,551** rows with `date <= 20170228` from `transactions_v2.csv` were inserted and explicitly flagged as `v2_correction`.
* **Sentinel values**: 32 rows exhibit `membership_expire_date < 19900101` (epoch/sentinel placeholders).

---

## 4. User Listening Logs

* **Total rows inserted**: 25,344,676
* **Distinct members with logs**: 85,971 (88.0% of cohort universe)
* **Date Coverage**:
  * `v1`: 24,011,896 rows (2015-01-01 to 2017-02-28)
  * `v2`: 1,332,780 rows (2017-03-01 to 2017-03-31)
* **Overlap / Quality**:
  * Overlapping dates between `v1` and `v2`: **0**
  * Missing dates: **0**
* **Log Inactivity**:
  * Cohort A users with 0 listening logs: **6,157** (12.3%)
  * Cohort B users with 0 listening logs: **5,833** (11.7%)
  * *Interpretation*: Inactive or lapsed subscribers who continued auto-renewing or cancelled without listening. These users are preserved; missing logs are a legitimate predictive signal for churn.

---

## 5. Members Profile Coverage

* **Total rows inserted**: 86,556
* **Distinct members**: 86,556
* **Coverage**: 88.6% of the 97,677 user universe have profile metadata in `members_v3.csv`.
* **Missingness**: 11,121 users (11.4%) do not appear in `members_v3.csv`. They will have `is_missing_member = 1`.

---

## 6. Documented Limitations & Leakage Guards for Stage 2

1. **`v2_correction` Exclusion in Cohort A**:
   `transactions_v2.csv` was released after February 28, 2017. Because we cannot verify whether the 17,551 correction records were knowable as of 2017-02-28, Stage 2 **must exclude** rows where `source_file = 'v2_correction'` when building features for Cohort A.
2. **Exclusion of `membership_expire_date` as a Raw Feature**:
   Expiration dates are subject to retrospective updates upon plan cancellations or adjustments. `days_since_last_tx` (calculated from `transaction_date`) is point-in-time safe, whereas raw `membership_expire_date` carries risk of post-cutoff adjustment.

---

## 7. Recommendation

Stage 1 has fully satisfied all integrity, isolation, and schema invariants.  
**Recommendation**: **APPROVED to proceed to Stage 2 (Feature Engineering).**


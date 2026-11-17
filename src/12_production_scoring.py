"""
Stage 6 — Production Scoring & Customer Retention Segmentation
==============================================================
KKBox Churn Prediction Challenge

PURPOSE
-------
1. Load the production-validated XGBoost model and preprocessor pipeline.
2. Load prospective customer features (e.g. Cohort B or any live feature file).
3. Perform strict validation checks:
   - Duplicate customer IDs (msno).
   - Missing required feature columns.
   - Target exclusion guarantee (is_churn is never fed to the model).
   - Output probability validity (range [0, 1], no NaN/Inf).
4. Generate churn probability scores and rank customers from highest to lowest risk.
5. Assign commercial risk segments:
   - High Risk: Top 5% of customer base
   - Medium Risk: Next 15% of customer base (5% - 20%)
   - Low Risk: Remaining 80% of customer base (20% - 100%)
6. Export customer scoring artifact to outputs/customer_churn_scores.csv.
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import joblib

# Setup clean console logging to sys.stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout
)
logger = logging.getLogger("ProductionScoring")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_PATH  = PROJECT_ROOT / "data" / "features" / "cohort_b_features.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "customer_churn_scores.csv"
DEFAULT_MODEL_PATH  = PROJECT_ROOT / "models" / "xgboost_churn_model.joblib"
DEFAULT_PREP_PATH   = PROJECT_ROOT / "models" / "churn_preprocessor.joblib"


def validate_input_data(df: pd.DataFrame, expected_features: list, id_col: str = "msno") -> pd.DataFrame:
    """
    Validates input feature dataset before inference.
    Raises ValueError if critical data quality criteria are violated.
    """
    logger.info("Running pre-inference validation checks on input data...")
    
    # Check 1: Non-empty dataset
    if df.empty:
        raise ValueError("Input dataset is completely empty (0 rows).")
    logger.info(f"  [PASS] Dataset contains {len(df):,} customer rows.")

    # Check 2: Customer ID column exists
    if id_col not in df.columns:
        raise ValueError(f"Required identifier column '{id_col}' not found in input columns.")
    logger.info(f"  [PASS] Identifier column '{id_col}' confirmed.")

    # Check 3: Duplicate customer IDs
    duplicates = df[id_col].duplicated().sum()
    if duplicates > 0:
        raise ValueError(f"Input data contains {duplicates:,} duplicate '{id_col}' records. Each customer must be unique.")
    logger.info(f"  [PASS] All {len(df):,} customer IDs are unique (zero duplicates).")

    # Check 4: Missing required feature columns
    missing_cols = set(expected_features) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Input data is missing {len(missing_cols)} required feature columns: {sorted(missing_cols)}")
    logger.info(f"  [PASS] All {len(expected_features)} required feature columns are present.")

    # Check 5: Target isolation guarantee
    if "is_churn" in df.columns:
        logger.info("  [NOTICE] Target column 'is_churn' detected in input file. Strictly excluding it from feature inputs.")
    
    return df


def validate_probabilities(probabilities: np.ndarray, expected_count: int) -> None:
    """
    Validates generated model probability outputs.
    """
    logger.info("Running post-inference validation checks on predicted probabilities...")
    
    # Check length
    if len(probabilities) != expected_count:
        raise ValueError(f"Prediction count ({len(probabilities)}) does not match input customer count ({expected_count}).")

    # Check for NaNs or Infinite values
    if np.isnan(probabilities).any():
        nan_count = np.isnan(probabilities).sum()
        raise ValueError(f"Predictions contain {nan_count} NaN values.")

    if np.isinf(probabilities).any():
        inf_count = np.isinf(probabilities).sum()
        raise ValueError(f"Predictions contain {inf_count} Infinite values.")

    # Check probability range [0.0, 1.0]
    min_prob = float(np.min(probabilities))
    max_prob = float(np.max(probabilities))
    if min_prob < 0.0 or max_prob > 1.0:
        raise ValueError(f"Probabilities out of bounds [0, 1]: min={min_prob}, max={max_prob}")

    logger.info(f"  [PASS] All {expected_count:,} probabilities valid. Range: [{min_prob:.4f}, {max_prob:.4f}].")


def score_customers(
    input_path: Path,
    output_path: Path,
    model_path: Path,
    preprocessor_path: Path,
    id_col: str = "msno"
) -> dict:
    """
    Production scoring pipeline: loads model, scores customers, segments by risk, and exports CSV.
    """
    logger.info("=" * 70)
    logger.info("KKBOX CHURN PREDICTION — PRODUCTION SCORING PIPELINE")
    logger.info(f"Execution Timestamp: {datetime.now(timezone.utc).isoformat()}")
    logger.info("=" * 70)

    # 1. Load Preprocessor and Model
    logger.info(f"Loading preprocessor artifact from: {preprocessor_path}")
    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Preprocessor file not found at {preprocessor_path}")
    prep_data = joblib.load(preprocessor_path)
    preprocessor = prep_data["preprocessor"]
    expected_features = prep_data["feature_cols"]

    logger.info(f"Loading trained XGBoost model from: {model_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model = joblib.load(model_path)

    # 2. Load Customer Features
    logger.info(f"Loading prospective customer features from: {input_path}")
    if not input_path.exists():
        raise FileNotFoundError(f"Input customer file not found at {input_path}")
    raw_df = pd.read_csv(input_path)

    # 3. Input Validation
    validate_input_data(raw_df, expected_features, id_col=id_col)

    # 4. Extract Features (strictly excluding ID and any target label)
    X = raw_df[expected_features].copy()
    customer_ids = raw_df[id_col].copy()

    # 5. Preprocessing & Model Inference
    logger.info("Applying production transformation pipeline...")
    X_transformed = preprocessor.transform(X)

    logger.info("Generating churn probability estimates with XGBoost...")
    churn_probabilities = model.predict_proba(X_transformed)[:, 1]

    # 6. Output Probability Validation
    validate_probabilities(churn_probabilities, len(raw_df))

    # 7. Customer Ranking and Risk Segmentation
    logger.info("Segmenting customer base into retention action tiers...")
    scored_df = pd.DataFrame({
        id_col: customer_ids,
        "churn_probability": np.round(churn_probabilities, 6)
    })

    # Sort descending by churn risk
    scored_df = scored_df.sort_values("churn_probability", ascending=False).reset_index(drop=True)
    
    # Assign discrete 1-based risk rank
    scored_df["risk_rank"] = np.arange(1, len(scored_df) + 1)

    # Assign retention tiers based on business specifications:
    # High Risk   : Top 5%
    # Medium Risk : Next 15% (percentile 5% to 20%)
    # Low Risk    : Remaining 80% (percentile 20% to 100%)
    total_users = len(scored_df)
    high_cutoff_rank = int(np.ceil(0.05 * total_users))
    med_cutoff_rank  = int(np.ceil(0.20 * total_users))

    def assign_tier(rank: int) -> str:
        if rank <= high_cutoff_rank:
            return "High Risk"
        elif rank <= med_cutoff_rank:
            return "Medium Risk"
        else:
            return "Low Risk"

    scored_df["risk_category"] = scored_df["risk_rank"].apply(assign_tier)

    # 8. Export to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored_df.to_csv(output_path, index=False)
    logger.info(f"[SUCCESS] Scored customer dataset exported to: {output_path}")

    # 9. Summary Statistics
    high_count = (scored_df["risk_category"] == "High Risk").sum()
    med_count  = (scored_df["risk_category"] == "Medium Risk").sum()
    low_count  = (scored_df["risk_category"] == "Low Risk").sum()

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_scored": total_users,
        "high_risk_count": int(high_count),
        "high_risk_pct": float(round(high_count / total_users * 100, 2)),
        "medium_risk_count": int(med_count),
        "medium_risk_pct": float(round(med_count / total_users * 100, 2)),
        "low_risk_count": int(low_count),
        "low_risk_pct": float(round(low_count / total_users * 100, 2)),
        "prob_min": float(round(scored_df["churn_probability"].min(), 6)),
        "prob_max": float(round(scored_df["churn_probability"].max(), 6)),
        "prob_mean": float(round(scored_df["churn_probability"].mean(), 6)),
        "prob_median": float(round(scored_df["churn_probability"].median(), 6)),
        "high_risk_min_score": float(round(scored_df[scored_df["risk_category"] == "High Risk"]["churn_probability"].min(), 4)),
        "medium_risk_min_score": float(round(scored_df[scored_df["risk_category"] == "Medium Risk"]["churn_probability"].min(), 4)),
        "output_file": str(output_path),
    }

    logger.info("\n" + "=" * 50)
    logger.info("PRODUCTION SCORING SUMMARY STATISTICS")
    logger.info("=" * 50)
    logger.info(f"Total Customers Scored  : {summary['total_scored']:,}")
    logger.info(f"  - High Risk (Top 5%)  : {summary['high_risk_count']:,} ({summary['high_risk_pct']}%) | Min Score: {summary['high_risk_min_score']:.4f}")
    logger.info(f"  - Medium Risk (15%)   : {summary['medium_risk_count']:,} ({summary['medium_risk_pct']}%) | Min Score: {summary['medium_risk_min_score']:.4f}")
    logger.info(f"  - Low Risk (80%)      : {summary['low_risk_count']:,} ({summary['low_risk_pct']}%)")
    logger.info(f"Probability Distribution:")
    logger.info(f"  - Min Probability     : {summary['prob_min']:.6f}")
    logger.info(f"  - Mean Probability    : {summary['prob_mean']:.6f}")
    logger.info(f"  - Median Probability  : {summary['prob_median']:.6f}")
    logger.info(f"  - Max Probability     : {summary['prob_max']:.6f}")
    logger.info("=" * 50)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Production scoring and risk segmentation for KKBox Churn Prediction.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="Path to customer features CSV.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Path to output scored CSV.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Path to trained XGBoost model joblib.")
    parser.add_argument("--preprocessor", type=Path, default=DEFAULT_PREP_PATH, help="Path to fitted preprocessor joblib.")
    args = parser.parse_args()

    try:
        score_customers(
            input_path=args.input,
            output_path=args.output,
            model_path=args.model,
            preprocessor_path=args.preprocessor
        )
    except Exception as e:
        logger.error(f"Production scoring failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

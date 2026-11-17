"""
KKBox Churn Prediction — Streamlit Dashboard
=============================================

Interactive dashboard for exploring KKBox churn prediction results.

Reads pre-computed dashboard CSVs from data/dashboard/. Does NOT load
raw data, staging database, or model files directly.

Usage:
  streamlit run app/dashboard.py --server.port 8501

Requirements:
  streamlit>=1.35.0
  pandas
  numpy
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# ===========================================================================
# PAGE CONFIG
# ===========================================================================
st.set_page_config(
    page_title="KKBox Churn Prediction Dashboard",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===========================================================================
# PATHS
# ===========================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DASH_DIR     = PROJECT_ROOT / "data" / "dashboard"

PREDICTIONS_CSV       = DASH_DIR / "predictions.csv"
SEGMENT_SUMMARY_CSV   = DASH_DIR / "segment_summary.csv"
FEATURE_IMPORTANCE_CSV= DASH_DIR / "feature_importance.csv"
MODEL_METRICS_CSV     = DASH_DIR / "model_metrics.csv"

# ===========================================================================
# DATA LOADING (cached)
# ===========================================================================
@st.cache_data
def load_predictions():
    if not PREDICTIONS_CSV.exists():
        return None
    return pd.read_csv(PREDICTIONS_CSV)

@st.cache_data
def load_segment_summary():
    if not SEGMENT_SUMMARY_CSV.exists():
        return None
    return pd.read_csv(SEGMENT_SUMMARY_CSV)

@st.cache_data
def load_feature_importance():
    if not FEATURE_IMPORTANCE_CSV.exists():
        return None
    return pd.read_csv(FEATURE_IMPORTANCE_CSV)

@st.cache_data
def load_model_metrics():
    if not MODEL_METRICS_CSV.exists():
        return None
    return pd.read_csv(MODEL_METRICS_CSV)

predictions    = load_predictions()
segment_df     = load_segment_summary()
feat_imp_df    = load_feature_importance()
model_metrics  = load_model_metrics()

data_ok = all(x is not None for x in [predictions, segment_df, feat_imp_df, model_metrics])

# ===========================================================================
# HEALTH ENDPOINT (Streamlit's built-in: /_stcore/health)
# ===========================================================================
# No explicit health route needed — Streamlit exposes /_stcore/health by default.

# ===========================================================================
# SIDEBAR
# ===========================================================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/6d/KKBOX_logo.svg/320px-KKBOX_logo.svg.png",
             use_container_width=True)
    st.title("KKBox Churn Prediction")
    st.caption("WSDM 2018 Kaggle Challenge — ML Pipeline v1.0")
    st.markdown("---")
    st.markdown("""
**Navigation**
- 📊 Overview Metrics
- 🎯 Risk Segments
- 🔍 Feature Importance
- 📈 Score Distribution
    """)
    st.markdown("---")
    if data_ok:
        st.success("✅ All dashboard data loaded")
    else:
        st.error("❌ Dashboard data missing. Run `src/12_score_and_export.py` first.")

# ===========================================================================
# MAIN CONTENT
# ===========================================================================
st.title("🎵 KKBox Churn Prediction Dashboard")
st.caption("Champion model: XGBoost | Evaluation: Cohort B (50,000 users, OOT — March/April 2017)")

if not data_ok:
    st.error(
        "Dashboard data files are missing. "
        "Run `python src/12_score_and_export.py` to generate them."
    )
    st.code("cd <project_root>\n.\\venv\\python.exe src/12_score_and_export.py")
    st.stop()

# ---------------------------------------------------------------------------
# SECTION 1: KEY METRICS (extracted from model_metrics.csv)
# ---------------------------------------------------------------------------
st.markdown("## 📊 Model Performance — Cohort B (Out-of-Time Test)")

xgb_oot = model_metrics[
    (model_metrics["model"] == "XGBoost") &
    (model_metrics["split"].str.contains("OOT"))
]

def get_metric(df, metric_name, default="N/A"):
    row = df[df["metric"] == metric_name]
    if row.empty:
        return default
    return round(float(row["value"].iloc[0]), 4)

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("ROC-AUC", f"{get_metric(xgb_oot, 'roc_auc'):.4f}", help="Area under ROC curve (0.5=random, 1.0=perfect)")
with col2:
    st.metric("PR-AUC", f"{get_metric(xgb_oot, 'pr_auc'):.4f}", help="Area under Precision-Recall curve")
with col3:
    st.metric("F1 Score", f"{get_metric(xgb_oot, 'f1'):.4f}", help="F1 at threshold=0.64")
with col4:
    st.metric("Precision", f"{get_metric(xgb_oot, 'precision'):.4f}", help="Precision at threshold=0.64")
with col5:
    st.metric("Recall", f"{get_metric(xgb_oot, 'recall'):.4f}", help="Recall at threshold=0.64")

st.markdown("---")

# ---------------------------------------------------------------------------
# SECTION 2: RISK SEGMENT OVERVIEW
# ---------------------------------------------------------------------------
st.markdown("## 🎯 Customer Risk Segmentation")

col_seg, col_metrics = st.columns([1, 1])

with col_seg:
    st.subheader("Risk Segment Distribution")

    # Segment pie chart using st.bar_chart approximation
    seg_display = segment_df.set_index("segment")[["customer_count"]]
    color_map = {"High Risk": "🔴", "Medium Risk": "🟡", "Low Risk": "🟢"}

    for _, row in segment_df.iterrows():
        seg   = row["segment"]
        count = int(row["customer_count"])
        pct   = float(row["pct_of_total"])
        avg_p = float(row["avg_churn_probability"])
        cr    = float(row["actual_churn_rate"]) if pd.notna(row.get("actual_churn_rate")) else None
        icon  = color_map.get(seg, "⚪")

        with st.container():
            c1, c2, c3 = st.columns([2, 1, 1])
            with c1:
                st.markdown(f"{icon} **{seg}**")
            with c2:
                st.markdown(f"{count:,} users ({pct}%)")
            with c3:
                cr_str = f"Churn rate: {cr:.1%}" if cr is not None else ""
                st.markdown(cr_str)

with col_metrics:
    st.subheader("Segment Statistics")
    display_cols = ["segment", "customer_count", "pct_of_total", "avg_churn_probability"]
    if "actual_churn_rate" in segment_df.columns:
        display_cols.append("actual_churn_rate")

    st.dataframe(
        segment_df[display_cols].rename(columns={
            "segment": "Segment",
            "customer_count": "Customers",
            "pct_of_total": "% of Total",
            "avg_churn_probability": "Avg Churn Prob",
            "actual_churn_rate": "Actual Churn Rate",
        }),
        use_container_width=True,
        hide_index=True,
    )

    if "actual_churn_rate" in segment_df.columns:
        rates = segment_df.set_index("segment")["actual_churn_rate"]
        high  = rates.get("High Risk", 0)
        med   = rates.get("Medium Risk", 0)
        low_  = rates.get("Low Risk", 0)
        monotone_ok = high > med > low_
        if monotone_ok:
            st.success("✅ Monotonic risk: High > Medium > Low (actual churn rate)")
        else:
            st.warning("⚠️ Risk segments are not monotonic — check predictions")

st.markdown("---")

# ---------------------------------------------------------------------------
# SECTION 3: FEATURE IMPORTANCE
# ---------------------------------------------------------------------------
st.markdown("## 🔍 Feature Importance")

col_fi1, col_fi2 = st.columns([1, 1])

with col_fi1:
    st.subheader("Top 15 by Mean |SHAP|")
    top15_shap = feat_imp_df.nlargest(15, "mean_abs_shap")[["feature", "mean_abs_shap"]].reset_index(drop=True)
    top15_shap.index += 1
    st.bar_chart(top15_shap.set_index("feature")["mean_abs_shap"])

with col_fi2:
    st.subheader("Top 15 by XGBoost Weight")
    top15_xgb = feat_imp_df.nlargest(15, "xgb_weight")[["feature", "xgb_weight"]].reset_index(drop=True)
    top15_xgb.index += 1
    st.bar_chart(top15_xgb.set_index("feature")["xgb_weight"])

st.subheader("Full Feature Importance Table")
st.dataframe(
    feat_imp_df[["rank", "feature", "mean_abs_shap", "xgb_weight"]].rename(columns={
        "rank": "Rank",
        "feature": "Feature",
        "mean_abs_shap": "Mean |SHAP|",
        "xgb_weight": "XGB Weight",
    }),
    use_container_width=True,
    hide_index=True,
)

st.markdown("---")

# ---------------------------------------------------------------------------
# SECTION 4: SCORE DISTRIBUTION
# ---------------------------------------------------------------------------
st.markdown("## 📈 Churn Probability Distribution")

col_hist, col_stats = st.columns([2, 1])

with col_hist:
    st.subheader("Score Distribution (Cohort B)")
    # Create histogram bins manually
    bins = np.linspace(0, 1, 51)
    hist_data = []
    for seg, color in [("High Risk", "🔴"), ("Medium Risk", "🟡"), ("Low Risk", "🟢")]:
        sub = predictions[predictions["risk_category"] == seg]["churn_probability"]
        hist_df = pd.cut(sub, bins=bins).value_counts().sort_index()
        hist_data.append(hist_df.values)

    # Simple histogram using bar_chart
    all_hist = pd.cut(predictions["churn_probability"], bins=bins).value_counts().sort_index()
    chart_df = pd.DataFrame({
        "Score Bucket": [f"{b.left:.2f}" for b in all_hist.index],
        "Count": all_hist.values
    }).set_index("Score Bucket")
    st.bar_chart(chart_df)

with col_stats:
    st.subheader("Distribution Statistics")
    prob_col = predictions["churn_probability"]
    stats_data = {
        "Statistic": ["Min", "5th Pct", "Median", "Mean", "95th Pct", "Max"],
        "Value": [
            f"{prob_col.min():.4f}",
            f"{prob_col.quantile(0.05):.4f}",
            f"{prob_col.median():.4f}",
            f"{prob_col.mean():.4f}",
            f"{prob_col.quantile(0.95):.4f}",
            f"{prob_col.max():.4f}",
        ]
    }
    st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)

    # Actual churn rate by score bucket
    if "is_churn" in predictions.columns:
        st.subheader("Actual Churn Rate by Segment")
        cr_data = []
        for seg in ["High Risk", "Medium Risk", "Low Risk"]:
            sub = predictions[predictions["risk_category"] == seg]
            if len(sub) > 0:
                cr = sub["is_churn"].mean()
                cr_data.append({"Segment": seg, "Actual Churn Rate": f"{cr:.2%}"})
        st.dataframe(pd.DataFrame(cr_data), use_container_width=True, hide_index=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# SECTION 5: MODEL COMPARISON
# ---------------------------------------------------------------------------
st.markdown("## 🏆 Model Comparison — Cohort B (OOT)")

pivot = model_metrics[
    model_metrics["metric"].isin(["roc_auc", "pr_auc", "f1", "threshold"])
].pivot_table(
    index=["model", "split"],
    columns="metric",
    values="value",
    aggfunc="first"
).reset_index().round(4)

if not pivot.empty:
    st.dataframe(pivot, use_container_width=True, hide_index=True)
else:
    st.info("Model comparison table is empty — check model_metrics.csv")

st.markdown("---")

# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------
st.caption(
    "KKBox Churn Prediction Challenge | XGBoost champion | "
    "Cohort A: 50,000 users (Feb 2017) | Cohort B: 50,000 users (Mar 2017) | "
    "Dashboard data: `data/dashboard/*.csv`"
)


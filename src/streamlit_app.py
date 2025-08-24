import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import datetime
import re

ART = Path("artifacts")

# --- NEW: Centralized Label Mapping & Helper ---
LABEL_MAP = {
    # Core Columns
    "customer_id": "Customer ID",
    "churn_proba": "Churn Probability",
    "churned": "Churned",
    "segment_label": "Segment",
    "plan": "Plan",
    "region": "Region",
    "age": "Age",
    # Feature Columns
    "tx_count": "Transaction Count",
    "amt_sum": "Total Spend ($)",
    "recency_days": "Days Since Last Transaction",
    # Chart/Table Specific
    "amount": "Sales Amount ($)",
    "date": "Date",
    "count": "Number of Customers",
    "importance": "Feature Importance (SHAP)",
}

def clean_feature_name(name):
    """Cleans up feature names from the preprocessor for display."""
    name = re.sub(r'^(num__|cat__)', '', name)
    name = name.replace('_', ' ').strip()
    # Capitalize all words
    return name.title()

st.set_page_config(page_title="Churn & Sales Dashboard", layout="wide")

# ---------------------------
# Load artifacts
# ---------------------------
@st.cache_data
def load_core_artifacts():
    scores = pd.read_parquet(ART / "churn_scores.parquet")
    # UPDATED: Load the new features file name
    feats_path = ART / "customer_features_with_segments.parquet"
    if feats_path.exists():
        feats = pd.read_parquet(feats_path)
    else:
        # Fallback for old name, if needed
        feats = pd.read_parquet(ART / "customer_features.parquet")

    monthly = pd.read_parquet(ART / "monthly_sales.parquet")
    return scores, feats, monthly

@st.cache_data
def load_optional_artifacts():
    seg_summary = None
    shap_top = None
    if (ART / "segment_summary.parquet").exists():
        seg_summary = pd.read_parquet(ART / "segment_summary.parquet")
    if (ART / "shap_top_features.parquet").exists():
        shap_top = pd.read_parquet(ART / "shap_top_features.parquet")
    return seg_summary, shap_top

scores, feats, monthly = load_core_artifacts()
segment_summary, shap_top = load_optional_artifacts()

# --- Get last updated time ---
last_updated_str = "Not available"
scores_path = ART / "churn_scores.parquet"
if scores_path.exists():
    last_updated_ts = scores_path.stat().st_mtime
    last_updated_str = datetime.datetime.fromtimestamp(last_updated_ts).strftime('%Y-%m-%d %H:%M:%S')


# Optionally enhance scores with basic columns
if "customer_id" in feats.columns and "customer_id" in scores.columns:
    cols = ["customer_id"]
    # UPDATED: Changed 'segment_k5' to 'segment'
    extra = [c for c in ["plan","region","age","tx_count","amt_sum","recency_days","segment","segment_label","churned"] if c in feats.columns]
    scores = scores.merge(feats[cols + extra], on="customer_id", how="left")

# ---------------------------
# Header and KPIs
# ---------------------------
st.title("Customer Churn & Sales Dashboard")

col1, col2, col3, col4 = st.columns(4)

if "churned" in feats.columns:
    active_customers = int((feats["churned"] == 0).sum())
    churn_rate = float(feats["churned"].mean() * 100)
else:
    active_customers = len(feats)
    churn_rate = float(scores["churn_proba"].mean() * 100)

revenue_total = float(feats["amt_sum"].sum()) if "amt_sum" in feats.columns else 0.0
arpu = float(feats["amt_sum"].mean()) if "amt_sum" in feats.columns else 0.0

col1.metric("Active Customers", f"{active_customers:,}")
col2.metric("Churn Rate (est.)", f"{churn_rate:.1f}%")
col3.metric("Revenue (agg)", f"${revenue_total:,.0f}")
col4.metric("Average Revenue Per User.", f"${arpu:,.2f}")

# ---------------------------
# Sales trend
# ---------------------------
st.subheader("Sales Trend")
if "date" in monthly.columns:
    monthly = monthly.copy()
    monthly["date"] = pd.to_datetime(monthly["date"])
    fig_sales = px.line(
        monthly, x="date", y="amount", title="Monthly Sales", markers=True,
        labels=LABEL_MAP
    )
    st.plotly_chart(fig_sales, use_container_width=True)
else:
    st.info("monthly_sales.parquet missing 'date' column. Please regenerate artifacts.")

# ---------------------------
# Segmentation
# ---------------------------
st.header("Segmentation")

if "segment_label" in feats.columns and feats["segment_label"].nunique() > 1:
    # --- New: Use human-readable labels ---
    st.subheader("Segment Distribution")
    seg_counts = feats["segment_label"].value_counts().reset_index()
    seg_counts.columns = ["segment_label", "count"]

    # Define a logical order for segments from best to worst
    seg_order = ["Champions", "Loyal Customers", "Potential Loyalists", "Needs Attention", "At Risk"]
    ordered_labels = [l for l in seg_order if l in seg_counts["segment_label"].values]
    other_labels = sorted([l for l in seg_counts["segment_label"].values if l not in ordered_labels])
    final_order = ordered_labels + other_labels

    fig_seg = px.bar(
        seg_counts, x="segment_label", y="count", title="Customer Segment Distribution",
        category_orders={"segment_label": final_order},
        labels=LABEL_MAP
    )
    fig_seg.update_xaxes(title_text=LABEL_MAP.get("segment_label", "Segment"))
    st.plotly_chart(fig_seg, use_container_width=True)

    if segment_summary is not None:
        st.subheader("Segment KPI Summary")
        display_summary = segment_summary.copy()
        if 'label' in display_summary.columns:
            # Use labels for a more readable summary table, sort it logically, and rename columns
            display_summary = display_summary.set_index('label').loc[final_order]
        st.dataframe(display_summary.rename(columns=LABEL_MAP), use_container_width=True)

    # Filter by segment
    options = ["All"] + final_order
    sel_seg_label = st.selectbox("Filter by segment", options=options)

    if sel_seg_label != "All":
        sub_scores = scores[scores["segment_label"] == sel_seg_label]
        st.caption(f"Displaying {len(sub_scores):,} customers in segment: '{sel_seg_label}'")
        if "churn_proba" in sub_scores.columns and not sub_scores.empty:
            fig_seg_hist = px.histogram(
                sub_scores, x="churn_proba", nbins=30,
                title=f"Churn Probabilities — Segment {sel_seg_label}",
                labels=LABEL_MAP
            )
            st.plotly_chart(fig_seg_hist, use_container_width=True)
elif "segment" in feats.columns:
    # --- Fallback: Use original numeric segments if labels don't exist ---
    st.info("Human-readable segment labels not found. Showing numeric segments. Re-run the training script to generate new labels.")
    # (The original code for numeric segments would go here, but is omitted for clarity as the new version is preferred)
else:
    st.info("No segments found. Re-run training with KMeans enabled in models.py.")

# ---------------------------
# Explainability
# ---------------------------
st.header("Model Explainability")
if shap_top is not None and not shap_top.empty:
    # Clean up feature names for better readability
    shap_display = shap_top.copy()
    shap_display["feature"] = shap_display["feature"].apply(clean_feature_name)

    fig_imp = px.bar(
        shap_display.sort_values("importance", ascending=True).head(15), # Show top 15
        x="importance", y="feature", orientation="h",
        title="Top Features Driving Churn Predictions",
        labels=LABEL_MAP
    )
    fig_imp.update_yaxes(title_text=None)
    st.plotly_chart(fig_imp, use_container_width=True)
    st.caption("Higher bars indicate features with greater influence on churn predictions.")
else:
    st.info("SHAP importances not available. Ensure training produced shap_top_features.parquet.")

# ---------------------------
# High-risk customers
# ---------------------------
st.subheader("High-Risk Customers")
threshold = st.slider("Churn probability threshold", 0.0, 1.0, 0.6, 0.01)

# Define columns to show, using original names
cols_to_show = ["customer_id", "churn_proba"]
for c in ["segment_label", "plan", "region", "age", "tx_count", "amt_sum", "recency_days"]:
    if c in scores.columns:
        cols_to_show.append(c)

high_risk = scores[scores["churn_proba"] >= threshold].sort_values("churn_proba", ascending=False)
st.caption(f"Customers above threshold: {len(high_risk):,}")

# Prepare dataframe for display with clean labels
if not high_risk.empty:
    display_df = high_risk[cols_to_show].head(1000).rename(columns=LABEL_MAP)
    st.dataframe(display_df, use_container_width=True)
else:
    st.info("No customers found above the selected threshold.")

# ---------------------------
# Customer lookup
# ---------------------------
st.subheader("Customer Lookup")
lookup_id = st.text_input("Enter customer_id to view details")
if lookup_id:
    row = scores[scores["customer_id"] == lookup_id]
    if row.empty:
        st.warning("Customer not found.")
    else:
        # Use the same cols_to_show and rename logic
        display_row = row[cols_to_show].rename(columns=LABEL_MAP)
        st.dataframe(display_row, use_container_width=True)

# ---------------------------
# Footer
# ---------------------------
st.markdown(f"""
---
*Dashboard powered by data-driven insights.*
- **Data Last Refreshed:** {last_updated_str}
- **How to use:** Use the filters and interactive charts to explore customer behavior, identify at-risk segments, and understand the key drivers of churn.
- For the latest insights, ensure the model training script has been run recently.
""")
# dashboard/app.py
# Churn Sentinel — Streamlit Dashboard
# Run: streamlit run dashboard/app.py

import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.express as px
import plotly.graph_objects as go
import shap

from src.config import (
    XGB_MODEL_PATH, SHAP_EXPLAINER_PATH,
    FEATURE_NAMES_PATH, EMAIL_LOG_PATH,
    CHURN_THRESHOLD, PROCESSED_DIR
)

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title = "Churn Sentinel",
    page_icon  = "🛡️",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0F0F1A; color: #EAEAEA; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1A1A2E;
        border-right: 1px solid #2E2E4E;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #1A1A2E;
        border: 1px solid #2E2E4E;
        border-radius: 8px;
        padding: 16px;
    }
    [data-testid="stMetricValue"] { color: #E9C46A; font-size: 2rem; }
    [data-testid="stMetricLabel"] { color: #AAAACC; }

    /* Headers */
    h1, h2, h3 { color: #E9C46A !important; }

    /* Risk badge colors */
    .high-risk  { color: #E63946; font-weight: bold; }
    .med-risk   { color: #F4A261; font-weight: bold; }
    .low-risk   { color: #2A9D8F; font-weight: bold; }

    /* Dataframe */
    [data-testid="stDataFrame"] { border: 1px solid #2E2E4E; }

    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        background-color: #1A1A2E;
        color: #AAAACC;
        border-radius: 4px 4px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #E9C46A !important;
        color: #0F0F1A !important;
        font-weight: bold;
    }

    /* Email card */
    .email-card {
        background-color: #1A1A2E;
        border: 1px solid #2E2E4E;
        border-radius: 8px;
        padding: 20px;
        margin: 10px 0;
    }

    /* Divider */
    hr { border-color: #2E2E4E; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
# LOAD ARTIFACTS (cached)
# ════════════════════════════════════════════════════════
@st.cache_resource
def load_artifacts():
    model         = joblib.load(XGB_MODEL_PATH)
    explainer     = joblib.load(SHAP_EXPLAINER_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)
    return model, explainer, feature_names


@st.cache_data
def load_test_data():
    X_test = pd.read_csv(f"{PROCESSED_DIR}/X_test.csv")
    y_test = pd.read_csv(f"{PROCESSED_DIR}/y_test.csv").squeeze()
    return X_test, y_test


@st.cache_data
def load_email_log():
    if not os.path.exists(EMAIL_LOG_PATH):
        return []
    with open(EMAIL_LOG_PATH, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


# ════════════════════════════════════════════════════════
# SCORE ALL USERS
# ════════════════════════════════════════════════════════
@st.cache_data
def get_risk_table(_model, X_test, y_test):
    probs    = _model.predict_proba(X_test)[:, 1]
    preds    = (probs >= CHURN_THRESHOLD).astype(int)

    df = pd.DataFrame({
        "User ID"      : [f"USR_{i:04d}" for i in range(len(X_test))],
        "Churn Prob"   : probs,
        "Risk Score %"  : (probs * 100).round(1),
        "Flagged"      : preds,
        "True Label"   : y_test.values,
        "tenure"       : X_test["tenure"].values,
        "MonthlyCharges": X_test["MonthlyCharges"].values,
        "active_services": X_test["active_services"].values,
    })

    def risk_label(p):
        if p >= 0.80: return "🔴 High"
        if p >= 0.65: return "🟡 Medium"
        return "🟢 Low"

    df["Risk Level"] = df["Churn Prob"].apply(risk_label)
    return df.sort_values("Churn Prob", ascending=False)


# ════════════════════════════════════════════════════════
# SHAP PLOTS
# ════════════════════════════════════════════════════════
def plot_shap_bar(shap_values, X_test, feature_names):
    """Global SHAP bar chart using plotly."""
    mean_shap = np.abs(shap_values.values).mean(axis=0)
    top_idx   = np.argsort(mean_shap)[-15:]

    fig = go.Figure(go.Bar(
        x           = mean_shap[top_idx],
        y           = [feature_names[i] for i in top_idx],
        orientation = "h",
        marker_color= "#F4A261",
        marker_line = dict(color="#FFFFFF", width=0.3)
    ))
    fig.update_layout(
        title       = "Global Feature Importance (Mean |SHAP|)",
        xaxis_title = "Mean |SHAP Value|",
        plot_bgcolor= "#1A1A2E",
        paper_bgcolor="#1A1A2E",
        font        = dict(color="#EAEAEA"),
        height      = 500
    )
    return fig


def plot_user_shap(shap_values, user_idx, feature_names):
    """Single user SHAP waterfall as plotly bar."""
    sv     = shap_values.values[user_idx]
    top_n  = 12
    sorted_idx = np.argsort(np.abs(sv))[-top_n:]

    values  = sv[sorted_idx]
    names   = [feature_names[i] for i in sorted_idx]
    colors  = ["#E63946" if v > 0 else "#2A9D8F" for v in values]

    fig = go.Figure(go.Bar(
        x           = values,
        y           = names,
        orientation = "h",
        marker_color= colors,
        marker_line = dict(color="#FFFFFF", width=0.3)
    ))
    fig.update_layout(
        title        = f"SHAP Explanation — User {user_idx:04d}",
        xaxis_title  = "SHAP Value (impact on churn probability)",
        plot_bgcolor = "#1A1A2E",
        paper_bgcolor= "#1A1A2E",
        font         = dict(color="#EAEAEA"),
        height       = 400
    )
    fig.add_vline(x=0, line_color="#EAEAEA", line_width=1)
    return fig


def plot_risk_distribution(risk_df):
    """Histogram of churn probability distribution."""
    fig = px.histogram(
        risk_df, x="Risk Score %",
        nbins       = 30,
        color_discrete_sequence=["#E63946"],
        title       = "Churn Probability Distribution"
    )
    fig.add_vline(
        x=CHURN_THRESHOLD * 100,
        line_dash="dash",
        line_color="#E9C46A",
        annotation_text=f"Threshold ({CHURN_THRESHOLD*100:.0f}%)",
        annotation_font_color="#E9C46A"
    )
    fig.update_layout(
        plot_bgcolor = "#1A1A2E",
        paper_bgcolor= "#1A1A2E",
        font         = dict(color="#EAEAEA"),
        height       = 350
    )
    return fig


def plot_model_comparison():
    """Bar chart comparing baseline vs XGBoost metrics."""
    try:
        baseline = joblib.load("outputs/baseline_metrics.pkl")
        xgb      = joblib.load("outputs/xgb_metrics.pkl")
    except Exception:
        return None

    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    labels  = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name   = "Logistic Regression",
        x      = labels,
        y      = [baseline[m]*100 for m in metrics],
        marker_color="#E63946", opacity=0.85
    ))
    fig.add_trace(go.Bar(
        name   = "XGBoost",
        x      = labels,
        y      = [xgb[m]*100 for m in metrics],
        marker_color="#F4A261", opacity=0.85
    ))
    fig.update_layout(
        barmode      = "group",
        title        = "Model Comparison — Baseline vs XGBoost",
        yaxis_title  = "Score (%)",
        yaxis_range  = [50, 100],
        plot_bgcolor = "#1A1A2E",
        paper_bgcolor= "#1A1A2E",
        font         = dict(color="#EAEAEA"),
        height       = 400,
        legend       = dict(
            bgcolor="#1A1A2E",
            bordercolor="#2E2E4E"
        )
    )
    return fig


# ════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════
def render_sidebar(risk_df):
    st.sidebar.markdown("# 🛡️ Churn Sentinel")
    st.sidebar.markdown("*Predictive Retention System*")
    st.sidebar.markdown("---")

    # Navigation
    page = st.sidebar.radio(
        "Navigate",
        ["📊 Overview",
         "👥 Risk Table",
         "🔍 User Explorer",
         "✉️ Email Log",
         "📈 Model Performance"]
    )

    st.sidebar.markdown("---")

    # Quick stats
    flagged = risk_df[risk_df["Flagged"] == 1]
    st.sidebar.metric("Total Users",   f"{len(risk_df):,}")
    st.sidebar.metric("Flagged Users", f"{len(flagged):,}")
    st.sidebar.metric("Churn Rate",
                      f"{risk_df['True Label'].mean()*100:.1f}%")
    st.sidebar.metric("Threshold",     f"{CHURN_THRESHOLD*100:.0f}%")

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Stack:** XGBoost + SHAP + LangChain + Groq"
    )
    st.sidebar.markdown(
        "**Model:** llama-3.1-8b-instant"
    )

    return page


# ════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ════════════════════════════════════════════════════════
def page_overview(risk_df, email_log):
    st.title("🛡️ Churn Sentinel — Dashboard")
    st.markdown(
        "*Predictive churn management powered by "
        "XGBoost + Multi-Agent AI*"
    )
    st.markdown("---")

    # KPI Metrics Row
    flagged      = risk_df[risk_df["Flagged"] == 1]
    high_risk    = risk_df[risk_df["Churn Prob"] >= 0.80]
    emails_sent  = len(email_log)
    disc_count   = sum(1 for e in email_log if e.get("eligible"))

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Users",      f"{len(risk_df):,}")
    col2.metric("🔴 Flagged",       f"{len(flagged):,}")
    col3.metric("🔴 High Risk",     f"{len(high_risk):,}")
    col4.metric("✉️ Emails Sent",   f"{emails_sent:,}")
    col5.metric("💰 Discounts",     f"{disc_count:,}")

    st.markdown("---")

    # Charts Row
    col_left, col_right = st.columns(2)

    with col_left:
        st.plotly_chart(
            plot_risk_distribution(risk_df),
            use_container_width=True
        )

    with col_right:
        # Risk level pie
        risk_counts = risk_df["Risk Level"].value_counts()
        fig_pie = px.pie(
            values = risk_counts.values,
            names  = risk_counts.index,
            title  = "Risk Level Distribution",
            color_discrete_sequence=["#E63946","#F4A261","#2A9D8F"]
        )
        fig_pie.update_layout(
            plot_bgcolor = "#1A1A2E",
            paper_bgcolor= "#1A1A2E",
            font         = dict(color="#EAEAEA"),
            height       = 350
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # Email log preview
    if email_log:
        st.markdown("### ✉️ Recent Emails Generated")
        recent = email_log[-5:][::-1]
        for em in recent:
            col_a, col_b, col_c, col_d = st.columns([2,1,1,3])
            col_a.write(f"**{em.get('user_id','N/A')}**")
            col_b.write(f"{em.get('churn_prob',0)*100:.1f}%")
            col_c.write(f"{em.get('risk_type','N/A')}")
            col_d.write(f"📧 {em.get('subject','N/A')[:45]}")
    else:
        st.info("💡 Run the pipeline first: "
                "`python src/pipeline.py --test`")


# ════════════════════════════════════════════════════════
# PAGE 2 — RISK TABLE
# ════════════════════════════════════════════════════════
def page_risk_table(risk_df):
    st.title("👥 Customer Risk Table")
    st.markdown("---")

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        filter_flagged = st.checkbox("Show flagged only", value=True)
    with col2:
        min_prob = st.slider("Min churn probability",
                             0.0, 1.0, float(CHURN_THRESHOLD), 0.05)
    with col3:
        sort_col = st.selectbox("Sort by",
                                ["Churn Prob", "tenure",
                                 "MonthlyCharges"])

    # Apply filters
    filtered = risk_df.copy()
    if filter_flagged:
        filtered = filtered[filtered["Flagged"] == 1]
    filtered = filtered[filtered["Churn Prob"] >= min_prob]
    filtered = filtered.sort_values(sort_col, ascending=False)

    st.markdown(f"**Showing {len(filtered):,} users**")

    # Display table
    display_cols = ["User ID", "Risk Score %", "Risk Level",
                    "tenure", "MonthlyCharges",
                    "active_services", "True Label"]
    st.dataframe(
        filtered[display_cols].reset_index(drop=True),
        use_container_width=True,
        height=500
    )

    # Download button
    csv = filtered.to_csv(index=False)
    st.download_button(
        label     = "⬇️ Download Risk Table CSV",
        data      = csv,
        file_name = "churn_risk_table.csv",
        mime      = "text/csv"
    )


# ════════════════════════════════════════════════════════
# PAGE 3 — USER EXPLORER
# ════════════════════════════════════════════════════════
def page_user_explorer(risk_df, model,
                        explainer, feature_names, X_test):
    st.title("🔍 User Explorer")
    st.markdown("*Inspect individual SHAP explanations*")
    st.markdown("---")

    # User selector
    flagged_users = risk_df[risk_df["Flagged"] == 1]["User ID"].tolist()

    if not flagged_users:
        st.warning("No flagged users found. Run the pipeline first.")
        return

    selected = st.selectbox(
        "Select a flagged user to inspect",
        flagged_users
    )

    user_idx = int(selected.split("_")[1])

    # User details
    user_row  = risk_df[risk_df["User ID"] == selected].iloc[0]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Churn Probability",
                f"{user_row['Risk Score %']:.1f}%")
    col2.metric("Risk Level",
                user_row["Risk Level"])
    col3.metric("Tenure",
                f"{user_row['tenure']:.0f} months")
    col4.metric("Monthly Charges",
                f"${user_row['MonthlyCharges']:.2f}")

    st.markdown("---")

    # SHAP explanation
    st.markdown("### 🔍 SHAP Feature Explanation")
    st.markdown(
        "*Red bars = push toward churn | "
        "Green bars = push toward retention*"
    )

    with st.spinner("Computing SHAP values..."):
        shap_values = explainer(X_test)

    st.plotly_chart(
        plot_user_shap(shap_values, user_idx, feature_names),
        use_container_width=True
    )

    # Raw feature values
    with st.expander("📋 Raw Feature Values"):
        feature_df = pd.DataFrame({
            "Feature" : feature_names,
            "Value"   : X_test.iloc[user_idx].values,
            "SHAP"    : shap_values.values[user_idx]
        }).sort_values("SHAP", key=abs, ascending=False)
        st.dataframe(feature_df.head(20),
                     use_container_width=True)


# ════════════════════════════════════════════════════════
# PAGE 4 — EMAIL LOG
# ════════════════════════════════════════════════════════
def page_email_log(email_log):
    st.title("✉️ Generated Emails")
    st.markdown("---")

    if not email_log:
        st.warning("No emails found.")
        st.code("python src/pipeline.py --test")
        return

    st.markdown(f"**{len(email_log)} emails generated**")

    # Filter by risk type
    risk_filter = st.selectbox(
        "Filter by risk type",
        ["All", "disengagement", "support_issue"]
    )

    filtered_log = email_log
    if risk_filter != "All":
        filtered_log = [e for e in email_log
                        if e.get("risk_type") == risk_filter]

    # Display emails
    for i, em in enumerate(filtered_log):
        churn_pct = em.get("churn_prob", 0) * 100
        disc      = em.get("discount_pct", 0)
        risk      = em.get("risk_type", "N/A")

        # Color code by risk
        color = "#E63946" if churn_pct >= 80 else "#F4A261"

        with st.expander(
            f"📧 {em.get('user_id','N/A')} — "
            f"{churn_pct:.1f}% risk — "
            f"{em.get('subject','N/A')[:50]}"
        ):
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Churn Risk",  f"{churn_pct:.1f}%")
            col2.metric("Risk Type",   risk)
            col3.metric("Discount",    f"{disc}%")
            col4.metric("Status",      em.get("status","N/A"))

            st.markdown("**Subject:**")
            st.info(em.get("subject", "N/A"))

            st.markdown("**Email Body:**")
            st.markdown(
                f"<div class='email-card'>"
                f"{em.get('body','N/A').replace(chr(10),'<br>')}"
                f"</div>",
                unsafe_allow_html=True
            )

            if em.get("top_reasons"):
                st.markdown("**Top Risk Reasons:**")
                for r in em["top_reasons"]:
                    st.markdown(f"• {r}")


# ════════════════════════════════════════════════════════
# PAGE 5 — MODEL PERFORMANCE
# ════════════════════════════════════════════════════════
def page_model_performance(model, explainer,
                            feature_names, X_test):
    st.title("📈 Model Performance")
    st.markdown("---")

    # Model comparison chart
    fig = plot_model_comparison()
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Run baseline + XGBoost training first")

    st.markdown("---")

    # Global SHAP
    st.markdown("### 🔍 Global SHAP Feature Importance")
    with st.spinner("Computing SHAP values for test set..."):
        shap_values = explainer(X_test.iloc[:200])

    st.plotly_chart(
        plot_shap_bar(shap_values, X_test, feature_names),
        use_container_width=True
    )

    # Metrics table
    st.markdown("### 📊 Detailed Metrics")
    try:
        baseline = joblib.load("outputs/baseline_metrics.pkl")
        xgb      = joblib.load("outputs/xgb_metrics.pkl")

        metrics_df = pd.DataFrame({
            "Metric"              : ["Accuracy","Precision",
                                     "Recall","F1","ROC-AUC"],
            "Logistic Regression" : [f"{baseline[m]*100:.2f}%"
                                     for m in ["accuracy","precision",
                                               "recall","f1","roc_auc"]],
            "XGBoost"             : [f"{xgb[m]*100:.2f}%"
                                     for m in ["accuracy","precision",
                                               "recall","f1","roc_auc"]],
            "Improvement"         : [
                f"{(xgb[m]-baseline[m])*100:+.2f}%"
                for m in ["accuracy","precision",
                          "recall","f1","roc_auc"]
            ]
        })
        st.dataframe(metrics_df, use_container_width=True)
    except Exception:
        st.warning("Run models first to see metrics")


# ════════════════════════════════════════════════════════
# MAIN APP
# ════════════════════════════════════════════════════════
def main():
    # Load everything
    try:
        model, explainer, feature_names = load_artifacts()
        X_test, y_test                  = load_test_data()
        email_log                       = load_email_log()
        risk_df = get_risk_table(model, X_test, y_test)
    except FileNotFoundError as e:
        st.error(f"❌ Missing file: {e}")
        st.info("Run the full pipeline first:\n"
                "```\npython src/pipeline.py --test\n```")
        st.stop()

    # Render sidebar + get page selection
    page = render_sidebar(risk_df)

    # Route to correct page
    if page == "📊 Overview":
        page_overview(risk_df, email_log)

    elif page == "👥 Risk Table":
        page_risk_table(risk_df)

    elif page == "🔍 User Explorer":
        page_user_explorer(
            risk_df, model, explainer, feature_names, X_test
        )

    elif page == "✉️ Email Log":
        page_email_log(email_log)

    elif page == "📈 Model Performance":
        page_model_performance(
            model, explainer, feature_names, X_test
        )


if __name__ == "__main__":
    main()
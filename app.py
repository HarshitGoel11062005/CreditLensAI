import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CreditLens AI",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# PROFESSIONAL UI
# ============================================================

st.markdown("""
<style>
    .main {
        background: #f7f9fc;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .hero {
        padding: 1.4rem 1.6rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #0f172a, #1e3a5f);
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 25px rgba(15, 23, 42, 0.12);
    }

    .hero h1 {
        margin: 0;
        font-size: 2.2rem;
    }

    .hero p {
        margin: 0.5rem 0 0 0;
        opacity: 0.88;
        font-size: 1rem;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        margin-top: 1rem;
        margin-bottom: 0.8rem;
    }

    .metric-card {
        background: white;
        border-radius: 15px;
        padding: 1.1rem 1.2rem;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
        min-height: 115px;
    }

    .metric-label {
        color: #64748b;
        font-size: 0.85rem;
        font-weight: 600;
    }

    .metric-value {
        color: #0f172a;
        font-size: 1.8rem;
        font-weight: 750;
        margin-top: 0.25rem;
    }

    .metric-sub {
        color: #64748b;
        font-size: 0.78rem;
        margin-top: 0.2rem;
    }

    .risk-low {
        color: #15803d;
        font-weight: 700;
    }

    .risk-medium {
        color: #b45309;
        font-weight: 700;
    }

    .risk-high {
        color: #b91c1c;
        font-weight: 700;
    }

    .insight-box {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 15px;
        padding: 1rem 1.1rem;
        margin-bottom: 0.7rem;
    }

    .small-note {
        color: #64748b;
        font-size: 0.78rem;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid #e5e7eb;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("models/creditlens_model.pkl")

model = load_model()

# ============================================================
# FEATURES — EXACT MODEL INPUT STRUCTURE
# ============================================================

BASE_FEATURES = [
    "monthly_revenue",
    "monthly_expenses",
    "total_debt",
    "monthly_emi",
    "average_balance",
    "late_payment_count",
    "total_transactions",
    "avg_transaction_value",
    "revenue_growth",
    "expense_growth",
    "cashflow_volatility",
    "credit_utilization"
]

MODEL_FEATURES = [
    "monthly_revenue",
    "monthly_expenses",
    "total_debt",
    "monthly_emi",
    "average_balance",
    "late_payment_count",
    "total_transactions",
    "avg_transaction_value",
    "revenue_growth",
    "expense_growth",
    "cashflow_volatility",
    "credit_utilization",
    "debt_to_income",
    "expense_ratio",
    "cashflow",
    "repayment_score",
    "transaction_consistency",
    "cash_reserve_ratio"
]

# ============================================================
# FEATURE ENGINEERING
# ============================================================

def engineer_features(df):
    df = df.copy()

    df["debt_to_income"] = (
        df["monthly_emi"] /
        df["monthly_revenue"].replace(0, np.nan)
    )

    df["expense_ratio"] = (
        df["monthly_expenses"] /
        df["monthly_revenue"].replace(0, np.nan)
    )

    df["cashflow"] = (
        df["monthly_revenue"] - df["monthly_expenses"]
    )

    df["repayment_score"] = (
        1 / (1 + df["late_payment_count"])
    )

    df["transaction_consistency"] = (
        df["total_transactions"] /
        (df["avg_transaction_value"] + 1)
    )

    df["cash_reserve_ratio"] = (
        df["average_balance"] /
        (df["monthly_expenses"] + 1)
    )

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    return df

# ============================================================
# CREDIT INTELLIGENCE SCORE
# ============================================================

def calculate_credit_score(row):
    score = 100

    if row["debt_to_income"] > 0.50:
        score -= 20
    elif row["debt_to_income"] > 0.35:
        score -= 10

    if row["expense_ratio"] > 0.85:
        score -= 20
    elif row["expense_ratio"] > 0.70:
        score -= 10

    if row["late_payment_count"] >= 5:
        score -= 20
    elif row["late_payment_count"] >= 3:
        score -= 10
    elif row["late_payment_count"] >= 1:
        score -= 5

    if row["cashflow_volatility"] > 0.60:
        score -= 15
    elif row["cashflow_volatility"] > 0.40:
        score -= 8

    if row["credit_utilization"] > 0.80:
        score -= 15
    elif row["credit_utilization"] > 0.60:
        score -= 8

    if row["revenue_growth"] > 0.10:
        score += 5
    elif row["revenue_growth"] < -0.10:
        score -= 10

    return max(0, min(100, score))


def risk_category(score):
    if score >= 75:
        return "Low Risk"
    elif score >= 50:
        return "Medium Risk"
    return "High Risk"


# ============================================================
# FINANCIAL HEALTH SCORE
# ============================================================

def calculate_financial_health(row):
    score = 0

    if row["revenue_growth"] >= 0.05:
        score += 20
    elif row["revenue_growth"] >= -0.05:
        score += 12
    else:
        score += 5

    if row["cashflow"] > 0:
        score += 20
    else:
        score += 5

    if row["debt_to_income"] < 0.30:
        score += 20
    elif row["debt_to_income"] < 0.50:
        score += 12
    else:
        score += 5

    if row["late_payment_count"] == 0:
        score += 20
    elif row["late_payment_count"] <= 2:
        score += 12
    else:
        score += 5

    if row["transaction_consistency"] > 1:
        score += 20
    else:
        score += 10

    return min(100, score)


# ============================================================
# DEMO DATA
# ============================================================

def demo_data():
    return pd.DataFrame({
        "business_id": ["SME001", "SME002", "SME003", "SME004", "SME005"],
        "monthly_revenue": [800000, 450000, 1200000, 600000, 950000],
        "monthly_expenses": [500000, 380000, 650000, 520000, 580000],
        "total_debt": [1200000, 1500000, 900000, 1800000, 1100000],
        "monthly_emi": [80000, 120000, 60000, 150000, 70000],
        "average_balance": [450000, 180000, 750000, 160000, 520000],
        "late_payment_count": [1, 4, 0, 5, 1],
        "total_transactions": [95, 70, 130, 65, 110],
        "avg_transaction_value": [8500, 6200, 9200, 5400, 8100],
        "revenue_growth": [0.12, -0.08, 0.18, -0.15, 0.10],
        "expense_growth": [0.08, 0.15, 0.05, 0.20, 0.07],
        "cashflow_volatility": [0.25, 0.62, 0.18, 0.70, 0.28],
        "credit_utilization": [0.42, 0.78, 0.30, 0.85, 0.38]
    })


# ============================================================
# ANALYSIS PIPELINE
# ============================================================

def analyze_data(input_df):
    df = input_df.copy()

    missing = [c for c in BASE_FEATURES if c not in df.columns]
    if missing:
        return None, "MISSING", missing

    df[BASE_FEATURES] = df[BASE_FEATURES].apply(
        pd.to_numeric, errors="coerce"
    )

    if df[BASE_FEATURES].isna().any().any():
        return None, "INVALID", None

    df = engineer_features(df)

    X = df[MODEL_FEATURES]

    df["ml_prediction"] = model.predict(X)
    df["ml_probability"] = model.predict_proba(X)[:, 1]

    df["credit_score"] = df.apply(
        calculate_credit_score,
        axis=1
    )

    df["risk_category"] = df["credit_score"].apply(
        risk_category
    )

    df["financial_health_score"] = df.apply(
        calculate_financial_health,
        axis=1
    )

    if len(df) >= 5:
        anomaly_features = [
            "monthly_revenue",
            "monthly_expenses",
            "total_debt",
            "average_balance",
            "total_transactions",
            "avg_transaction_value"
        ]

        anomaly_model = IsolationForest(
            contamination=min(0.05, max(1 / len(df), 0.01)),
            random_state=42
        )

        df["anomaly_prediction"] = anomaly_model.fit_predict(
            df[anomaly_features]
        )

        df["anomaly_status"] = df["anomaly_prediction"].map({
            1: "Normal",
            -1: "Anomaly"
        })
    else:
        df["anomaly_status"] = "Need 5+ records"

    return df, "OK", None


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def money(value):
    return f"₹{value:,.0f}"


def score_progress(label, value):
    st.markdown(f"**{label}**")
    st.progress(int(max(0, min(100, value))))
    st.caption(f"{int(value)}/100")


def risk_class(risk):
    if risk == "Low Risk":
        return "risk-low"
    elif risk == "Medium Risk":
        return "risk-medium"
    return "risk-high"


# ============================================================
# SESSION STATE
# ============================================================

if "analysis_df" not in st.session_state:
    demo, _, _ = analyze_data(demo_data())
    st.session_state.analysis_df = demo

if "source_name" not in st.session_state:
    st.session_state.source_name = "Demo SME dataset"


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("## 💳 CreditLens AI")
st.sidebar.caption("SME Credit Intelligence Platform")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Risk Analysis",
        "Anomaly Detection",
        "AI Copilot"
    ]
)

st.sidebar.divider()

st.sidebar.markdown("### 📁 Business Data")

uploaded_file = st.sidebar.file_uploader(
    "Upload SME CSV",
    type=["csv"],
    help="CSV must contain the 12 base financial features."
)

if uploaded_file is not None:

    if st.session_state.get("uploaded_name") != uploaded_file.name:

        try:
            input_df = pd.read_csv(uploaded_file)
            analyzed, status, details = analyze_data(input_df)

            if status == "MISSING":
                st.sidebar.error(
                    "Missing columns: " + ", ".join(details)
                )

            elif status == "INVALID":
                st.sidebar.error(
                    "Some financial columns contain invalid values."
                )

            else:
                st.session_state.analysis_df = analyzed
                st.session_state.source_name = uploaded_file.name
                st.session_state.uploaded_name = uploaded_file.name
                st.sidebar.success(
                    f"{len(analyzed)} records analyzed."
                )

        except Exception as e:
            st.sidebar.error(f"Could not process CSV: {e}")

demo_csv = demo_data().to_csv(index=False)

st.sidebar.download_button(
    "⬇️ Download Sample CSV",
    data=demo_csv,
    file_name="creditlens_sample_sme_data.csv",
    mime="text/csv"
)

st.sidebar.divider()
st.sidebar.caption(
    "Prototype model trained on synthetic SME data."
)


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="hero">
    <h1>💳 CreditLens AI</h1>
    <p>AI-Based SME Credit Intelligence & Financial Risk Assessment</p>
</div>
""", unsafe_allow_html=True)

st.caption(
    f"Current dataset: **{st.session_state.source_name}**"
)


df = st.session_state.analysis_df


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.markdown(
        '<div class="section-title">📊 Executive Risk Dashboard</div>',
        unsafe_allow_html=True
    )

    avg_credit = int(round(df["credit_score"].mean()))
    avg_health = int(round(df["financial_health_score"].mean()))
    high_risk = int((df["risk_category"] == "High Risk").sum())
    anomaly_count = int((df["anomaly_status"] == "Anomaly").sum())

    c1, c2, c3, c4 = st.columns(4)

    cards = [
        ("Average Credit Score", f"{avg_credit}/100", "Credit intelligence"),
        ("Financial Health", f"{avg_health}/100", "Business health"),
        ("High-Risk Businesses", str(high_risk), "Requires attention"),
        ("Anomalies Detected", str(anomaly_count), "Unusual patterns")
    ]

    for col, (label, value, sub) in zip(
        [c1, c2, c3, c4], cards
    ):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-sub">{sub}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.divider()

    # Overall scores
    left, right = st.columns(2)

    with left:
        st.markdown("### 🎯 Portfolio Credit Score")
        score_progress("Average Credit Intelligence", avg_credit)

    with right:
        st.markdown("### ❤️ Portfolio Financial Health")
        score_progress("Average Financial Health", avg_health)

    st.divider()

    # Financial overview
    st.markdown("### 💰 Financial Overview")

    financial_data = pd.DataFrame({
        "Metric": [
            "Monthly Revenue",
            "Monthly Expenses",
            "Total Debt",
            "Average Balance"
        ],
        "Average Amount": [
            df["monthly_revenue"].mean(),
            df["monthly_expenses"].mean(),
            df["total_debt"].mean(),
            df["average_balance"].mean()
        ]
    })

    chart_col, summary_col = st.columns([2, 1])

    with chart_col:
        st.bar_chart(
            financial_data.set_index("Metric")
        )

    with summary_col:
        avg_revenue = df["monthly_revenue"].mean()
        avg_expenses = df["monthly_expenses"].mean()
        avg_cashflow = df["cashflow"].mean()

        st.markdown("#### Portfolio Snapshot")
        st.metric("Avg Revenue", money(avg_revenue))
        st.metric("Avg Expenses", money(avg_expenses))
        st.metric("Avg Cashflow", money(avg_cashflow))

    st.divider()

    # Risk distribution
    st.markdown("### 🎯 Risk Distribution")

    risk_counts = (
        df["risk_category"]
        .value_counts()
        .reindex(
            ["Low Risk", "Medium Risk", "High Risk"],
            fill_value=0
        )
    )

    rc1, rc2 = st.columns([1, 2])

    with rc1:
        st.dataframe(
            risk_counts.rename("Businesses"),
            use_container_width=True
        )

    with rc2:
        st.bar_chart(risk_counts)

    st.divider()

    # Attention required
    st.markdown("### 🚨 Businesses Requiring Attention")

    attention = df[
        (df["risk_category"] == "High Risk") |
        (df["anomaly_status"] == "Anomaly")
    ].copy()

    attention_cols = [
        c for c in [
            "business_id",
            "credit_score",
            "financial_health_score",
            "risk_category",
            "ml_probability",
            "anomaly_status"
        ]
        if c in attention.columns
    ]

    if len(attention) > 0:
        st.dataframe(
            attention[attention_cols],
            use_container_width=True
        )
    else:
        st.success(
            "No high-risk or anomalous businesses require immediate attention."
        )

    st.caption(
        "Prototype note: risk predictions are based on the current synthetic training dataset."
    )


# ============================================================
# RISK ANALYSIS
# ============================================================

elif page == "Risk Analysis":

    st.markdown(
        '<div class="section-title">🔍 Business-Level Credit Risk Analysis</div>',
        unsafe_allow_html=True
    )

    if "business_id" in df.columns:
        options = df["business_id"].astype(str).tolist()

        selected = st.selectbox(
            "Select Business",
            options
        )

        row = df[
            df["business_id"].astype(str) == selected
        ].iloc[0]
    else:
        selected_index = st.selectbox(
            "Select Business Record",
            range(len(df))
        )
        row = df.iloc[selected_index]

    st.markdown(f"### 🏢 Business Profile: {row.get('business_id', 'Selected Record')}")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Credit Score",
        f"{int(row['credit_score'])}/100"
    )

    c2.metric(
        "Financial Health",
        f"{int(row['financial_health_score'])}/100"
    )

    c3.metric(
        "Risk Category",
        row["risk_category"]
    )

    c4.metric(
        "High-Risk Probability",
        f"{row['ml_probability'] * 100:.1f}%"
    )

    st.divider()

    # Score bars
    p1, p2 = st.columns(2)

    with p1:
        score_progress(
            "Credit Intelligence Score",
            row["credit_score"]
        )

    with p2:
        score_progress(
            "Financial Health Score",
            row["financial_health_score"]
        )

    st.divider()

    st.markdown("### 📌 Key Financial Indicators")

    k1, k2, k3, k4 = st.columns(4)

    k1.metric(
        "Debt-to-Income",
        f"{row['debt_to_income']:.2f}"
    )

    k2.metric(
        "Expense Ratio",
        f"{row['expense_ratio']:.2f}"
    )

    k3.metric(
        "Repayment Score",
        f"{row['repayment_score'] * 100:.0f}/100"
    )

    k4.metric(
        "Credit Utilization",
        f"{row['credit_utilization'] * 100:.1f}%"
    )

    st.divider()

    # Automated explanation
    positive = []
    risks = []

    if row["revenue_growth"] >= 0.05:
        positive.append("Positive revenue growth")
    else:
        risks.append("Revenue growth is weak or negative")

    if row["cashflow"] > 0:
        positive.append("Positive operating cash-flow")
    else:
        risks.append("Negative operating cash-flow")

    if row["debt_to_income"] < 0.30:
        positive.append("Healthy debt-to-income level")
    elif row["debt_to_income"] >= 0.50:
        risks.append("High debt-to-income ratio")

    if row["late_payment_count"] == 0:
        positive.append("No late payments")
    elif row["late_payment_count"] >= 3:
        risks.append("Multiple late payments")

    if row["credit_utilization"] > 0.80:
        risks.append("High credit utilization")
    elif row["credit_utilization"] < 0.60:
        positive.append("Controlled credit utilization")

    ex1, ex2 = st.columns(2)

    with ex1:
        st.markdown("### ✅ Positive Factors")
        if positive:
            for item in positive:
                st.success(item)
        else:
            st.info("No strong positive indicators identified.")

    with ex2:
        st.markdown("### ⚠️ Risk Factors")
        if risks:
            for item in risks:
                st.warning(item)
        else:
            st.success("No major rule-based risk factors identified.")

    st.divider()

    st.markdown("### 💰 Business Financial Data")

    raw_cols = [
        c for c in BASE_FEATURES
        if c in row.index
    ]

    st.dataframe(
        pd.DataFrame([row[raw_cols]]),
        use_container_width=True
    )


# ============================================================
# ANOMALY DETECTION
# ============================================================

elif page == "Anomaly Detection":

    st.markdown(
        '<div class="section-title">🚨 Financial Anomaly Detection</div>',
        unsafe_allow_html=True
    )

    anomaly_count = int(
        (df["anomaly_status"] == "Anomaly").sum()
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Records Analyzed",
        len(df)
    )

    c2.metric(
        "Anomalies Detected",
        anomaly_count
    )

    c3.metric(
        "Normal Records",
        int((df["anomaly_status"] == "Normal").sum())
    )

    st.divider()

    if len(df) < 5:
        st.warning(
            "Upload at least 5 business records for batch anomaly detection."
        )

    anomaly_cols = [
        c for c in [
            "business_id",
            "monthly_revenue",
            "monthly_expenses",
            "total_debt",
            "average_balance",
            "credit_score",
            "risk_category",
            "anomaly_status"
        ]
        if c in df.columns
    ]

    st.markdown("### 📋 Anomaly Monitoring Table")

    st.dataframe(
        df[anomaly_cols],
        use_container_width=True
    )

    anomalies = df[
        df["anomaly_status"] == "Anomaly"
    ]

    if len(anomalies) > 0:
        st.error(
            f"⚠️ {len(anomalies)} unusual financial record(s) detected."
        )

        st.dataframe(
            anomalies[anomaly_cols],
            use_container_width=True
        )
    elif len(df) >= 5:
        st.success(
            "No unusual financial records detected in the current batch."
        )

    st.info(
        "Current prototype uses Isolation Forest on the uploaded batch. "
        "For production, a separately trained anomaly model should be persisted."
    )


# ============================================================
# AI COPILOT
# ============================================================

elif page == "AI Copilot":

    st.markdown(
        '<div class="section-title">🤖 CreditLens AI Copilot</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Ask questions about the current SME portfolio."
    )

    st.info(
        "The current Copilot is a rule-based prototype. "
        "An LLM-powered Copilot will be added in the next development phase."
    )

    question = st.text_input(
        "Ask CreditLens",
        placeholder="Which businesses are high risk?"
    )

    if question:

        q = question.lower()

        if "high risk" in q:
            high = df[
                df["risk_category"] == "High Risk"
            ]

            st.success(
                f"CreditLens found {len(high)} high-risk business(es)."
            )

            if len(high) > 0:
                cols = [
                    c for c in [
                        "business_id",
                        "credit_score",
                        "financial_health_score"
                    ]
                    if c in high.columns
                ]
                st.dataframe(
                    high[cols],
                    use_container_width=True
                )

        elif "anomal" in q:
            count = int(
                (df["anomaly_status"] == "Anomaly").sum()
            )

            st.info(
                f"CreditLens detected {count} anomalous record(s)."
            )

        elif "score" in q:
            st.info(
                f"The average Credit Intelligence Score is "
                f"{df['credit_score'].mean():.1f}/100."
            )

        elif "health" in q:
            st.info(
                f"The average Financial Health Score is "
                f"{df['financial_health_score'].mean():.1f}/100."
            )

        elif "best" in q or "lowest risk" in q:
            best = df.sort_values(
                "credit_score",
                ascending=False
            ).head(5)

            st.success(
                "Here are the strongest businesses by Credit Intelligence Score."
            )

            cols = [
                c for c in [
                    "business_id",
                    "credit_score",
                    "financial_health_score",
                    "risk_category"
                ]
                if c in best.columns
            ]

            st.dataframe(
                best[cols],
                use_container_width=True
            )

        else:
            st.info(
                "Try asking about high-risk businesses, anomalies, "
                "credit score, financial health, or lowest-risk businesses."
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "CreditLens AI • SME Credit Intelligence Prototype • "
    "AI/ML model trained on synthetic data"
)

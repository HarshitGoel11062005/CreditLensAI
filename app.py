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
    layout="wide"
)

# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("models/creditlens_model.pkl")

model = load_model()

# ============================================================
# FEATURES — MUST MATCH THE COLAB MODEL
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
# FEATURE ENGINEERING — SAME AS COLAB
# ============================================================

def engineer_features(df):
    df = df.copy()

    df["debt_to_income"] = (
        df["monthly_emi"] / df["monthly_revenue"].replace(0, np.nan)
    )

    df["expense_ratio"] = (
        df["monthly_expenses"] / df["monthly_revenue"].replace(0, np.nan)
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
# CREDIT INTELLIGENCE SCORE — SAME LOGIC AS COLAB
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
# FINANCIAL HEALTH SCORE — SAME LOGIC AS COLAB
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
# PREPARE ANALYSIS
# ============================================================

def analyze_data(input_df):
    df = input_df.copy()

    missing = [col for col in BASE_FEATURES if col not in df.columns]
    if missing:
        return None, missing

    df[BASE_FEATURES] = df[BASE_FEATURES].apply(
        pd.to_numeric, errors="coerce"
    )

    if df[BASE_FEATURES].isna().any().any():
        return "INVALID_VALUES", None

    df = engineer_features(df)

    X = df[MODEL_FEATURES]

    df["ml_prediction"] = model.predict(X)
    df["ml_probability"] = model.predict_proba(X)[:, 1]

    df["credit_score"] = df.apply(calculate_credit_score, axis=1)
    df["risk_category"] = df["credit_score"].apply(risk_category)
    df["financial_health_score"] = df.apply(
        calculate_financial_health, axis=1
    )

    # Batch anomaly detection.
    # IsolationForest needs multiple records to identify unusual records.
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

    return df, None

# ============================================================
# HEADER
# ============================================================

st.title("💳 CreditLens AI")
st.subheader("AI-Based SME Credit Intelligence & Financial Risk Assessment")
st.write(
    "Analyze SME financial data, evaluate credit risk, "
    "measure financial health, and identify unusual financial patterns."
)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("CreditLens AI")

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
st.sidebar.caption("Prototype • AI/ML Credit Intelligence")

# ============================================================
# DATA INPUT
# ============================================================

if "analysis_df" not in st.session_state:
    st.session_state.analysis_df = demo_data()

if page in ["Dashboard", "Risk Analysis", "Anomaly Detection"]:
    st.sidebar.subheader("Business Data")

    uploaded_file = st.sidebar.file_uploader(
        "Upload SME CSV",
        type=["csv"]
    )

    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            analyzed, error = analyze_data(input_df)

            if error is not None:
                if error == "INVALID_VALUES":
                    st.sidebar.error(
                        "Some required financial columns contain invalid values."
                    )
                else:
                    st.sidebar.error(
                        "Missing columns: " + ", ".join(error)
                    )
            else:
                st.session_state.analysis_df = analyzed
                st.sidebar.success(
                    f"{len(analyzed)} business records analyzed."
                )
        except Exception as e:
            st.sidebar.error(f"Could not process CSV: {e}")

    if uploaded_file is None and not st.session_state.analysis_df.equals(demo_data()):
        pass

    if uploaded_file is None:
        analyzed, _ = analyze_data(st.session_state.analysis_df)
        st.session_state.analysis_df = analyzed

# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    df = st.session_state.analysis_df

    st.header("📊 Financial Risk Dashboard")

    avg_credit = int(round(df["credit_score"].mean()))
    avg_health = int(round(df["financial_health_score"].mean()))
    high_risk_count = int((df["risk_category"] == "High Risk").sum())
    anomaly_count = int((df["anomaly_status"] == "Anomaly").sum())

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Avg Credit Score", f"{avg_credit}/100")
    col2.metric("Financial Health", f"{avg_health}/100")
    col3.metric("High-Risk Businesses", high_risk_count)
    col4.metric("Anomalies", anomaly_count)

    st.divider()

    st.subheader("📈 Financial Overview")

    overview = pd.DataFrame({
        "Metric": ["Revenue", "Expenses", "Debt", "Average Balance"],
        "Amount": [
            df["monthly_revenue"].mean(),
            df["monthly_expenses"].mean(),
            df["total_debt"].mean(),
            df["average_balance"].mean()
        ]
    })

    st.bar_chart(overview.set_index("Metric"))

    st.subheader("🎯 Risk Distribution")

    risk_counts = (
        df["risk_category"]
        .value_counts()
        .reindex(["Low Risk", "Medium Risk", "High Risk"], fill_value=0)
    )

    st.bar_chart(risk_counts)

    st.subheader("Business Risk Table")

    display_cols = [
        c for c in [
            "business_id",
            "credit_score",
            "financial_health_score",
            "risk_category",
            "ml_probability",
            "anomaly_status"
        ] if c in df.columns
    ]

    st.dataframe(
        df[display_cols],
        use_container_width=True
    )

    st.caption(
        "Prototype note: current model was trained on synthetic SME data."
    )

# ============================================================
# RISK ANALYSIS
# ============================================================

elif page == "Risk Analysis":

    df = st.session_state.analysis_df

    st.header("🔍 Credit Risk Analysis")

    if "business_id" in df.columns:
        selected = st.selectbox(
            "Select Business",
            df["business_id"].astype(str).tolist()
        )
        row = df[df["business_id"].astype(str) == selected].iloc[0]
    else:
        selected_index = st.selectbox(
            "Select Business Record",
            range(len(df))
        )
        row = df.iloc[selected_index]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Credit Score", f"{int(row['credit_score'])}/100")
    col2.metric(
        "Financial Health",
        f"{int(row['financial_health_score'])}/100"
    )
    col3.metric("Risk Category", row["risk_category"])
    col4.metric(
        "ML High-Risk Probability",
        f"{row['ml_probability'] * 100:.1f}%"
    )

    st.divider()

    st.subheader("Financial Indicators")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Debt-to-Income",
        f"{row['debt_to_income']:.2f}"
    )

    c2.metric(
        "Expense Ratio",
        f"{row['expense_ratio']:.2f}"
    )

    c3.metric(
        "Repayment Score",
        f"{row['repayment_score'] * 100:.0f}/100"
    )

    st.subheader("Why this score?")

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

    if positive:
        st.write("### ✅ Positive Factors")
        for item in positive:
            st.success(item)

    if risks:
        st.write("### ⚠️ Risk Factors")
        for item in risks:
            st.warning(item)

    st.subheader("Business Financial Data")

    raw_cols = [c for c in BASE_FEATURES if c in row.index]
    st.dataframe(
        pd.DataFrame([row[raw_cols]]),
        use_container_width=True
    )

# ============================================================
# ANOMALY DETECTION
# ============================================================

elif page == "Anomaly Detection":

    df = st.session_state.analysis_df

    st.header("🚨 Financial Anomaly Detection")

    anomaly_count = int((df["anomaly_status"] == "Anomaly").sum())

    col1, col2 = st.columns(2)

    col1.metric("Records Analyzed", len(df))
    col2.metric("Anomalies Detected", anomaly_count)

    st.divider()

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
        ] if c in df.columns
    ]

    st.dataframe(
        df[anomaly_cols],
        use_container_width=True
    )

    anomalies = df[df["anomaly_status"] == "Anomaly"]

    if len(anomalies) > 0:
        st.error(
            f"⚠️ {len(anomalies)} unusual financial record(s) detected."
        )
        st.dataframe(
            anomalies[anomaly_cols],
            use_container_width=True
        )
    else:
        st.success("No unusual financial records detected.")

    st.info(
        "Isolation Forest is currently applied to the uploaded batch. "
        "A production version should use a separately trained anomaly model."
    )

# ============================================================
# AI COPILOT
# ============================================================

elif page == "AI Copilot":

    st.header("🤖 CreditLens AI Copilot")

    st.write(
        "Ask questions about the analyzed financial risk profile."
    )

    df = st.session_state.analysis_df

    question = st.text_input(
        "Ask CreditLens:",
        placeholder="Which businesses are high risk?"
    )

    if question:

        q = question.lower()

        if "high risk" in q:
            high = df[df["risk_category"] == "High Risk"]
            st.info(
                f"CreditLens found {len(high)} high-risk business(es) "
                f"in the current dataset."
            )
        elif "anomal" in q:
            count = int((df["anomaly_status"] == "Anomaly").sum())
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
        else:
            st.info(
                "For the current prototype, try questions about "
                "high risk businesses, anomalies, scores, or financial health."
            )

# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption(
    "CreditLens AI • Prototype for SME Credit Intelligence • "
    "Model trained on synthetic data"
)

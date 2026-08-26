import streamlit as st
import pandas as pd
import numpy as np


# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="CreditLens AI",
    page_icon="💳",
    layout="wide"
)


# ============================================
# HEADER
# ============================================

st.title("💳 CreditLens AI")

st.subheader(
    "AI-Based SME Credit Intelligence & Financial Risk Assessment"
)

st.write(
    "Analyze SME financial data, evaluate credit risk, "
    "identify anomalies, and understand financial health."
)


# ============================================
# SIDEBAR
# ============================================

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


# ============================================
# DASHBOARD
# ============================================

if page == "Dashboard":

    st.header("📊 Financial Risk Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Credit Score",
        "81/100"
    )

    col2.metric(
        "Financial Health",
        "84/100"
    )

    col3.metric(
        "Risk Level",
        "Low"
    )

    col4.metric(
        "Anomalies",
        "2"
    )

    st.divider()

    st.subheader("Financial Overview")

    chart_data = pd.DataFrame({
        "Month": [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun"
        ],

        "Revenue": [
            650000,
            680000,
            710000,
            690000,
            760000,
            810000
        ],

        "Expenses": [
            420000,
            430000,
            450000,
            440000,
            470000,
            490000
        ]
    })

    st.line_chart(
        chart_data.set_index("Month")
    )

    st.subheader("Risk Summary")

    col1, col2 = st.columns(2)

    with col1:

        st.success(
            "Stable revenue growth"
        )

        st.success(
            "Strong repayment behavior"
        )

    with col2:

        st.warning(
            "Debt utilization is increasing"
        )

        st.warning(
            "Cash-flow volatility detected"
        )


# ============================================
# RISK ANALYSIS
# ============================================

elif page == "Risk Analysis":

    st.header("🔍 Credit Risk Analysis")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Credit Intelligence Score",
            "81/100"
        )

        st.metric(
            "Debt-to-Income",
            "0.31"
        )

        st.metric(
            "Repayment Score",
            "91/100"
        )

    with col2:

        st.metric(
            "Financial Health",
            "84/100"
        )

        st.metric(
            "Cash-flow Stability",
            "78/100"
        )

        st.metric(
            "Revenue Stability",
            "86/100"
        )

    st.divider()

    st.subheader("Why this score?")

    st.write("### Positive Factors")

    st.success("Stable revenue growth")

    st.success("Strong repayment history")

    st.success("Consistent transaction activity")

    st.write("### Risk Factors")

    st.warning("Debt utilization has increased")

    st.warning("Cash reserves are declining")


# ============================================
# ANOMALY DETECTION
# ============================================

elif page == "Anomaly Detection":

    st.header("🚨 Transaction Anomaly Detection")

    anomaly_data = pd.DataFrame({

        "Transaction": [
            "TXN-001",
            "TXN-002",
            "TXN-003",
            "TXN-004",
            "TXN-005"
        ],

        "Amount": [
            12000,
            18500,
            15000,
            850000,
            21000
        ],

        "Status": [
            "Normal",
            "Normal",
            "Normal",
            "Anomaly",
            "Normal"
        ]
    })

    st.dataframe(
        anomaly_data,
        use_container_width=True
    )

    st.error(
        "⚠️ Unusual transaction detected: ₹8,50,000"
    )


# ============================================
# AI COPILOT
# ============================================

elif page == "AI Copilot":

    st.header("🤖 CreditLens AI Copilot")

    st.write(
        "Ask questions about the financial risk profile."
    )

    question = st.text_input(
        "Ask CreditLens:"
    )

    if question:

        st.info(
            "The AI Copilot will analyze the business "
            "financial profile and generate an explanation."
        )

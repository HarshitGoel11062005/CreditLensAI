import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from io import BytesIO

# Optional PDF report dependency.
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


# ============================================================
# CREDITLENS AI — ALL-IN-ONE PRODUCT APP
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
    .main { background: #f7f9fc; }
    .block-container { padding-top: 1.8rem; padding-bottom: 2rem; }

    .hero {
        padding: 1.5rem 1.7rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #0f172a, #1e3a5f);
        color: white;
        margin-bottom: 1.4rem;
        box-shadow: 0 8px 25px rgba(15,23,42,.12);
    }

    .hero h1 { margin: 0; font-size: 2.25rem; }
    .hero p { margin: .45rem 0 0; opacity: .88; }

    .section-title {
        font-size: 1.35rem;
        font-weight: 750;
        margin: .8rem 0;
    }

    .metric-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 15px;
        padding: 1rem 1.15rem;
        min-height: 110px;
        box-shadow: 0 4px 14px rgba(15,23,42,.05);
    }

    .metric-label { color:#64748b; font-size:.83rem; font-weight:650; }
    .metric-value { color:#0f172a; font-size:1.75rem; font-weight:800; margin-top:.25rem; }
    .metric-sub { color:#64748b; font-size:.76rem; }

    .insight {
        background:white;
        border:1px solid #e5e7eb;
        border-radius:14px;
        padding:1rem;
        margin-bottom:.65rem;
    }

    .small-note { color:#64748b; font-size:.78rem; }
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
# MODEL FEATURES
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

MODEL_FEATURES = BASE_FEATURES + [
    "debt_to_income",
    "expense_ratio",
    "cashflow",
    "repayment_score",
    "transaction_consistency",
    "cash_reserve_ratio"
]


# ============================================================
# SMART COLUMN NORMALIZATION
# ============================================================

ALIASES = {
    "revenue": "monthly_revenue",
    "monthly sales": "monthly_revenue",
    "sales": "monthly_revenue",
    "income": "monthly_revenue",
    "monthly income": "monthly_revenue",
    "turnover": "monthly_revenue",

    "expenses": "monthly_expenses",
    "monthly expense": "monthly_expenses",
    "costs": "monthly_expenses",
    "monthly costs": "monthly_expenses",

    "debt": "total_debt",
    "loan": "total_debt",
    "total loan": "total_debt",
    "outstanding debt": "total_debt",
    "outstanding loan": "total_debt",

    "emi": "monthly_emi",
    "monthly loan payment": "monthly_emi",
    "loan payment": "monthly_emi",
    "monthly emi": "monthly_emi",

    "balance": "average_balance",
    "bank balance": "average_balance",
    "average bank balance": "average_balance",
    "avg balance": "average_balance",

    "late payments": "late_payment_count",
    "late payment count": "late_payment_count",
    "delayed payments": "late_payment_count",
    "missed payments": "late_payment_count",

    "transactions": "total_transactions",
    "transaction count": "total_transactions",
    "number of transactions": "total_transactions",

    "average transaction": "avg_transaction_value",
    "avg transaction": "avg_transaction_value",
    "average transaction value": "avg_transaction_value",

    "revenue growth": "revenue_growth",
    "sales growth": "revenue_growth",
    "growth": "revenue_growth",

    "expense growth": "expense_growth",

    "cash flow volatility": "cashflow_volatility",
    "cashflow volatility": "cashflow_volatility",
    "cash volatility": "cashflow_volatility",

    "credit utilization": "credit_utilization",
    "credit utilisation": "credit_utilization",
    "utilization": "credit_utilization",
    "utilisation": "credit_utilization"
}


def normalize_columns(df):
    df = df.copy()
    rename_map = {}

    for col in df.columns:
        cleaned = str(col).strip().lower().replace("_", " ")
        if cleaned in ALIASES:
            rename_map[col] = ALIASES[cleaned]

    df.rename(columns=rename_map, inplace=True)
    return df


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
        df["monthly_revenue"] -
        df["monthly_expenses"]
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
    if score >= 50:
        return "Medium Risk"
    return "High Risk"


# ============================================================
# FINANCIAL HEALTH
# ============================================================

def calculate_financial_health(row):
    score = 0

    if row["revenue_growth"] >= 0.05:
        score += 20
    elif row["revenue_growth"] >= -0.05:
        score += 12
    else:
        score += 5

    score += 20 if row["cashflow"] > 0 else 5

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

    # Kept consistent with the current Colab prototype.
    score += 20 if row["transaction_consistency"] > 1 else 10

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
# FLEXIBLE ANALYSIS ENGINE
# ============================================================

DEFAULTS = {
    "monthly_revenue": 500000,
    "monthly_expenses": 350000,
    "total_debt": 800000,
    "monthly_emi": 60000,
    "average_balance": 300000,
    "late_payment_count": 0,
    "total_transactions": 80,
    "avg_transaction_value": 7500,
    "revenue_growth": 0.05,
    "expense_growth": 0.05,
    "cashflow_volatility": 0.30,
    "credit_utilization": 0.50
}


def analyze_data(input_df):
    df = normalize_columns(input_df)

    if df.empty:
        return None, "EMPTY", None

    # If a useful business ID exists under a common name, preserve it.
    if "business_id" not in df.columns:
        for possible in ["business", "business name", "company", "company name", "id"]:
            if possible in df.columns:
                df["business_id"] = df[possible].astype(str)
                break

    # Convert supplied model fields.
    supplied = []
    estimated = []

    for col in BASE_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            supplied.append(col)

    # Handle a simple profit field if available.
    profit_candidates = ["monthly_profit", "profit", "net profit"]
    if "monthly_expenses" not in df.columns:
        for p in profit_candidates:
            if p in df.columns and "monthly_revenue" in df.columns:
                profit = pd.to_numeric(df[p], errors="coerce")
                df["monthly_expenses"] = (
                    df["monthly_revenue"] - profit
                )
                supplied.append("monthly_expenses")
                break

    if not supplied:
        return None, "NO_FINANCIAL_DATA", None

    # Rows with no supplied financial information are removed.
    df = df.dropna(
        how="all",
        subset=[c for c in BASE_FEATURES if c in df.columns]
    ).reset_index(drop=True)

    if df.empty:
        return None, "NO_FINANCIAL_DATA", None

    # Fill missing fields with transparent prototype defaults.
    for col in BASE_FEATURES:
        if col not in df.columns:
            df[col] = DEFAULTS[col]
            estimated.append(col)
        else:
            missing = df[col].isna()
            if missing.any():
                df.loc[missing, col] = DEFAULTS[col]
                estimated.append(col)

    if "business_id" not in df.columns:
        df["business_id"] = [
            f"SME-{i + 1:04d}"
            for i in range(len(df))
        ]

    df = engineer_features(df)

    # Actual trained Random Forest.
    X = df[MODEL_FEATURES]
    df["ml_prediction"] = model.predict(X)
    df["ml_probability"] = model.predict_proba(X)[:, 1]

    # CreditLens business scores.
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

    # Data coverage.
    supplied_unique = set(
        c for c in supplied if c in BASE_FEATURES
    )

    completeness = round(
        100 * len(supplied_unique) / len(BASE_FEATURES),
        1
    )

    if completeness >= 90:
        confidence = "High"
    elif completeness >= 60:
        confidence = "Medium"
    else:
        confidence = "Low"

    df["data_completeness"] = completeness
    df["prediction_confidence"] = confidence

    # Batch anomaly detection.
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
            contamination=min(
                0.05,
                max(1 / len(df), 0.01)
            ),
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

    # Phase 6: enrich the existing anomaly result with explainable
    # financial reasons, severity and recommended actions.
    df = detect_financial_anomalies(df)

    info = {
        "supplied_columns": sorted(supplied_unique),
        "estimated_columns": sorted(set(estimated)),
        "completeness": completeness,
        "confidence": confidence
    }

    return df, "OK", info




# ============================================================
# PHASE 6 — ADVANCED FINANCIAL ANOMALY INTELLIGENCE
# ============================================================

def detect_financial_anomalies(df):
    """
    Phase 6 anomaly engine.

    Combines:
    1. Isolation Forest batch anomaly detection already used by CreditLens.
    2. Transparent financial rules for common abnormal patterns.
    3. Relative-to-portfolio comparisons using robust medians.
    4. Severity scoring and human-readable reasons.

    Important:
    Anomaly != fraud. It means the financial pattern is unusual
    compared with the analyzed portfolio and/or defined rules.
    """
    df = df.copy()

    # Default outputs so the UI remains stable for small datasets.
    df["anomaly_reasons"] = ""
    df["anomaly_severity"] = "Normal"
    df["anomaly_score"] = 0
    df["anomaly_recommendation"] = "No immediate anomaly action required."
    df["rule_anomaly"] = False

    numeric_cols = [
        "monthly_revenue",
        "monthly_expenses",
        "total_debt",
        "average_balance",
        "total_transactions",
        "avg_transaction_value",
        "revenue_growth",
        "expense_growth",
        "cashflow",
        "debt_to_income",
        "credit_utilization",
        "cashflow_volatility",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Portfolio reference values. Median is used because it is less
    # sensitive to extreme observations than the mean.
    medians = {}
    for col in numeric_cols:
        if col in df.columns:
            medians[col] = float(df[col].median())

    def safe_ratio(value, reference):
        if reference is None or abs(reference) < 1e-9:
            return 1.0
        return float(value) / abs(float(reference))

    for idx, row in df.iterrows():
        reasons = []
        recommendations = []
        score = 0
        critical_count = 0

        revenue = float(row["monthly_revenue"])
        expenses = float(row["monthly_expenses"])
        debt = float(row["total_debt"])
        balance = float(row["average_balance"])
        transactions = float(row["total_transactions"])
        avg_txn = float(row["avg_transaction_value"])
        revenue_growth = float(row["revenue_growth"])
        expense_growth = float(row["expense_growth"])
        cashflow = float(row["cashflow"])
        dti = float(row["debt_to_income"])
        utilization = float(row["credit_utilization"])
        volatility = float(row["cashflow_volatility"])

        # --------------------------------------------------------
        # A. Revenue abnormality
        # --------------------------------------------------------
        rev_median = medians.get("monthly_revenue", 0)
        if rev_median > 0:
            rev_ratio = safe_ratio(revenue, rev_median)
            if rev_ratio < 0.50:
                reasons.append("Revenue is substantially below the portfolio median.")
                recommendations.append("Review sales performance and near-term cash inflows.")
                score += 3
                critical_count += 1
            elif rev_ratio > 2.50:
                reasons.append("Revenue is substantially above the portfolio median.")
                recommendations.append("Validate the revenue spike and confirm it is sustainable.")
                score += 2

        if revenue_growth <= -0.15:
            reasons.append("Revenue is declining materially.")
            recommendations.append("Investigate the cause of the revenue decline.")
            score += 3
            critical_count += 1
        elif revenue_growth < -0.05:
            reasons.append("Revenue growth is negative.")
            recommendations.append("Monitor revenue recovery closely.")
            score += 1

        # --------------------------------------------------------
        # B. Expense spike / margin pressure
        # --------------------------------------------------------
        exp_median = medians.get("monthly_expenses", 0)
        if exp_median > 0:
            exp_ratio = safe_ratio(expenses, exp_median)
            if exp_ratio > 2.00:
                reasons.append("Expenses are substantially above the portfolio median.")
                recommendations.append("Review major cost increases and unusual outflows.")
                score += 3
                critical_count += 1
            elif exp_ratio > 1.50:
                reasons.append("Expenses are elevated relative to the portfolio.")
                recommendations.append("Review operating expenses for unusual increases.")
                score += 1

        if expense_growth >= 0.20:
            reasons.append("Expense growth is unusually high.")
            recommendations.append("Check for recent cost spikes and recurring expense increases.")
            score += 2
            critical_count += 1
        elif expense_growth >= 0.10:
            reasons.append("Expenses are growing faster than normal.")
            score += 1

        if cashflow < 0:
            reasons.append("Monthly cashflow is negative.")
            recommendations.append("Prioritize restoring positive operating cashflow.")
            score += 3
            critical_count += 1

        # --------------------------------------------------------
        # C. Liquidity / balance abnormality
        # --------------------------------------------------------
        bal_median = medians.get("average_balance", 0)
        if bal_median > 0:
            bal_ratio = safe_ratio(balance, bal_median)
            if bal_ratio < 0.35:
                reasons.append("Average balance is unusually low.")
                recommendations.append("Maintain a stronger cash reserve for near-term obligations.")
                score += 2
                critical_count += 1
            elif bal_ratio > 3.00:
                reasons.append("Average balance is unusually high.")
                recommendations.append("Validate the balance movement and source of funds.")
                score += 1

        if expenses > 0 and balance / expenses < 0.25:
            reasons.append("Cash reserves cover less than roughly one quarter of monthly expenses.")
            recommendations.append("Build a larger operating cash buffer.")
            score += 2

        # --------------------------------------------------------
        # D. Debt / credit stress
        # --------------------------------------------------------
        debt_median = medians.get("total_debt", 0)
        if debt_median > 0:
            debt_ratio = safe_ratio(debt, debt_median)
            if debt_ratio > 2.50:
                reasons.append("Debt is substantially above the portfolio median.")
                recommendations.append("Review debt structure and repayment capacity.")
                score += 2
                critical_count += 1

        if dti > 0.75:
            reasons.append("Debt servicing is very high relative to revenue.")
            recommendations.append("Reduce debt-servicing pressure where feasible.")
            score += 3
            critical_count += 1
        elif dti > 0.50:
            reasons.append("Debt servicing is elevated relative to revenue.")
            recommendations.append("Monitor EMI and debt levels closely.")
            score += 1

        if utilization > 0.90:
            reasons.append("Credit utilization is extremely high.")
            recommendations.append("Reduce reliance on available credit where possible.")
            score += 3
            critical_count += 1
        elif utilization > 0.80:
            reasons.append("Credit utilization is high.")
            recommendations.append("Keep credit utilization under closer control.")
            score += 1

        # --------------------------------------------------------
        # E. Transaction behavior
        # --------------------------------------------------------
        txn_median = medians.get("total_transactions", 0)
        if txn_median > 0:
            txn_ratio = safe_ratio(transactions, txn_median)
            if txn_ratio > 3.00:
                reasons.append("Transaction volume is unusually high.")
                recommendations.append("Review transaction activity for unusual business behavior.")
                score += 2
            elif txn_ratio < 0.25:
                reasons.append("Transaction volume is unusually low.")
                recommendations.append("Check whether operating activity has slowed.")
                score += 2

        avg_txn_median = medians.get("avg_transaction_value", 0)
        if avg_txn_median > 0:
            avg_txn_ratio = safe_ratio(avg_txn, avg_txn_median)
            if avg_txn_ratio > 3.00:
                reasons.append("Average transaction value is unusually high.")
                recommendations.append("Validate large-value transactions and their business purpose.")
                score += 2
            elif avg_txn_ratio < 0.25:
                reasons.append("Average transaction value is unusually low.")
                score += 1

        # --------------------------------------------------------
        # F. Cashflow volatility
        # --------------------------------------------------------
        if volatility > 0.75:
            reasons.append("Cashflow volatility is extremely high.")
            recommendations.append("Investigate irregular inflows and outflows.")
            score += 3
            critical_count += 1
        elif volatility > 0.60:
            reasons.append("Cashflow volatility is high.")
            recommendations.append("Monitor irregular cash movements.")
            score += 2

        # --------------------------------------------------------
        # G. Isolation Forest result
        # --------------------------------------------------------
        if "anomaly_status" in df.columns and row["anomaly_status"] == "Anomaly":
            score += 2
            reasons.insert(
                0,
                "Isolation Forest identified this record as unusual relative to the analyzed batch."
            )
            recommendations.append(
                "Review the complete financial record before taking action."
            )

        rule_anomaly = score >= 2
        df.at[idx, "rule_anomaly"] = rule_anomaly
        df.at[idx, "anomaly_score"] = int(min(score, 10))

        # Severity is intentionally conservative.
        if score >= 7 or critical_count >= 2:
            severity = "High"
        elif score >= 3:
            severity = "Medium"
        elif score >= 1:
            severity = "Low"
        else:
            severity = "Normal"

        df.at[idx, "anomaly_severity"] = severity
        df.at[idx, "anomaly_reasons"] = " | ".join(dict.fromkeys(reasons))

        if recommendations:
            df.at[idx, "anomaly_recommendation"] = " ".join(
                dict.fromkeys(recommendations)
            )[:500]
        elif severity == "Normal":
            df.at[idx, "anomaly_recommendation"] = (
                "No immediate anomaly action required."
            )

    return df


def anomaly_summary(df):
    """Return portfolio-level Phase 6 anomaly metrics."""
    severity_counts = (
        df["anomaly_severity"]
        .value_counts()
        .reindex(["High", "Medium", "Low", "Normal"], fill_value=0)
    )

    return {
        "total": len(df),
        "high": int(severity_counts["High"]),
        "medium": int(severity_counts["Medium"]),
        "low": int(severity_counts["Low"]),
        "normal": int(severity_counts["Normal"]),
        "flagged": int(
            (df["anomaly_severity"] != "Normal").sum()
        ),
    }


# ============================================================
# PHASE 4 — EXPLAINABLE AI
# ============================================================

def explain_risk(row):
    """
    Explain the CreditLens result using:
    1. Financial/rule-based drivers that directly affect the CreditLens score.
    2. Random Forest feature importance as model-level context.

    This is intentionally transparent and does not claim that feature
    importance alone is a causal explanation.
    """
    risk_drivers = []
    positive_drivers = []

    # Rule-based explanations aligned with calculate_credit_score().
    dti = float(row["debt_to_income"])
    expense_ratio = float(row["expense_ratio"])
    late = float(row["late_payment_count"])
    volatility = float(row["cashflow_volatility"])
    utilization = float(row["credit_utilization"])
    growth = float(row["revenue_growth"])
    cashflow = float(row["cashflow"])

    if dti > 0.50:
        risk_drivers.append(("Debt-to-Income", dti, "High", "EMI is high relative to monthly revenue."))
    elif dti > 0.35:
        risk_drivers.append(("Debt-to-Income", dti, "Medium", "Debt servicing is becoming significant relative to revenue."))
    elif dti < 0.30:
        positive_drivers.append(("Debt-to-Income", dti, "Healthy", "Debt servicing is relatively low compared with revenue."))

    if expense_ratio > 0.85:
        risk_drivers.append(("Expense Ratio", expense_ratio, "High", "A large share of revenue is being consumed by expenses."))
    elif expense_ratio > 0.70:
        risk_drivers.append(("Expense Ratio", expense_ratio, "Medium", "Operating expenses are relatively high compared with revenue."))
    elif expense_ratio < 0.70:
        positive_drivers.append(("Expense Ratio", expense_ratio, "Healthy", "Expenses are relatively controlled compared with revenue."))

    if late >= 5:
        risk_drivers.append(("Late Payments", late, "High", "Frequent late payments indicate repayment stress."))
    elif late >= 3:
        risk_drivers.append(("Late Payments", late, "Medium", "Multiple late payments may indicate repayment pressure."))
    elif late == 0:
        positive_drivers.append(("Late Payments", late, "Healthy", "No late payments were reported."))

    if volatility > 0.60:
        risk_drivers.append(("Cashflow Volatility", volatility, "High", "Cashflow is highly variable."))
    elif volatility > 0.40:
        risk_drivers.append(("Cashflow Volatility", volatility, "Medium", "Cashflow shows noticeable variability."))
    elif volatility < 0.40:
        positive_drivers.append(("Cashflow Volatility", volatility, "Healthy", "Cashflow volatility is relatively controlled."))

    if utilization > 0.80:
        risk_drivers.append(("Credit Utilization", utilization, "High", "A high proportion of available credit is being used."))
    elif utilization > 0.60:
        risk_drivers.append(("Credit Utilization", utilization, "Medium", "Credit utilization is moderately high."))
    elif utilization < 0.60:
        positive_drivers.append(("Credit Utilization", utilization, "Healthy", "Credit utilization is relatively controlled."))

    if growth < -0.10:
        risk_drivers.append(("Revenue Growth", growth, "High", "Revenue is declining materially."))
    elif growth < 0.05:
        risk_drivers.append(("Revenue Growth", growth, "Medium", "Revenue growth is weak or limited."))
    elif growth > 0.10:
        positive_drivers.append(("Revenue Growth", growth, "Strong", "Revenue is growing strongly."))

    if cashflow <= 0:
        risk_drivers.append(("Cashflow", cashflow, "High", "Monthly expenses are at or above monthly revenue."))
    elif cashflow > 0:
        positive_drivers.append(("Cashflow", cashflow, "Healthy", "Monthly revenue exceeds monthly expenses."))

    # Model-level feature importance.
    model_importance = []
    if hasattr(model, "feature_importances_"):
        importance_map = dict(zip(MODEL_FEATURES, model.feature_importances_))
        for feature, importance in sorted(
            importance_map.items(), key=lambda x: x[1], reverse=True
        )[:8]:
            model_importance.append((feature, float(importance)))

    # Make a compact human-readable summary.
    if risk_drivers:
        top = risk_drivers[:3]
        summary = "The strongest observed risk signals are " + ", ".join(
            item[0] for item in top
        ) + "."
    else:
        summary = "No major rule-based risk drivers were triggered."

    recommendations = []
    if dti > 0.50:
        recommendations.append("Reduce debt servicing pressure relative to revenue.")
    elif dti > 0.35:
        recommendations.append("Monitor EMI and debt levels closely.")

    if expense_ratio > 0.85:
        recommendations.append("Reduce operating expenses and protect positive cashflow.")
    elif expense_ratio > 0.70:
        recommendations.append("Look for opportunities to improve expense efficiency.")

    if late >= 3:
        recommendations.append("Improve repayment discipline and avoid further late payments.")
    elif late >= 1:
        recommendations.append("Maintain consistent on-time repayment.")

    if utilization > 0.80:
        recommendations.append("Reduce credit utilization where feasible.")
    elif utilization > 0.60:
        recommendations.append("Keep credit utilization under closer control.")

    if growth < 0:
        recommendations.append("Focus on stabilizing revenue and reversing the decline.")

    if cashflow <= 0:
        recommendations.append("Prioritize restoring positive monthly cashflow.")

    if not recommendations:
        recommendations.append("Maintain current financial discipline and monitor trends.")

    return {
        "risk_drivers": risk_drivers,
        "positive_drivers": positive_drivers,
        "model_importance": model_importance,
        "summary": summary,
        "recommendations": recommendations[:5],
    }


def format_driver_value(name, value):
    if name in {"Debt-to-Income", "Expense Ratio", "Cashflow Volatility"}:
        return f"{value:.2f}"
    if name == "Credit Utilization":
        return f"{value * 100:.1f}%"
    if name == "Revenue Growth":
        return f"{value * 100:.1f}%"
    if name == "Cashflow":
        return f"₹{value:,.0f}"
    if name == "Late Payments":
        return f"{int(value)}"
    return f"{value:.2f}"


# ============================================================
# REPORT GENERATION
# ============================================================

def create_pdf_report(row, explanation=None):
    """Create a professional CreditLens AI PDF risk report."""
    if not REPORTLAB_AVAILABLE:
        return None

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=35,
        leftMargin=35,
        topMargin=35,
        bottomMargin=35
    )

    styles = getSampleStyleSheet()
    story = []
    business = str(row.get("business_id", "Selected Business"))

    title_style = styles["Title"]
    heading_style = styles["Heading2"]
    body_style = styles["BodyText"]

    story.append(Paragraph("CreditLens AI", title_style))
    story.append(Paragraph("SME Credit & Financial Intelligence Report", heading_style))
    story.append(Paragraph(f"Business: {business}", body_style))
    story.append(Spacer(1, 14))

    summary = [
        ["Metric", "Result"],
        ["Credit Intelligence Score", f"{int(row['credit_score'])}/100"],
        ["Risk Category", str(row["risk_category"])],
        ["Financial Health", f"{int(row['financial_health_score'])}/100"],
        ["ML High-Risk Probability", f"{row['ml_probability']*100:.1f}%"],
        ["Data Completeness", f"{row['data_completeness']:.0f}%"],
        ["Prediction Confidence", str(row["prediction_confidence"])],
    ]
    table = Table(summary, colWidths=[230, 180])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a5f")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(table)
    story.append(Spacer(1, 16))

    story.append(Paragraph("Key Financial Indicators", heading_style))
    indicators = [
        ["Indicator", "Value"],
        ["Monthly Revenue", f"₹{row['monthly_revenue']:,.0f}"],
        ["Monthly Expenses", f"₹{row['monthly_expenses']:,.0f}"],
        ["Cashflow", f"₹{row['cashflow']:,.0f}"],
        ["Total Debt", f"₹{row['total_debt']:,.0f}"],
        ["Monthly EMI", f"₹{row['monthly_emi']:,.0f}"],
        ["Debt-to-Income", f"{row['debt_to_income']:.2f}"],
        ["Expense Ratio", f"{row['expense_ratio']:.2f}"],
        ["Late Payments", str(int(row["late_payment_count"]))],
        ["Credit Utilization", f"{row['credit_utilization']*100:.1f}%"],
    ]
    table2 = Table(indicators, colWidths=[230, 180])
    table2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#334155")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(table2)
    story.append(Spacer(1, 16))

    # Phase 4 explanation in the report
    if explanation:
        story.append(Paragraph("Explainable AI — Key Drivers", heading_style))
        driver_rows = [["Driver", "Assessment"]]
        for item in explanation.get("risks", [])[:6]:
            driver_rows.append([item["factor"], item["impact"]])
        if len(driver_rows) == 1:
            driver_rows.append(["No major rule-based risk driver", "Positive"])
        driver_table = Table(driver_rows, colWidths=[300, 110])
        driver_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#475569")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("PADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(driver_table)
        story.append(Spacer(1, 10))

        if explanation.get("positive"):
            story.append(Paragraph("Positive Factors", heading_style))
            for item in explanation["positive"][:6]:
                story.append(Paragraph("• " + item, body_style))
            story.append(Spacer(1, 8))

        if explanation.get("recommendations"):
            story.append(Paragraph("Recommended Actions", heading_style))
            for item in explanation["recommendations"][:6]:
                story.append(Paragraph("• " + item, body_style))
            story.append(Spacer(1, 8))

    story.append(Paragraph(
        "Prototype disclaimer: the current ML model was trained on synthetic SME data. "
        "This report is for project/prototype analysis and should not be used as an "
        "automated lending decision.",
        body_style
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ============================================================

# ============================================================
# PHASE 7 — HISTORICAL FINANCIAL ANALYSIS
# ============================================================

DATE_ALIASES = {
    "date": "analysis_date",
    "datetime": "analysis_date",
    "timestamp": "analysis_date",
    "month": "analysis_date",
    "period": "analysis_date",
    "transaction date": "analysis_date",
    "transaction_date": "analysis_date",
    "statement date": "analysis_date",
    "statement_date": "analysis_date",
}

def normalize_historical_columns(df):
    df = df.copy()
    rename_map = {}
    for col in df.columns:
        cleaned = str(col).strip().lower().replace("_", " ")
        if cleaned in DATE_ALIASES:
            rename_map[col] = DATE_ALIASES[cleaned]
    df.rename(columns=rename_map, inplace=True)
    return df


def detect_date_column(df):
    candidates = [
        "analysis_date", "date", "month", "period",
        "transaction_date", "statement_date", "datetime", "timestamp"
    ]
    for col in candidates:
        if col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().sum() >= 2:
                return col
    for col in df.columns:
        name = str(col).strip().lower().replace("_", " ")
        if any(token in name for token in ["date", "month", "period", "time"]):
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().sum() >= 2:
                return col
    return None


def build_historical_analysis(raw_df):
    df = normalize_historical_columns(normalize_columns(raw_df))
    date_col = detect_date_column(df)
    if date_col is None:
        return None, "NO_DATE", None

    df["analysis_date"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df[df["analysis_date"].notna()].copy()
    if len(df) < 2:
        return None, "INSUFFICIENT_DATES", None

    if "business_id" not in df.columns:
        for possible in ["business", "business name", "company", "company name", "id"]:
            if possible in df.columns:
                df["business_id"] = df[possible].astype(str)
                break
    if "business_id" not in df.columns:
        df["business_id"] = "Portfolio"

    numeric_cols = [
        "monthly_revenue", "monthly_expenses", "total_debt",
        "monthly_emi", "average_balance", "late_payment_count",
        "total_transactions", "avg_transaction_value"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    defaults = {
        "monthly_revenue": 0.0, "monthly_expenses": 0.0,
        "total_debt": 0.0, "monthly_emi": 0.0,
        "average_balance": 0.0, "late_payment_count": 0.0,
        "total_transactions": 0.0, "avg_transaction_value": 0.0
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = df[col].fillna(default)

    df["cashflow"] = df["monthly_revenue"] - df["monthly_expenses"]
    df["period"] = df["analysis_date"].dt.to_period("M").dt.to_timestamp()

    historical = (
        df.groupby("period", as_index=False)
        .agg(
            monthly_revenue=("monthly_revenue", "sum"),
            monthly_expenses=("monthly_expenses", "sum"),
            total_debt=("total_debt", "mean"),
            monthly_emi=("monthly_emi", "mean"),
            average_balance=("average_balance", "mean"),
            late_payment_count=("late_payment_count", "sum"),
            total_transactions=("total_transactions", "sum"),
            avg_transaction_value=("avg_transaction_value", "mean"),
            cashflow=("cashflow", "sum"),
        )
        .sort_values("period")
        .reset_index(drop=True)
    )

    # Avoid double-counting monthly totals when the same monthly value
    # is repeated across transaction rows.
    repeated = df.groupby("period").agg(
        revenue_unique=("monthly_revenue", "nunique"),
        expense_unique=("monthly_expenses", "nunique"),
        row_count=("monthly_revenue", "size")
    )
    for period, check in repeated.iterrows():
        mask = historical["period"] == period
        rows = df.loc[df["period"] == period]
        if check["row_count"] > 1 and check["revenue_unique"] == 1:
            historical.loc[mask, "monthly_revenue"] = rows["monthly_revenue"].iloc[0]
        if check["row_count"] > 1 and check["expense_unique"] == 1:
            historical.loc[mask, "monthly_expenses"] = rows["monthly_expenses"].iloc[0]

    historical["cashflow"] = historical["monthly_revenue"] - historical["monthly_expenses"]

    for new_col, source in [
        ("revenue_mom", "monthly_revenue"),
        ("expense_mom", "monthly_expenses"),
        ("debt_mom", "total_debt"),
        ("cashflow_mom", "cashflow")
    ]:
        historical[new_col] = (
            historical[source].pct_change()
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
        )

    historical["revenue_3m_avg"] = historical["monthly_revenue"].rolling(3, min_periods=1).mean()
    historical["expense_3m_avg"] = historical["monthly_expenses"].rolling(3, min_periods=1).mean()

    signals = []
    if len(historical) >= 2:
        latest = historical.iloc[-1]
        previous = historical.iloc[-2]

        if latest["revenue_mom"] <= -0.15:
            signals.append(("Revenue Decline", "High",
                            f"Revenue fell {abs(latest['revenue_mom'])*100:.1f}% from the previous period."))
        elif latest["revenue_mom"] < 0:
            signals.append(("Revenue Weakness", "Medium",
                            f"Revenue declined {abs(latest['revenue_mom'])*100:.1f}% from the previous period."))

        if latest["expense_mom"] >= 0.15:
            signals.append(("Expense Spike", "High",
                            f"Expenses increased {latest['expense_mom']*100:.1f}% from the previous period."))

        if latest["cashflow"] < 0:
            signals.append(("Negative Cashflow", "High",
                            "Latest-period expenses exceed revenue."))
        elif latest["cashflow"] < previous["cashflow"]:
            signals.append(("Cashflow Deterioration", "Medium",
                            "Cashflow has weakened versus the previous period."))

        if latest["debt_mom"] >= 0.15:
            signals.append(("Debt Increase", "High",
                            f"Debt increased {latest['debt_mom']*100:.1f}% from the previous period."))

        if latest["average_balance"] < latest["monthly_expenses"] * 0.25:
            signals.append(("Low Liquidity Buffer", "Medium",
                            "Average balance is below 25% of latest monthly expenses."))

    if len(historical) >= 3:
        recent = historical.tail(3)
        if (recent["monthly_revenue"].iloc[-1] < recent["monthly_revenue"].iloc[0]
                and recent["cashflow"].iloc[-1] < recent["cashflow"].iloc[0]):
            signals.append(("Deteriorating Trend", "High",
                            "Revenue and cashflow both deteriorated across recent periods."))

    if not signals:
        signals.append(("Stable Trend", "Low",
                        "No major historical warning signal was detected."))

    def pct_change(first, last):
        return np.nan if first == 0 else (last - first) / abs(first)

    first, last = historical.iloc[0], historical.iloc[-1]
    info = {
        "periods": len(historical),
        "start_period": first["period"],
        "end_period": last["period"],
        "revenue_change": pct_change(first["monthly_revenue"], last["monthly_revenue"]),
        "expense_change": pct_change(first["monthly_expenses"], last["monthly_expenses"]),
        "cashflow_change": pct_change(first["cashflow"], last["cashflow"]),
        "debt_change": pct_change(first["total_debt"], last["total_debt"]),
        "signals": signals,
        "date_column": date_col,
    }
    return historical, "OK", info


def build_business_history(raw_df, selected_business):
    df = normalize_historical_columns(normalize_columns(raw_df))
    date_col = detect_date_column(df)
    if date_col is None:
        return None
    df["analysis_date"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df[df["analysis_date"].notna()].copy()

    if "business_id" not in df.columns:
        for possible in ["business", "business name", "company", "company name", "id"]:
            if possible in df.columns:
                df["business_id"] = df[possible].astype(str)
                break
    if "business_id" not in df.columns:
        return None

    df["business_id"] = df["business_id"].astype(str)
    return df[df["business_id"] == str(selected_business)].copy()


def historical_signal_count(historical):
    if historical is None or len(historical) < 2:
        return 0
    latest = historical.iloc[-1]
    score = 0
    if latest["revenue_mom"] <= -0.15:
        score += 2
    elif latest["revenue_mom"] < 0:
        score += 1
    if latest["expense_mom"] >= 0.15:
        score += 2
    if latest["cashflow"] < 0:
        score += 2
    if latest["debt_mom"] >= 0.15:
        score += 2
    if latest["average_balance"] < latest["monthly_expenses"] * 0.25:
        score += 1
    return min(score, 9)


# SESSION STATE
# ============================================================

if "analysis_df" not in st.session_state:
    demo, _, _ = analyze_data(demo_data())
    st.session_state.analysis_df = demo

if "source_name" not in st.session_state:
    st.session_state.source_name = "Demo SME dataset"

if "uploaded_name" not in st.session_state:
    st.session_state.uploaded_name = None

if "raw_input_df" not in st.session_state:
    st.session_state.raw_input_df = None


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
        "Data Intelligence",
        "Anomaly Detection",
        "Historical Analysis",
        "Credit Simulator",
        "Risk Report",
        "AI Copilot"
    ]
)

st.sidebar.divider()

st.sidebar.markdown("### 📁 Upload Financial Data")

uploaded_file = st.sidebar.file_uploader(
    "CSV or Excel file",
    type=["csv", "xlsx"],
    help="CreditLens can work with partial financial datasets."
)

if uploaded_file is not None:
    if st.session_state.uploaded_name != uploaded_file.name:

        try:
            if uploaded_file.name.lower().endswith(".xlsx"):
                input_df = pd.read_excel(uploaded_file)
            else:
                input_df = pd.read_csv(uploaded_file)

            st.session_state.raw_input_df = input_df.copy()

            analyzed, status, info = analyze_data(input_df)

            if status != "OK":
                st.sidebar.error(
                    "No usable financial information was found."
                )
            else:
                st.session_state.analysis_df = analyzed
                st.session_state.source_name = uploaded_file.name
                st.session_state.uploaded_name = uploaded_file.name

                st.sidebar.success(
                    f"{len(analyzed)} record(s) analyzed."
                )

                st.sidebar.caption(
                    f"Data coverage: {info['completeness']:.0f}%"
                )

                if info["confidence"] == "Low":
                    st.sidebar.warning(
                        "Low coverage: some model inputs are estimated."
                    )
                elif info["confidence"] == "Medium":
                    st.sidebar.info(
                        "Medium coverage: some model inputs are estimated."
                    )
                else:
                    st.sidebar.success("High data coverage.")

        except Exception as e:
            st.sidebar.error(f"Could not process file: {e}")

# Sample download.
sample_csv = demo_data().to_csv(index=False)

st.sidebar.download_button(
    "⬇️ Download Sample CSV",
    data=sample_csv,
    file_name="creditlens_sample_sme_data.csv",
    mime="text/csv"
)

st.sidebar.divider()
st.sidebar.caption(
    "Prototype ML model trained on synthetic data."
)


# ============================================================
# GLOBAL HEADER
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

    cols = st.columns(4)

    card_data = [
        ("Average Credit Score", f"{avg_credit}/100", "Credit intelligence"),
        ("Financial Health", f"{avg_health}/100", "Portfolio health"),
        ("High-Risk Businesses", str(high_risk), "Needs attention"),
        ("Anomalies Detected", str(anomaly_count), "Unusual patterns")
    ]

    for col, (label, value, sub) in zip(cols, card_data):
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

    left, right = st.columns(2)

    with left:
        st.markdown("### 🎯 Credit Score")
        st.progress(avg_credit)
        st.caption(f"{avg_credit}/100")

    with right:
        st.markdown("### ❤️ Financial Health")
        st.progress(avg_health)
        st.caption(f"{avg_health}/100")

    st.divider()

    st.markdown("### 💰 Portfolio Financial Overview")

    overview = pd.DataFrame({
        "Metric": [
            "Monthly Revenue",
            "Monthly Expenses",
            "Total Debt",
            "Average Balance"
        ],
        "Average": [
            df["monthly_revenue"].mean(),
            df["monthly_expenses"].mean(),
            df["total_debt"].mean(),
            df["average_balance"].mean()
        ]
    })

    chart, snapshot = st.columns([2, 1])

    with chart:
        st.bar_chart(overview.set_index("Metric"))

    with snapshot:
        st.markdown("#### Portfolio Snapshot")
        st.metric("Avg Revenue", f"₹{df['monthly_revenue'].mean():,.0f}")
        st.metric("Avg Expenses", f"₹{df['monthly_expenses'].mean():,.0f}")
        st.metric("Avg Cashflow", f"₹{df['cashflow'].mean():,.0f}")

    st.divider()

    st.markdown("### 🎯 Risk Distribution")

    risk_counts = (
        df["risk_category"]
        .value_counts()
        .reindex(
            ["Low Risk", "Medium Risk", "High Risk"],
            fill_value=0
        )
    )

    r1, r2 = st.columns([1, 2])

    with r1:
        st.dataframe(
            risk_counts.rename("Businesses"),
            use_container_width=True
        )

    with r2:
        st.bar_chart(risk_counts)

    st.divider()

    st.markdown("### 🚨 Businesses Requiring Attention")

    attention = df[
        (df["risk_category"] == "High Risk") |
        (df["anomaly_status"] == "Anomaly")
    ]

    attention_cols = [
        "business_id",
        "credit_score",
        "financial_health_score",
        "risk_category",
        "ml_probability",
        "anomaly_status"
    ]

    st.dataframe(
        attention[[c for c in attention_cols if c in attention.columns]],
        use_container_width=True
    )

    if "data_completeness" in df.columns:
        coverage = float(df["data_completeness"].iloc[0])
        confidence = df["prediction_confidence"].iloc[0]

        if coverage < 100:
            st.warning(
                f"Data coverage: {coverage:.0f}% • "
                f"Prediction confidence: {confidence}. "
                "Missing inputs are estimated in this prototype."
            )


# ============================================================
# RISK ANALYSIS
# ============================================================

elif page == "Risk Analysis":

    st.markdown(
        '<div class="section-title">🔍 Business-Level Risk Analysis</div>',
        unsafe_allow_html=True
    )

    selected = st.selectbox(
        "Select Business",
        df["business_id"].astype(str).tolist()
    )

    row = df[
        df["business_id"].astype(str) == selected
    ].iloc[0]

    st.markdown(
        f"### 🏢 {row['business_id']}"
    )

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
        "Risk",
        row["risk_category"]
    )

    c4.metric(
        "ML High-Risk Probability",
        f"{row['ml_probability']*100:.1f}%"
    )

    st.divider()

    k1, k2, k3, k4 = st.columns(4)

    k1.metric("Debt-to-Income", f"{row['debt_to_income']:.2f}")
    k2.metric("Expense Ratio", f"{row['expense_ratio']:.2f}")
    k3.metric("Repayment Score", f"{row['repayment_score']*100:.0f}/100")
    k4.metric("Credit Utilization", f"{row['credit_utilization']*100:.1f}%")

    st.divider()

    positive = []
    risks = []

    if row["revenue_growth"] >= 0.05:
        positive.append("Positive revenue growth")
    else:
        risks.append("Weak or negative revenue growth")

    if row["cashflow"] > 0:
        positive.append("Positive operating cash-flow")
    else:
        risks.append("Negative operating cash-flow")

    if row["debt_to_income"] < 0.30:
        positive.append("Healthy debt-to-income")
    elif row["debt_to_income"] >= 0.50:
        risks.append("High debt-to-income")

    if row["late_payment_count"] == 0:
        positive.append("No late payments")
    elif row["late_payment_count"] >= 3:
        risks.append("Multiple late payments")

    if row["credit_utilization"] > 0.80:
        risks.append("High credit utilization")
    elif row["credit_utilization"] < 0.60:
        positive.append("Controlled credit utilization")

    e1, e2 = st.columns(2)

    with e1:
        st.markdown("### ✅ Positive Factors")
        for item in positive:
            st.success(item)
        if not positive:
            st.info("No strong positive indicators found.")

    with e2:
        st.markdown("### ⚠️ Risk Factors")
        for item in risks:
            st.warning(item)
        if not risks:
            st.success("No major rule-based risk factors found.")

    st.divider()

    # ========================================================
    # PHASE 4 — EXPLAINABLE AI
    # ========================================================
    explanation = explain_risk(row)

    st.markdown("### 🧠 Explainable AI — Why this result?")

    st.info(
        f"**CreditLens explanation:** {explanation['summary']} "
        "These explanations combine transparent financial rules with "
        "Random Forest model feature importance."
    )

    ex_left, ex_right = st.columns(2)

    with ex_left:
        st.markdown("#### 🔴 Top Risk Drivers")
        if explanation["risk_drivers"]:
            for name, value, severity, detail in explanation["risk_drivers"][:6]:
                if severity == "High":
                    st.error(
                        f"**{name} — {format_driver_value(name, value)}**  \n"
                        f"{detail}"
                    )
                else:
                    st.warning(
                        f"**{name} — {format_driver_value(name, value)}**  \n"
                        f"{detail}"
                    )
        else:
            st.success("No major rule-based risk drivers were triggered.")

    with ex_right:
        st.markdown("#### 🟢 Positive Factors")
        if explanation["positive_drivers"]:
            for name, value, strength, detail in explanation["positive_drivers"][:6]:
                st.success(
                    f"**{name} — {format_driver_value(name, value)}**  \n"
                    f"{detail}"
                )
        else:
            st.info("No strong positive indicators were identified.")

    st.markdown("#### 📊 Model Feature Importance")
    if explanation["model_importance"]:
        imp_df = pd.DataFrame(
            explanation["model_importance"],
            columns=["Feature", "Importance"]
        )
        imp_df["Importance"] = imp_df["Importance"] * 100
        imp_df["Importance"] = imp_df["Importance"].round(2)

        chart_col, table_col = st.columns([2, 1])
        with chart_col:
            st.bar_chart(
                imp_df.set_index("Feature")["Importance"],
                use_container_width=True
            )
        with table_col:
            st.dataframe(
                imp_df.rename(columns={"Importance": "Importance (%)"}),
                use_container_width=True,
                hide_index=True
            )
    else:
        st.info("Model feature importance is unavailable for this model.")

    st.markdown("#### 💡 Recommended Actions")
    for recommendation in explanation["recommendations"]:
        st.write(f"• {recommendation}")

    st.caption(
        "Explainability note: Random Forest feature importance shows which "
        "features contributed most to the model's overall decisions across "
        "the trained dataset; it is not a causal explanation for an individual "
        "prediction. The risk-driver cards above use transparent CreditLens "
        "financial rules for business-level interpretation."
    )

    st.divider()

    st.markdown("### 💰 Financial Profile")

    profile = pd.DataFrame({
        "Metric": [
            "Monthly Revenue",
            "Monthly Expenses",
            "Cashflow",
            "Total Debt",
            "Monthly EMI",
            "Average Balance",
            "Late Payments",
            "Revenue Growth",
            "Expense Growth"
        ],
        "Value": [
            f"₹{row['monthly_revenue']:,.0f}",
            f"₹{row['monthly_expenses']:,.0f}",
            f"₹{row['cashflow']:,.0f}",
            f"₹{row['total_debt']:,.0f}",
            f"₹{row['monthly_emi']:,.0f}",
            f"₹{row['average_balance']:,.0f}",
            int(row["late_payment_count"]),
            f"{row['revenue_growth']*100:.1f}%",
            f"{row['expense_growth']*100:.1f}%"
        ]
    })

    st.dataframe(profile, use_container_width=True)

    st.caption(
        f"Data coverage: {row['data_completeness']:.0f}% • "
        f"Prediction confidence: {row['prediction_confidence']}"
    )


# ============================================================
# DATA INTELLIGENCE
# ============================================================

elif page == "Data Intelligence":

    st.markdown(
        '<div class="section-title">🧠 Smart Data Intelligence</div>',
        unsafe_allow_html=True
    )

    st.write(
        "CreditLens accepts different column names and does not require "
        "the engineered model columns to be supplied by the user."
    )

    coverage = float(df["data_completeness"].iloc[0])
    confidence = df["prediction_confidence"].iloc[0]

    c1, c2, c3 = st.columns(3)

    c1.metric("Data Coverage", f"{coverage:.0f}%")
    c2.metric("Prediction Confidence", confidence)
    c3.metric("Records", len(df))

    st.divider()

    supplied = []
    estimated = []

    for col in BASE_FEATURES:
        if col in df.columns:
            # A field is considered supplied when it was part of
            # the original user dataset. This information is stored
            # in session state when an upload is used.
            supplied.append(col)

    st.markdown("### 🔄 CreditLens Processing Pipeline")

    st.write(
        "User Data → Column Mapping → Missing-Input Handling → "
        "Feature Engineering → Random Forest → Credit Score → "
        "Financial Health → Anomaly Detection"
    )

    st.divider()

    st.markdown("### 📋 Current Dataset")

    st.dataframe(
        df.head(100),
        use_container_width=True
    )

    st.divider()

    st.markdown("### ⚠️ Important Data Limitation")

    st.warning(
        "If important financial fields are missing, this prototype uses "
        "neutral fallback estimates. This allows the application to work "
        "with incomplete data, but those estimates must be replaced with "
        "validated imputation or real extracted data before production use."
    )


# ============================================================
# ANOMALY DETECTION
# ============================================================

elif page == "Anomaly Detection":

    st.markdown(
        '<div class="section-title">🚨 Advanced Financial Anomaly Intelligence</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Phase 6 combines batch-level machine learning with transparent "
        "financial rules to identify unusual revenue, expense, liquidity, "
        "debt and transaction patterns."
    )

    summary = anomaly_summary(df)

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Records Analyzed", summary["total"])
    c2.metric("Flagged", summary["flagged"])
    c3.metric("High Severity", summary["high"])
    c4.metric("Medium Severity", summary["medium"])
    c5.metric("Normal", summary["normal"])

    st.divider()

    if len(df) < 5:
        st.warning(
            "Isolation Forest batch detection is most meaningful with at least "
            "5 records. Phase 6 rule-based checks can still highlight unusual "
            "financial patterns."
        )

    # --------------------------------------------------------
    # Severity overview
    # --------------------------------------------------------
    st.markdown("### 📊 Anomaly Severity Overview")

    severity_chart = pd.Series(
        {
            "High": summary["high"],
            "Medium": summary["medium"],
            "Low": summary["low"],
            "Normal": summary["normal"],
        },
        name="Businesses"
    )

    chart_col, info_col = st.columns([2, 1])

    with chart_col:
        st.bar_chart(severity_chart)

    with info_col:
        st.markdown("#### How CreditLens interprets severity")
        st.write("🔴 **High** — multiple or critical abnormal signals")
        st.write("🟠 **Medium** — meaningful unusual financial pattern")
        st.write("🟡 **Low** — mild unusual pattern")
        st.write("🟢 **Normal** — no material anomaly signal")

    st.divider()

    # --------------------------------------------------------
    # Main anomaly table
    # --------------------------------------------------------
    st.markdown("### 🔎 Anomaly Investigation")

    display_cols = [
        "business_id",
        "anomaly_status",
        "anomaly_severity",
        "anomaly_score",
        "credit_score",
        "risk_category",
        "monthly_revenue",
        "monthly_expenses",
        "cashflow",
        "total_debt",
        "average_balance",
        "revenue_growth",
        "expense_growth",
    ]

    investigation = df[
        [c for c in display_cols if c in df.columns]
    ].copy()

    st.dataframe(
        investigation,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # --------------------------------------------------------
    # Flagged records
    # --------------------------------------------------------
    flagged = df[
        df["anomaly_severity"] != "Normal"
    ].copy()

    if len(flagged) > 0:

        st.markdown(
            f"### 🚨 {len(flagged)} Record(s) Requiring Anomaly Review"
        )

        flagged_cols = [
            "business_id",
            "anomaly_severity",
            "anomaly_score",
            "anomaly_reasons",
            "anomaly_recommendation",
        ]

        st.dataframe(
            flagged[
                [c for c in flagged_cols if c in flagged.columns]
            ],
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        # ----------------------------------------------------
        # Individual anomaly investigation
        # ----------------------------------------------------
        selected_anomaly = st.selectbox(
            "Select an anomalous business for detailed investigation",
            flagged["business_id"].astype(str).tolist()
        )

        anomaly_row = flagged[
            flagged["business_id"].astype(str) == selected_anomaly
        ].iloc[0]

        st.markdown(
            f"### 🏢 Investigation: {anomaly_row['business_id']}"
        )

        a1, a2, a3, a4 = st.columns(4)

        a1.metric(
            "Severity",
            anomaly_row["anomaly_severity"]
        )
        a2.metric(
            "Anomaly Score",
            f"{int(anomaly_row['anomaly_score'])}/10"
        )
        a3.metric(
            "Credit Score",
            f"{int(anomaly_row['credit_score'])}/100"
        )
        a4.metric(
            "ML Risk Probability",
            f"{anomaly_row['ml_probability'] * 100:.1f}%"
        )

        st.markdown("#### 🔍 Why was this record flagged?")

        reasons_text = str(
            anomaly_row.get("anomaly_reasons", "")
        ).strip()

        if reasons_text:
            for reason in [
                r.strip() for r in reasons_text.split("|") if r.strip()
            ]:
                if anomaly_row["anomaly_severity"] == "High":
                    st.error(f"⚠️ {reason}")
                elif anomaly_row["anomaly_severity"] == "Medium":
                    st.warning(f"⚠️ {reason}")
                else:
                    st.info(f"ℹ️ {reason}")
        else:
            st.success("No specific anomaly reason was triggered.")

        st.markdown("#### 💡 Recommended Review Action")

        st.info(
            str(
                anomaly_row.get(
                    "anomaly_recommendation",
                    "Review the financial record."
                )
            )
        )

        st.markdown("#### 💰 Financial Signals")

        signal_df = pd.DataFrame(
            {
                "Signal": [
                    "Monthly Revenue",
                    "Monthly Expenses",
                    "Cashflow",
                    "Total Debt",
                    "Average Balance",
                    "Revenue Growth",
                    "Expense Growth",
                    "Debt-to-Income",
                    "Credit Utilization",
                    "Cashflow Volatility",
                    "Transaction Count",
                    "Average Transaction Value",
                ],
                "Value": [
                    f"₹{anomaly_row['monthly_revenue']:,.0f}",
                    f"₹{anomaly_row['monthly_expenses']:,.0f}",
                    f"₹{anomaly_row['cashflow']:,.0f}",
                    f"₹{anomaly_row['total_debt']:,.0f}",
                    f"₹{anomaly_row['average_balance']:,.0f}",
                    f"{anomaly_row['revenue_growth'] * 100:.1f}%",
                    f"{anomaly_row['expense_growth'] * 100:.1f}%",
                    f"{anomaly_row['debt_to_income']:.2f}",
                    f"{anomaly_row['credit_utilization'] * 100:.1f}%",
                    f"{anomaly_row['cashflow_volatility']:.2f}",
                    f"{anomaly_row['total_transactions']:,.0f}",
                    f"₹{anomaly_row['avg_transaction_value']:,.0f}",
                ]
            }
        )

        st.dataframe(
            signal_df,
            use_container_width=True,
            hide_index=True
        )

    else:
        st.success(
            "No material financial anomalies were identified in the current dataset."
        )

    st.divider()

    # --------------------------------------------------------
    # Downloadable anomaly report
    # --------------------------------------------------------
    st.markdown("### 📥 Export Anomaly Analysis")

    export_cols = [
        "business_id",
        "anomaly_status",
        "anomaly_severity",
        "anomaly_score",
        "anomaly_reasons",
        "anomaly_recommendation",
        "credit_score",
        "risk_category",
        "ml_probability",
    ]

    export_df = df[
        [c for c in export_cols if c in df.columns]
    ].copy()

    csv_bytes = export_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download Anomaly Analysis CSV",
        data=csv_bytes,
        file_name="creditlens_phase6_anomaly_analysis.csv",
        mime="text/csv"
    )

    st.info(
        "Phase 6 note: an anomaly indicates an unusual financial pattern; "
        "it does not by itself indicate fraud, default or illegal activity. "
        "The current Isolation Forest is trained on the analyzed batch. "
        "For production, use a separately validated anomaly model and "
        "historical transaction-level data."
    )


# ============================================================

# CREDIT SIMULATOR
# ============================================================


elif page == "Historical Analysis":

    st.markdown(
        '<div class="section-title">📈 Historical Financial Analysis</div>',
        unsafe_allow_html=True
    )
    st.write(
        "Analyze multi-period financial performance and identify early-warning "
        "signals in revenue, expenses, cashflow, debt and liquidity."
    )

    historical_source = st.session_state.get("raw_input_df")

    if historical_source is None:
        st.info(
            "Upload a CSV or Excel file containing at least two dated periods "
            "to activate historical analysis."
        )
    else:
        historical, hist_status, hist_info = build_historical_analysis(
            historical_source
        )

        if hist_status != "OK":
            if hist_status == "NO_DATE":
                st.warning(
                    "No usable date column was found. Use a column such as "
                    "`date`, `month`, `period`, `transaction_date`, or "
                    "`statement_date`."
                )
            else:
                st.warning("At least two valid dated records are required.")
        else:
            st.success(
                f"Detected {hist_info['periods']} monthly periods using "
                f"**{hist_info['date_column']}**."
            )

            latest = historical.iloc[-1]

            def fmt_change(v):
                return "N/A" if pd.isna(v) else f"{v*100:+.1f}%"

            h1, h2, h3, h4 = st.columns(4)
            h1.metric(
                "Latest Revenue",
                f"₹{latest['monthly_revenue']:,.0f}",
                fmt_change(latest["revenue_mom"])
            )
            h2.metric(
                "Latest Expenses",
                f"₹{latest['monthly_expenses']:,.0f}",
                fmt_change(latest["expense_mom"])
            )
            h3.metric(
                "Latest Cashflow",
                f"₹{latest['cashflow']:,.0f}",
                fmt_change(latest["cashflow_mom"])
            )
            h4.metric(
                "Latest Debt",
                f"₹{latest['total_debt']:,.0f}",
                fmt_change(latest["debt_mom"])
            )

            st.divider()
            st.markdown("### 📊 Financial Trend")

            trend_choice = st.selectbox(
                "Select trend to visualize",
                [
                    "Revenue vs Expenses",
                    "Cashflow",
                    "Debt",
                    "Average Balance",
                    "Transactions"
                ],
                key="historical_trend_choice"
            )

            trend_df = historical.set_index("period")

            if trend_choice == "Revenue vs Expenses":
                st.line_chart(
                    trend_df[["monthly_revenue", "monthly_expenses"]].rename(
                        columns={
                            "monthly_revenue": "Revenue",
                            "monthly_expenses": "Expenses"
                        }
                    )
                )
            elif trend_choice == "Cashflow":
                st.line_chart(
                    trend_df[["cashflow"]].rename(columns={"cashflow": "Cashflow"})
                )
            elif trend_choice == "Debt":
                st.line_chart(
                    trend_df[["total_debt"]].rename(columns={"total_debt": "Total Debt"})
                )
            elif trend_choice == "Average Balance":
                st.line_chart(
                    trend_df[["average_balance"]].rename(
                        columns={"average_balance": "Average Balance"}
                    )
                )
            else:
                st.line_chart(
                    trend_df[["total_transactions"]].rename(
                        columns={"total_transactions": "Transactions"}
                    )
                )

            st.divider()
            st.markdown("### ⚠️ Historical Early-Warning Signals")

            signals = hist_info["signals"]
            high_count = sum(s[1] == "High" for s in signals)
            medium_count = sum(s[1] == "Medium" for s in signals)

            a1, a2, a3 = st.columns(3)
            a1.metric("High Alerts", high_count)
            a2.metric("Medium Alerts", medium_count)
            a3.metric(
                "Trend Warning Score",
                f"{historical_signal_count(historical)}/9"
            )

            for name, severity, detail in signals:
                if severity == "High":
                    st.error(f"**{name}** — {detail}")
                elif severity == "Medium":
                    st.warning(f"**{name}** — {detail}")
                else:
                    st.success(f"**{name}** — {detail}")

            # Business-level view when multiple businesses are present.
            raw_norm = normalize_columns(historical_source)
            if "business_id" not in raw_norm.columns:
                for possible in [
                    "business", "business name", "company",
                    "company name", "id"
                ]:
                    if possible in raw_norm.columns:
                        raw_norm["business_id"] = raw_norm[possible].astype(str)
                        break

            if "business_id" in raw_norm.columns:
                businesses = sorted(
                    raw_norm["business_id"].astype(str).dropna().unique()
                )

                if len(businesses) > 1:
                    st.divider()
                    st.markdown("### 🏢 Business-Level Historical View")

                    selected_business = st.selectbox(
                        "Select Business",
                        businesses,
                        key="historical_business_selector"
                    )

                    business_raw = build_business_history(
                        historical_source,
                        selected_business
                    )

                    if business_raw is not None and len(business_raw) >= 2:
                        bh, bs, _ = build_historical_analysis(business_raw)
                        if bs == "OK":
                            st.line_chart(
                                bh.set_index("period")[
                                    ["monthly_revenue", "monthly_expenses"]
                                ].rename(
                                    columns={
                                        "monthly_revenue": "Revenue",
                                        "monthly_expenses": "Expenses"
                                    }
                                )
                            )
                            latest_b = bh.iloc[-1]
                            b1, b2, b3 = st.columns(3)
                            b1.metric(
                                "Latest Revenue",
                                f"₹{latest_b['monthly_revenue']:,.0f}"
                            )
                            b2.metric(
                                "Latest Cashflow",
                                f"₹{latest_b['cashflow']:,.0f}"
                            )
                            b3.metric(
                                "Latest Debt",
                                f"₹{latest_b['total_debt']:,.0f}"
                            )

            st.divider()
            st.markdown("### 📋 Historical Performance Table")

            display_hist = historical[
                [
                    "period", "monthly_revenue", "monthly_expenses",
                    "cashflow", "total_debt", "average_balance",
                    "total_transactions", "revenue_mom", "expense_mom"
                ]
            ].copy()

            display_hist["period"] = display_hist["period"].dt.strftime("%Y-%m")
            display_hist["Revenue MoM (%)"] = (
                display_hist.pop("revenue_mom") * 100
            ).round(1)
            display_hist["Expense MoM (%)"] = (
                display_hist.pop("expense_mom") * 100
            ).round(1)

            display_hist = display_hist.rename(
                columns={
                    "period": "Period",
                    "monthly_revenue": "Revenue",
                    "monthly_expenses": "Expenses",
                    "cashflow": "Cashflow",
                    "total_debt": "Total Debt",
                    "average_balance": "Average Balance",
                    "total_transactions": "Transactions"
                }
            )

            st.dataframe(
                display_hist,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "⬇️ Download Historical Analysis CSV",
                data=historical.to_csv(index=False).encode("utf-8"),
                file_name="creditlens_historical_analysis.csv",
                mime="text/csv"
            )

            st.info(
                "Historical analysis is an early-warning trend layer. "
                "It does not prove that a future financial event will occur."
            )


elif page == "Credit Simulator":

    st.markdown(
        '<div class="section-title">🧪 Credit Score Simulator</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Test how changes in business finances could affect the "
        "CreditLens score. This is a scenario tool, not a guaranteed future score."
    )

    selected = st.selectbox(
        "Select Business",
        df["business_id"].astype(str).tolist()
    )

    base_row = df[
        df["business_id"].astype(str) == selected
    ].iloc[0]

    st.markdown("### Change Financial Conditions")

    c1, c2 = st.columns(2)

    with c1:
        sim_revenue = st.number_input(
            "Monthly Revenue",
            min_value=0.0,
            value=float(base_row["monthly_revenue"]),
            step=10000.0
        )

        sim_expenses = st.number_input(
            "Monthly Expenses",
            min_value=0.0,
            value=float(base_row["monthly_expenses"]),
            step=10000.0
        )

        sim_debt = st.number_input(
            "Total Debt",
            min_value=0.0,
            value=float(base_row["total_debt"]),
            step=10000.0
        )

        sim_emi = st.number_input(
            "Monthly EMI",
            min_value=0.0,
            value=float(base_row["monthly_emi"]),
            step=5000.0
        )

    with c2:
        sim_late = st.number_input(
            "Late Payments",
            min_value=0,
            value=int(base_row["late_payment_count"]),
            step=1
        )

        sim_util = st.slider(
            "Credit Utilization",
            0.0,
            1.0,
            float(base_row["credit_utilization"]),
            0.01
        )

        sim_growth = st.slider(
            "Revenue Growth",
            -0.50,
            0.50,
            float(base_row["revenue_growth"]),
            0.01
        )

        sim_volatility = st.slider(
            "Cashflow Volatility",
            0.0,
            1.0,
            float(base_row["cashflow_volatility"]),
            0.01
        )

    scenario = pd.DataFrame([{
        "monthly_revenue": sim_revenue,
        "monthly_expenses": sim_expenses,
        "total_debt": sim_debt,
        "monthly_emi": sim_emi,
        "average_balance": float(base_row["average_balance"]),
        "late_payment_count": sim_late,
        "total_transactions": float(base_row["total_transactions"]),
        "avg_transaction_value": float(base_row["avg_transaction_value"]),
        "revenue_growth": sim_growth,
        "expense_growth": float(base_row["expense_growth"]),
        "cashflow_volatility": sim_volatility,
        "credit_utilization": sim_util
    }])

    scenario = engineer_features(scenario)

    scenario_score = calculate_credit_score(
        scenario.iloc[0]
    )

    scenario_health = calculate_financial_health(
        scenario.iloc[0]
    )

    scenario_risk = risk_category(scenario_score)

    scenario_probability = model.predict_proba(
        scenario[MODEL_FEATURES]
    )[:, 1][0]

    st.divider()

    s1, s2, s3, s4 = st.columns(4)

    s1.metric(
        "Scenario Credit Score",
        f"{scenario_score}/100",
        delta=int(scenario_score - base_row["credit_score"])
    )

    s2.metric(
        "Scenario Health",
        f"{scenario_health}/100",
        delta=int(scenario_health - base_row["financial_health_score"])
    )

    s3.metric(
        "Scenario Risk",
        scenario_risk
    )

    s4.metric(
        "ML Risk Probability",
        f"{scenario_probability*100:.1f}%"
    )

    st.info(
        "The simulator recalculates CreditLens features and runs the "
        "trained Random Forest on the scenario."
    )


# ============================================================
# RISK REPORT
# ============================================================

elif page == "Risk Report":

    st.markdown(
        '<div class="section-title">📄 CreditLens Risk Report</div>',
        unsafe_allow_html=True
    )

    selected = st.selectbox(
        "Select Business",
        df["business_id"].astype(str).tolist()
    )

    row = df[
        df["business_id"].astype(str) == selected
    ].iloc[0]

    st.markdown(f"### {row['business_id']}")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Credit Score",
        f"{int(row['credit_score'])}/100"
    )

    c2.metric(
        "Risk",
        row["risk_category"]
    )

    c3.metric(
        "Financial Health",
        f"{int(row['financial_health_score'])}/100"
    )

    st.divider()

    report_cols = [
        "business_id",
        "credit_score",
        "risk_category",
        "financial_health_score",
        "ml_probability",
        "data_completeness",
        "prediction_confidence"
    ]

    st.dataframe(
        pd.DataFrame([
            row[[c for c in report_cols if c in row.index]]
        ]),
        use_container_width=True
    )

    if REPORTLAB_AVAILABLE:
        pdf_bytes = create_pdf_report(row, explain_risk(row))

        st.download_button(
            "⬇️ Download PDF Risk Report",
            data=pdf_bytes,
            file_name=f"CreditLens_{row['business_id']}_Risk_Report.pdf",
            mime="application/pdf"
        )
    else:
        st.warning(
            "Install reportlab to enable PDF report generation."
        )

    st.caption(
        "Prototype report only. Not a lending decision."
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
        "This version uses deterministic portfolio logic. "
        "The next AI phase can connect an LLM to these verified results."
    )

    question = st.text_input(
        "Ask CreditLens",
        placeholder="Why is SME004 high risk?"
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

            st.dataframe(
                high[
                    [
                        c for c in [
                            "business_id",
                            "credit_score",
                            "financial_health_score",
                            "ml_probability"
                        ]
                        if c in high.columns
                    ]
                ],
                use_container_width=True
            )

        elif "anomal" in q:

            count = int(
                (df["anomaly_status"] == "Anomaly").sum()
            )

            st.info(
                f"CreditLens detected {count} anomalous record(s)."
            )

        elif "average" in q and "revenue" in q:

            st.info(
                f"Average monthly revenue is "
                f"₹{df['monthly_revenue'].mean():,.0f}."
            )

        elif "score" in q:

            st.info(
                f"Average Credit Intelligence Score is "
                f"{df['credit_score'].mean():.1f}/100."
            )

        elif "health" in q:

            st.info(
                f"Average Financial Health Score is "
                f"{df['financial_health_score'].mean():.1f}/100."
            )

        elif "best" in q or "lowest risk" in q:

            best = df.sort_values(
                "credit_score",
                ascending=False
            ).head(5)

            st.success(
                "Strongest businesses by Credit Intelligence Score:"
            )

            st.dataframe(
                best[
                    [
                        c for c in [
                            "business_id",
                            "credit_score",
                            "financial_health_score",
                            "risk_category"
                        ]
                        if c in best.columns
                    ]
                ],
                use_container_width=True
            )

        else:
            st.info(
                "Try: 'Which businesses are high risk?', "
                "'Show anomalies', 'What is the average score?', "
                "'What is the financial health?', or "
                "'Which businesses are lowest risk?'"
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "CreditLens AI • SME Credit Intelligence Prototype • "
    "ML model trained on synthetic data • Not a lending decision"
)

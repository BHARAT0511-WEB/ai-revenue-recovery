import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# CONFIG
st.set_page_config(
    page_title="RecoverAI",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

FEATURES = [
    "amount",
    "payment_method",
    "failure_reason",
    "previous_transactions",
    "previous_successes",
    "retry_count",
    "last_attempt_hours_ago",
    "customer_tenure_days"
]

NUMERIC_FEATURES = [
    "amount",
    "previous_transactions",
    "previous_successes",
    "retry_count",
    "last_attempt_hours_ago",
    "customer_tenure_days"
]

CATEGORICAL_FEATURES = [
    "payment_method",
    "failure_reason"
]

TARGET_OPTIONS = [
    "recovered",
    "recovery",
    "is_recovered",
    "success"
]

# STYLING
st.markdown("""
<style>
    .stApp {
        background: #0B1120;
        color: #E5E7EB;
    }

    [data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid #243044;
    }

    [data-testid="stSidebar"] * {
        color: #E5E7EB;
    }

    .main-title {
        color: #F8FAFC;
        font-size: 38px;
        font-weight: 800;
        letter-spacing: -1px;
        margin-bottom: 0;
    }

    .subtitle {
        color: #94A3B8;
        font-size: 16px;
        margin-bottom: 18px;
    }

    .landing-hero {
        background: linear-gradient(135deg, #172554, #1E3A8A, #312E81);
        border: 1px solid #334E92;
        border-radius: 20px;
        padding: 38px;
        margin-bottom: 28px;
    }

    .landing-hero h1 {
        color: #FFFFFF;
        font-size: 38px;
        margin: 0 0 10px 0;
    }

    .landing-hero p {
        color: #CBD5E1;
        font-size: 17px;
        margin: 0;
    }

    .dashboard-card {
        background: #151F32;
        border: 1px solid #27364F;
        border-radius: 16px;
        padding: 22px;
        min-height: 150px;
        margin-bottom: 12px;
    }

    .dashboard-card h3 {
        color: #F8FAFC;
        margin: 0 0 10px 0;
    }

    .dashboard-card p {
        color: #94A3B8;
        font-size: 14px;
    }

    .insight-box {
        background: #151F32;
        border: 1px solid #27364F;
        border-radius: 14px;
        color: #E5E7EB;
        padding: 18px;
        margin-bottom: 12px;
    }

    .result-box {
        background: #102A22;
        border: 1px solid #166534;
        border-radius: 14px;
        color: #DCFCE7;
        padding: 22px;
        margin-top: 18px;
    }

    div[data-testid="stMetric"] {
        background: #151F32;
        border: 1px solid #27364F;
        border-radius: 14px;
        padding: 16px;
    }

    div[data-testid="stMetricLabel"] {
        color: #94A3B8;
    }

    div[data-testid="stMetricValue"] {
        color: #F8FAFC;
        font-weight: 700;
    }

    .stButton > button {
        background: #2563EB;
        border: 1px solid #3B82F6;
        border-radius: 10px;
        color: white;
        font-weight: 700;
        padding: 10px;
    }

    .stButton > button:hover {
        background: #1D4ED8;
        border: 1px solid #60A5FA;
        color: white;
    }

    .stDataFrame {
        border: 1px solid #27364F;
        border-radius: 12px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# DATA AND MODEL
@st.cache_data
def load_default_data():
    return pd.read_csv("transactions.csv")


def load_data():

    uploaded_file = st.sidebar.file_uploader(
        "Upload transaction CSV",
        type=["csv"],
        help="Upload a CSV file with the required transaction columns."
    )

    if uploaded_file is not None:

        if uploaded_file.size > 5 * 1024 * 1024:
            st.sidebar.error(
                "File is too large. Maximum allowed size is 5 MB."
            )
            st.stop()

        try:
            uploaded_df = pd.read_csv(uploaded_file)

            if uploaded_df.empty:
                st.sidebar.error("Uploaded CSV file is empty.")
                st.stop()

            return uploaded_df

        except Exception:
            st.sidebar.error(
                "Unable to read the uploaded CSV. "
                "Please upload a valid CSV file."
            )
            st.stop()

    return load_default_data()
    
def clean_target(series):
    mapping = {
        "yes": 1,
        "no": 0,
        "true": 1,
        "false": 0,
        "recovered": 1,
        "failed": 0,
        "success": 1,
        "failure": 0
    }

    if series.dtype == "object":
        series = series.astype(str).str.lower().map(mapping)

    return pd.to_numeric(series, errors="coerce")


@st.cache_resource
def train_model(data):

    missing = [column for column in FEATURES if column not in data.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    target_column = next(
        (column for column in TARGET_OPTIONS if column in data.columns),
        None
    )

    if target_column is None:
        for column in data.columns:
            values = data[column].dropna().unique()

            if len(values) == 2 and column not in FEATURES:
                target_column = column
                break

    if target_column is None:
        raise ValueError("Could not find a recovery target column.")

    X = data[FEATURES].copy()
    y = clean_target(data[target_column])

    valid_rows = y.notna()
    X = X.loc[valid_rows]
    y = y.loc[valid_rows].astype(int)

    if y.nunique() != 2:
        raise ValueError(
            "Target column must contain both recovered and failed records."
        )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", NUMERIC_FEATURES),
            (
                "category",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES
            )
        ]
    )

    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    stratify_value = y if y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=stratify_value
    )

    pipeline.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, pipeline.predict(X_test))

    return pipeline, accuracy, target_column

# HELPERS
def simulate_recovery_outcome(probability, action, payment_id):
    seed = sum(ord(char) for char in str(payment_id))
    rng = np.random.default_rng(seed)

    action_multiplier = {
        "SCHEDULE_RETRY": 1.10,
        "SEND_REMINDER": 1.05,
        "SEND_PAYMENT_LINK": 1.12,
        "REQUEST_PAYMENT_UPDATE": 0.92,
        "ESCALATE": 0.70,
        "HUMAN_REVIEW": 0.65,
        "WAIT": 0.00,
        "DO_NOT_CONTACT": 0.00
    }

    multiplier = action_multiplier.get(action, 1.0)

    adjusted_probability = min(
        probability * multiplier,
        0.95
    )

    recovered = rng.random() < adjusted_probability

    return recovered, adjusted_probability
    
def money(value):
    return f"₹{value:,.0f}"

def get_action(
    probability,
    retry_count,
    failure_reason,
    consent,
    opted_out,
    last_attempt_hours_ago,
    amount
):
    hard_failures = [
        "hard_decline",
        "blocked_card",
        "fraud_suspected"
    ]

    technical_failures = [
        "bank_timeout",
        "gateway_error",
        "network_error",
        "technical_error"
    ]

    if opted_out:
        return {
            "action": "DO_NOT_CONTACT",
            "reason": "Customer opted out of recovery communication.",
            "policy_status": "BLOCKED",
            "next_step": "No automated outreach."
        }

    if not consent:
        return {
            "action": "HUMAN_REVIEW",
            "reason": "Communication consent is unavailable.",
            "policy_status": "BLOCKED",
            "next_step": "Do not send an automated message."
        }

    if amount >= 50000:
        return {
            "action": "HUMAN_REVIEW",
            "reason": "High-value transaction requires manual review.",
            "policy_status": "ESCALATED",
            "next_step": "Send case to recovery specialist."
        }

    if failure_reason in hard_failures:
        return {
            "action": "ESCALATE",
            "reason": "Hard decline or risk-related failure cannot be auto-retried.",
            "policy_status": "BLOCKED",
            "next_step": "Create a human-review case."
        }

    if retry_count >= 2:
        return {
            "action": "ESCALATE",
            "reason": "Maximum automatic retry limit reached.",
            "policy_status": "BLOCKED",
            "next_step": "Stop retries and route to human support."
        }

    if last_attempt_hours_ago < 24:
        return {
            "action": "WAIT",
            "reason": "Cooldown period is active; avoid repeated outreach.",
            "policy_status": "BLOCKED",
            "next_step": "Re-evaluate after 24 hours."
        }

    if failure_reason == "expired_card":
        return {
            "action": "REQUEST_PAYMENT_UPDATE",
            "reason": "The saved card may be expired.",
            "policy_status": "ALLOWED",
            "next_step": "Send a secure payment-method update link."
        }

    if failure_reason in ["authentication_failed", "3ds_failed"]:
        return {
            "action": "SEND_PAYMENT_LINK",
            "reason": "Customer authentication is required to complete payment.",
            "policy_status": "ALLOWED",
            "next_step": "Send a secure payment link."
        }

    if failure_reason in ["upi_pending", "checkout_abandoned"]:
        return {
            "action": "SEND_REMINDER",
            "reason": "Customer may complete the payment with a reminder.",
            "policy_status": "ALLOWED",
            "next_step": "Send a payment reminder with a secure link."
        }

    if failure_reason in technical_failures and probability >= 0.50:
        return {
            "action": "SCHEDULE_RETRY",
            "reason": "Temporary technical issue; one bounded retry is appropriate.",
            "policy_status": "ALLOWED",
            "next_step": "Retry after 6 hours."
        }

    if probability >= 0.60:
        return {
            "action": "SEND_PAYMENT_LINK",
            "reason": "Recovery probability is moderate to high.",
            "policy_status": "ALLOWED",
            "next_step": "Send a secure alternate payment link."
        }

    return {
        "action": "ESCALATE",
        "reason": "Recovery probability is low; avoid repeated automated actions.",
        "policy_status": "ESCALATED",
        "next_step": "Route to support/recovery team."
    }

def simulate_recovery_outcome(probability, action, payment_id):
    seed = sum(ord(character) for character in str(payment_id))

    rng = np.random.default_rng(seed)

    action_multiplier = {
        "SCHEDULE_RETRY": 1.10,
        "SEND_REMINDER": 1.05,
        "SEND_PAYMENT_LINK": 1.12,
        "REQUEST_PAYMENT_UPDATE": 0.92,
        "ESCALATE": 0.70,
        "HUMAN_REVIEW": 0.65,
        "WAIT": 0.00,
        "DO_NOT_CONTACT": 0.00
    }

    multiplier = action_multiplier.get(action, 1.0)

    adjusted_probability = min(
        probability * multiplier,
        0.95
    )

    recovered = rng.random() < adjusted_probability

    return recovered, adjusted_probability
    
def action_column(data):

    actions = []

    for _, row in data.iterrows():

        decision = get_action(
            probability=row["recovery_probability"],
            retry_count=row["retry_count"],
            failure_reason=row["failure_reason"],
            consent=True,
            opted_out=False,
            last_attempt_hours_ago=24,
            amount=row["amount"]
        )

        actions.append(decision["action"])

    return actions

def format_table(data, columns):

    output = data[columns].copy()

    if "amount" in output.columns:
        output["amount"] = output["amount"].map(lambda x: f"₹{x:,.2f}")

    if "failed_revenue" in output.columns:
        output["failed_revenue"] = output["failed_revenue"].map(
            lambda x: f"₹{x:,.2f}"
        )

    if "expected_recovery" in output.columns:
        output["expected_recovery"] = output["expected_recovery"].map(
            lambda x: f"₹{x:,.2f}"
        )

    if "recovery_probability" in output.columns:
        output["recovery_probability"] = output["recovery_probability"].map(
            lambda x: f"{x * 100:.1f}%"
        )

    if "recovery_rate" in output.columns:
        output["recovery_rate"] = output["recovery_rate"].map(
            lambda x: f"{x:.1f}%"
        )

    return output

def grouped_analysis(data, group_column):

    analysis = (
        data
        .groupby(group_column)
        .agg(
            failed_transactions=("amount", "count"),
            failed_revenue=("amount", "sum"),
            expected_recovery=("expected_recovery", "sum")
        )
        .reset_index()
    )

    analysis["recovery_rate"] = (
        analysis["expected_recovery"]
        .div(analysis["failed_revenue"])
        .mul(100)
        .fillna(0)
    )

    return analysis.sort_values(
        "expected_recovery",
        ascending=False
    )

# PAGE FUNCTIONS
def show_landing(metrics):

    st.markdown("""
    <div class="landing-hero">
        <h1>💰 Revenue Recovery AI</h1>
        <p>
            Identify failed payments, predict recovery probability,
            and recommend the best action using machine learning.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Choose your workspace")

    cards = [
        (
            "📊 Overview",
            "View failed revenue, recovery opportunity, priority payments and trends.",
            "Open Revenue Overview",
            "Overview"
        ),
        (
            "🔍 Payment Analyzer",
            "Analyze a single failed payment and receive an AI recommendation.",
            "Analyze a Payment",
            "Payment Analyzer"
        ),
        (
            "📋 Transaction History",
            "Filter and inspect every transaction in the recovery pipeline.",
            "Open Transaction History",
            "Transaction History"
        ),
        (
            "🤖 AI Insights",
            "Find the best payment channels and most recoverable failure types.",
            "View AI Insights",
            "AI Insights"
        ),
        (
            "⚡ Recovery Agent",
            "Run safe AI recovery actions, simulate outcomes, and track recovered revenue.",
            "Open Recovery Agent",
            "Recovery Agent"
        )
        (
            "📜 Audit Logs",
            "Review every AI decision, policy check, recovery action, and outcome.",
            "Open Audit Logs",
            "Audit Logs"
        )
    ]

    left, right = st.columns(2)

    for index, (title, description, button, page_name) in enumerate(cards):

        with left if index % 2 == 0 else right:

            st.markdown(
                f"""
                <div class="dashboard-card">
                    <h3>{title}</h3>
                    <p>{description}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            if st.button(
                button,
                key=f"open_{index}",
                use_container_width=True
            ):
                st.session_state.selected_page = page_name
                st.session_state.app_screen = "dashboard"
                st.rerun()

    st.divider()

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Transactions", f"{metrics['transactions']:,}")

    with c2:
        st.metric("Revenue at Risk", money(metrics["failed_revenue"]))

    with c3:
        st.metric("AI Recovery Potential", money(metrics["expected_recovery"]))

    st.caption(
        "Privacy note: This prototype uses anonymized transaction data. "
        "Do not upload card numbers, CVV, passwords, OTPs, bank credentials, "
        "or personally identifiable customer information."
    )

def show_overview(data, failed_data, metrics, accuracy):

    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #172554, #1D4ED8);
        border: 1px solid #334E92;
        border-radius: 18px;
        padding: 25px;
        margin-bottom: 20px;
    ">
        <div style="color:#BFDBFE;font-size:13px;font-weight:700;">
            AI REVENUE RECOVERY COMMAND CENTER
        </div>
        <div style="color:#FFFFFF;font-size:28px;font-weight:800;margin-top:7px;">
            Recover more revenue from failed payments
        </div>
        <div style="color:#DBEAFE;font-size:15px;margin-top:8px;">
            {metrics['failed_transactions']:,} failed payments need attention.
            Estimated recovery opportunity: <b>{money(metrics['expected_recovery'])}</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("💳 Total Revenue", money(metrics["total_revenue"]))

    with c2:
        st.metric("⚠️ Revenue at Risk", money(metrics["failed_revenue"]))

    with c3:
        st.metric(
            "✨ AI Recovery Opportunity",
            money(metrics["expected_recovery"])
        )

    with c4:
        st.metric("🤖 Model Accuracy", f"{accuracy * 100:.1f}%")

    st.divider()
    st.subheader("🎯 Priority Recovery Actions")

    if failed_data.empty:
        st.success("Great news! No failed payments are available.")

    else:
        priority = (
            failed_data
            .sort_values("expected_recovery", ascending=False)
            .head(5)
            .copy()
        )

        priority["AI Action"] = priority["action"]
        priority.insert(0, "Rank", range(1, len(priority) + 1))

        priority_total = priority["expected_recovery"].sum()

        st.markdown(f"""
        <div style="
            text-align:center;
            background:linear-gradient(135deg,#172554,#1E3A8A);
            border:1px solid #334E92;
            border-radius:16px;
            padding:18px;
            margin:12px 0 20px 0;
        ">
            <div style="color:#BFDBFE;font-size:13px;font-weight:700;">
                HIGH-PRIORITY RECOVERY QUEUE
            </div>
            <div style="color:#FFFFFF;font-size:25px;font-weight:800;margin-top:6px;">
                {money(priority_total)} Recovery Opportunity
            </div>
            <div style="color:#CBD5E1;font-size:14px;margin-top:7px;">
                Top failed payments ranked by expected revenue recovery.
            </div>
        </div>
        """, unsafe_allow_html=True)

        display = format_table(
            priority,
            [
                "Rank",
                "amount",
                "payment_method",
                "failure_reason",
                "retry_count",
                "recovery_probability",
                "expected_recovery",
                "AI Action"
            ]
        ).rename(columns={
            "amount": "Payment Amount",
            "payment_method": "Payment Method",
            "failure_reason": "Failure Reason",
            "retry_count": "Retries",
            "recovery_probability": "Recovery Probability",
            "expected_recovery": "Expected Recovery"
        })

        _, center, _ = st.columns([0.25, 4.5, 0.25])

        with center:
            st.markdown(
                "<h3 style='text-align:center;'>Highest-Priority Failed Payments</h3>",
                unsafe_allow_html=True
            )

            st.dataframe(
                display,
                use_container_width=True,
                hide_index=True,
                height=250
            )

    csv_data = priority.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download Priority Recovery Queue",
        data=csv_data,
        file_name="priority_recovery_queue.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.divider()
    st.subheader("📊 Revenue Recovery Performance")

    left, right = st.columns(2)

    with left:
        revenue_chart = pd.DataFrame(
            {
                "Revenue": [
                    metrics["failed_revenue"],
                    metrics["recovered_revenue"],
                    metrics["expected_recovery"]
                ]
            },
            index=[
                "Revenue at Risk",
                "Already Recovered",
                "AI Opportunity"
            ]
        )

        st.bar_chart(revenue_chart)

    with right:
        if not failed_data.empty:
            method_chart = (
                failed_data
                .groupby("payment_method")["expected_recovery"]
                .sum()
                .sort_values(ascending=False)
            )

            st.bar_chart(method_chart)
        else:
            st.info("No failed-payment method data available.")

    st.divider()
    st.subheader("🔎 Recovery Breakdown")

    method_tab, reason_tab = st.tabs(
        ["💳 Payment Methods", "⚠️ Failure Reasons"]
    )

    with method_tab:
        if not failed_data.empty:
            method_data = grouped_analysis(
                failed_data,
                "payment_method"
            )

            st.dataframe(
                format_table(
                    method_data,
                    [
                        "payment_method",
                        "failed_transactions",
                        "failed_revenue",
                        "expected_recovery",
                        "recovery_rate"
                    ]
                ).rename(columns={
                    "payment_method": "Payment Method",
                    "failed_transactions": "Failed Transactions",
                    "failed_revenue": "Failed Revenue",
                    "expected_recovery": "Expected Recovery",
                    "recovery_rate": "Recovery Rate"
                }),
                use_container_width=True,
                hide_index=True
            )

    with reason_tab:
        if not failed_data.empty:
            reason_data = grouped_analysis(
                failed_data,
                "failure_reason"
            )

            st.dataframe(
                format_table(
                    reason_data,
                    [
                        "failure_reason",
                        "failed_transactions",
                        "failed_revenue",
                        "expected_recovery",
                        "recovery_rate"
                    ]
                ).rename(columns={
                    "failure_reason": "Failure Reason",
                    "failed_transactions": "Failed Transactions",
                    "failed_revenue": "Failed Revenue",
                    "expected_recovery": "Expected Recovery",
                    "recovery_rate": "Recovery Rate"
                }),
                use_container_width=True,
                hide_index=True
            )


def show_payment_analyzer(data, pipeline):

    st.subheader("🔍 AI Payment Analyzer")
    st.caption(
        "Enter failed payment details to receive an AI-based recovery recommendation."
    )

    left, right = st.columns(2)

    with left:
        amount = st.number_input(
            "Payment Amount (₹)",
            min_value=1.0,
            value=2500.0,
            step=100.0
        )

        payment_method = st.selectbox(
            "Payment Method",
            sorted(data["payment_method"].dropna().astype(str).unique())
        )

        failure_reason = st.selectbox(
            "Failure Reason",
            sorted(data["failure_reason"].dropna().astype(str).unique())
        )

    with right:
        previous_transactions = st.number_input(
            "Previous Transactions",
            min_value=0,
            value=10
        )

        previous_successes = st.number_input(
            "Previous Successful Transactions",
            min_value=0,
            value=8
        )

        retry_count = st.number_input(
            "Previous Retry Attempts",
            min_value=0,
            value=0
        )

        communication_consent = st.checkbox(
            "Customer has communication consent",
            value=True
        )

        customer_tenure_days = st.number_input(
            "Customer Tenure (Days)",
            min_value=1,
            value=180,
            step=1
        )
        
        customer_opted_out = st.checkbox(
            "Customer opted out of recovery messages",
            value=False
        )

        last_attempt_hours_ago = st.number_input(
            "Hours Since Last Recovery Attempt",
            min_value=0,
            value=24,
            step=1
        )

    if st.button("🚀 Analyze Payment", use_container_width=True):

        if amount <= 0:
            st.error("Payment amount must be greater than ₹0.")
            st.stop()

        if amount > 10000000:
            st.error("Payment amount is above the allowed limit.")
            st.stop()

        if previous_successes > previous_transactions:
            st.error(
                "Previous successful transactions cannot be greater than "
                "total previous transactions."
            )
            st.stop()

        if retry_count > 10:
            st.error("Retry count cannot be greater than 10.")
            st.stop()
       
        input_data = pd.DataFrame([{
            "amount": amount,
            "payment_method": payment_method,
            "failure_reason": failure_reason,
            "previous_transactions": previous_transactions,
            "previous_successes": previous_successes,
            "retry_count": retry_count
        }])

        probability = float(
            pipeline.predict_proba(input_data)[0][1]
        )

        expected = amount * probability

        decision = get_action(
            probability=probability,
            retry_count=retry_count,
            failure_reason=failure_reason,
            consent=communication_consent,
            opted_out=customer_opted_out,
            last_attempt_hours_ago=last_attempt_hours_ago,
            amount=amount
        )
 
        action = decision["action"]
        reason = decision["reason"]
        policy_status = decision["policy_status"]
        next_step = decision["next_step"]

        st.divider()

        a, b, c = st.columns(3)

        with a:
            st.metric(
                "Recovery Probability",
                f"{probability * 100:.1f}%"
            )

        with b:
            st.metric("Recommended Action", action)

        with c:
            st.metric("Expected Recovery", f"₹{expected:,.2f}")

        st.markdown(f"""
        <div class="result-box">
            <h3>🤖 AI Recovery Decision</h3>
            <p><b>Recommended Action:</b> {action}</p>
            <p><b>Policy Status:</b> {policy_status}</p>
            <p><b>Decision Reason:</b> {reason}</p>
            <p><b>Next Step:</b> {next_step}</p>
            <p><b>Predicted Recovery Opportunity:</b> ₹{expected:,.2f}</p>
        </div>
        """, unsafe_allow_html=True)

        st.info(
            f"""
            **Why this recommendation?**

            - Recovery probability: {probability * 100:.1f}%
            - Previous successful payments: {previous_successes}
            - Previous payment attempts: {previous_transactions}
            - Retry attempts already used: {retry_count}
            - Selected failure reason: {failure_reason}
            """
        )

def show_history(data):

    st.subheader("📋 Transaction History")

    c1, c2, c3 = st.columns(3)

    with c1:
        status = st.selectbox(
            "Status",
            ["All", "Recovered", "Failed"]
        )

    with c2:
        method = st.selectbox(
            "Payment Method",
            ["All"] + sorted(data["payment_method"].dropna().astype(str).unique())
        )

    with c3:
        reason = st.selectbox(
            "Failure Reason",
            ["All"] + sorted(data["failure_reason"].dropna().astype(str).unique())
        )

    filtered = data.copy()

    if status != "All":
        filtered = filtered[filtered["status"] == status]

    if method != "All":
        filtered = filtered[filtered["payment_method"] == method]

    if reason != "All":
        filtered = filtered[filtered["failure_reason"] == reason]

    display = format_table(
        filtered,
        [
            "amount",
            "payment_method",
            "failure_reason",
            "previous_transactions",
            "previous_successes",
            "retry_count",
            "recovery_probability",
            "expected_recovery",
            "status"
        ]
    ).rename(columns={
        "amount": "Amount",
        "payment_method": "Payment Method",
        "failure_reason": "Failure Reason",
        "previous_transactions": "Previous Transactions",
        "previous_successes": "Previous Successes",
        "retry_count": "Retry Count",
        "recovery_probability": "Recovery Probability",
        "expected_recovery": "Expected Recovery",
        "status": "Status"
    })

    st.caption(f"Showing {len(display):,} transactions")

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True
    )

def show_ai_insights(failed_data, metrics, accuracy):

    st.subheader("🤖 AI Revenue Insights")

    if failed_data.empty:
        st.info("No failed transactions are available for AI insights.")
        return

    method_data = grouped_analysis(
        failed_data,
        "payment_method"
    )

    reason_data = grouped_analysis(
        failed_data,
        "failure_reason"
    )

    best_method = method_data.iloc[0]
    worst_method = method_data.iloc[-1]
    best_reason = reason_data.iloc[0]
    worst_reason = reason_data.iloc[-1]

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(f"""
        <div class="insight-box">
            <h3>🏆 Best Recovery Channel</h3>
            <p><b>{best_method["payment_method"]}</b> has the highest
            recovery opportunity: <b>{money(best_method["expected_recovery"])}</b>.</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="insight-box">
            <h3>⚠️ Highest Risk Channel</h3>
            <p><b>{worst_method["payment_method"]}</b> has the lowest
            recovery potential: <b>{worst_method["recovery_rate"]:.1f}%</b>.</p>
        </div>
        """, unsafe_allow_html=True)

    c3, c4 = st.columns(2)

    with c3:
        st.markdown(f"""
        <div class="insight-box">
            <h3>💡 Most Recoverable Failure</h3>
            <p><b>{best_reason["failure_reason"]}</b> offers the strongest
            opportunity: <b>{money(best_reason["expected_recovery"])}</b>.</p>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="insight-box">
            <h3>🚨 Priority Failure Type</h3>
            <p><b>{worst_reason["failure_reason"]}</b> has the lowest
            recovery rate: <b>{worst_reason["recovery_rate"]:.1f}%</b>.</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.subheader("🎯 Recommended Business Strategy")

    if metrics["recovery_rate"] >= 40:
        st.success(
            "Strong recovery potential. Prioritize automated retries "
            "for high-probability payments."
        )
    elif metrics["recovery_rate"] >= 25:
        st.warning(
            "Moderate recovery potential. Combine reminders with retries."
        )
    else:
        st.error(
            "Lower recovery potential. Prioritize escalation and alternative methods."
        )

    st.write(f"""
    - **{metrics["failed_transactions"]:,}** failed transactions require attention.
    - Estimated recoverable revenue: **{money(metrics["expected_recovery"])}**.
    - Model test accuracy: **{accuracy * 100:.1f}%**.
    - Estimated recovery rate: **{metrics["recovery_rate"]:.1f}%**.
    """)

def show_recovery_agent(failed_data):

    st.subheader("⚡ Recovery Agent")

    st.caption(
        "Run policy-controlled recovery actions on failed payments "
        "using a simulated recovery executor."
    )

    if failed_data.empty:
        st.success("No failed payments require recovery action.")
        return

    actionable = failed_data[
        failed_data["policy_status"] == "ALLOWED"
    ].copy()
    
    st.divider()
    st.subheader("🛡️ Blocked and Escalated Cases")

    if blocked.empty:
        st.success("No recovery cases were blocked by policy.")
    else:
        blocked_display = format_table(
            blocked,
            [
                "payment_id",
                "amount",
                "failure_reason",
                "retry_count",
                "action",
                "policy_status",
                "reason",
                "next_step"
            ]
        ).rename(columns={
            "payment_id": "Payment ID",
            "amount": "Amount",
            "failure_reason": "Failure Reason",
            "retry_count": "Previous Retries",
            "action": "Agent Decision",
            "policy_status": "Policy Status",
            "reason": "Why Blocked / Escalated",
            "next_step": "Next Step"
        })

        st.dataframe(
            blocked_display,
            use_container_width=True,
            hide_index=True
        )
    
    blocked = failed_data[
        failed_data["policy_status"] != "ALLOWED"
    ].copy()

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Actionable Payments", f"{len(actionable):,}")

    with c2:
        st.metric("Blocked / Escalated", f"{len(blocked):,}")

    with c3:
        st.metric(
            "Actionable Revenue",
            money(float(actionable["amount"].sum()))
        )

    st.divider()
    st.subheader("🎯 AI Recovery Queue")

    if actionable.empty:
        st.warning(
            "No payments are currently eligible for automated recovery. "
            "All cases are blocked, waiting, or escalated."
        )
    else:
        display = format_table(
            actionable,
            [
                "payment_id",
                "amount",
                "payment_method",
                "failure_reason",
                "retry_count",
                "recovery_probability",
                "expected_recovery",
                "action",
                "reason",
                "next_step"
            ]
        ).rename(columns={
            "payment_id": "Payment ID",
            "amount": "Amount",
            "payment_method": "Payment Method",
            "failure_reason": "Failure Reason",
            "retry_count": "Previous Retries",
            "recovery_probability": "Recovery Probability",
            "expected_recovery": "Predicted Recovery Value",
            "action": "Recommended Action",
            "reason": "Decision Reason",
            "next_step": "Next Step"
        })

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True
        )

    st.divider()
    st.subheader("▶ Execute Simulated Recovery")

    st.info(
        "Demo mode: This prototype does not trigger real payments, emails, "
        "or WhatsApp messages. It simulates recovery outcomes using synthetic data."
    )

    if st.button(
        "▶ Run Simulated Recovery Batch",
        use_container_width=True,
        disabled=actionable.empty
    ):

        results = actionable.copy()

        simulation = results.apply(
            lambda row: simulate_recovery_outcome(
                probability=row["recovery_probability"],
                action=row["action"],
                payment_id=row["payment_id"]
            ),
            axis=1
        )

        results["simulated_recovered"] = simulation.map(
            lambda outcome: outcome[0]
        )

        results["adjusted_probability"] = simulation.map(
            lambda outcome: outcome[1]
        )

        results["recovered_amount"] = np.where(
            results["simulated_recovered"],
            results["amount"],
            0
        )

        results["execution_status"] = np.where(
            results["simulated_recovered"],
            "RECOVERED",
            "NOT_RECOVERED"
        )

        results["event_time"] = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        results["agent_name"] = "RecoverAI Recovery Agent"

        results["decision_reason"] = results["reason"]

        results["risk_level"] = np.select(
            [
                results["recovery_probability"] >= 0.75,
                results["recovery_probability"] >= 0.50
            ],
            [
                "HIGH_RECOVERY_OPPORTUNITY",
                "MEDIUM_RECOVERY_OPPORTUNITY"
            ],
            default="LOW_RECOVERY_OPPORTUNITY"
        )

        audit_columns = [
            "event_time",
            "agent_name",
            "payment_id",
            "amount",
            "payment_method",
            "failure_reason",
            "recovery_probability",
            "risk_level",
            "action",
            "policy_status",
            "decision_reason",
            "next_step",
            "execution_status",
            "recovered_amount"
        ]

        audit_log = results[audit_columns].copy()

        st.session_state.recovery_results = results
        st.session_state.audit_log = audit_log

        st.success("Recovery batch simulation completed successfully.")
        st.rerun()

    if "recovery_results" in st.session_state:

        results = st.session_state.recovery_results.copy()

        recovered_count = int(results["simulated_recovered"].sum())
        recovered_amount = float(results["recovered_amount"].sum())

        recovery_rate = (
            recovered_count / len(results) * 100
            if len(results) > 0 else 0
        )

        st.divider()
        st.subheader("📈 Recovery Execution Results")

        r1, r2, r3, r4 = st.columns(4)

        with r1:
            st.metric(
                "Payments Processed",
                f"{len(results):,}"
            )

        with r2:
            st.metric(
                "Payments Recovered",
                f"{recovered_count:,}"
            )

        with r3:
            st.metric(
                "Recovered Revenue",
                money(recovered_amount)
            )

        with r4:
            st.metric(
                "Recovery Rate",
                f"{recovery_rate:.1f}%"
            )

        result_display = format_table(
            results,
            [
                "payment_id",
                "amount",
                "failure_reason",
                "recovery_probability",
                "action",
                "execution_status",
                "recovered_amount"
            ]
        ).rename(columns={
            "payment_id": "Payment ID",
            "amount": "Amount",
            "failure_reason": "Failure Reason",
            "recovery_probability": "Recovery Probability",
            "action": "Action Taken",
            "execution_status": "Outcome",
            "recovered_amount": "Recovered Amount"
        })

        st.dataframe(
            result_display,
            use_container_width=True,
            hide_index=True
        )

        csv_data = results.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇️ Download Recovery Results",
            data=csv_data,
            file_name="recovery_execution_results.csv",
            mime="text/csv",
            use_container_width=True
        )

        def show_audit_logs():

            st.subheader("📜 Recovery Audit Logs")

            st.caption(
                "A complete record of AI recovery decisions, policy checks, "
                "simulated actions, and recovery outcomes."
            )

            if "audit_log" not in st.session_state:
                st.info(
                    "No audit records are available yet. "
                    "Open Recovery Agent and run a simulated recovery batch."
                )
                return

            audit_log = st.session_state.audit_log.copy()

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                 st.metric(
                     "Total Decisions",
                     f"{len(audit_log):,}"
            )

            with c2:
                allowed_count = int(
                    (audit_log["policy_status"] == "ALLOWED").sum()
                )
                st.metric(
                    "Actions Allowed",
                    f"{allowed_count:,}"
                )

            with c3:
                recovered_count = int(
                    (audit_log["execution_status"] == "RECOVERED").sum()
                )
                st.metric(
                    "Recovered Payments",
                    f"{recovered_count:,}"
                )

            with c4:
                recovered_amount = float(
                    audit_log["recovered_amount"].sum()
                )
                st.metric(
                    "Recovered Revenue",
                    money(recovered_amount)
                )

            st.divider()

            st.subheader("Decision and Execution Trail")

            display = format_table(
                audit_log,
                [
                    "event_time",
                    "payment_id",
                    "amount",
                    "payment_method",
                    "failure_reason",
                    "recovery_probability",
                    "risk_level",
                    "action",
                    "policy_status",
                    "decision_reason",
                    "next_step",
                    "execution_status",
                    "recovered_amount"
                ]
            ).rename(columns={
                "event_time": "Timestamp",
                "payment_id": "Payment ID",
                "amount": "Payment Amount",
                "payment_method": "Payment Method",
                "failure_reason": "Failure Reason",
                "recovery_probability": "Recovery Probability",
                "risk_level": "Opportunity Level",
                "action": "Agent Action",
                "policy_status": "Policy Status",
                "decision_reason": "Decision Reason",
                "next_step": "Next Step",
                "execution_status": "Execution Outcome",
                "recovered_amount": "Recovered Amount"
            })

            st.dataframe(
                display,
                use_container_width=True,
                hide_index=True,
                height=4 50
            )

            csv_data = audit_log.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="⬇️ Download Complete Audit Log",
                data=csv_data,
                file_name="recoverai_audit_log.csv",
                mime="text/csv",
                use_container_width=True
            )

# APP SETUP
try:
    df = load_data()
    pipeline, accuracy, target_column = train_model(df)

except Exception as error:
    st.error("Application setup failed.")
    st.exception(error)
    st.stop()


df_analysis = df.copy()
if "payment_id" not in df_analysis.columns:
    df_analysis["payment_id"] = [
        f"PAY_{index + 1:04d}"
        for index in range(len(df_analysis))
    ]
df_analysis["_target"] = clean_target(df_analysis[target_column])

df_analysis = df_analysis[
    df_analysis["_target"].isin([0, 1])
].copy()

df_analysis["status"] = np.where(
    df_analysis["_target"] == 1,
    "Recovered",
    "Failed"
)

df_analysis["recovery_probability"] = (
    pipeline.predict_proba(df_analysis[FEATURES])[:, 1]
)

df_analysis["expected_recovery"] = (
    df_analysis["amount"] * df_analysis["recovery_probability"]
)

failed_df = df_analysis[df_analysis["_target"] == 0].copy()

# Create an AI recovery plan for every failed payment
def create_recovery_plan(row):
    decision = get_action(
        probability=row["recovery_probability"],
        retry_count=row["retry_count"],
        failure_reason=row["failure_reason"],
        consent=True,
        opted_out=False,
        last_attempt_hours_ago=24,
        amount=row["amount"]
    )

    return pd.Series(decision)

recovery_plan = failed_df.apply(
    create_recovery_plan,
    axis=1
)

failed_df = pd.concat(
    [failed_df, recovery_plan],
    axis=1
)

st.write(
    failed_df[
        [
            "amount",
            "failure_reason",
            "recovery_probability",
            "action",
            "policy_status",
            "next_step"
        ]
    ].head()
)

recovered_df = df_analysis[df_analysis["_target"] == 1].copy()

total_revenue = float(df_analysis["amount"].sum())
failed_revenue = float(failed_df["amount"].sum())
recovered_revenue = float(recovered_df["amount"].sum())
expected_recovery = float(failed_df["expected_recovery"].sum())

metrics = {
    "transactions": len(df_analysis),
    "failed_transactions": len(failed_df),
    "total_revenue": total_revenue,
    "failed_revenue": failed_revenue,
    "recovered_revenue": recovered_revenue,
    "expected_recovery": expected_recovery,
    "recovery_rate": (
        expected_recovery / failed_revenue * 100
        if failed_revenue > 0 else 0
    )
}

# NAVIGATION
if "app_screen" not in st.session_state:
    st.session_state.app_screen = "landing"

if "selected_page" not in st.session_state:
    st.session_state.selected_page = "Overview"

if st.session_state.app_screen == "landing":
    show_landing(metrics)
    st.stop()

page = st.session_state.selected_page

st.markdown(
    '<div class="main-title">💰 Revenue Recovery AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    f'<div class="subtitle">Current workspace: {page}</div>',
    unsafe_allow_html=True
)

st.sidebar.markdown("## 💰 RecoverAI")
st.sidebar.caption("AI Revenue Recovery Platform")

if st.sidebar.button(
    "← Back to Dashboard",
    use_container_width=True
):
    st.session_state.app_screen = "landing"
    st.rerun()

st.sidebar.divider()
st.sidebar.metric("Transactions", f"{metrics['transactions']:,}")
st.sidebar.metric("Model Accuracy", f"{accuracy * 100:.1f}%")

# PAGE ROUTER
if page == "Overview":
    show_overview(df_analysis, failed_df, metrics, accuracy)

elif page == "Payment Analyzer":
    show_payment_analyzer(df_analysis, pipeline)

elif page == "Transaction History":
    show_history(df_analysis)

elif page == "AI Insights":
    show_ai_insights(failed_df, metrics, accuracy)

elif page == "Recovery Agent":
    show_recovery_agent(failed_df)

elif page == "Audit Logs":
    show_audit_logs()

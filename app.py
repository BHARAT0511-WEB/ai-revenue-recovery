import streamlit as st
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline  
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# PAGE CONFIG
st.set_page_config(
    page_title="AI Revenue Recovery",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# PROFESSIONAL STYLING
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
        font-size: 42px;
        font-weight: 800;
        color: #F8FAFC;
        margin-bottom: 0;
        letter-spacing: -1px;
    }

    .subtitle {
        color: #94A3B8;
        font-size: 17px;
        margin-bottom: 22px;
    }

    .landing-hero {
        padding: 38px;
        border-radius: 20px;
        background: linear-gradient(135deg, #172554 0%, #1E3A8A 55%, #312E81 100%);
        border: 1px solid #334E92;
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
        min-height: 175px;
        margin-bottom: 14px;
    }

    .dashboard-card h3 {
        color: #F8FAFC;
        margin-top: 0;
        margin-bottom: 10px;
    }

    .dashboard-card p {
        color: #94A3B8;
        font-size: 14px;
    }

    .insight-box {
        padding: 18px;
        border-radius: 14px;
        border: 1px solid #27364F;
        background: #151F32;
        color: #E5E7EB;
        margin-bottom: 12px;
    }

    .result-box {
        padding: 22px;
        border-radius: 14px;
        border: 1px solid #166534;
        background: #102A22;
        color: #DCFCE7;
        margin-top: 18px;
    }

    div[data-testid="stMetric"] {
        background: #151F32;
        border: 1px solid #27364F;
        padding: 16px;
        border-radius: 14px;
    }

    div[data-testid="stMetricLabel"] {
        color: #94A3B8;
    }

    div[data-testid="stMetricValue"] {
        color: #F8FAFC;
        font-weight: 700;
    }

    .stButton > button {
        border-radius: 10px;
        border: 1px solid #3B82F6;
        background: #2563EB;
        color: white;
        font-weight: 700;
        padding: 0.65rem 1rem;
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

# LOAD DATA
@st.cache_data
def load_data():

    df = pd.read_csv("transactions.csv")

    return df

# TRAIN MODEL
@st.cache_resource
def train_model(df):

    feature_columns = [
        "amount",
        "payment_method",
        "failure_reason",
        "previous_transactions",
        "previous_successes",
        "retry_count"
    ]

    missing = [
        col for col in feature_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    possible_targets = [
        "recovered",
        "recovery",
        "is_recovered",
        "success"
    ]

    target_column = None

    for col in possible_targets:

        if col in df.columns:
            target_column = col
            break

    if target_column is None:

        for col in df.columns:

            unique_values = df[col].dropna().unique()

            if len(unique_values) == 2 and col not in feature_columns:
                target_column = col
                break

    if target_column is None:
        raise ValueError(
            "Could not find recovery target column."
        )

    X = df[feature_columns].copy()
    y = df[target_column].copy()

    if y.dtype == "object":

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

        y = (
            y.astype(str)
            .str.lower()
            .map(mapping)
        )

    y = pd.to_numeric(
        y,
        errors="coerce"
    )

    valid = y.notna()

    X = X.loc[valid].copy()
    y = y.loc[valid].astype(int)

    numerical_features = [
        "amount",
        "previous_transactions",
        "previous_successes",
        "retry_count"
    ]

    categorical_features = [
        "payment_method",
        "failure_reason"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                "passthrough",
                numerical_features
            ),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features
            )
        ]
    )

    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    pipeline.fit(
        X_train,
        y_train
    )

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    return (
        pipeline,
        accuracy,
        target_column
    )

# LOAD APPLICATION
try:

    df = load_data()

    pipeline, accuracy, target_column = train_model(df)

except Exception as e:

    st.error("Application setup failed.")
    st.exception(e)
    st.stop()

# PREPARE ANALYTICS
df_analysis = df.copy()

df_analysis["_target"] = pd.to_numeric(
    df_analysis[target_column],
    errors="coerce"
)

df_analysis["status"] = np.where(
    df_analysis["_target"] == 1,
    "Recovered",
    "Failed"
)

# PREDICT ALL TRANSACTIONS
feature_columns = [
    "amount",
    "payment_method",
    "failure_reason",
    "previous_transactions",
    "previous_successes",
    "retry_count"
]

df_analysis["recovery_probability"] = (
    pipeline.predict_proba(
        df_analysis[feature_columns]
    )[:, 1]
)


df_analysis["expected_recovery"] = (
    df_analysis["amount"]
    * df_analysis["recovery_probability"]
)

# BUSINESS METRICS
total_revenue = float(
    df_analysis["amount"].sum()
)

failed_df = df_analysis[
    df_analysis["_target"] == 0
].copy()

recovered_df = df_analysis[
    df_analysis["_target"] == 1
].copy()

failed_revenue = float(
    failed_df["amount"].sum()
)

actual_recovered_revenue = float(
    recovered_df["amount"].sum()
)

expected_recovery = float(
    failed_df["expected_recovery"].sum()
)

recovery_rate = (
    expected_recovery / failed_revenue * 100
    if failed_revenue > 0
    else 0
)

transaction_count = len(df_analysis)

failed_transactions = len(failed_df)

recovered_transactions = len(recovered_df)

# APP NAVIGATION STATE
if "app_screen" not in st.session_state:
    st.session_state.app_screen = "landing"

if "selected_page" not in st.session_state:
    st.session_state.selected_page = "Overview"


# LANDING PAGE
if st.session_state.app_screen == "landing":

    st.markdown(
        """
        <div class="landing-hero">
            <h1>💰 Revenue Recovery AI</h1>
            <p>
                Identify failed payments, predict recovery probability,
                and take the best revenue recovery action using AI.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Choose your workspace")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="dashboard-card">
                <h3>📊 Overview</h3>
                <p>
                    View revenue at risk, recovery opportunity,
                    and the most important payment insights.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button(
            "Open Revenue Overview",
            key="open_overview",
            use_container_width=True
        ):
            st.session_state.selected_page = "Overview"
            st.session_state.app_screen = "dashboard"
            st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            """
            <div class="dashboard-card">
                <h3>📋 Transaction History</h3>
                <p>
                    Filter, explore, and review all payment
                    transaction records in detail.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button(
            "Open Transaction History",
            key="open_history",
            use_container_width=True
        ):
            st.session_state.selected_page = "Transaction History"
            st.session_state.app_screen = "dashboard"
            st.rerun()

    with col2:
        st.markdown(
            """
            <div class="dashboard-card">
                <h3>🔍 Payment Analyzer</h3>
                <p>
                    Enter a failed payment's details and get an
                    AI-powered recommendation: retry, remind, or escalate.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button(
            "Analyze a Payment",
            key="open_analyzer",
            use_container_width=True
        ):
            st.session_state.selected_page = "Payment Analyzer"
            st.session_state.app_screen = "dashboard"
            st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            """
            <div class="dashboard-card">
                <h3>🤖 AI Insights</h3>
                <p>
                    Discover the best recovery channel, risk areas,
                    and recommended business strategy.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button(
            "View AI Insights",
            key="open_insights",
            use_container_width=True
        ):
            st.session_state.selected_page = "AI Insights"
            st.session_state.app_screen = "dashboard"
            st.rerun()

    st.divider()

    a, b, c = st.columns(3)

    with a:
        st.metric(
            "Total Transactions",
            f"{transaction_count:,}"
        )

    with b:
        st.metric(
            "Revenue at Risk",
            f"₹{failed_revenue:,.0f}"
        )

    with c:
        st.metric(
            "AI Recovery Potential",
            f"₹{expected_recovery:,.0f}"
        )

    st.stop()


# DASHBOARD HEADER
page = st.session_state.selected_page

st.markdown(
    '<div class="main-title">💰 Revenue Recovery AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    f'<div class="subtitle">Current workspace: {page}</div>',
    unsafe_allow_html=True
)

st.divider()


# DASHBOARD SIDEBAR
st.sidebar.markdown("## 💰 RecoverAI")
st.sidebar.caption("AI Revenue Recovery Platform")

if st.sidebar.button(
    "← Back to Dashboard",
    use_container_width=True
):
    st.session_state.app_screen = "landing"
    st.rerun()

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Payment Analyzer",
        "Transaction History",
        "AI Insights"
    ],
    index=[
        "Overview",
        "Payment Analyzer",
        "Transaction History",
        "AI Insights"
    ].index(st.session_state.selected_page)
)

st.session_state.selected_page = page

st.sidebar.divider()

st.sidebar.metric(
    "Transactions",
    f"{transaction_count:,}"
)

st.sidebar.metric(
    "Model Accuracy",
    f"{accuracy * 100:.1f}%"
)

# OVERVIEW
if page == "Overview":

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #0F172A 0%, #1D4ED8 100%);
            padding: 28px;
            border-radius: 18px;
            margin-bottom: 24px;
            color: white;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.18);
        ">
            <div style="font-size: 14px; color: #BFDBFE; font-weight: 600;">
                AI REVENUE RECOVERY COMMAND CENTER
            </div>
            <div style="font-size: 30px; font-weight: 800; margin-top: 7px;">
                Recover more revenue from failed payments
            </div>
            <div style="font-size: 16px; color: #DBEAFE; margin-top: 10px;">
                <b>{failed_transactions:,}</b> failed transactions need attention.
                AI estimates a recovery opportunity of
                <b>₹{expected_recovery:,.0f}</b>.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # KPI CARDS
    c1, c2, c3, c4 = st.columns(4)

    failed_percent = (
        failed_revenue / total_revenue * 100
        if total_revenue > 0
        else 0
    )

    with c1:
        st.metric(
            "💳 Total Revenue",
            f"₹{total_revenue:,.0f}",
            f"{transaction_count:,} transactions"
        )

    with c2:
        st.metric(
            "⚠️ Revenue at Risk",
            f"₹{failed_revenue:,.0f}",
            f"{failed_percent:.1f}% of total revenue",
            delta_color="inverse"
        )

    with c3:
        st.metric(
            "✨ AI Recovery Opportunity",
            f"₹{expected_recovery:,.0f}",
            f"{recovery_rate:.1f}% recovery potential"
        )

    with c4:
        st.metric(
            "🤖 Model Accuracy",
            f"{accuracy * 100:.1f}%",
            "Held-out test data"
        )

    st.divider()

    # ACTION SECTION
    st.subheader("🎯 Priority Recovery Actions")

    if failed_df.empty:
        st.success(
            "Great news! No failed transactions are currently "
            "available in the dataset."
        )

    else:
        priority_transactions = (
            failed_df
            .sort_values(
                by="expected_recovery",
                ascending=False
            )
            .head(5)
            .copy()
        )

        priority_amount = float(
            priority_transactions["expected_recovery"].sum()
        )

        p1, p2 = st.columns([1.15, 1.85])

        with p1:
            st.markdown(
                f"""
                <div class="insight-box">
                    <h3 style="margin-top: 0;">🚀 Act Now</h3>

                    <p style="font-size: 16px;">
                        Focus on the top
                        <b>{len(priority_transactions)}</b>
                        failed transactions first.
                    </p>

                    <p style="font-size: 22px; font-weight: 800; color: #16A34A;">
                        ₹{priority_amount:,.0f}
                    </p>

                    <p class="small-text">
                        Estimated recovery from the highest-priority payments.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            if recovery_rate >= 40:
                st.success(
                    "Strong recovery potential detected. "
                    "Prioritize automated retries for high-value payments."
                )
            elif recovery_rate >= 25:
                st.warning(
                    "Moderate recovery potential. Use reminders "
                    "before retrying medium-probability payments."
                )
            else:
                st.error(
                    "Low recovery potential. Escalate high-value "
                    "failures and offer alternate payment methods."
                )

        with p2:
            priority_display = priority_transactions[
                [
                    "amount",
                    "payment_method",
                    "failure_reason",
                    "retry_count",
                    "recovery_probability",
                    "expected_recovery"
                ]
            ].copy()

            priority_display["recovery_probability"] = (
                priority_display["recovery_probability"]
                .mul(100)
                .round(1)
                .astype(str)
                + "%"
            )

            priority_display["amount"] = (
                "₹"
                + priority_display["amount"]
                .round(2)
                .astype(str)
            )

            priority_display["expected_recovery"] = (
                "₹"
                + priority_display["expected_recovery"]
                .round(2)
                .astype(str)
            )

            priority_display = priority_display.rename(
                columns={
                    "amount": "Payment Amount",
                    "payment_method": "Payment Method",
                    "failure_reason": "Failure Reason",
                    "retry_count": "Retry Count",
                    "recovery_probability": "Recovery Probability",
                    "expected_recovery": "Expected Recovery"
                }
            )

            st.markdown("#### Highest-Priority Failed Payments")

            st.dataframe(
                priority_display,
                use_container_width=True,
                hide_index=True
            )

    st.divider()

    # CHART SECTION
    st.subheader("📊 Revenue Recovery Performance")

    left_chart, right_chart = st.columns(2)

    with left_chart:
        st.markdown("#### Revenue Distribution")

        revenue_chart = pd.DataFrame(
            {
                "Revenue (₹)": [
                    failed_revenue,
                    actual_recovered_revenue,
                    expected_recovery
                ]
            },
            index=[
                "Revenue at Risk",
                "Already Recovered",
                "AI Recovery Opportunity"
            ]
        )

        st.bar_chart(
            revenue_chart,
            color="#2563EB"
        )

        st.caption(
            "AI Recovery Opportunity is the estimated value "
            "recoverable from currently failed payments."
        )

    with right_chart:
        st.markdown("#### Recovery Opportunity by Payment Method")

        if not failed_df.empty:
            method_chart = (
                failed_df
                .groupby("payment_method")
                .agg(
                    expected_recovery=(
                        "expected_recovery",
                        "sum"
                    )
                )
                .sort_values(
                    by="expected_recovery",
                    ascending=False
                )
            )

            st.bar_chart(
                method_chart,
                color="#16A34A"
            )
        else:
            st.info("No failed payment methods available to analyze.")

    st.divider()

    # BUSINESS ANALYSIS
    st.subheader("🔎 Recovery Breakdown")

    tab1, tab2 = st.tabs(
        [
            "💳 Payment Methods",
            "⚠️ Failure Reasons"
        ]
    )

    with tab1:

        if not failed_df.empty:
            method_analysis = (
                failed_df
                .groupby("payment_method")
                .agg(
                    failed_transactions=("amount", "count"),
                    failed_revenue=("amount", "sum"),
                    expected_recovery=(
                        "expected_recovery",
                        "sum"
                    )
                )
                .reset_index()
            )

            method_analysis["recovery_rate"] = (
                method_analysis["expected_recovery"]
                .div(method_analysis["failed_revenue"])
                .mul(100)
                .round(2)
            )

            method_analysis = method_analysis.sort_values(
                by="expected_recovery",
                ascending=False
            )

            method_display = method_analysis.copy()

            method_display["failed_revenue"] = (
                "₹"
                + method_display["failed_revenue"]
                .round(2)
                .astype(str)
            )

            method_display["expected_recovery"] = (
                "₹"
                + method_display["expected_recovery"]
                .round(2)
                .astype(str)
            )

            method_display["recovery_rate"] = (
                method_display["recovery_rate"]
                .astype(str)
                + "%"
            )

            method_display = method_display.rename(
                columns={
                    "payment_method": "Payment Method",
                    "failed_transactions": "Failed Transactions",
                    "failed_revenue": "Failed Revenue",
                    "expected_recovery": "Expected Recovery",
                    "recovery_rate": "Recovery Rate"
                }
            )

            st.dataframe(
                method_display,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No payment-method data available.")

    with tab2:

        if not failed_df.empty:
            reason_analysis = (
                failed_df
                .groupby("failure_reason")
                .agg(
                    failed_transactions=("amount", "count"),
                    failed_revenue=("amount", "sum"),
                    expected_recovery=(
                        "expected_recovery",
                        "sum"
                    )
                )
                .reset_index()
            )

            reason_analysis["recovery_rate"] = (
                reason_analysis["expected_recovery"]
                .div(reason_analysis["failed_revenue"])
                .mul(100)
                .round(2)
            )

            reason_analysis = reason_analysis.sort_values(
                by="expected_recovery",
                ascending=False
            )

            reason_display = reason_analysis.copy()

            reason_display["failed_revenue"] = (
                "₹"
                + reason_display["failed_revenue"]
                .round(2)
                .astype(str)
            )

            reason_display["expected_recovery"] = (
                "₹"
                + reason_display["expected_recovery"]
                .round(2)
                .astype(str)
            )

            reason_display["recovery_rate"] = (
                reason_display["recovery_rate"]
                .astype(str)
                + "%"
            )

            reason_display = reason_display.rename(
                columns={
                    "failure_reason": "Failure Reason",
                    "failed_transactions": "Failed Transactions",
                    "failed_revenue": "Failed Revenue",
                    "expected_recovery": "Expected Recovery",
                    "recovery_rate": "Recovery Rate"
                }
            )

            st.dataframe(
                reason_display,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No failure-reason data available.")

# PAYMENT ANALYZER
elif page == "Payment Analyzer":

    st.subheader("🔍 AI Payment Analyzer")

    st.write(
        "Enter payment details to receive an AI-powered "
        "recovery recommendation."
    )

    col1, col2 = st.columns(2)

    with col1:

        amount = st.number_input(
            "Payment Amount (₹)",
            min_value=1.0,
            value=2500.0,
            step=100.0
        )

        payment_method = st.selectbox(
            "Payment Method",
            sorted(
                df["payment_method"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
        )

        failure_reason = st.selectbox(
            "Failure Reason",
            sorted(
                df["failure_reason"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
        )

    with col2:

        previous_transactions = st.number_input(
            "Previous Transactions",
            min_value=0,
            value=10,
            step=1
        )

        previous_successes = st.number_input(
            "Previous Successful Transactions",
            min_value=0,
            value=8,
            step=1
        )

        retry_count = st.number_input(
            "Previous Retry Attempts",
            min_value=0,
            value=0,
            step=1
        )

    if st.button(
        "🚀 Analyze Payment",
        use_container_width=True
    ):

        input_data = pd.DataFrame(
            [{
                "amount": amount,
                "payment_method": payment_method,
                "failure_reason": failure_reason,
                "previous_transactions": previous_transactions,
                "previous_successes": previous_successes,
                "retry_count": retry_count
            }]
        )

        probability = float(
            pipeline.predict_proba(
                input_data
            )[0][1]
        )

        expected = (
            amount * probability
        )

        # Action policy
        if retry_count >= 2:

            action = "ESCALATE"

            reason = (
                "Retry limit has been reached. "
                "Escalation is recommended."
            )

        elif probability >= 0.75:

            action = "RETRY"

            reason = (
                "High recovery probability. "
                "Immediate payment retry is recommended."
            )

        elif probability >= 0.50:

            action = "REMINDER"

            reason = (
                "Moderate recovery probability. "
                "Send a customer reminder before retry."
            )

        else:

            action = "ESCALATE"

            reason = (
                "Low recovery probability. "
                "Escalation or alternate payment method "
                "is recommended."
            )

        st.divider()

        r1, r2, r3 = st.columns(3)

        with r1:

            st.metric(
                "Recovery Probability",
                f"{probability * 100:.1f}%"
            )

        with r2:

            st.metric(
                "Recommended Action",
                action
            )

        with r3:

            st.metric(
                "Expected Recovery",
                f"₹{expected:,.2f}"
            )

        st.markdown(
            f"""
            <div class="result-box">

            <h3>🤖 AI Recommendation</h3>

            <p><b>Action:</b> {action}</p>

            <p><b>Reason:</b> {reason}</p>

            <p>
            <b>Recovery Probability:</b>
            {probability * 100:.1f}%
            </p>

            <p>
            <b>Expected Revenue Recovery:</b>
            ₹{expected:,.2f}
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

# TRANSACTION HISTORY
elif page == "Transaction History":

    st.subheader("📋 Transaction History")

    col1, col2, col3 = st.columns(3)

    with col1:

        status_filter = st.selectbox(
            "Status",
            [
                "All",
                "Recovered",
                "Failed"
            ]
        )

    with col2:

        method_filter = st.selectbox(
            "Payment Method",
            [
                "All"
            ] + sorted(
                df["payment_method"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
        )

    with col3:

        reason_filter = st.selectbox(
            "Failure Reason",
            [
                "All"
            ] + sorted(
                df["failure_reason"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
        )

    filtered = df_analysis.copy()

    if status_filter != "All":

        filtered = filtered[
            filtered["status"] == status_filter
        ]

    if method_filter != "All":

        filtered = filtered[
            filtered["payment_method"]
            == method_filter
        ]

    if reason_filter != "All":

        filtered = filtered[
            filtered["failure_reason"]
            == reason_filter
        ]

    display_columns = [
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

    history = filtered[
        display_columns
    ].copy()

    history["recovery_probability"] = (
        history["recovery_probability"]
        .mul(100)
        .round(1)
    )

    history["expected_recovery"] = (
        history["expected_recovery"]
        .round(2)
    )

    history = history.rename(
        columns={
            "amount": "Amount",
            "payment_method": "Payment Method",
            "failure_reason": "Failure Reason",
            "previous_transactions": "Previous Transactions",
            "previous_successes": "Previous Successes",
            "retry_count": "Retry Count",
            "recovery_probability": "Recovery Probability (%)",
            "expected_recovery": "Expected Recovery (₹)",
            "status": "Status"
        }
    )

    st.write(
        f"Showing **{len(history):,}** transactions"
    )

    st.dataframe(
        history,
        use_container_width=True,
        hide_index=True
    )

# AI INSIGHTS
elif page == "AI Insights":

    st.subheader("🤖 AI Revenue Insights")

    prediction_features = [
        "amount",
        "payment_method",
        "failure_reason",
        "previous_transactions",
        "previous_successes",
        "retry_count"
    ]

    # Empty failed-data safeguard
    if failed_df.empty:
        st.info("No failed transactions are available for AI insights.")
        st.stop()

    failed_input = failed_df[prediction_features].copy()

    # Model predictions for failed transactions
    recovery_probability = pipeline.predict_proba(
        failed_input
    )[:, 1]

    # Payment-method analysis
    method_analysis = (
        failed_df
        .groupby("payment_method")
        .agg(
            failed_transactions=("payment_method", "count"),
            failed_revenue=("amount", "sum")
        )
        .reset_index()
    )

    failed_method_probability = pd.DataFrame({
        "payment_method": failed_df["payment_method"].values,
        "recovery_probability": recovery_probability
    })

    method_probability = (
        failed_method_probability
        .groupby("payment_method", as_index=False)["recovery_probability"]
        .mean()
    )

    method_analysis = method_analysis.merge(
        method_probability,
        on="payment_method",
        how="left"
    )

    method_analysis["recovery_rate"] = (
        method_analysis["recovery_probability"]
        .fillna(0)
        .mul(100)
        .round(2)
    )

    method_analysis["failed_revenue"] = (
        method_analysis["failed_revenue"]
        .round(2)
    )

    # Failure-reason analysis
    reason_analysis = (
        failed_df
        .groupby("failure_reason")
        .agg(
            failed_transactions=("failure_reason", "count"),
            failed_revenue=("amount", "sum")
        )
        .reset_index()
    )

    reason_probability = pd.DataFrame({
        "failure_reason": failed_df["failure_reason"].values,
        "recovery_probability": recovery_probability
    })

    reason_probability = (
        reason_probability
        .groupby("failure_reason", as_index=False)["recovery_probability"]
        .mean()
    )

    reason_analysis = reason_analysis.merge(
        reason_probability,
        on="failure_reason",
        how="left"
    )

    reason_analysis["recovery_rate"] = (
        reason_analysis["recovery_probability"]
        .fillna(0)
        .mul(100)
        .round(2)
    )

    reason_analysis["failed_revenue"] = (
        reason_analysis["failed_revenue"]
        .round(2)
    )

    # Find best/worst groups
    best_method = method_analysis.loc[
        method_analysis["recovery_rate"].idxmax()
    ]

    worst_method = method_analysis.loc[
        method_analysis["recovery_rate"].idxmin()
    ]

    best_reason = reason_analysis.loc[
        reason_analysis["recovery_rate"].idxmax()
    ]

    worst_reason = reason_analysis.loc[
        reason_analysis["recovery_rate"].idxmin()
    ]

    i1, i2 = st.columns(2)

    with i1:
        st.markdown(
            f"""
            <div class="insight-box">
                <h3>🏆 Best Recovery Channel</h3>
                <p>
                    <b>{best_method["payment_method"]}</b>
                    has the highest expected recovery rate of
                    <b>{best_method["recovery_rate"]:.1f}%</b>.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with i2:
        st.markdown(
            f"""
            <div class="insight-box">
                <h3>⚠️ Highest Risk Channel</h3>
                <p>
                    <b>{worst_method["payment_method"]}</b>
                    has the lowest expected recovery rate of
                    <b>{worst_method["recovery_rate"]:.1f}%</b>.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    i3, i4 = st.columns(2)

    with i3:
        st.markdown(
            f"""
            <div class="insight-box">
                <h3>💡 Most Recoverable Failure</h3>
                <p>
                    <b>{best_reason["failure_reason"]}</b>
                    shows the strongest expected recovery rate:
                    <b>{best_reason["recovery_rate"]:.1f}%</b>.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with i4:
        st.markdown(
            f"""
            <div class="insight-box">
                <h3>🚨 Priority Failure Type</h3>
                <p>
                    <b>{worst_reason["failure_reason"]}</b>
                    has the lowest expected recovery rate:
                    <b>{worst_reason["recovery_rate"]:.1f}%</b>.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    st.subheader("🎯 Recommended Business Strategy")

    if recovery_rate >= 40:
        st.success(
            "The AI model indicates strong recovery potential. "
            "Prioritize high-probability failed payments for automated retries."
        )
    elif recovery_rate >= 25:
        st.warning(
            "Recovery potential is moderate. Combine automated retries "
            "with customer reminders."
        )
    else:
        st.error(
            "Recovery potential is relatively low. Prioritize escalation "
            "and alternate payment methods."
        )

    st.write(
        f"""
        Based on the current dataset:

        - **{failed_transactions:,}** failed transactions require attention.
        - Expected recoverable revenue is approximately **₹{expected_recovery:,.2f}**.
        - The model's test accuracy is **{accuracy * 100:.1f}%**.
        - Estimated recovery rate is **{recovery_rate:.2f}%**.
        """
    )

import streamlit as st
import pandas as pd
import numpy as np

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
    "retry_count"
]

NUMERIC_FEATURES = [
    "amount",
    "previous_transactions",
    "previous_successes",
    "retry_count"
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
def money(value):
    return f"₹{value:,.0f}"

def get_action(probability, retry_count):

    if retry_count >= 2:
        return "ESCALATE", "Retry limit reached. Escalation is recommended."

    if probability >= 0.75:
        return "RETRY", "High recovery probability. Retry immediately."

    if probability >= 0.50:
        return "REMINDER", "Moderate probability. Send a reminder before retry."

    return "ESCALATE", "Low probability. Offer alternate payment options."

def action_column(data):
    return np.select(
        [
            data["retry_count"] >= 2,
            data["recovery_probability"] >= 0.75,
            data["recovery_probability"] >= 0.50
        ],
        ["ESCALATE", "RETRY", "SEND REMINDER"],
        default="ESCALATE"
    )

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

        priority["AI Action"] = action_column(priority)
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
        action, reason = get_action(probability, retry_count)

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
            <h3>🤖 AI Recommendation</h3>
            <p><b>Action:</b> {action}</p>
            <p><b>Reason:</b> {reason}</p>
            <p><b>Expected Revenue Recovery:</b> ₹{expected:,.2f}</p>
        </div>
        """, unsafe_allow_html=True)


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

# APP SETUP
try:
    df = load_data()
    pipeline, accuracy, target_column = train_model(df)

except Exception as error:
    st.error("Application setup failed.")
    st.exception(error)
    st.stop()


df_analysis = df.copy()
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

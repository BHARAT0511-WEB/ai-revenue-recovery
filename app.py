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

.main-title {
    font-size: 40px;
    font-weight: 800;
    margin-bottom: 0;
}

.subtitle {
    color: #6b7280;
    font-size: 17px;
    margin-bottom: 20px;
}

.section-title {
    font-size: 25px;
    font-weight: 700;
}

.insight-box {
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
    background: #fafafa;
    color: #222222;
    margin-bottom: 12px;
}

.result-box {
    padding: 22px;
    border-radius: 14px;
    border: 1px solid #e5e7eb;
    margin-top: 18px;
}

.small-text {
    color: #6b7280;
    font-size: 14px;
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

# HEADER
st.markdown(
    '<div class="main-title">💰 AI Revenue Recovery</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered payment failure recovery and revenue optimization'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

# SIDEBAR
st.sidebar.title("⚙️ Dashboard")

page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Payment Analyzer",
        "Transaction History",
        "AI Insights"
    ]
)

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

    st.subheader("📊 Revenue Recovery Overview")

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Total Revenue",
            f"₹{total_revenue:,.0f}"
        )

    with c2:

        st.metric(
            "Failed Revenue",
            f"₹{failed_revenue:,.0f}"
        )

    with c3:

        st.metric(
            "Expected Recovery",
            f"₹{expected_recovery:,.0f}"
        )

    with c4:

        st.metric(
            "Recovery Rate",
            f"{recovery_rate:.2f}%"
        )

    st.divider()

    # Revenue Distribution
    st.subheader("💰 Revenue Distribution")

    chart_data = pd.DataFrame(
        {
            "Revenue": [
                failed_revenue,
                actual_recovered_revenue
            ]
        },
        index=[
            "Failed Revenue",
            "Recovered Revenue"
        ]
    )

    st.bar_chart(
        chart_data
    )

    # Payment Method
    st.subheader("💳 Recovery by Payment Method")

    method_analysis = (
        failed_df
        .groupby("payment_method")
        .agg(
            failed_revenue=("amount", "sum"),
            expected_recovery=("expected_recovery", "sum"),
            transactions=("amount", "count")
        )
        .reset_index()
    )

    method_analysis["recovery_rate"] = (
        method_analysis["expected_recovery"]
        / method_analysis["failed_revenue"]
        * 100
    )

    st.dataframe(
        method_analysis,
        use_container_width=True,
        hide_index=True
    )

    st.bar_chart(
        method_analysis.set_index(
            "payment_method"
        )[
            ["failed_revenue", "expected_recovery"]
        ]
    )

    # Failure Reasons
    st.subheader("⚠️ Failure Reason Analysis")

    reason_analysis = (
        failed_df
        .groupby("failure_reason")
        .agg(
            transactions=("amount", "count"),
            failed_revenue=("amount", "sum"),
            expected_recovery=("expected_recovery", "sum")
        )
        .reset_index()
    )

    reason_analysis["recovery_rate"] = (
        reason_analysis["expected_recovery"]
        / reason_analysis["failed_revenue"]
        * 100
    )

    st.dataframe(
        reason_analysis,
        use_container_width=True,
        hide_index=True
    )

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

    # Payment method analysis
    method_analysis = (
        failed_df.groupby("payment_method")
        .agg(
            failed_transactions=("payment_method", "count"),
            failed_revenue=("amount", "sum")
        )
        .reset_index()
    )

    # Generate actual recovery probabilities from the trained model
    prediction_features = [
        "amount",
        "payment_method",
        "failure_reason",
        "previous_transactions",
        "previous_successes",
        "retry_count"
    ]

    failed_input = failed_df[prediction_features].copy()

    recovery_probability = pipeline.predict_proba(
        failed_input
    )[:, 1]

# Actual failed transaction data by payment method
method_analysis = (
    failed_df
    .groupby("payment_method")
    .agg(
        failed_transactions=("payment_method", "count"),
        failed_revenue=("amount", "sum")
    )
    .reset_index()
)
 # Calculate model-based recovery probability by payment method
method_probability = (
    failed_method_probability
    .groupby("payment_method", as_index=False)["recovery_probability"]
    .mean()
) 

# Combine actual transaction data with model predictions
method_analysis = method_analysis.merge(
    method_probability,
    on="payment_method",
    how="left"
)

# Convert probability to percentage
method_analysis["recovery_rate"] = (
    method_analysis["recovery_probability"].fillna(0) * 100
).round(2) 

method_analysis["failed_revenue"] = (
    method_analysis["failed_revenue"].round(2)
)

# Best and worst payment method
if len(method_analysis) > 0:

    best_method = method_analysis.loc[
        method_analysis["recovery_rate"].idxmax()
    ]

    worst_method = method_analysis.loc[
        method_analysis["recovery_rate"].idxmin()
    ]

    # Failure Reason Analysis
    reason_analysis = (
        failed_df.groupby("failure_reason")
        .agg(
            failed_transactions=("failure_reason", "count"),
            failed_revenue=("amount", "sum")
        )
        .reset_index()
    )

    # Use actual model predictions
    reason_probability = pd.DataFrame({
        "failure_reason": failed_df["failure_reason"].values,
        "recovery_probability": recovery_probability
    })

    reason_probability = (
        reason_probability
        .groupby("failure_reason")["recovery_probability"]
        .mean()
        .reset_index()
    )

    reason_analysis = reason_analysis.merge(
        reason_probability,
        on="failure_reason",
        how="left"
    )

    reason_analysis["recovery_rate"] = (
        reason_analysis["recovery_probability"] * 100
    ).round(2)

    reason_analysis["failed_revenue"] = (
        reason_analysis["failed_revenue"].round(2)
    )
    # Best failure reason
    if len(reason_analysis) > 0:

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
            "Prioritize high-probability failed payments for "
            "automated retries."
        )

    elif recovery_rate >= 25:

        st.warning(
            "Recovery potential is moderate. Combine automated "
            "retries with customer reminders."
        )

    else:

        st.error(
            "Recovery potential is relatively low. "
            "Prioritize escalation and alternate payment methods."
        )

    st.write(
        f"""
        Based on the current dataset:

        - **{failed_transactions:,}** failed transactions require attention.
        - Expected recoverable revenue is approximately
          **₹{expected_recovery:,.2f}**.
        - The model's test accuracy is
          **{accuracy * 100:.1f}%**.
        - Estimated recovery rate is
          **{recovery_rate:.2f}%**.
        """
    )

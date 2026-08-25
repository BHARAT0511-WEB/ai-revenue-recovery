import streamlit as st
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Revenue Recovery",
    page_icon="💰",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 700;
}

.subtitle {
    font-size: 18px;
    color: #666;
}

.result-box {
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #ddd;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    df = pd.read_csv("transactions.csv")

    return df


# =========================================================
# TRAIN MODEL
# =========================================================

@st.cache_resource
def train_model(df):

    # Expected feature columns
    feature_columns = [
        "amount",
        "payment_method",
        "failure_reason",
        "previous_transactions",
        "previous_successes",
        "retry_count"
    ]

    # Check columns
    missing = [
        col for col in feature_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns in transactions.csv: {missing}"
        )

    # Find target column
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

        # Try to find a binary column
        for col in df.columns:

            unique_values = df[col].dropna().unique()

            if len(unique_values) == 2:

                if col not in feature_columns:
                    target_column = col
                    break

    if target_column is None:
        raise ValueError(
            "Could not find recovery target column in transactions.csv."
        )

    X = df[feature_columns].copy()
    y = df[target_column].copy()

    # Convert target to 0/1 if necessary
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

    y = pd.to_numeric(y, errors="coerce")

    valid_rows = y.notna()

    X = X.loc[valid_rows].copy()
    y = y.loc[valid_rows].astype(int)

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

    return pipeline, accuracy, target_column


# =========================================================
# LOAD + TRAIN
# =========================================================

try:

    df = load_data()

    pipeline, accuracy, target_column = train_model(df)

except Exception as e:

    st.error("Model setup failed.")

    st.exception(e)

    st.stop()


# =========================================================
# HEADER
# =========================================================

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


# =========================================================
# DASHBOARD METRICS
# =========================================================

# Failed transactions
failed_df = df.copy()

# Identify failed/recovery rows
if target_column in failed_df.columns:

    target_numeric = pd.to_numeric(
        failed_df[target_column],
        errors="coerce"
    )

    failed_df = failed_df[target_numeric == 0]


# Total failed revenue
if "amount" in failed_df.columns:

    failed_revenue = float(
        failed_df["amount"].sum()
    )

else:

    failed_revenue = 0.0


# Predict recovery probability for failed transactions
if len(failed_df) > 0:

    prediction_features = failed_df[
        [
            "amount",
            "payment_method",
            "failure_reason",
            "previous_transactions",
            "previous_successes",
            "retry_count"
        ]
    ].copy()

    probabilities = pipeline.predict_proba(
        prediction_features
    )[:, 1]

    expected_recovery = float(
        (
            failed_df["amount"].values
            * probabilities
        ).sum()
    )

else:

    expected_recovery = 0.0


if failed_revenue > 0:

    recovery_rate = (
        expected_recovery
        / failed_revenue
    ) * 100

else:

    recovery_rate = 0.0


col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Failed Revenue",
        f"₹{failed_revenue:,.0f}"
    )

with col2:

    st.metric(
        "Expected Recovery",
        f"₹{expected_recovery:,.0f}"
    )

with col3:

    st.metric(
        "Recovery Rate",
        f"{recovery_rate:.2f}%"
    )

with col4:

    st.metric(
        "AI Model Accuracy",
        f"{accuracy * 100:.1f}%"
    )


st.divider()


# =========================================================
# PAYMENT ANALYSIS
# =========================================================

st.subheader("🔍 Analyze Failed Payment")

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


# =========================================================
# PREDICTION
# =========================================================

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

    try:

        probability = float(
            pipeline.predict_proba(
                input_data
            )[0][1]
        )

        expected_recovery = (
            amount * probability
        )

        # AI action
        if probability >= 0.75:

            action = "Retry Payment"

            reason = (
                "High recovery probability. "
                "A payment retry is recommended."
            )

        elif probability >= 0.50:

            action = "Smart Retry"

            reason = (
                "Moderate recovery probability. "
                "Retry with optimized timing."
            )

        elif probability >= 0.30:

            action = "Customer Reminder"

            reason = (
                "Recovery probability is moderate-low. "
                "Send a payment reminder before retrying."
            )

        else:

            action = "Manual Review"

            reason = (
                "Low recovery probability. "
                "Manual review or alternative payment method "
                "is recommended."
            )


        # =================================================
        # RESULT
        # =================================================

        st.divider()

        st.subheader("🤖 AI Decision")

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
                f"₹{expected_recovery:,.2f}"
            )


        st.markdown(
            f"""
            <div class="result-box">

            <h3>AI Recommendation</h3>

            <p>
            <b>Action:</b> {action}
            </p>

            <p>
            <b>Reason:</b> {reason}
            </p>

            <p>
            <b>Recovery Probability:</b>
            {probability * 100:.1f}%
            </p>

            <p>
            <b>Expected Revenue Recovery:</b>
            ₹{expected_recovery:,.2f}
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    except Exception as e:

        st.error(
            "Prediction failed."
        )

        st.exception(e)


# =========================================================
# DATASET INFORMATION
# =========================================================

with st.expander("📊 Dataset & Model Information"):

    st.write(
        f"**Dataset rows:** {len(df):,}"
    )

    st.write(
        f"**Dataset columns:** {len(df.columns)}"
    )

    st.write(
        f"**Target column:** `{target_column}`"
    )

    st.write(
        f"**Model:** Random Forest Classifier"
    )

    st.write(
        f"**Test Accuracy:** {accuracy * 100:.2f}%"
    )

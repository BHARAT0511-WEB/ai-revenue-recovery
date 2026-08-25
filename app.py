import streamlit as st
import pandas as pd
import joblib


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load("recovery_model.pkl")


pipeline = load_model()


# =========================================================
# RECOVERY POLICY
# =========================================================

def recovery_action(probability, retry_count):

    if retry_count >= 3:
        return "ESCALATE"

    if probability >= 0.70:
        return "RETRY"

    elif probability >= 0.40:
        return "REMINDER"

    else:
        return "ESCALATE"


# =========================================================
# PAGE CONFIGURATION
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
# BUSINESS METRICS
# =========================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Failed Revenue",
        "₹5.20 L"
    )

with col2:
    st.metric(
        "Expected Recovery",
        "₹2.17 L"
    )

with col3:
    st.metric(
        "Recovery Rate",
        "41.74%"
    )

with col4:
    st.metric(
        "AI Model Accuracy",
        "91.3%"
    )

st.divider()


# =========================================================
# PAYMENT INPUT
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
        [
            "UPI",
            "Card",
            "Net Banking",
            "Wallet"
        ]
    )

    failure_reason = st.selectbox(
        "Failure Reason",
        [
            "Network Error",
            "Insufficient Funds",
            "Bank Timeout",
            "Authentication Failure",
            "Expired Card"
        ]
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
# ANALYZE PAYMENT
# =========================================================

if st.button(
    "🚀 Analyze Payment",
    use_container_width=True
):

    # Create model input
    input_data = pd.DataFrame([{
        "amount": amount,
        "payment_method": payment_method,
        "failure_reason": failure_reason,
        "previous_transactions": previous_transactions,
        "previous_successes": previous_successes,
        "retry_count": retry_count
    }])

    try:

        # -------------------------------------------------
        # ML PREDICTION
        # -------------------------------------------------

        probability = pipeline.predict_proba(
            input_data
        )[0][1]

        probability = float(probability)


        # -------------------------------------------------
        # RECOVERY ACTION
        # -------------------------------------------------

        action = recovery_action(
            probability,
            retry_count
        )


        # -------------------------------------------------
        # EXPECTED RECOVERY
        # -------------------------------------------------

        expected_recovery = (
            float(amount) * probability
        )


        # -------------------------------------------------
        # EXPLANATION
        # -------------------------------------------------

        if action == "RETRY":

            reason = (
                "High probability of payment recovery. "
                "Retrying the payment is recommended."
            )

        elif action == "REMINDER":

            reason = (
                "Medium probability of recovery. "
                "A customer reminder is recommended."
            )

        else:

            reason = (
                "Low recovery probability or retry limit "
                "reached. Escalation is recommended."
            )


        # -------------------------------------------------
        # DISPLAY RESULTS
        # -------------------------------------------------

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


        # -------------------------------------------------
        # RECOMMENDATION BOX
        # -------------------------------------------------

        st.markdown(
            f"""
            <div class="result-box">

            <h3>🤖 AI Recommendation</h3>

            <p>
            <b>Action:</b> {action}
            </p>

            <p>
            <b>Recovery Probability:</b>
            {probability * 100:.1f}%
            </p>

            <p>
            <b>Reason:</b>
            {reason}
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

        st.code(str(e))

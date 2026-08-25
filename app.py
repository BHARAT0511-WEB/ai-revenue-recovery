
import streamlit as st
import requests

# -----------------------------
# Page configuration
# -----------------------------

st.set_page_config(
    page_title="AI Revenue Recovery",
    page_icon="💰",
    layout="wide"
)

# -----------------------------
# Custom styling
# -----------------------------

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

# -----------------------------
# Header
# -----------------------------

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

# -----------------------------
# Dashboard metrics
# -----------------------------

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

# -----------------------------
# Payment analysis section
# -----------------------------

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

# -----------------------------
# Analyze button
# -----------------------------

if st.button(
    "🚀 Analyze Payment",
    use_container_width=True
):

    payload = {
        "amount": amount,
        "payment_method": payment_method,
        "failure_reason": failure_reason,
        "previous_transactions": previous_transactions,
        "previous_successes": previous_successes,
        "retry_count": retry_count
    }

    try:

        response = requests.post(
            "http://127.0.0.1:8001/predict",
            json=payload,
            timeout=10
        )

        if response.status_code == 200:

            result = response.json()

            probability = result["recovery_probability"]
            action = result["recommended_action"]
            expected_recovery = result["expected_recovery"]
            reason = result["reason"]

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

                <p><b>Action:</b> {action}</p>

                <p><b>Reason:</b> {reason}</p>

                <p>
                <b>Expected Revenue Recovery:</b>
                ₹{expected_recovery:,.2f}
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.error(
                f"API Error: {response.status_code}"
            )

    except Exception as e:

        st.error(
            "Could not connect to FastAPI. "
            "Make sure the API server is running on port 8001."
        )

        st.code(str(e))

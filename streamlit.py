import streamlit as st
import requests

API_URL = "http://localhost:8000"


HEADERS = {
    "X-API-Key": "fruad-secret-key-2026"
}

st.set_page_config(
    page_title="🔍 Fraud Detection",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 Credit Card Fraud Detection")
st.write("Upload a CSV file containing transaction records for fraud detection.")

st.divider()

uploaded_file = st.file_uploader(
    "📂 Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    st.success(f"Loaded: {uploaded_file.name}")

    if st.button("🔍 Detect Fraud", type="primary"):

        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                "text/csv"
            )
        }

        with st.spinner("Analyzing transactions..."):

            try:

                response = requests.post(
                    f"{API_URL}/detect",
                    headers=HEADERS,
                    files=files
                )

                if response.status_code == 200:

                    data = response.json()

                    st.success("✅ Analysis Complete")

                    st.metric(
                        "Total Transactions",
                        data["total_transactions"]
                    )

                    st.divider()

                    for i, result in enumerate(data["results"], start=1):

                        with st.expander(f"Transaction {i}"):

                            st.write(f"**Time:** {result['Time']}")
                            st.write(f"**Amount:** ${result['Amount']}")

                            if result["Prediction"] == "Fraud":
                                st.error("🚨 Fraud Detected")
                            else:
                                st.success("✅ Legitimate")

                            st.write(f"**Confidence:** {result['Confidence']}")
                            st.write(f"**Risk:** {result['Risk']}")
                            st.info(result["Recommendation"])

                else:
                    st.error(response.json()["detail"])

            except Exception as e:
                st.error(str(e))
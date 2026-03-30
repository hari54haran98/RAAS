"""
DAY 27: Streamlit UI with JWT Authentication
"""

import streamlit as st
import requests

st.set_page_config(page_title="RAAS - Banking Q&A", layout="wide")

API_URL = "http://localhost:8000"

if "token" not in st.session_state:
    st.session_state.token = None

st.title("🏦 RAAS - Banking Document Q&A")
st.info("🔐 **Demo Login:** `demo` / `demo123`")

if st.session_state.token is None:
    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")

        if st.button("Login", key="login_btn"):
            try:
                response = requests.post(
                    f"{API_URL}/token",
                    data={"username": username, "password": password}
                )
                if response.status_code == 200:
                    st.session_state.token = response.json()["access_token"]
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("Invalid username or password")
            except Exception as e:
                st.error(f"Connection error: {e}")

    with tab2:
        reg_user = st.text_input("Username", key="reg_user")
        reg_pass = st.text_input("Password", type="password", key="reg_pass")
        reg_email = st.text_input("Email", key="reg_email")
        reg_name = st.text_input("Full Name", key="reg_name")

        if st.button("Register", key="reg_btn"):
            try:
                response = requests.post(
                    f"{API_URL}/register",
                    json={
                        "username": reg_user,
                        "password": reg_pass,
                        "email": reg_email,
                        "full_name": reg_name
                    }
                )
                if response.status_code == 200:
                    st.success("Registration successful! Please login.")
                else:
                    st.error("Registration failed")
            except Exception as e:
                st.error(f"Connection error: {e}")

else:
    st.success(f"✅ Logged in as: {st.session_state.username}")

    if st.button("Logout"):
        st.session_state.token = None
        st.rerun()

    st.divider()

    question = st.text_area("💬 Ask a question:", height=100)

    if st.button("🔍 Ask", type="primary"):
        with st.spinner("Processing..."):
            try:
                headers = {"Authorization": f"Bearer {st.session_state.token}"}
                response = requests.post(
                    f"{API_URL}/ask",
                    json={"question": question},
                    headers=headers,
                    timeout=30
                )

                if response.status_code == 200:
                    data = response.json()

                    st.subheader("📚 Answer")
                    st.success(data.get("answer", "No answer"))

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Confidence", f"{data.get('confidence', 0):.0%}")
                    with col2:
                        st.metric("Hallucination Score", f"{data.get('hallucination_score', 0):.2f}")

                    if data.get("sources"):
                        st.subheader("📌 Sources")
                        for src in data["sources"]:
                            st.write(f"📄 {src.get('document', 'Unknown')} (Page {src.get('page', '?')})")
                else:
                    st.error(f"Error: {response.status_code}")
            except Exception as e:
                st.error(f"Error: {e}")

st.divider()
st.caption("RAAS v3.0 — FAISS + BM25 | Groq LLM | Hallucination Detection | JWT Auth")
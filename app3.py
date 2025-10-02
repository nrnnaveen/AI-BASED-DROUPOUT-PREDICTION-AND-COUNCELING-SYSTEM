# app.py
import streamlit as st
import pandas as pd
import numpy as np
from model import train_and_evaluate, save_model, load_model, predict_df
from generate_sample_data import generate
import plotly.express as px
import os
from dotenv import load_dotenv
import smtplib
from email.message import EmailMessage

load_dotenv()
st.set_page_config(page_title="𝙂𝙪𝙖𝙧𝙙𝙞𝙖𝙣𝘼𝙄", layout="wide")

# ---------------- Styled Title & Subheader ----------------
st.markdown("""
<h1 style='text-align: center; color: #2E86C1; font-size: 48px;'>
𝙂𝙪𝙖𝙧𝙙𝙞𝙖𝙣𝘼𝙄 – ᴡᴀᴛᴄʜ. ᴅᴇᴛᴇᴄᴛ. ᴘʀᴏᴛᴇᴄᴛ
</h1>
<p style='text-align: center; color: #555; font-size: 20px;'>Zero Dropouts, Infinite Potential</p>
""", unsafe_allow_html=True)

# ---------------- Sidebar ----------------
st.sidebar.markdown("## 📊 Data / Model Selection")
data_option = st.sidebar.radio("Choose data", ("Use sample data", "Upload CSV"))
model_option = st.sidebar.selectbox("Model action", ("Train new model", "Load existing model (dropout_model.joblib)"))

# ---------------- Data Handling ----------------
if data_option == "Use sample data":
    st.sidebar.write("Using generated sample data (2000 rows)")
    df = generate(2000)
else:
    uploaded_file = st.sidebar.file_uploader("Upload students CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success(f"Loaded {len(df)} rows")
    else:
        st.sidebar.info("No file uploaded. Using sample data.")
        df = generate(2000)

# Required columns check
required_cols = [
    "student_id", "gender", "scholarship", "attendance_pct", "avg_assignment_pct",
    "avg_test_pct", "fee_delay_days", "num_attempts", "prior_arrears", "engagement_score", "dropout_risk"
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.warning(f"Missing columns in dataset: {missing}. The sample data includes all required columns.")

# Type enforcement
expected_types = {
    "student_id": str,
    "gender": str,
    "scholarship": int,
    "attendance_pct": float,
    "avg_assignment_pct": float,
    "avg_test_pct": float,
    "fee_delay_days": int,
    "num_attempts": int,
    "prior_arrears": int,
    "engagement_score": float
}
for col, dtype in expected_types.items():
    if col in df.columns:
        df[col] = df[col].astype(dtype)

st.write("### Sample of dataset (showing up to 1000 rows)")
st.dataframe(df.head(1000))

# ---------------- Model Handling ----------------
model = None
metrics = None

if model_option == "Train new model":
    st.subheader("Train model")
    train_button = st.button("Train model on selected dataset")
    if train_button:
        model, metrics = train_and_evaluate(df)
        save_model(model, "dropout_model.joblib")
        st.success("✅ Model trained and saved to dropout_model.joblib")
        st.write("**Metrics**")
        st.json(metrics)
else:
    model = load_model("dropout_model.joblib")
    st.success("✅ Loaded model dropout_model.joblib")

# ---------------- Initialize high_risk ----------------
high_risk = pd.DataFrame()  # avoid NameError if predictions haven't run yet

# ---------------- Prediction & Risk ----------------
if model is not None:
    st.subheader("Predict & Explore")
    predict_option = st.radio("Prediction source", ("Predict on dataset shown above", "Upload new students to predict"))

    if predict_option == "Predict on dataset shown above":
        pred_df = predict_df(model, df)
    else:
        new_file = st.file_uploader("Upload new students (no label needed)", type=["csv"], key="predupload")
        if new_file is not None:
            new_df = pd.read_csv(new_file)
            pred_df = predict_df(model, new_df)
        else:
            st.info("Upload new rows to predict; otherwise predictions will run on current dataset.")
            pred_df = predict_df(model, df)

    # Risk level calculation
    def assign_risk_level(p):
        if p < 0.3:
            return "Low"
        elif p < 0.6:
            return "Medium"
        else:
            return "High"

    pred_df["risk_level"] = pred_df["risk_proba"].apply(assign_risk_level)

    # Color function for risk
    def color_risk(val):
        colors = {"High": "#ff4d4d", "Medium": "#ffc107", "Low": "#28a745"}
        return f"background-color: {colors.get(val, 'white')}; color: white; font-weight: bold;"

    # ---------------- Metrics Cards ----------------
    col1, col2, col3 = st.columns(3)
    col1.metric("High-Risk Students", len(pred_df[pred_df["risk_level"]=="High"]))
    col2.metric("Average Risk Probability", round(pred_df["risk_proba"].mean(), 2))
    col3.metric("Low-Risk Students", len(pred_df[pred_df["risk_level"]=="Low"]))

    # Top 10 predictions table
    st.write("### Predictions (top 10 rows with Risk Levels)")
    styled = pred_df.sort_values("risk_proba", ascending=False).head(10).style.applymap(color_risk, subset=["risk_level"])
    st.dataframe(styled, use_container_width=True)

    # Risk distribution histogram
    st.write("### Risk distribution")
    fig = px.histogram(
        pred_df, x="risk_proba", nbins=30,
        color="risk_level",
        color_discrete_map={"Low": "green", "Medium": "orange", "High": "red"},
        title="Predicted risk probability distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Risk pie chart
    st.write("### Risk Level Distribution")
    pie_fig = px.pie(pred_df, names="risk_level",
                     color="risk_level",
                     color_discrete_map={"Low":"green","Medium":"orange","High":"red"},
                     title="Risk Level Distribution")
    st.plotly_chart(pie_fig, use_container_width=True)

    # ---------------- Dynamic High-Risk Table ----------------
    if "threshold" not in st.session_state:
        st.session_state.threshold = 0.5

    threshold = st.slider("Risk threshold (label as high-risk)", 0.0, 1.0, st.session_state.threshold, key="threshold_slider")
    st.session_state.threshold = threshold

    def get_high_risk_df(df, threshold):
        df["risk_level"] = df["risk_proba"].apply(assign_risk_level)
        return df[df["risk_proba"] >= threshold].sort_values("risk_proba", ascending=False)

    high_risk = get_high_risk_df(pred_df, threshold)

    st.write(f"High-risk students (risk_proba >= {threshold}): {len(high_risk)}")
    st.dataframe(
        high_risk[["student_id","attendance_pct","avg_test_pct","engagement_score","risk_proba","risk_level"]]
        .head(50)
        .style.applymap(color_risk, subset=["risk_level"]),
        use_container_width=True
    )

    # Download buttons
    st.download_button("📥 Download High-risk CSV", high_risk.to_csv(index=False), file_name="high_risk_students.csv", mime="text/csv")
    st.download_button("📥 Download ALL Predictions", pred_df.to_csv(index=False), file_name="all_predictions.csv", mime="text/csv")

    # ---------------- Counseling Email Section ----------------
    st.write("## Counseling / Outreach")
    st.write("Compose a counseling email message and send to a list. (Prototype only)")

    with st.expander("Compose email"):
        enable_email = st.checkbox("Enable Email Sending (use with caution)", value=False)
        use_real_emails = st.checkbox(
            "Send to actual student emails (requires 'email' column in dataset)", value=False
        )

        subject = st.text_input("Subject", "Counseling: Support available to help your academic progress")
        body_template = st.text_area(
            "Body template",
            "Dear Student {student_id},\n\nWe noticed your attendance is {attendance_pct}% "
            "and recent test average is {avg_test_pct}%. We are concerned and want to offer support.\n\nRegards,\nCounseling Team"
        )

        def send_email(to_email, subject, body):
            try:
                host = os.getenv("EMAIL_HOST")
                port = int(os.getenv("EMAIL_PORT", "587"))
                user = os.getenv("EMAIL_USER")
                password = os.getenv("EMAIL_PASSWORD")
                from_addr = os.getenv("FROM_ADDRESS", user)

                if not all([host, port, user, password]):
                    st.error("SMTP credentials are missing! Check your .env file.")
                    return False

                msg = EmailMessage()
                msg["Subject"] = subject
                msg["From"] = from_addr
                msg["To"] = to_email
                msg.set_content(body)

                with smtplib.SMTP(host, port) as s:
                    s.starttls()
                    s.login(user, password)
                    s.send_message(msg)
                return True
            except Exception as e:
                st.error(f"Failed to send email to {to_email}: {e}")
                return False

        if enable_email and st.button("Send emails (demo)"):
            if high_risk.empty:
                st.info("No high-risk students to send to. Run predictions first.")
            else:
                sent, failed = 0, []
                for _, row in high_risk.iterrows():
                    student_id = row.get("student_id", "Unknown")
                    # Decide recipient
                    if use_real_emails and "email" in row:
                        to_email = row["email"]
                    else:
                        # Default safe test email
                        to_email = "ffnrnindian@gmail.com"

                    body = body_template.format(
                        student_id=student_id,
                        attendance_pct=row.get("attendance_pct", "N/A"),
                        avg_test_pct=row.get("avg_test_pct", "N/A"),
                        risk_proba=round(row.get("risk_proba", 0), 3)
                    )

                    if send_email(to_email, subject, body):
                        sent += 1
                    else:
                        failed.append(to_email)

                st.success(f"✅ Sent {sent} emails. Failures: {len(failed)}")
                if failed:
                    st.write("Failed to send to:", failed[:10])
        elif not enable_email:
            st.info("Email sending disabled by default. Tick the box to enable.")

# ---------------- Footer ----------------
st.markdown("""
<div style='text-align:center; padding:10px; font-size:14px; background: linear-gradient(to right, #f8f9fa, #e9ecef);'>
© 2025 GuardianAI | Developed by <b>GEN Z CODERS</b> | 💌 Contact: ffnrnindian@gmail.com
</div>
""", unsafe_allow_html=True)

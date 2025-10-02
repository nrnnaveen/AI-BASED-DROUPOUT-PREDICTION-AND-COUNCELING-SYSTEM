# app.py
import streamlit as st
import pandas as pd
import numpy as np
from model import train_and_evaluate, save_model, load_model, predict_df, build_pipeline
from generate_sample_data import generate
import plotly.express as px
import os
from dotenv import load_dotenv
import smtplib
from email.message import EmailMessage

load_dotenv()
st.set_page_config(page_title="𝙂𝙪𝙖𝙧𝙙𝙞𝙖𝙣𝘼𝙄", layout="wide")

st.title("𝙂𝙪𝙖𝙧𝙙𝙞𝙖𝙣𝘼𝙄 – ᴡᴀᴛᴄʜ. ᴅᴇᴛᴇᴄᴛ. ᴘʀᴏᴛᴇᴄᴛ")
st.subheader("Zero Dropouts, Infinite Potential")

# Sidebar controls
st.sidebar.header("Data / Model")
data_option = st.sidebar.radio("Choose data", ("Use sample data", "Upload CSV"))
model_option = st.sidebar.selectbox("Model action", ("Train new model", "Load existing model (dropout_model.joblib)"))

# ---- Data Handling ----
if data_option == "Use sample data":
    st.sidebar.write("Using generated sample data (2000 rows)")
    df = generate(2000)
else:
    uploaded_file = st.sidebar.file_uploader("Upload students CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success(f"Loaded {len(df)} rows")
        except Exception as e:
            st.sidebar.error(f"❌ Could not read CSV: {e}")
            df = generate(2000)
    else:
        st.sidebar.info("No file uploaded. Using sample data.")
        df = generate(2000)

# Required columns
required_cols = [
    "student_id", "gender", "scholarship", "attendance_pct", "avg_assignment_pct",
    "avg_test_pct", "fee_delay_days", "num_attempts", "prior_arrears", "engagement_score", "dropout_risk"
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.warning(f"Missing columns in dataset: {missing}. The sample data includes all required columns.")

# Datatype validation
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
        try:
            df[col] = df[col].astype(dtype)
        except Exception:
            st.warning(f"⚠️ Column {col} could not be converted to {dtype.__name__}")

st.write("### Sample of dataset (showing up to 1000 rows)")
st.dataframe(df.head(1000))

# ---- Model Handling ----
model = None
metrics = None

if model_option == "Train new model":
    st.subheader("Train model")
    train_button = st.button("Train model on selected dataset")
    if train_button:
        try:
            with st.spinner("Training model..."):
                model, metrics = train_and_evaluate(df)
                save_model(model, "dropout_model.joblib")
            st.success("✅ Model trained and saved to dropout_model.joblib")
            st.write("**Metrics**")
            st.json(metrics)
        except Exception as e:
            st.error(f"❌ Training failed: {e}")
else:
    try:
        model = load_model("dropout_model.joblib")
        st.success("✅ Loaded model dropout_model.joblib")
    except Exception as e:
        st.error("❌ Could not load model (dropout_model.joblib). Train a new model or upload a model file.")
        st.write(e)

# ---- Prediction ----
if model is not None:
    st.subheader("Predict & Explore")
    predict_option = st.radio("Prediction source", ("Predict on dataset shown above", "Upload new students to predict"))

    pred_df = None
    try:
        if predict_option == "Predict on dataset shown above":
            pred_df = predict_df(model, df)
        else:
            new_file = st.file_uploader("Upload new students (no label needed)", type=["csv"], key="predupload")
            if new_file is not None:
                new_df = pd.read_csv(new_file)
                missing_new = [c for c in expected_types.keys() if c not in new_df.columns]
                if missing_new:
                    st.error(f"Uploaded file is missing columns: {missing_new}")
                else:
                    pred_df = predict_df(model, new_df)
            else:
                st.info("Upload new rows to predict; otherwise predictions will run on current dataset.")
                pred_df = predict_df(model, df)
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        pred_df = None

    if pred_df is not None:
        # ---- Risk Levels & Colour Coding ----
        def assign_risk_level(p):
            if p < 0.3:
                return "Low"
            elif p < 0.6:
                return "Medium"
            else:
                return "High"

        pred_df["risk_level"] = pred_df["risk_proba"].apply(assign_risk_level)

        def color_risk(val):
            if val == "High":
                return "background-color: #ffcccc; color: #a30000;"   # red
            elif val == "Medium":
                return "background-color: #fff3cd; color: #856404;"   # yellow
            else:
                return "background-color: #d4edda; color: #155724;"   # green

        # Display top 10 predictions with color-coded risk
        st.write("### Predictions (top 10 rows with Risk Levels)")
        styled = pred_df.sort_values("risk_proba", ascending=False).head(10).style.applymap(color_risk, subset=["risk_level"])
        st.dataframe(styled, use_container_width=True)

        # Risk distribution histogram (color-coded)
        st.write("### Risk distribution")
        fig = px.histogram(pred_df, x="risk_proba", nbins=30, title="Predicted risk probability distribution",
                           color="risk_level", color_discrete_map={"Low": "green", "Medium": "orange", "High": "red"})
        st.plotly_chart(fig, use_container_width=True)

        # High-risk subset (controlled by slider)
        threshold = st.slider("Risk threshold (label as high-risk)", 0.0, 1.0, 0.5)
        high_risk = pred_df[pred_df["risk_proba"] >= threshold].sort_values("risk_proba", ascending=False)
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

        # Feature importance (approx)
        st.write("### Model feature importances (approx)")
        try:
            rf = None
            if hasattr(model, "named_steps"):
                if "rf" in model.named_steps:
                    rf = model.named_steps["rf"]
                    pre = model.named_steps["pre"]
                else:
                    for name, step in model.named_steps.items():
                        if hasattr(step, "named_steps") and "rf" in step.named_steps:
                            rf = step.named_steps["rf"]
                            pre = step.named_steps.get("pre", model.named_steps["pre"])
            if rf is not None:
                preprocessor = model.named_steps["pre"]
                num_features = preprocessor.transformers_[0][2]
                try:
                    ohe = preprocessor.transformers_[1][1].named_steps["onehot"]
                    cat_names = list(ohe.get_feature_names_out(preprocessor.transformers_[1][2]))
                except Exception:
                    cat_names = preprocessor.transformers_[1][2]
                feat_names = list(num_features) + cat_names
                fi = pd.DataFrame({"feature": feat_names, "importance": rf.feature_importances_}).sort_values("importance", ascending=False)
                st.dataframe(fi.head(20))
                fig2 = px.bar(fi.head(15), x="importance", y="feature", orientation="h", title="Top features")
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("Model does not expose feature importances in an accessible way.")
        except Exception as e:
            st.write("Could not compute feature importances:", e)

        # ---- Counseling Email ----
        st.write("## Counseling / Outreach")
        st.write("Compose a counseling email message and send to a list. (Prototype only)")

        with st.expander("Compose email"):
            enable_email = st.checkbox("Enable Email Sending (use with caution)", value=False)
            subject = st.text_input("Subject", "Counseling: Support available to help your academic progress")
            body_template = st.text_area("Body template", 
                                         "Dear Student {student_id},\n\nWe noticed your attendance is {attendance_pct}% "
                                         "and recent test average is {avg_test_pct}%. We are concerned and want to offer support.\n\nRegards,\nCounseling Team")

            def send_email(to_email, subject, body):
                host = os.getenv("EMAIL_HOST")
                port = int(os.getenv("EMAIL_PORT", "587"))
                user = os.getenv("EMAIL_USER")
                password = os.getenv("EMAIL_PASSWORD")
                from_addr = os.getenv("FROM_ADDRESS", user)
                if not (host and user and password):
                    raise EnvironmentError("Missing EMAIL_HOST/EMAIL_USER/EMAIL_PASSWORD environment variables.")
                msg = EmailMessage()
                msg["Subject"] = subject
                msg["From"] = from_addr
                msg["To"] = to_email
                msg.set_content(body)
                with smtplib.SMTP(host, port) as s:
                    s.starttls()
                    s.login(user, password)
                    s.send_message(msg)

            if enable_email and st.button("Send emails (demo)"):
                if high_risk.empty:
                    st.info("No high-risk students to send to.")
                else:
                    failed, sent = [], 0
                    for _, row in high_risk.iterrows():
                        student_id = row.get("student_id")
                        to_email = f"{student_id}@example.edu"
                        body = body_template.format(
                            student_id=student_id,
                            attendance_pct=row.get("attendance_pct"),
                            avg_test_pct=row.get("avg_test_pct"),
                            risk_proba=round(row.get("risk_proba"), 3)
                        )
                        try:
                            send_email(to_email, subject, body)
                            sent += 1
                        except Exception as e:
                            failed.append((to_email, str(e)))
                    st.success(f"✅ Sent {sent} emails. Failures: {len(failed)}")
                    if failed:
                        st.write(failed[:10])
            elif not enable_email:
                st.info("Email sending disabled by default. Tick the box to enable.")

# ---- Footer ----
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f5f5f5;
        color: #555;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        border-top: 1px solid #ddd;
    }
    </style>
    <div class="footer">
        © 2025 GuardianAI | Developed by <b>GEN Z CODERS</b>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("---")

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

# Sidebar controls
st.sidebar.header("Data / Model")
data_option = st.sidebar.radio("Choose data", ("Use sample data", "Upload CSV"))
model_option = st.sidebar.selectbox("Model action", ("Train new model", "Load existing model (dropout_model.joblib)"))

uploaded_df = None
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

# Ensure required columns exist or suggest mapping
required_cols = [
    "student_id", "gender", "scholarship", "attendance_pct", "avg_assignment_pct",
    "avg_test_pct", "fee_delay_days", "num_attempts", "prior_arrears", "engagement_score", "dropout_risk"
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.warning(f"Missing columns in dataset: {missing}. The sample data includes all required columns. If using your own CSV, ensure these columns exist (or rename accordingly).")
    # Show first rows to help user map columns
st.write("### Sample of dataset")
st.dataframe(df.head())

model = None
metrics = None
if model_option == "Train new model":
    st.subheader("Train model")
    train_button = st.button("Train model on selected dataset")
    if train_button:
        with st.spinner("Training model..."):
            model, metrics = train_and_evaluate(df)
            save_model(model, "dropout_model.joblib")
        st.success("Model trained and saved to dropout_model.joblib")
        st.write("**Metrics**")
        st.json(metrics)
else:
    # load existing
    try:
        model = load_model("dropout_model.joblib")
        st.success("Loaded model dropout_model.joblib")
    except Exception as e:
        st.error("Could not load model (dropout_model.joblib). Train a new model or upload a model file.")
        st.write(e)

# If model available, allow prediction
if model is not None:
    st.subheader("Predict & Explore")
    # Let user pick a subset or upload new student rows to predict
    predict_option = st.radio("Prediction source", ("Predict on dataset shown above", "Upload new students to predict"))
    if predict_option == "Predict on dataset shown above":
        pred_df = predict_df(model, df)
    else:
        new_file = st.file_uploader("Upload new students (no label needed)", type=["csv"], key="predupload")
        if new_file is not None:
            new_df = pd.read_csv(new_file)
            # attempt to fill missing columns (alert if missing)
            missing_new = [c for c in ["student_id","gender","scholarship","attendance_pct","avg_assignment_pct","avg_test_pct","fee_delay_days","num_attempts","prior_arrears","engagement_score"] if c not in new_df.columns]
            if missing_new:
                st.error(f"Uploaded file is missing columns: {missing_new}")
                pred_df = None
            else:
                pred_df = predict_df(model, new_df)
        else:
            st.info("Upload new rows to predict; otherwise predictions will run on current dataset.")
            pred_df = predict_df(model, df)

    if pred_df is not None:
        st.write("### Predictions (top 10 rows)")
        st.dataframe(pred_df.sort_values("risk_proba", ascending=False).head(10))

        # Summary stats
        st.write("### Risk distribution")
        fig = px.histogram(pred_df, x="risk_proba", nbins=30, title="Predicted risk probability distribution")
        st.plotly_chart(fig, use_container_width=True)

        # Show high-risk students
        threshold = st.slider("Risk threshold (label as high-risk)", 0.0, 1.0, 0.5)
        high_risk = pred_df[pred_df["risk_proba"] >= threshold].sort_values("risk_proba", ascending=False)
        st.write(f"High-risk students (risk_proba >= {threshold}): {len(high_risk)}")
        st.dataframe(high_risk[["student_id","attendance_pct","avg_test_pct","engagement_score","risk_proba"]].head(50))

        # Feature importance (approx via RF feature importances if RandomForest inside pipeline)
        st.write("### Model feature importances (approx)")
        try:
            # attempt to get feature importances from RandomForest inside pipeline
            rf = None
            # If pipeline: steps include pre -> rf
            if hasattr(model, "named_steps"):
                # if pipeline
                if "rf" in model.named_steps:
                    rf = model.named_steps["rf"]
                    pre = model.named_steps["pre"]
                else:
                    # nested pipeline
                    for name, step in model.named_steps.items():
                        if hasattr(step, "named_steps") and "rf" in step.named_steps:
                            rf = step.named_steps["rf"]
                            pre = step.named_steps["pre"] if "pre" in step.named_steps else model.named_steps["pre"]
            if rf is not None:
                # get feature names from preprocessor
                preprocessor = model.named_steps["pre"]
                num_features = preprocessor.transformers_[0][2]
                cat_pipe = preprocessor.transformers_[1][1]
                cat_names = []
                try:
                    ohe = preprocessor.transformers_[1][1].named_steps["onehot"]
                    cat_names = list(ohe.get_feature_names_out(preprocessor.transformers_[1][2]))
                except Exception:
                    cat_names = preprocessor.transformers_[1][2]
                feat_names = list(num_features) + cat_names
                importances = rf.feature_importances_
                fi = pd.DataFrame({"feature": feat_names, "importance": importances}).sort_values("importance", ascending=False)
                st.dataframe(fi.head(20))
                fig2 = px.bar(fi.head(15), x="importance", y="feature", orientation="h", title="Top features")
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("Model does not expose feature importances in an accessible way.")
        except Exception as e:
            st.write("Could not compute feature importances:", e)

        # Download high-risk list
        if not high_risk.empty:
            csv = high_risk.to_csv(index=False)
            st.download_button("Download high-risk CSV", csv, file_name="high_risk_students.csv", mime="text/csv")

        # Counseling email section
        st.write("## Counseling / Outreach")
        st.write("You can compose a counseling email message and send to a list. (This prototype uses SMTP; set credentials as environment variables.)")
        st.write("Set environment variables: EMAIL_HOST, EMAIL_PORT, EMAIL_USER, EMAIL_PASSWORD, FROM_ADDRESS")
        with st.expander("Compose email"):
            sample_subject = st.text_input("Subject", "Counseling: Support available to help your academic progress")
            sample_body = st.text_area("Body template (use {student_id}, {attendance_pct}, {avg_test_pct}, {risk_proba})",
                                       "Dear Student {student_id},\n\nWe noticed your attendance is {attendance_pct}% and recent test average is {avg_test_pct}%. We are concerned and want to offer support. Please contact the counseling team.\n\nRegards,\nCounseling Team")
            send_to_high_risk = st.checkbox("Send to all high-risk students above threshold", value=True)

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

            if st.button("Send emails (demo)"):
                # Note: sample dataset has no email column. For demo, generate fake emails using student_id
                if high_risk.empty:
                    st.info("No high-risk students to send to.")
                else:
                    failed = []
                    sent = 0
                    for _, row in high_risk.iterrows():
                        student_id = row.get("student_id")
                        to_email = f"{student_id}@example.edu"  # replace with real email column if available
                        body = sample_body.format(
                            student_id=student_id,
                            attendance_pct=row.get("attendance_pct"),
                            avg_test_pct=row.get("avg_test_pct"),
                            risk_proba=round(row.get("risk_proba"), 3)
                        )
                        try:
                            send_email(to_email, sample_subject, body)
                            sent += 1
                        except Exception as e:
                            failed.append((to_email, str(e)))
                    st.success(f"Sent {sent} emails. Failures: {len(failed)}")
                    if failed:
                        st.write(failed[:10])

st.write("---")

# app4.py
import streamlit as st
import pandas as pd
import numpy as np
from model3 import train_and_evaluate, save_model, load_model, predict_df
from generate_sample_datas import generate
import plotly.express as px
import os
from dotenv import load_dotenv
import smtplib
from email.message import EmailMessage

load_dotenv()
st.set_page_config(page_title="𝙂𝙪𝙖𝙧𝙙𝙞𝙖𝙣𝘼𝙄", layout="wide")

# ---------------- Styled Title ----------------
st.markdown("""
<h1 style='text-align: center; color: #2E86C1; font-size: 48px;'>
𝙂𝙪𝙖𝙧𝙙𝙞𝙖𝙣𝘼𝙄 – ᴡᴀᴛᴄʜ. ᴅᴇᴛᴇᴄᴛ. ᴘʀᴏᴛᴇᴄᴛ
</h1>
<p style='text-align: center; color: #555; font-size: 20px;'>Zero Dropouts, Infinite Potential</p>
""", unsafe_allow_html=True)

# ---------------- Sidebar ----------------
st.sidebar.markdown("## 📊 Data / Model Selection")
data_option = st.sidebar.radio("Choose data", ("Use sample data", "Upload Single CSV", "Upload multiple CSVs (Attendance, Tests, Fees)"))
model_option = st.sidebar.selectbox("Model action", ("Train new model", "Load existing model (dropout_model.joblib)"))

# ---------------- Data Handling ----------------
df = None

# Option 1: Sample Data
if data_option == "Use sample data":
    df = generate(2000)
    st.sidebar.success("✅ Generated sample dataset (2000 rows)")

# Option 2: Single CSV
elif data_option == "Upload Single CSV":
    uploaded_file = st.sidebar.file_uploader("Upload students CSV", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success(f"Loaded {len(df)} rows")
        except Exception as e:
            st.sidebar.error(f"Error reading file: {e}")
            df = generate(2000)
    else:
        st.sidebar.info("No file uploaded. Using sample data.")
        df = generate(2000)

# Option 3: Multiple CSVs
elif data_option == "Upload multiple CSVs (Attendance, Tests, Fees)":
    st.sidebar.info("Upload 3 CSVs with a common `student_id` column")
    att_file = st.sidebar.file_uploader("Upload Attendance CSV", type=["csv"], key="att")
    test_file = st.sidebar.file_uploader("Upload Tests CSV", type=["csv"], key="test")
    fee_file = st.sidebar.file_uploader("Upload Fees CSV", type=["csv"], key="fee")

    if att_file and test_file and fee_file:
        try:
            df_att = pd.read_csv(att_file)[["student_id", "attendance_pct"]]
            df_test = pd.read_csv(test_file)[["student_id", "avg_test_pct", "avg_assignment_pct"]]
            df_fee = pd.read_csv(fee_file)[["student_id", "fee_delay_days"]]

            df = df_att.merge(df_test, on="student_id", how="outer")
            df = df.merge(df_fee, on="student_id", how="outer")
            st.sidebar.success(f"✅ Merged dataset with {len(df)} rows")
        except Exception as e:
            st.sidebar.error(f"Error merging CSVs: {e}")
            df = generate(2000)
    else:
        st.sidebar.warning("⚠ Please upload all 3 files")
        df = generate(2000)

# Clean column names
df.columns = df.columns.str.strip()

# ---------------- Ensure required columns ----------------
required_cols = ["student_id","gender","scholarship","attendance_pct","avg_assignment_pct","avg_test_pct",
                 "fee_delay_days","num_attempts","prior_arrears","engagement_score","dropout_risk"]

# Fill missing required columns with default values if missing
for col in required_cols:
    if col not in df.columns:
        if col in ["gender", "student_id"]:
            df[col] = "Unknown"
        else:
            df[col] = 0

# Enforce types
expected_types = {"student_id": str, "gender": str, "scholarship": int, "attendance_pct": float, 
                  "avg_assignment_pct": float, "avg_test_pct": float, "fee_delay_days": int,
                  "num_attempts": int, "prior_arrears": int, "engagement_score": float,
                  "dropout_risk": int}

for col, dtype in expected_types.items():
    df[col] = df[col].astype(dtype)

st.write("### Sample dataset (up to 1000 rows)")
st.dataframe(df.head(1000))

# ---------------- Model Handling ----------------
model = None
metrics = None

if model_option == "Train new model":
    st.subheader("Train model")
    if st.button("Train model on selected dataset"):
        try:
            model, metrics = train_and_evaluate(df)
            save_model(model, "dropout_model.joblib")
            st.success("✅ Model trained and saved to dropout_model.joblib")
            st.json(metrics)
        except Exception as e:
            st.error(f"Training error: {e}")
else:
    model = load_model("dropout_model.joblib")
    if model:
        st.success("✅ Loaded model dropout_model.joblib")
    else:
        st.warning("No model found. Train a new model first.")

# ---------------- Prediction & Risk ----------------
if model:
    st.subheader("Predict & Explore")
    predict_option = st.radio("Prediction source", ("Predict on dataset shown above", "Upload new students to predict"))

    if predict_option == "Predict on dataset shown above":
        pred_df = predict_df(model, df)
    else:
        new_file = st.file_uploader("Upload new students CSV", type=["csv"], key="predupload")
        if new_file:
            new_df = pd.read_csv(new_file)
            pred_df = predict_df(model, new_df)
        else:
            st.info("Upload new CSV; predictions will run on current dataset.")
            pred_df = predict_df(model, df)

    # Assign risk levels
    def assign_risk_level(p):
        if p < 0.3: return "Low"
        elif p < 0.6: return "Medium"
        else: return "High"
    pred_df["risk_level"] = pred_df["risk_proba"].apply(assign_risk_level)

    # Color function for table
    def color_risk(val):
        colors = {"High":"#ff4d4d","Medium":"#ffc107","Low":"#28a745"}
        return f"background-color: {colors.get(val,'white')}; color:white; font-weight:bold;"

    # Metrics cards
    c1,c2,c3 = st.columns(3)
    c1.metric("High-Risk Students", len(pred_df[pred_df["risk_level"]=="High"]))
    c2.metric("Average Risk Probability", round(pred_df["risk_proba"].mean(),2))
    c3.metric("Low-Risk Students", len(pred_df[pred_df["risk_level"]=="Low"]))

    # Top 10 table
    st.write("### Predictions (Top 10)")
    styled = pred_df.sort_values("risk_proba", ascending=False).head(10).style.applymap(color_risk, subset=["risk_level"])
    st.dataframe(styled, use_container_width=True)

    # Histogram & Pie
    st.write("### Risk Distribution")
    fig = px.histogram(pred_df, x="risk_proba", nbins=30, color="risk_level",
                       color_discrete_map={"Low":"green","Medium":"orange","High":"red"},
                       title="Predicted risk probability")
    st.plotly_chart(fig, use_container_width=True)

    pie_fig = px.pie(pred_df, names="risk_level", color="risk_level",
                     color_discrete_map={"Low":"green","Medium":"orange","High":"red"},
                     title="Risk Level Distribution")
    st.plotly_chart(pie_fig, use_container_width=True)

    # High-risk students table
    if "threshold" not in st.session_state: st.session_state.threshold=0.5
    threshold = st.slider("Risk threshold",0.0,1.0,st.session_state.threshold)
    st.session_state.threshold=threshold
    high_risk = pred_df[pred_df["risk_proba"]>=threshold].sort_values("risk_proba",ascending=False)
    st.write(f"High-risk students (risk_proba>={threshold}): {len(high_risk)}")
    st.dataframe(high_risk[["student_id","attendance_pct","avg_test_pct","engagement_score","risk_proba","risk_level"]]
                 .head(50).style.applymap(color_risk, subset=["risk_level"]))

    # Download buttons
    st.download_button("📥 Download High-risk CSV", high_risk.to_csv(index=False), file_name="high_risk_students.csv", mime="text/csv")
    st.download_button("📥 Download All Predictions", pred_df.to_csv(index=False), file_name="all_predictions.csv", mime="text/csv")

# ---------------- Footer ----------------
st.markdown("""
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
  © 2025 GuardianAI | Developed by <b>GEN Z CODERS</b> | 💌 Contact: ffnrnindian@gmail.com
</div>
""", unsafe_allow_html=True)
st.write("---")

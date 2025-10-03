# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from dotenv import load_dotenv
import smtplib
from email.message import EmailMessage

from model3 import train_and_evaluate, save_model, load_model, predict_df
from generate_sample_datas import generate

load_dotenv()

# Streamlit UI Config
st.set_page_config(page_title="GuardianAI – Dropout Risk Dashboard", layout="wide")

st.title("𝙂𝙪𝙖𝙧𝙙𝙞𝙖𝙣𝘼𝙄 – ᴡᴀᴛᴄʜ. ᴅᴇᴛᴇᴄᴛ. ᴘʀᴏᴛᴇᴄᴛ")
st.subheader("AI-Based Dropout Prediction & Counseling — Prototype")

# -------------------------------
# Sidebar Options
# -------------------------------
st.sidebar.header("📂 Data Input Options")

data_option = st.sidebar.radio(
    "Choose data source:",
    ("Use Sample Data", "Upload Single CSV", "Upload multiple CSVs (Attendance, Tests, Fees)")
)

df = None

# --- Option 1: Sample Data
if data_option == "Use Sample Data":
    df = generate(2000)
    st.sidebar.success("✅ Generated sample dataset (2000 students)")

# --- Option 2: Single CSV Upload
elif data_option == "Upload Single CSV":
    uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV)", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success(f"✅ Uploaded dataset with {len(df)} rows")
        except Exception as e:
            st.sidebar.error(f"❌ Error reading file: {e}")
            df = generate(2000)

# --- Option 3: Multiple CSVs Upload
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
            st.sidebar.error(f"❌ Error merging CSVs: {e}")
            df = generate(2000)
    else:
        st.sidebar.warning("⚠ Please upload all 3 files")
        df = generate(2000)

# -------------------------------
# Data Preview
# -------------------------------
st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# -------------------------------
# Model Training & Prediction
# -------------------------------
st.subheader("⚙️ Train Model & Predict Dropout Risk")

if st.button("Train & Evaluate Model"):
    model, acc = train_and_evaluate(df)
    save_model(model)
    st.success(f"✅ Model trained with accuracy: {acc:.2f}")

if st.button("Predict Dropout Risk"):
    model = load_model()
    if model:
        df_pred = predict_df(model, df)
        st.dataframe(df_pred.head())

        # --- Visualization
        fig = px.histogram(df_pred, x="dropout_risk", color="dropout_risk",
                           title="Dropout Risk Distribution")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("❌ No trained model found. Please train the model first.")

# ---------------- Footer ----------------
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
      © 2025 GuardianAI | Developed by <b>GEN Z CODERS</b> | 💌 Contact: ffnrnindian@gmail.com
    </div>
    """,
    unsafe_allow_html=True
)

st.write("---")

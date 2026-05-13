import streamlit as st
import pandas as pd
import plotly.express as px

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Predictive Maintenance",
    page_icon="🔧",
    layout="wide"
)

# ---------------- LOAD DATA ----------------
df = pd.read_csv("Predictive_Maintenance_dataset.csv")

# ---------------- SIDEBAR ----------------
st.sidebar.title("About Project")

st.sidebar.info("""
AI-Based Predictive Maintenance
Proof-of-Concept System

Developed by:
Saeed Alghamdi

Laboratory Analyst &
Deputy Quality Manager

Saudi Standards, Metrology
and Quality Organization (SASO)
""")

st.sidebar.success("System Status: Online")

# ---------------- MAIN TITLE ----------------
st.title("🔧 AI Predictive Maintenance for Laboratory Equipment")

st.markdown("""
### Proof-of-Concept Decision Support System

This system is designed to support predictive maintenance planning for laboratory equipment using operational risk indicators and AI concepts.
""")

# ---------------- DATASET ----------------
st.header("Dataset Preview")

st.dataframe(df)

# ---------------- KPI SECTION ----------------
st.header("System Overview")

col1, col2, col3 = st.columns(3)

high_risk = len(df[df["Risk_Score"] >= 8])
medium_risk = len(df[(df["Risk_Score"] >= 5) & (df["Risk_Score"] < 8)])
low_risk = len(df[df["Risk_Score"] < 5])

col1.metric("High Risk Devices", high_risk)
col2.metric("Medium Risk Devices", medium_risk)
col3.metric("Low Risk Devices", low_risk)

# ---------------- CHART ----------------
st.header("Risk Score Distribution")

fig = px.bar(
    df,
    x="Device_Type",
    y="Risk_Score",
    color="Risk_Score",
    title="Equipment Risk Levels"
)

st.plotly_chart(fig, use_container_width=True)

# ---------------- INPUT SECTION ----------------
st.header("AI Prediction Input")

failure = st.selectbox("Failure", [0, 1])
predictability = st.slider("Predictability", 0, 3, 1)
repeatability = st.slider("Repeatability", 0, 3, 1)
operational_impact = st.slider("Operational Impact", 0, 4, 1)
sudden = st.selectbox("Sudden", [0, 1])
severity = st.slider("Severity", 0, 3, 1)
symptom_duration = st.slider("Symptom Duration", 0, 7, 1)

# ---------------- AI LOGIC ----------------
if st.button("Analyze Device"):

    risk_score = (
        failure +
        predictability +
        repeatability +
        operational_impact +
        sudden +
        severity +
        symptom_duration
    )

    if risk_score >= 8:
        status = "Critical"
        recommendation = "Immediate inspection required"
        priority = "High"

    elif risk_score >= 5:
        status = "Warning"
        recommendation = "Schedule inspection"
        priority = "Medium"

    else:
        status = "Normal"
        recommendation = "Continue monitoring"
        priority = "Low"

    st.header("AI Decision Output")

    st.error(f"AI Status: {status}")
    st.warning(f"Recommendation: {recommendation}")
    st.success(f"Priority Level: {priority}")

    st.write(f"Calculated Risk Score: {risk_score}")

# ---------------- DISCLAIMER ----------------
st.markdown("---")

st.caption("""
This system is a Proof-of-Concept AI decision support tool.
Final maintenance decisions require expert review.
""")

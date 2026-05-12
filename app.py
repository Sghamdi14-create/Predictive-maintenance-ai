import streamlit as st
import pandas as pd

st.set_page_config(page_title="AI Predictive Maintenance", layout="wide")

st.title("🔧 AI Predictive Maintenance for Laboratory Equipment")
st.caption("Proof-of-Concept Decision Support System")

df = pd.read_csv("Predictive_Maintenance_dataset.CSV", sep=";")

st.subheader("Dataset Preview")
st.dataframe(df)

st.subheader("Device Risk Dashboard")
st.bar_chart(df.set_index("Device_Type")["Risk_Score"])

st.subheader("AI Prediction Input")

failure = st.selectbox("Failure", [0, 1])
predictability = st.slider("Predictability", 1, 3, 2)
repeatability = st.slider("Repeatability", 0, 3, 1)
operational_impact = st.slider("Operational Impact", 0, 4, 3)
sudden = st.selectbox("Sudden", [0, 1])
severity = st.slider("Severity", 0, 4, 2)
symptom_duration = st.slider("Symptom Duration", 0, 10, 1)

risk_score = (
    failure * 2
    + predictability
    + repeatability
    + operational_impact
    + severity
    + sudden
    + (1 if symptom_duration >= 5 else 0)
)

if risk_score >= 8:
    status = "Critical"
    recommendation = "Immediate Maintenance Required"
    priority = "High"
elif risk_score >= 5:
    status = "Warning"
    recommendation = "Schedule Inspection"
    priority = "Medium"
else:
    status = "Normal"
    recommendation = "Continue Normal Operation"
    priority = "Low"

if st.button("Analyze Device"):
    st.subheader("AI Decision Output")
    st.success(f"AI Status: {status}")
    st.info(f"Recommendation: {recommendation}")
    st.warning(f"Priority Level: {priority}")
    st.write(f"Calculated Risk Score: {risk_score}")

st.caption("This is a Proof-of-Concept AI decision support tool. Final maintenance decisions require expert review.")

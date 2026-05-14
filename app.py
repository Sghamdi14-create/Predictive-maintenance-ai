import streamlit as st
import pandas as pd

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
st.image("logo.png", width=180)
st.set_page_config(
    page_title="AI Predictive Maintenance",
    page_icon="🔧",
    layout="wide"
)

df = pd.read_csv("Predictive_Maintenance_dataset 22.CSV", sep=";")
st.write(df.columns)
# تنظيف الأعمدة الرقمية لاستخدامها في XGBoost
model_columns = [
    "Predictability",
    "Repeatability",
    "Operational_impact",
    "Sudden",
    "Severity",
    "Symptom_Duration",
    "Failure"
]
for col in model_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# حذف الصفوف غير المكتملة
model_df = df.dropna(subset=model_columns)

# تحديد المدخلات والهدف
X = model_df[
        "Predictability",
        "Repeatability",
        "Operational_impact",
        "Sudden",
        "Severity",
        "Symptom_Duration"
]
y = model_df["Failure"]

# تدريب نموذج XGBoost
xgb_model = XGBClassifier(
    n_estimators=50,
    max_depth=3,
    learning_rate=0.1,
    random_state=42,
    eval_metric="logloss"
)

xgb_model.fit(X, y)
st.sidebar.title("About Project")
st.sidebar.info("""
AI-Based Predictive Maintenance  
Proof-of-Concept System

Developed by:  
Saeed Alghamdi

Laboratory Analyst & Deputy Quality Manager  
Saudi Standards, Metrology and Quality Organization (SASO)
""")
st.sidebar.success("System Status: Online")

st.title("🔧 AI Predictive Maintenance for Laboratory Equipment")

st.markdown("""
### Proof-of-Concept Decision Support System

This system supports predictive maintenance planning for laboratory equipment using operational risk indicators and AI decision logic.
""")

st.header("System Overview")
df["Risk_Score"] = pd.to_numeric(df["Risk_Score"], errors="coerce")
high_risk = len(df[df["Risk_Score"] >= 8])
medium_risk = len(df[(df["Risk_Score"] >= 5) & (df["Risk_Score"] < 8)])
low_risk = len(df[df["Risk_Score"] < 5])

col1, col2, col3 = st.columns(3)
col1.metric("High Risk Devices", high_risk)
col2.metric("Medium Risk Devices", medium_risk)
col3.metric("Low Risk Devices", low_risk)

st.header("Dataset Preview")
st.dataframe(df)

st.header("Device Risk Dashboard")
chart_data = df[["Device_Type", "Risk_Score"]].dropna()
st.bar_chart(chart_data.set_index("Device_Type"))

st.header("AI Prediction Input")

failure = st.selectbox("Failure", [0, 1])
predictability = st.slider("Predictability", 0, 3, 1)
repeatability = st.slider("Repeatability", 0, 3, 1)
operational_impact = st.slider("Operational Impact", 0, 4, 1)
sudden = st.selectbox("Sudden", [0, 1])
severity = st.slider("Severity", 0, 3, 1)
symptom_duration = st.slider("Symptom Duration", 0, 7, 1)

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
        st.error(f"AI Status: {status}")
    elif risk_score >= 5:
        status = "Warning"
        recommendation = "Schedule inspection"
        priority = "Medium"
        st.warning(f"AI Status: {status}")
    else:
        status = "Normal"
        recommendation = "Continue monitoring"
        priority = "Low"
        st.success(f"AI Status: {status}")

    st.info(f"Recommendation: {recommendation}")
    st.write(f"Priority Level: {priority}")
    st.write(f"Calculated Risk Score: {risk_score}")

st.markdown("---")
st.caption("This system is a Proof-of-Concept AI decision support tool. Final maintenance decisions require expert review.")

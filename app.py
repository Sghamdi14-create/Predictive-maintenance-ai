import streamlit as st
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# =========================
# Page Config
# =========================

st.set_page_config(
    page_title="AI Predictive Maintenance",
    page_icon="🔧",
    layout="wide"
)

# =========================
# Logo
# =========================

st.image("logo.png", width=180)

# =========================
# Read Dataset
# =========================

df = pd.read_csv(
    "Predictive_Maintenance_dataset 22.csv",
    encoding="latin1"
)

# تنظيف أسماء الأعمدة
df.columns = df.columns.str.strip()

# =========================
# Feature Columns
# =========================

feature_columns = [
    "Predictability",
    "Repeatability",
    "Operational_impact",
    "Sudden",
    "Severity",
    "Symptom_Duration"
]

# =========================
# Convert Numeric Columns
# =========================

for col in feature_columns + ["Failure", "Risk_Score"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# حذف الصفوف الفارغة
model_df = df.dropna(subset=feature_columns + ["Failure"])

# =========================
# Model Inputs
# =========================

X = model_df[feature_columns]
y = model_df["Failure"]

# =========================
# Train XGBoost Model
# =========================

xgb_model = XGBClassifier(
    n_estimators=50,
    max_depth=3,
    learning_rate=0.1,
    random_state=42,
    eval_metric="logloss"
)

xgb_model.fit(X, y)

# =========================
# Model Evaluation
# =========================

y_pred = xgb_model.predict(X)

accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)

# =========================
# Sidebar
# =========================

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

# =========================
# Model Performance
# =========================

st.sidebar.markdown("## Model Performance")

st.sidebar.success(f"XGBoost Accuracy: {accuracy:.2f}")
st.sidebar.info(f"Precision: {precision:.2f}")
st.sidebar.info(f"Recall: {recall:.2f}")
st.sidebar.info(f"F1 Score: {f1:.2f}")

st.sidebar.write(
    "Preliminary result based on the current proof-of-concept dataset."
)

# =========================
# Feature Importance
# =========================

importance_df = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": xgb_model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

st.sidebar.markdown("## Feature Importance")
st.sidebar.dataframe(importance_df)

# =========================
# Main Title
# =========================

st.title("🔧 AI Predictive Maintenance for Laboratory Equipment")

st.markdown("""
## Proof-of-Concept Decision Support System

This system supports predictive maintenance planning for laboratory equipment using operational risk indicators and an XGBoost machine learning model.
""")

# =========================
# System Overview
# =========================

st.header("System Overview")

high_risk = len(df[df["Risk_Score"] >= 8])

medium_risk = len(
    df[
        (df["Risk_Score"] >= 5) &
        (df["Risk_Score"] < 8)
    ]
)

low_risk = len(df[df["Risk_Score"] < 5])

col1, col2, col3 = st.columns(3)

col1.metric("High Risk Devices", high_risk)
col2.metric("Medium Risk Devices", medium_risk)
col3.metric("Low Risk Devices", low_risk)

# =========================
# Dataset Preview
# =========================

st.header("Dataset Preview")

st.dataframe(df)

# =========================
# Risk Dashboard
# =========================

st.header("Device Risk Dashboard")

chart_data = df[
    ["Device_Type", "Risk_Score"]
].dropna()

st.bar_chart(
    chart_data.set_index("Device_Type")
)

# =========================
# Feature Importance Chart
# =========================

st.header("XGBoost Feature Importance")

importance_chart = importance_df.set_index("Feature")

st.bar_chart(importance_chart)

# =========================
# Model Evaluation Summary
# =========================

st.header("Model Evaluation Summary")

m1, m2, m3, m4 = st.columns(4)

m1.metric("Accuracy", f"{accuracy:.2f}")
m2.metric("Precision", f"{precision:.2f}")
m3.metric("Recall", f"{recall:.2f}")
m4.metric("F1 Score", f"{f1:.2f}")

st.caption(
    "These metrics are preliminary and calculated on the current proof-of-concept dataset."
)

# =========================
# AI Prediction Section
# =========================

st.header("AI Predictive Analysis")

failure = st.selectbox("Failure", [0, 1])

predictability = st.slider(
    "Predictability",
    0,
    3,
    1
)

repeatability = st.slider(
    "Repeatability",
    0,
    3,
    1
)

operational_impact = st.slider(
    "Operational Impact",
    0,
    4,
    1
)

sudden = st.selectbox(
    "Sudden",
    [0, 1]
)

severity = st.slider(
    "Severity",
    0,
    3,
    1
)

symptom_duration = st.slider(
    "Symptom Duration",
    0,
    7,
    1
)

# =========================
# AI Prediction Button
# =========================

if st.button("Analyze Device Using XGBoost"):

    input_data = pd.DataFrame({
        "Predictability": [predictability],
        "Repeatability": [repeatability],
        "Operational_impact": [operational_impact],
        "Sudden": [sudden],
        "Severity": [severity],
        "Symptom_Duration": [symptom_duration]
    })

    prediction = xgb_model.predict(input_data)[0]

    probability = xgb_model.predict_proba(input_data)[0][1]

    risk_score = (
        failure +
        predictability +
        repeatability +
        operational_impact +
        sudden +
        severity +
        symptom_duration
    )

    # Status Logic

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

    # AI Results

    st.info(f"XGBoost Failure Prediction: {prediction}")

    st.write(
        f"Failure Probability: {probability:.2%}"
    )

    st.write(
        f"Recommendation: {recommendation}"
    )

    st.write(
        f"Priority Level: {priority}"
    )

    st.write(
        f"Calculated Risk Score: {risk_score}"
    )

# =========================
# Footer
# =========================

st.markdown("---")

st.caption(
    "This system is a Proof-of-Concept AI decision support tool using XGBoost. Final maintenance decisions require expert review."
)

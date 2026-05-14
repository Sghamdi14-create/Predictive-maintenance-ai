import streamlit as st
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Page setup
st.set_page_config(
    page_title="AI Predictive Maintenance",
    page_icon="🔧",
    layout="wide"
)

# Logo
st.image("logo.png", width=180)

# Load dataset
df = pd.read_csv("Predictive_Maintenance_dataset 22.CSV", sep=";")
df.columns = df.columns.str.strip()

# Model columns
feature_columns = [
    "Predictability",
    "Repeatability",
    "Operational_impact",
    "Sudden",
    "Severity",
    "Symptom_Duration"
]

target_column = "Failure"

model_columns = feature_columns + [target_column]

# Convert to numeric
for col in model_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["Risk_Score"] = pd.to_numeric(df["Risk_Score"], errors="coerce")

# Prepare model dataset
model_df = df.dropna(subset=model_columns)

X = model_df[feature_columns]
y = model_df[target_column]

# Train XGBoost model
xgb_model = XGBClassifier(
    n_estimators=50,
    max_depth=3,
    learning_rate=0.1,
    random_state=42,
    eval_metric="logloss"
)

xgb_model.fit(X, y)

# Model evaluation on proof-of-concept dataset
y_pred = xgb_model.predict(X)

accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred, zero_division=0)
recall = recall_score(y, y_pred, zero_division=0)
f1 = f1_score(y, y_pred, zero_division=0)

cm = confusion_matrix(y, y_pred)

feature_importance = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": xgb_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

# Sidebar
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

st.sidebar.markdown("### Model Performance")
st.sidebar.success(f"XGBoost Accuracy: {accuracy:.2f}")
st.sidebar.info(f"Precision: {precision:.2f}")
st.sidebar.info(f"Recall: {recall:.2f}")
st.sidebar.info(f"F1 Score: {f1:.2f}")
st.sidebar.caption("Preliminary results based on the current proof-of-concept dataset.")

st.sidebar.markdown("### Feature Importance")
st.sidebar.dataframe(feature_importance)

# Main title
st.title("🔧 AI Predictive Maintenance for Laboratory Equipment")

st.markdown("""
### Proof-of-Concept Decision Support System

This system supports predictive maintenance planning for laboratory equipment using operational risk indicators and an XGBoost machine learning model.
""")

# System Overview
st.header("System Overview")

high_risk = len(df[df["Risk_Score"] >= 8])
medium_risk = len(df[(df["Risk_Score"] >= 5) & (df["Risk_Score"] < 8)])
low_risk = len(df[df["Risk_Score"] < 5])

col1, col2, col3 = st.columns(3)

col1.metric("High Risk Devices", high_risk)
col2.metric("Medium Risk Devices", medium_risk)
col3.metric("Low Risk Devices", low_risk)

# Dataset preview
st.header("Dataset Preview")
st.dataframe(df)

# Device risk chart
st.header("Device Risk Dashboard")

chart_data = df[["Device_Type", "Risk_Score"]].dropna()
st.bar_chart(chart_data.set_index("Device_Type"))

# Feature importance chart
st.header("XGBoost Feature Importance")
st.bar_chart(feature_importance.set_index("Feature"))

# Confusion matrix
st.header("Model Evaluation Summary")

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

metric_col1.metric("Accuracy", f"{accuracy:.2f}")
metric_col2.metric("Precision", f"{precision:.2f}")
metric_col3.metric("Recall", f"{recall:.2f}")
metric_col4.metric("F1 Score", f"{f1:.2f}")

st.caption("These metrics are preliminary and calculated on the current proof-of-concept dataset.")

cm_df = pd.DataFrame(
    cm,
    index=["Actual 0", "Actual 1"],
    columns=["Predicted 0", "Predicted 1"]
)

st.subheader("Confusion Matrix")
st.dataframe(cm_df)

# AI Prediction Input
st.header("AI Prediction Input")

failure = st.selectbox("Observed Failure", [0, 1])

predictability = st.slider("Predictability", 0, 3, 1)
repeatability = st.slider("Repeatability", 0, 3, 1)
operational_impact = st.slider("Operational Impact", 0, 4, 1)
sudden = st.selectbox("Sudden", [0, 1])
severity = st.slider("Severity", 0, 3, 1)
symptom_duration = st.slider("Symptom Duration", 0, 7, 1)

if st.button("Analyze Device Using XGBoost"):

    input_data = pd.DataFrame(
        [[
            predictability,
            repeatability,
            operational_impact,
            sudden,
            severity,
            symptom_duration
        ]],
        columns=feature_columns
    )

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

    if probability >= 0.70:
        status = "Critical"
        recommendation = "Immediate inspection required"
        priority = "High"
        st.error(f"AI Status: {status}")

    elif probability >= 0.40:
        status = "Warning"
        recommendation = "Schedule inspection"
        priority = "Medium"
        st.warning(f"AI Status: {status}")

    else:
        status = "Normal"
        recommendation = "Continue monitoring"
        priority = "Low"
        st.success(f"AI Status: {status}")

    st.info(f"XGBoost Failure Prediction: {int(prediction)}")
    st.write(f"Failure Probability: {probability:.2%}")
    st.write(f"Recommendation: {recommendation}")
    st.write(f"Priority Level: {priority}")
    st.write(f"Calculated Risk Score: {risk_score}")

# Footer
st.markdown("---")
st.caption(
    "This system is a Proof-of-Concept AI decision support tool using XGBoost. Final maintenance decisions require expert review."
)

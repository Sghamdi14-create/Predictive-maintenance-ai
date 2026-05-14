import streamlit as st
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# إعداد الصفحة
st.set_page_config(
    page_title="AI Predictive Maintenance",
    page_icon="🔧",
    layout="wide"
)

# الشعار
st.image("logo.png", width=180)

# قراءة البيانات
df = pd.read_csv("Predictive_Maintenance_dataset 22.CSV", sep=";")

# تنظيف الأعمدة الرقمية
model_columns = [
    "Predictability",
    "Repeatability",
    "Operational_Impact",
    "Sudden",
    "Severity",
    "Symptom_Duration",
    "Failure"
]

for col in model_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# حذف الصفوف غير المكتملة
model_df = df.dropna(subset=model_columns)

# المدخلات والهدف
X = model_df[
    [
        "Predictability",
        "Repeatability",
        "Operational_Impact",
        "Sudden",
        "Severity",
        "Symptom_Duration"
    ]
]

y = model_df["Failure"]

# نموذج XGBoost
xgb_model = XGBClassifier(
    n_estimators=50,
    max_depth=3,
    learning_rate=0.1,
    random_state=42,
    eval_metric="logloss"
)

# تدريب النموذج
xgb_model.fit(X, y)

# تقييم النموذج
y_pred = xgb_model.predict(X)

model_accuracy = accuracy_score(y, y_pred)

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
st.sidebar.success(f"XGBoost Accuracy: {model_accuracy:.2f}")

# العنوان الرئيسي
st.title("🔧 AI Predictive Maintenance for Laboratory Equipment")

st.markdown("""
### Proof-of-Concept Decision Support System

This system supports predictive maintenance planning for laboratory equipment using operational risk indicators and an XGBoost machine learning model.
""")

# نظرة عامة
st.header("System Overview")

df["Risk_Score"] = pd.to_numeric(df["Risk_Score"], errors="coerce")

high_risk = len(df[df["Risk_Score"] >= 8])
medium_risk = len(df[(df["Risk_Score"] >= 5) & (df["Risk_Score"] < 8)])
low_risk = len(df[df["Risk_Score"] < 5])

col1, col2, col3 = st.columns(3)

col1.metric("High Risk Devices", high_risk)
col2.metric("Medium Risk Devices", medium_risk)
col3.metric("Low Risk Devices", low_risk)

# عرض البيانات
st.header("Dataset Preview")
st.dataframe(df)

# الرسم البياني
st.header("Device Risk Dashboard")

chart_data = df[["Device_Type", "Risk_Score"]].dropna()

st.bar_chart(chart_data.set_index("Device_Type"))

# إدخال المستخدم
st.header("AI Prediction Input")

failure = st.selectbox("Failure", [0, 1])

predictability = st.slider("Predictability", 0, 3, 1)

repeatability = st.slider("Repeatability", 0, 3, 1)

operational_impact = st.slider("Operational Impact", 0, 4, 1)

sudden = st.selectbox("Sudden", [0, 1])

severity = st.slider("Severity", 0, 3, 1)

symptom_duration = st.slider("Symptom Duration", 0, 7, 1)

# التنبؤ
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
        columns=[
            "Predictability",
            "Repeatability",
            "Operational_Impact",
            "Sudden",
            "Severity",
            "Symptom_Duration"
        ]
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

    st.info(f"XGBoost Failure Prediction: {prediction}")

    st.write(f"Failure Probability: {probability:.2%}")

    st.write(f"Recommendation: {recommendation}")

    st.write(f"Priority Level: {priority}")

    st.write(f"Calculated Risk Score: {risk_score}")

# Footer
st.markdown("---")

st.caption(
    "This system is a Proof-of-Concept AI decision support tool using XGBoost. Final maintenance decisions require expert review."
)

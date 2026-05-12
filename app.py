import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# إعداد الصفحة
st.set_page_config(
    page_title="AI Predictive Maintenance",
    layout="wide"
)

st.title("🔧 AI Predictive Maintenance for Laboratory Equipment")

# قراءة البيانات
df = pd.read_csv("Predictive_Maintenance_dataset.CSV", sep=';')

# عرض البيانات
st.subheader("Dataset Preview")
st.dataframe(df.head())

# اختيار الخصائص
features = [
    "Predictability",
    "Repeatability",
    "Operational_Impact",
    "Sudden",
    "Severity"
]

X = df[features]
y = df["Failure"]

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# تدريب النموذج
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# التنبؤ
predictions = model.predict(X_test)

# الدقة
accuracy = accuracy_score(y_test, predictions)

st.subheader("Model Accuracy")
st.success(f"Accuracy: {accuracy:.2f}")

st.subheader("Prediction Results")
results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": predictions
})

st.dataframe(results)

import streamlit as st
import requests
import matplotlib.pyplot as plt

st.set_page_config(page_title="Clinical AI", layout="centered")

st.title("🧠 AI Clinical Decision System")

st.subheader("Enter Patient Details")

data = {
    "Pregnancies": st.number_input("Pregnancies", 0, 10),
    "Glucose": st.number_input("Glucose"),
    "BloodPressure": st.number_input("Blood Pressure"),
    "SkinThickness": st.number_input("Skin Thickness"),
    "Insulin": st.number_input("Insulin"),
    "BMI": st.number_input("BMI"),
    "DiabetesPedigree": st.number_input("Diabetes Pedigree"),
    "Age": st.number_input("Age")
}

if st.button("Analyze Patient"):

    response = requests.post(
        "http://127.0.0.1:5000/predict",
        json=data
    )

    result = response.json()

    st.markdown("### 🧾 Patient Risk Summary")

    # Prediction
    if result["prediction"] == "Diabetic":
        st.error("⚠️ Diabetic Risk Detected")
    else:
        st.success("✅ No Diabetes Risk")

    # Risk level
    if result["risk_level"] == "High":
        st.error(f"HIGH (Score: {result['risk_score']})")
    elif result["risk_level"] == "Medium":
        st.warning(f"MEDIUM (Score: {result['risk_score']})")
    else:
        st.success(f"LOW (Score: {result['risk_score']})")

    # Progress bar
    st.progress(result["risk_score"] / 5)

    # Insights
    st.subheader("📌 Clinical Insights")
    for i in result["insights"]:
        st.write(f"- {i}")

    # Recommendations
    st.subheader("💡 Recommendations")
    for r in result["recommendations"]:
        st.write(f"- {r}")

    # Graph
    st.subheader("📈 Key Factors")

    features = []
    values = []

    if data["Glucose"] > 140:
        features.append("Glucose")
        values.append(data["Glucose"] / 200)

    if data["BMI"] > 25:
        features.append("BMI")
        values.append(data["BMI"] / 50)

    if data["Age"] > 40:
        features.append("Age")
        values.append(data["Age"] / 100)

    if len(features) == 0:
        features = ["Stable Health Indicators"]
        values = [0.2]
        st.success("All major health indicators are within safe range.")

    fig, ax = plt.subplots()
    ax.barh(features, values)
    ax.set_title("Patient Risk Contributors")

    st.pyplot(fig)
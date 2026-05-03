from model import ClinicalModel
from llm import generate_explanation

print("Starting LLM Test...\n")

# Initialize model
model = ClinicalModel()

# Sample input
sample_input = {
    "Pregnancies": 2,
    "Glucose": 150,
    "BloodPressure": 80,
    "SkinThickness": 25,
    "Insulin": 100,
    "BMI": 30,
    "DiabetesPedigree": 0.5,
    "Age": 45
}

# Prediction
prediction = model.predict(sample_input)

print("Prediction:", prediction)

# Generate LLM explanation
explanation = generate_explanation(sample_input, prediction)

print("\nLLM Explanation:\n")
print(explanation)
from flask import Flask, request, jsonify
import pickle
import pandas as pd
import json
from agent import run_agent

app = Flask(__name__)

# Load model
model = pickle.load(open("src/model.pkl", "rb"))

@app.route("/")
def home():
    return "Clinical AI API Running"

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    df = pd.DataFrame([data])

    prediction = int(model.predict(df)[0])

    # 🔥 Risk scoring logic
    glucose = data["Glucose"]
    bmi = data["BMI"]
    age = data["Age"]

    risk_score = 0

    if glucose > 140:
        risk_score += 2
    if bmi > 30:
        risk_score += 2
    if age > 40:
        risk_score += 1

    if risk_score >= 4:
        risk_level = "High"
    elif risk_score >= 2:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    # 🧠 Agent call
    agent_output = run_agent(data, prediction, risk_score, risk_level)

    try:
        agent_output = json.loads(agent_output)
    except:
        agent_output = {
            "insights": ["AI parsing failed"],
            "recommendations": ["Try again"]
        }

    return jsonify({
        "prediction": "Diabetic" if prediction == 1 else "Non-diabetic",
        "risk_score": risk_score,
        "risk_level": risk_level,
        "insights": agent_output["insights"],
        "recommendations": agent_output["recommendations"]
    })


if __name__ == "__main__":
    app.run(debug=True)
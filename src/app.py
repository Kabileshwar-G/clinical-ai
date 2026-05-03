from flask import Flask, request, jsonify
import pandas as pd
import pickle

from llm import generate_explanation

app = Flask(__name__)

# Load model
model = pickle.load(open("src/model.pkl", "rb"))

@app.route("/")
def home():
    return "Clinical AI API is running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    df = pd.DataFrame([data])

    prediction = model.predict(df)[0]

    explanation = generate_explanation(data, prediction)

    return jsonify({
        "prediction": int(prediction),
        "explanation": explanation
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
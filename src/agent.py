import os
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

def run_agent(data, prediction, risk_score, risk_level):

    prompt = f"""
You are a clinical AI assistant.

Patient Data:
{data}

Prediction: {"Diabetic" if prediction == 1 else "Non-diabetic"}
Risk Score: {risk_score}
Risk Level: {risk_level}

Tasks:
1. Explain WHY the prediction happened
2. Identify key risk factors
3. Give 3 actionable medical recommendations

Respond STRICTLY in JSON format:
{{
  "insights": ["..."],
  "recommendations": ["...", "...", "..."]
}}
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content
# AI-Powered Clinical Decision Support System

## Overview

This project presents an AI-based Clinical Decision Support System that predicts diabetes using machine learning and provides human-readable explanations using a hybrid LLM approach.

## Features

* Machine Learning Model (Logistic Regression & Random Forest)
* Model Evaluation (Accuracy, Precision, Recall, F1-score)
* Explainable AI (Feature Importance)
* Hybrid LLM Integration (LLM + fallback)
* REST API using Flask
* Docker-based Deployment

## Architecture

User Input → Flask API → ML Model → Explanation Layer → Response

## API Endpoints

### 1. Home

GET /

* Returns API status

### 2. Prediction

POST /predict

#### Input JSON

{
"Pregnancies": 2,
"Glucose": 150,
"BloodPressure": 80,
"SkinThickness": 25,
"Insulin": 100,
"BMI": 30,
"DiabetesPedigree": 0.5,
"Age": 45
}

#### Output

{
"prediction": 1,
"explanation": "Patient is likely diabetic..."
}

## Docker Usage

### Pull Image

docker pull kabileshwar/clinical-ai

### Run Container

docker run -p 5000:5000 kabileshwar/clinical-ai

## SDG Alignment

This project aligns with **SDG 3: Good Health and Well-being** by assisting early diagnosis of diabetes.

## Author

Kabileshwar

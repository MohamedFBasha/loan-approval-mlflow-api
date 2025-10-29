🚀 Loan Approval MLflow API
📘 Overview

Loan Approval MLflow API is a machine learning project that predicts whether a loan should be approved or not using the Loan Approval Dataset.
It trains three models — XGBoost, Random Forest, and Logistic Regression — then automatically selects the best-performing model through MLflow.
The chosen model is deployed via FastAPI, offering a RESTful API for real-time financial predictions.

🧠 Features

Trains and compares 3 ML models

Tracks experiments and metrics with MLflow

Automatically selects the best model

Deploys the model using FastAPI

Provides easy-to-use REST API for predictions

⚙️ Installation & Setup
1️⃣ Create and activate the environment
conda create -n mlflow_env python=3.10.9
conda activate mlflow_env

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run MLflow UI to track experiments
mlflow ui


This will start MLflow at http://127.0.0.1:5000

🚀 Run the API with FastAPI

Activate your environment and run:

conda activate mlflow_env
pip install uvicorn fastapi
uvicorn main:app --host 127.0.0.1 --port 8000 --reload


The API will be available at:
👉 http://127.0.0.1:8000

🧾 Example Request (Postman)

POST → http://127.0.0.1:8000/predict

JSON Body:
{
    "columns": [
        "no_of_dependents",
        "education",
        "self_employed",
        "income_annum",
        "loan_amount",
        "loan_term",
        "cibil_score",
        "residential_assets_value",
        "commercial_assets_value",
        "luxury_assets_value",
        "bank_asset_value"
    ],
    "data": [
        [-0.2914039977413991, 1.0040695112580063, -1.0040695112580063, 0.37255141623584065, 0.9487081783858019, 0.5456702102315741, -0.5218397554059292, 0.2450565069160716, 1.0302355939926513, -0.30197786093753753, -0.49064925537831916],
        [-1.4721434873239536, -0.995946982542167, 0.995946982542167, 1.1232492205160998, 1.6450682129252223, -0.8455606483516742, -1.129646438050548, -0.5482513082409929, -0.9573447306643038, 0.960558969153455, 0.22141599454712899]
    ]
}

✅ Response Example
{
  "predictions": [0, 0]
}

🧩 Tech Stack

Python 3.10.9

MLflow – for experiment tracking and model selection

FastAPI – for serving predictions

Uvicorn – for running the API server

Scikit-learn, XGBoost – for model building and training

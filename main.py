from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.xgboost
import pandas as pd

mlflow.set_tracking_uri("http://localhost:5000")
model_uri = "runs:/617eeeac97944137af6d39472acd54fe/xgboost_model"
model = mlflow.xgboost.load_model(model_uri)

app = FastAPI(title="Loan Approval Prediction API")

# نموذج الإدخال
class PredictInput(BaseModel):
    columns: list[str]
    data: list[list[float]]

    class Config:
        schema_extra = {
            "example": {
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
                    [
                        -0.2914, 1.0040, -1.0040,
                        0.3725, 0.9487, 0.5456,
                        -0.5218, 0.2450, 1.0302,
                        -0.3019, -0.4906
                    ]
                ]
            }
        }

@app.post("/predict")
def predict(input_data: PredictInput):
    df = pd.DataFrame(input_data.data, columns=input_data.columns)
    preds = model.predict(df)
    return {"predictions": preds.tolist()}

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import mlflow
import mlflow.xgboost
import joblib  

# ----------------------------------------------------
# ----------------------------------------------------
mlflow.set_tracking_uri("http://localhost:5000")
exp = mlflow.set_experiment("loan_approval_final")

print(f"🔍 Tracking URI: {mlflow.get_tracking_uri()}")
print(f"🧪 Active Experiment: {exp.name} (ID={exp.experiment_id})")


train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

train_df.columns = train_df.columns.str.strip()
test_df.columns = test_df.columns.str.strip()

y_train = train_df["loan_status"]
X_train = train_df.drop(columns=["loan_status"])
y_test = test_df["loan_status"]
X_test = test_df.drop(columns=["loan_status"])

# ----------------------------------------------------# 
model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# ----------------------------------------------------
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


joblib.dump(model, "xgboost_model.joblib")
print(" Model saved as 'xgboost_model.joblib'")


with mlflow.start_run(run_name="XGBoost_Model2") as run:
    run_id = run.info.run_id
    print(f"Run ID: {run_id}")

    mlflow.log_params({
        "n_estimators": 200,
        "learning_rate": 0.1,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8
    })
    mlflow.log_metric("accuracy", accuracy)

    mlflow.xgboost.log_model(
        xgb_model=model,
        artifact_path="xgboost_model"
    )

print(f"\n🚀 Model logged successfully under Run ID: {run_id}")
print(f"📁 Path: mlruns/{exp.experiment_id}/{run_id}/artifacts/xgboost_model")


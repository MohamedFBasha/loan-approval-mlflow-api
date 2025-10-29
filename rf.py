from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import mlflow
import mlflow.sklearn

# ----------------------------------------------------
# 📌 1. إعداد MLflow
# ----------------------------------------------------
mlflow.set_tracking_uri("http://localhost:5000")  
exp = mlflow.set_experiment("loan_approval_final")

print(f"🔍 Tracking URI: {mlflow.get_tracking_uri()}")
print(f"🧪 Active Experiment: {exp.name} (ID={exp.experiment_id})")

# ----------------------------------------------------
# 📂 2. تحميل البيانات
# ----------------------------------------------------
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

train_df.columns = train_df.columns.str.strip()
test_df.columns = test_df.columns.str.strip()

y_train = train_df["loan_status"]
X_train = train_df.drop(columns=["loan_status"])
y_test = test_df["loan_status"]
X_test = test_df.drop(columns=["loan_status"])

# ----------------------------------------------------
# ⚙️ 3. تدريب الموديل (Random Forest)
# ----------------------------------------------------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ----------------------------------------------------
# 📊 4. تقييم الأداء
# ----------------------------------------------------
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {accuracy:.4f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ----------------------------------------------------
# 🧾 5. تسجيل التجربة والموديل
# ----------------------------------------------------
with mlflow.start_run(run_name="RandomForest_Model") as run:
    run_id = run.info.run_id
    print(f"🆔 Run ID: {run_id}")

    # تسجيل المعاملات
    mlflow.log_params({
        "n_estimators": 200,
        "max_depth": 8,
        "min_samples_split": 4,
        "min_samples_leaf": 2,
        "random_state": 42
    })

    # تسجيل المقاييس
    mlflow.log_metric("accuracy", accuracy)

    # تسجيل الموديل على MLflow
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="rf_model"
    )

print(f"\n🚀 RandomForest model logged successfully under Run ID: {run_id}")
print(f"📁 Path: mlruns/{exp.experiment_id}/{run_id}/artifacts/rf_model")

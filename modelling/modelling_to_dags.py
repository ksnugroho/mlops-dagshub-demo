import os
import json
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
)

from dagshub import dagshub_logger
from preprocessing.preprocessing_pipeline import preprocessing_pipeline

# Inisialisasi MLflow untuk DagsHub
mlflow.set_tracking_uri("https://dagshub.com/ksnugroho/mlops-dagshub-demo.mlflow/")  # Ganti dengan username & repo kamu
mlflow.set_experiment("credit-risk-model-v1")

# Buat folder model jika belum ada
os.makedirs("model", exist_ok=True)

# Ambil data dari pipeline
train_df, _ = preprocessing_pipeline("data/train_cleaned.csv")
X = train_df.drop("Credit_Score", axis=1)
y = train_df["Credit_Score"]

# Split ulang untuk validasi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
input_example = X_train.iloc[:5]

# Hyperparameter tuning
param_grid = {
    "n_estimators": [100, 300, 505],
    "max_depth": [10, 20, 37]
}
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=3,
    n_jobs=-1,
    scoring='accuracy'
)
grid_search.fit(X_train, y_train)

# Ambil model terbaik
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# Logging ke MLflow
with mlflow.start_run():
    mlflow.log_param("n_estimators", best_params["n_estimators"])
    mlflow.log_param("max_depth", best_params["max_depth"])

    # Fit ulang (opsional)
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", acc)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig("model/training_confusion_matrix.png")
    mlflow.log_artifact("model/training_confusion_matrix.png")

    # Classification Report (HTML)
    html_report = classification_report(y_test, y_pred, output_dict=True)
    html_content = f"""
    <html>
    <head><title>Estimator Report</title></head>
    <body>
        <h2>Classification Report</h2>
        <pre>{json.dumps(html_report, indent=2)}</pre>
    </body>
    </html>
    """
    with open("model/estimator.html", "w") as f:
        f.write(html_content)
    mlflow.log_artifact("model/estimator.html")

    # Simpan metrik ke JSON
    with open("model/metric_info.json", "w") as f:
        json.dump({"accuracy": acc}, f)
    mlflow.log_artifact("model/metric_info.json")

    # Simpan model .pkl
    joblib.dump(best_model, "model/model.pkl")
    mlflow.log_artifact("model/model.pkl")

    # Log model ke MLflow (dengan input_example)
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model",
        input_example=input_example
    )

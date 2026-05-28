import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score
import mlflow
import mlflow.sklearn
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=5)
    args = parser.parse_args()

    try:
        if os.environ.get("SKIP_DAGSHUB") == "1":
            raise RuntimeError("Skipping DagsHub due to environment variable SKIP_DAGSHUB=1")
        import dagshub
        print("Initializing DagsHub Tracking in MLProject run...")
        dagshub.init(repo_owner='akubima', repo_name='Eksperimen_SML_Bima-Mukhlisin-Bil-Sajjad', mlflow=True)
    except Exception as e:
        print(f"Could not connect to DagsHub ({e}). Using local MLflow Tracking.")
        mlflow.set_tracking_uri("http://127.0.0.1:5000")

    # Check if managed by mlflow run
    if "MLFLOW_RUN_ID" not in os.environ:
        mlflow.set_experiment("CI_Retraining")
        run_context = mlflow.start_run(run_name="CI_Run")
    else:
        from contextlib import nullcontext
        run_context = nullcontext()

    # Load data
    data_path = os.path.join("heart_disease_preprocessing", "heart-disease-preprocessed.csv")
    if not os.path.exists(data_path):
        data_path = "heart-disease-preprocessed.csv"
        
    if not os.path.exists(data_path):
        data_path = os.path.join("..", "preprocessing", "heart_disease_preprocessing", "heart-disease-preprocessed.csv")
        
    print(f"Loading preprocessed data from {data_path}...")
    df = pd.read_csv(data_path)
    X = df.drop(columns=['target'])
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    with run_context:
        print(f"Training model with n_estimators={args.n_estimators}, max_depth={args.max_depth}...")
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        
        model = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=42)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        
        mlflow.sklearn.log_model(model, "model")
        run_id = mlflow.active_run().info.run_id
        print(f"CI Run finished. Run ID: {run_id}. Accuracy: {acc:.4f}")

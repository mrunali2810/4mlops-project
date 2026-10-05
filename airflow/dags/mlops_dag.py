"""Airflow DAG: download -> preprocess -> train -> evaluate -> (deploy)."""
import json
import os
import shutil
import sys
from datetime import datetime

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator

# Folder containing src/, data/ (override with env var if needed)
PROJECT_DIR = os.environ.get("MLOPS_PROJECT_DIR",
                             os.path.expanduser("~/4mlops-project"))
sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))

DATA_URL = ("https://raw.githubusercontent.com/mwaskom/seaborn-data/"
            "master/iris.csv")
RAW = os.path.join(PROJECT_DIR, "data", "data.csv")
PROCESSED = os.path.join(PROJECT_DIR, "data", "processed")
MODELS = os.path.join(PROJECT_DIR, "models")
DEPLOYED = os.path.join(PROJECT_DIR, "deployed")
BEST = os.path.join(DEPLOYED, "best_metrics.json")


def download_data():
    import pandas as pd
    try:
        df = pd.read_csv(DATA_URL).rename(columns={"species": "target"})
        df.columns = ["sepal_length", "sepal_width", "petal_length",
                      "petal_width", "target"]
        os.makedirs(os.path.dirname(RAW), exist_ok=True)
        df.to_csv(RAW, index=False)
        print("Downloaded", len(df), "rows")
    except Exception as exc:  # fall back to the local copy
        print("Download failed, using existing data.csv:", exc)


def preprocess_task():
    from preprocess import preprocess
    preprocess(RAW, PROCESSED)


def train_task():
    from train import train
    train(PROCESSED, MODELS)


def evaluate_task():
    """Branch: deploy only if accuracy improved over the deployed model."""
    with open(os.path.join(MODELS, "metrics.json")) as f:
        new_acc = json.load(f)["accuracy"]
    old_acc = -1.0
    if os.path.exists(BEST):
        with open(BEST) as f:
            old_acc = json.load(f)["accuracy"]
    print(f"new={new_acc:.4f} previous={old_acc:.4f}")
    return "deploy_model" if new_acc > old_acc else "skip_deployment"


def deploy_task():
    os.makedirs(DEPLOYED, exist_ok=True)
    shutil.copy(os.path.join(MODELS, "model.joblib"),
                os.path.join(DEPLOYED, "model.joblib"))
    shutil.copy(os.path.join(MODELS, "metrics.json"), BEST)
    print("Model deployed to", DEPLOYED)  # replace with a real deploy call


with DAG(
    dag_id="mlops_training_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["mlops", "assignment4"],
) as dag:
    t1 = PythonOperator(task_id="download_data", python_callable=download_data)
    t2 = PythonOperator(task_id="preprocess", python_callable=preprocess_task)
    t3 = PythonOperator(task_id="train_model", python_callable=train_task)
    t4 = BranchPythonOperator(task_id="evaluate_model",
                              python_callable=evaluate_task)
    deploy = PythonOperator(task_id="deploy_model",
                            python_callable=deploy_task)
    skip = EmptyOperator(task_id="skip_deployment")

    t1 >> t2 >> t3 >> t4 >> [deploy, skip]

"""Train a model and save it with its metrics."""
import json
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def train(processed_dir="data/processed", model_dir="models",
          n_estimators=100, seed=42):
    train_df = pd.read_csv(os.path.join(processed_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(processed_dir, "test.csv"))

    X_train, y_train = train_df.drop(columns=["target"]), train_df["target"]
    X_test, y_test = test_df.drop(columns=["target"]), test_df["target"]

    model = RandomForestClassifier(n_estimators=n_estimators, random_state=seed)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))

    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "model.joblib"))
    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump({"accuracy": acc}, f)
    print(f"Model trained. Accuracy = {acc:.4f}")
    return acc


if __name__ == "__main__":
    train()

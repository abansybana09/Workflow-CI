"""
Model Training untuk MLProject (Kriteria 3 - Workflow CI)
Dataset: Iris

File ini adalah versi MLProject dari modelling.py
dengan argumen CLI untuk parameterisasi.

Author: Mohamad Aban Sybana
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

mlflow.sklearn.autolog()


def load_data():
    """Load dataset iris preprocessing"""
    data_path = 'iris_preprocessing/iris_preprocessing.csv'
    df = pd.read_csv(data_path)
    feature_cols = [c for c in df.columns if c not in ['species', 'split']]
    train_df = df[df['split'] == 'train']
    test_df = df[df['split'] == 'test']
    return (
        train_df[feature_cols], test_df[feature_cols],
        train_df['species'], test_df['species']
    )


def main(n_estimators, max_depth, test_size, random_state):
    # Jika dijalankan via `mlflow run`, MLFLOW_RUN_ID sudah di-set otomatis
    # Gunakan active run jika ada, hindari konflik nested run
    env_run_id = os.environ.get('MLFLOW_RUN_ID')
    if not env_run_id:
        mlflow.set_experiment("Iris_CI_Training")

    X_train, X_test, y_train, y_test = load_data()
    print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    run_ctx = mlflow.start_run(run_id=env_run_id) if env_run_id else mlflow.start_run(run_name="CI_Run")

    with run_ctx:
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth if max_depth > 0 else None,
            random_state=random_state
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred,
              target_names=['setosa', 'versicolor', 'virginica']))

        run_id = mlflow.active_run().info.run_id
        print(f"Run ID: {run_id}")

        # Simpan run ID agar bisa di-build Docker pada step Advanced
        with open("run_id.txt", "w") as f:
            f.write(run_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=5)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--random_state', type=int, default=42)
    args = parser.parse_args()

    main(args.n_estimators, args.max_depth, args.test_size, args.random_state)

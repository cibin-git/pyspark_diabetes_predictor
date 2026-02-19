"""Train a simple sklearn model and log to MLflow.

Saves:
- artifacts/test.pkl  (X_test, y_test)
- models/model.pkl    (joblib dump)
Logs model and metrics to MLflow
"""
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import mlflow
import mlflow.sklearn
import os
import argparse


def main(n_estimators: int = 100, random_state: int = 42):
    data = load_diabetes()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    os.makedirs("models", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

    with mlflow.start_run():
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("random_state", random_state)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)

        # log model to MLflow
        mlflow.sklearn.log_model(model, "model")

        # save local artifacts for quick use
        joblib.dump((X_test, y_test), "artifacts/test.pkl")
        joblib.dump(model, "models/model.pkl")

        print(f"Training complete. MSE={mse:.4f}, R2={r2:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    main(n_estimators=args.n_estimators, random_state=args.random_state)

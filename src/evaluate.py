"""Evaluate saved model on test split saved by `train.py`."""
import joblib
from sklearn.metrics import mean_squared_error, r2_score


def main():
    try:
        X_test, y_test = joblib.load("artifacts/test.pkl")
    except Exception as e:
        raise SystemExit("Could not load artifacts/test.pkl — run train first")

    model = joblib.load("models/model.pkl")
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"Evaluation results — MSE: {mse:.4f}, R2: {r2:.4f}")


if __name__ == "__main__":
    main()

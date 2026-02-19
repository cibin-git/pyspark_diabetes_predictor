#!/usr/bin/env bash
# Simple helper to run train -> evaluate
python -m src.train
python -m src.evaluate
echo "Done. Start MLflow UI with: mlflow ui --port 5000"

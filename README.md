# Local MLOps Pipeline (minimal)

This project provides a minimal local MLOps pipeline using MLflow for tracking, a training script, an evaluation script, and a simple FastAPI serving endpoint.

Quick start

1. Create a virtualenv and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Train a model (this logs to MLflow and saves artifacts):

```bash
python -m src.train
```

3. View MLflow UI (runs locally):

```bash
mlflow ui --port 5000
# open http://127.0.0.1:5000
```

4. Evaluate the saved model:

```bash
python -m src.evaluate
```

5. Serve the model with FastAPI:

```bash
uvicorn src.serve:app --reload --port 8000
# POST JSON to http://127.0.0.1:8000/predict with {"features": [f1, f2, ...]}
```

Files added

- `src/train.py` — training + MLflow logging
- `src/evaluate.py` — evaluation using saved test split
- `src/serve.py` — FastAPI model server
- `requirements.txt` — Python deps

Notes

- MLflow uses local filesystem for tracking and artifacts by default; no extra services required for basic use.
- This is a minimal scaffold to run locally; tell me if you want Docker, MLflow server, or a data/versioning layer (DVC, MinIO).

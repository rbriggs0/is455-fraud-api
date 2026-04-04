# Fraud API

This repository is a small [FastAPI](https://fastapi.tiangolo.com/) service that loads a fraud-detection model and feature list exported from `Deployment.ipynb` (or an equivalent training notebook).

## Repository layout

| Path | Purpose |
|------|---------|
| `api/main.py` | Production entry point: defines the FastAPI app, loads artifacts at startup, and implements `/health` and `/score`. Uvicorn should target `api.main:app`. |
| `api/main.ipynb` | Notebook mirror of the same logic for local exploration in Jupyter. **Deployments still use `main.py`**; the notebook helps when `__file__` is not available and resolves the project root by finding `fraud_model.sav`. |
| `requirements.txt` | Python dependencies for the API: `fastapi`, `uvicorn`, `pandas`, `scikit-learn`, and `joblib` (used to load `.sav` / `.pkl` artifacts). |

## Model and feature artifacts (repo root)

Place these next to `requirements.txt` (the parent of the `api/` package). Paths are resolved from `api/main.py` as `Path(__file__).resolve().parent.parent`.

| File | Role |
|------|------|
| `fraud_model.sav` | **Required at runtime.** Serialized trained model (loaded with `joblib`). The service calls `predict` and `predict_proba` on this object. |
| `fraud_features.pkl` | **Required at runtime.** List (or sequence) of feature column names in scoring order. Request rows are aligned to these columns; any missing columns are filled with `None` before prediction. |
| `fraud_model.pkl` | **Not used by this codebase by default.** If your notebook exports the model as `.pkl` instead of `.sav`, it is the same kind of joblib-serialized object; to use it you would either rename/copy it to `fraud_model.sav` or change `MODEL_PATH` in `api/main.py` to point at `fraud_model.pkl`. |

The `.sav` and `.pkl` extensions are both common for joblib dumps; this project’s `main.py` is wired to `fraud_model.sav` and `fraud_features.pkl` only.

## Run locally

```bash
pip install -r requirements.txt
uvicorn api.main:app --reload
```

## Endpoints

- `GET /health` — Liveness and artifact status (`model_loaded`, `features_loaded`, and the configured filenames).
- `POST /score` — Score a batch of rows; returns sorted results by fraud risk.

Example request body:

```json
{
  "rows": [
    {
      "feature_a": 1,
      "feature_b": "x"
    }
  ],
  "top_n": 50
}
```

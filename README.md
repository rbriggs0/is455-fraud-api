# Fraud API

This folder contains a small FastAPI service that loads the fraud model exported from `Deployment.ipynb`.

## Required files

Place these two files in this folder before running the API:

- `fraud_model.sav`
- `fraud_features.pkl`

## Run locally

```bash
pip install -r requirements.txt
uvicorn api.main:app --reload
```

## Endpoints

- `GET /health`
- `POST /score`

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

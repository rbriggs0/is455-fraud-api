from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "fraud_model.sav"
FEATURES_PATH = BASE_DIR / "fraud_features.pkl"

app = FastAPI(title="Fraud Scoring API")

model = None
feature_columns = None


def load_artifacts() -> None:
    global model, feature_columns
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model file: {MODEL_PATH.name}")
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing feature file: {FEATURES_PATH.name}")

    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURES_PATH)


@app.on_event("startup")
def startup_event() -> None:
    load_artifacts()


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "model_loaded": model is not None,
        "features_loaded": feature_columns is not None,
        "model_path": MODEL_PATH.name,
        "features_path": FEATURES_PATH.name,
    }


@app.post("/score")
def score(payload: dict) -> dict:
    if model is None or feature_columns is None:
        raise HTTPException(status_code=500, detail="Model artifacts are not loaded.")

    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise HTTPException(status_code=400, detail="Payload must include a non-empty 'rows' list.")

    df = pd.DataFrame(rows)

    missing = [col for col in feature_columns if col not in df.columns]
    for col in missing:
        df[col] = None

    df = df[feature_columns]

    if not hasattr(model, "predict"):
        raise HTTPException(status_code=500, detail="Loaded model does not support predict.")
    if not hasattr(model, "predict_proba"):
        raise HTTPException(status_code=500, detail="Loaded model does not support predict_proba.")

    pred_labels = model.predict(df)
    scores = model.predict_proba(df)[:, 1]

    scored = pd.DataFrame(rows).copy()
    scored["fraud_risk"] = scores
    # Binary is_fraud: class 1 = fraud (matches sklearn .predict on trained pipeline)
    scored["predicted_fraud"] = (pred_labels == 1)
    scored = scored.sort_values("fraud_risk", ascending=False)

    top_n = int(payload.get("top_n", 50))
    top_n = max(1, min(top_n, len(scored)))

    # Native bool/float for JSON (numpy/pandas types are not always JSON-serializable)
    results: list[dict[str, Any]] = []
    for rec in scored.head(top_n).to_dict(orient="records"):
        out = dict(rec)
        if "predicted_fraud" in out:
            out["predicted_fraud"] = bool(out["predicted_fraud"])
        if "fraud_risk" in out:
            out["fraud_risk"] = float(out["fraud_risk"])
        results.append(out)

    return {
        "count": len(scored),
        "top_n": top_n,
        "results": results,
    }

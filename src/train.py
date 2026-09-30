"""Retrain the final model from the command line.

Run from the repo root:
    python -m src.train

Uses the hyperparameters the notebook found (reports/best_params.json), fits on the
same 80/20 split (random_state=42, stratified on smoker), prints test metrics and
saves the pipeline to models/model_pipeline.pkl.
"""
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from src.preprocessing import build_pipeline, load_and_clean, split_xy

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "XGBoost": XGBRegressor,
    "Gradient Boosting": GradientBoostingRegressor,
    "Random Forest": RandomForestRegressor,
}


def main():
    cfg = json.loads((ROOT / "reports" / "best_params.json").read_text())
    model = MODELS[cfg["model"]](random_state=42, **cfg["params"])

    df = load_and_clean(ROOT / "data" / "insurance.csv")
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=X["smoker"]
    )

    pipe = build_pipeline(model, scale=False).fit(X_train, y_train)
    pred = pipe.predict(X_test)
    print(f"model: {cfg['model']}")
    print(f"test R2   : {r2_score(y_test, pred):.3f}")
    print(f"test RMSE : {np.sqrt(mean_squared_error(y_test, pred)):,.0f}")
    print(f"test MAE  : {mean_absolute_error(y_test, pred):,.0f}")

    out = ROOT / "models" / "model_pipeline.pkl"
    joblib.dump(pipe, out)
    print("saved ->", out)


if __name__ == "__main__":
    main()

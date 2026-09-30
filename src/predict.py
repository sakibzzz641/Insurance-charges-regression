"""Predict yearly charges for one person with the saved pipeline.

Run from the repo root:
    python -m src.predict --age 35 --sex male --bmi 32 --children 1 --smoker yes --region southeast
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd

import src.preprocessing  # noqa: F401  (needed so the pickled pipeline can be loaded)

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--age", type=int, required=True)
    p.add_argument("--sex", choices=["male", "female"], required=True)
    p.add_argument("--bmi", type=float, required=True)
    p.add_argument("--children", type=int, required=True)
    p.add_argument("--smoker", choices=["yes", "no"], required=True)
    p.add_argument("--region", choices=["northeast", "northwest", "southeast", "southwest"], required=True)
    a = p.parse_args()

    model = joblib.load(ROOT / "models" / "model_pipeline.pkl")
    row = pd.DataFrame([vars(a)])
    print(f"predicted yearly charges: {model.predict(row)[0]:,.0f}")


if __name__ == "__main__":
    main()

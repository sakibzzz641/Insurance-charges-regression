"""Cleaning, feature engineering and the sklearn preprocessor for the insurance data."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "charges"
NUMERIC = ["age", "bmi", "children", "bmi_obese", "smoker_obese"]
CATEGORICAL = ["sex", "smoker", "region"]


def load_and_clean(path="data/insurance.csv"):
    """Read the csv, tidy the text columns and drop exact duplicate rows."""
    df = pd.read_csv(path)
    for col in ["sex", "smoker", "region"]:
        df[col] = df[col].str.lower().str.strip()
    return df.drop_duplicates().reset_index(drop=True)


def add_features(df):
    """Two engineered columns.

    bmi_obese    : 1 if BMI >= 30 (standard obesity cut-off)
    smoker_obese : 1 if the person is a smoker AND obese - this group has by far
                   the highest charges, a plain linear model can't see that on its own.
    Purely row-wise, so it's safe to run before the train/test split.
    """
    out = df.copy()
    out["bmi_obese"] = (out["bmi"] >= 30).astype(int)
    out["smoker_obese"] = ((out["smoker"] == "yes") & (out["bmi"] >= 30)).astype(int)
    return out


def build_preprocessor(scale=True):
    """ColumnTransformer: scale numeric columns, one-hot the categorical ones.

    Fit this on the training data only (put it inside a Pipeline).
    """
    num = StandardScaler() if scale else "passthrough"
    return ColumnTransformer(
        [
            ("num", num, NUMERIC),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL),
        ]
    )


def split_xy(df):
    return df.drop(columns=[TARGET]), df[TARGET]


def build_pipeline(model, scale=True):
    """Full leakage-safe pipeline: feature engineering -> preprocessing -> model.

    Takes the raw dataframe columns (age, sex, bmi, children, smoker, region),
    so the saved pipeline can be used on new rows directly.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer

    return Pipeline(
        [
            ("fe", FunctionTransformer(add_features)),
            ("prep", build_preprocessor(scale)),
            ("model", model),
        ]
    )

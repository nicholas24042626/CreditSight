from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

RATING_GROUP_MAPPING = {
    "Investment-High": ["AAA", "AA", "A"],
    "Investment-Low": ["BBB"],
    "Speculative": ["BB", "B"],
    "Distressed": ["CCC", "CC", "C", "D"],
}

CLASS_ORDER = ["Investment-High", "Investment-Low", "Speculative", "Distressed"]
IDENTIFIER_ALIASES = [
    "name",
    "company name",
    "issuer name",
    "security name",
    "symbol",
    "ticker",
    "date",
    "rating agency name",
]
TARGET_ALIASES = ["rating", "credit rating", "issuer rating", "rating grade"]
MODEL_NAME_MAP = {
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "logistic_regression": "Logistic Regression",
    "xgboost": "XGBoost",
}

MODEL_ALIAS_MAP = {
    "decision tree": "decision_tree",
    "random forest": "random_forest",
    "logistic regression": "logistic_regression",
    "xgboost": "xgboost",
    "decision_tree": "decision_tree",
    "random_forest": "random_forest",
    "logistic_regression": "logistic_regression",
}

# This file keeps the shared data rules in one place for training and prediction.

def normalize_text(value: object) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_column_name(value: object) -> str:
    return normalize_text(value).lower()


def normalize_model_key(value: object) -> str:
    key = normalize_column_name(value)
    return MODEL_ALIAS_MAP.get(key, key.replace(" ", "_"))


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {column: normalize_text(column) for column in df.columns}
    return df.rename(columns=renamed)


def find_column(df: pd.DataFrame, aliases: List[str]) -> str | None:
    lookup = {normalize_column_name(column): column for column in df.columns}
    for alias in aliases:
        column = lookup.get(normalize_column_name(alias))
        if column:
            return column
    return None


def load_tabular_file(file_path: str) -> pd.DataFrame:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    raise ValueError("Unsupported file type. Please upload a CSV or Excel file.")


def create_rating_group(rating_value: object) -> str | None:
    rating = normalize_text(rating_value).upper()
    if not rating:
        return None

    for group_name, ratings in RATING_GROUP_MAPPING.items():
        if rating in ratings:
            return group_name
    return None


def clean_rating_label(value: object) -> str:
    return normalize_text(value)


def coerce_feature_types(
    frame: pd.DataFrame,
    numeric_columns: List[str],
    categorical_columns: List[str],
) -> pd.DataFrame:
    cleaned = frame.copy()

    for column in numeric_columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    for column in categorical_columns:
        cleaned[column] = cleaned[column].astype("string").fillna("Missing").astype(str)

    return cleaned


def infer_numeric_like_columns(frame: pd.DataFrame, exclude_columns: List[str] | None = None) -> pd.DataFrame:
    cleaned = frame.copy()
    exclude = set(exclude_columns or [])

    for column in cleaned.columns:
        if column in exclude:
            continue
        if pd.api.types.is_numeric_dtype(cleaned[column]):
            continue

        converted = pd.to_numeric(cleaned[column], errors="coerce")
        non_missing = cleaned[column].notna().sum()
        if non_missing == 0:
            continue

        convertible = converted.notna().sum()
        if convertible / non_missing >= 0.8:
            cleaned[column] = converted

    return cleaned


def build_preprocessor(numeric_columns: List[str], categorical_columns: List[str]) -> ColumnTransformer:
    try:
        one_hot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        one_hot = OneHotEncoder(handle_unknown="ignore", sparse=False)

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", one_hot),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, numeric_columns),
            ("categorical", categorical_pipe, categorical_columns),
        ],
        remainder="drop",
    )


def split_feature_columns(
    df: pd.DataFrame,
    target_column: str,
    drop_columns: List[str],
) -> Tuple[List[str], List[str]]:
    usable = df.drop(columns=[target_column] + drop_columns, errors="ignore")

    numeric_columns = [
        column
        for column in usable.columns
        if pd.api.types.is_numeric_dtype(usable[column])
    ]
    categorical_columns = [
        column
        for column in usable.columns
        if column not in numeric_columns
    ]

    return numeric_columns, categorical_columns


def build_model_artifact(
    model_name: str,
    pipeline: Pipeline,
    label_encoder: LabelEncoder,
    feature_columns: List[str],
    numeric_columns: List[str],
    categorical_columns: List[str],
    target_column: str,
    dataset_summary: Dict[str, object],
    metrics: Dict[str, object],
):
    return {
        "model_name": model_name,
        "model_display_name": MODEL_NAME_MAP.get(model_name, model_name),
        "pipeline": pipeline,
        "label_encoder": label_encoder,
        "feature_columns": feature_columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "target_column": target_column,
        "class_labels": CLASS_ORDER,
        "encoder_class_labels": list(label_encoder.classes_),
        "dataset_summary": dataset_summary,
        "metrics": metrics,
    }


def save_artifact(artifact: Dict[str, object], file_path: str) -> None:
    joblib.dump(artifact, file_path)


def load_artifact(file_path: str) -> Dict[str, object]:
    return joblib.load(file_path)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]) -> Dict[str, object]:
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
        output_dict=True,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )

    matrix = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    return {
        "accuracy": float(report_dict["accuracy"]),
        "weighted_f1": float(report_dict["weighted avg"]["f1-score"]),
        "macro_f1": float(report_dict["macro avg"]["f1-score"]),
        "classification_report_dict": report_dict,
        "classification_report_text": report_text,
        "confusion_matrix": matrix,
    }

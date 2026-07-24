from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from common import (
    CLASS_ORDER,
    IDENTIFIER_ALIASES,
    MODEL_NAME_MAP,
    TARGET_ALIASES,
    build_model_artifact,
    build_preprocessor,
    clean_rating_label,
    coerce_feature_types,
    create_rating_group,
    evaluate_predictions,
    find_column,
    infer_numeric_like_columns,
    load_tabular_file,
    normalize_text,
    save_artifact,
    sanitize_json_value,
    split_feature_columns,
    standardize_columns,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the CreditSight models.")
    parser.add_argument("--data", required=True, help="Path to the training dataset file.")
    parser.add_argument(
        "--target-column",
        default=None,
        help="Optional name of the rating column to convert into RatingGroup.",
    )
    parser.add_argument("--output-dir", default="models", help="Folder to save trained models.")
    return parser.parse_args()


def prepare_training_frame(raw_df: pd.DataFrame, target_column_override: str | None = None) -> tuple[pd.DataFrame, dict]:
    # This section cleans the raw dataset before any model sees it.
    df = standardize_columns(raw_df)
    summary = {
        "rows_before_cleaning": int(len(df)),
        "missing_values": df.isna().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
    }

    rating_column = None
    if target_column_override:
        rating_column = find_column(df, [target_column_override])
    if not rating_column:
        rating_column = find_column(df, TARGET_ALIASES)
    if not rating_column:
        raise ValueError(
            "The training dataset must include a rating column such as 'Rating', or you must pass --target-column."
        )

    cleaned = df.copy()
    cleaned[rating_column] = cleaned[rating_column].apply(clean_rating_label)
    cleaned["RatingGroup"] = cleaned[rating_column].apply(create_rating_group)
    cleaned = cleaned.dropna(subset=["RatingGroup"]).copy()

    drop_columns = []
    for alias in IDENTIFIER_ALIASES:
        column = find_column(cleaned, [alias])
        if column and column not in drop_columns:
            drop_columns.append(column)

    if rating_column not in drop_columns:
        drop_columns.append(rating_column)

    cleaned = cleaned.drop(columns=drop_columns, errors="ignore")
    summary["rows_after_target_mapping"] = int(len(cleaned))
    summary["identifier_columns_removed"] = drop_columns

    if len(cleaned) == 0:
        raise ValueError("No rows remained after mapping the rating target. Check the rating labels in Dataset A.")

    cleaned = infer_numeric_like_columns(cleaned, exclude_columns=["RatingGroup"])

    feature_columns = [column for column in cleaned.columns if column != "RatingGroup"]
    if not feature_columns:
        raise ValueError("No usable feature columns were found after cleaning Dataset A.")

    numeric_columns, categorical_columns = split_feature_columns(cleaned, "RatingGroup", [])
    summary["numeric_feature_columns"] = numeric_columns
    summary["categorical_feature_columns"] = categorical_columns
    summary["feature_columns"] = feature_columns

    return cleaned, summary


def train_single_model(
    model_name: str,
    estimator,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train_encoded,
    y_test_encoded,
    label_encoder: LabelEncoder,
    feature_columns: list[str],
    numeric_columns: list[str],
    categorical_columns: list[str],
    target_column: str,
    dataset_summary: dict,
    output_dir: Path,
    scale_numeric: bool = False,
):
    # This section builds and trains one model at a time.
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(numeric_columns, categorical_columns, scale_numeric=scale_numeric)),
            ("model", estimator),
        ]
    )

    pipeline.fit(X_train, y_train_encoded)
    predicted_encoded = pipeline.predict(X_test)
    y_test_labels = label_encoder.inverse_transform(y_test_encoded)
    y_pred_labels = label_encoder.inverse_transform(predicted_encoded.astype(int))
    metrics = evaluate_predictions(y_test_labels, y_pred_labels, CLASS_ORDER)

    artifact = build_model_artifact(
        model_name=model_name,
        pipeline=pipeline,
        label_encoder=label_encoder,
        feature_columns=feature_columns,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        target_column=target_column,
        dataset_summary=dataset_summary,
        metrics=metrics,
        model_parameters=sanitize_json_value(estimator.get_params(deep=False)),
    )
    artifact["shap_reference_rows"] = X_train.head(100).to_dict(orient="records")

    model_path = output_dir / f"{model_name}.joblib"
    save_artifact(artifact, str(model_path))

    return {
        "model_name": model_name,
        "model_display_name": MODEL_NAME_MAP.get(model_name, model_name),
        "model_path": str(model_path),
        "model_parameters": estimator.get_params(deep=False),
        "metrics": {
            "accuracy": metrics["accuracy"],
            "weighted_f1": metrics["weighted_f1"],
            "macro_f1": metrics["macro_f1"],
        },
        "confusion_matrix": metrics["confusion_matrix"],
        "classification_report_text": metrics["classification_report_text"],
    }


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = load_tabular_file(str(data_path))
    cleaned, summary = prepare_training_frame(raw_df, args.target_column)

    target_labels = cleaned["RatingGroup"].tolist()
    label_encoder = LabelEncoder()
    label_encoder.fit(CLASS_ORDER)
    y_encoded = label_encoder.transform(target_labels)

    feature_columns = summary["feature_columns"]
    numeric_columns = summary["numeric_feature_columns"]
    categorical_columns = summary["categorical_feature_columns"]

    X = cleaned[feature_columns].copy()
    X = coerce_feature_types(X, numeric_columns, categorical_columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.30,
        random_state=42,
        stratify=y_encoded,
    )

    models = {
        "decision_tree": {
            "estimator": DecisionTreeClassifier(
                random_state=42,
                criterion="gini",
                max_depth=None,
                max_features="log2",
                min_samples_split=2,
                min_samples_leaf=1,
                class_weight="balanced",
                splitter="random",
            ),
            "scale_numeric": False,
        },
        "random_forest": {
            "estimator": RandomForestClassifier(
                n_estimators=100,
                criterion="gini",
                max_depth=20,
                min_samples_leaf=1,
                min_samples_split=5,
                max_features="sqrt",
                bootstrap=False,
                class_weight=None,
                random_state=42,
            ),
            "scale_numeric": False,
        },
        "logistic_regression": {
            "estimator": LogisticRegression(
                random_state=42,
                max_iter=2000,
                C=100.0,
                class_weight=None,
                penalty="l2",
                solver="lbfgs",
            ),
            "scale_numeric": True,
        },
        "xgboost": {
            "estimator": XGBClassifier(
                random_state=42,
                eval_metric="mlogloss",
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=1.0,
            ),
            "scale_numeric": False,
        },
    }

    results = []
    for model_name, model_spec in models.items():
        results.append(
            train_single_model(
                model_name=model_name,
                estimator=model_spec["estimator"],
                X_train=X_train,
                X_test=X_test,
                y_train_encoded=y_train,
                y_test_encoded=y_test,
                label_encoder=label_encoder,
                feature_columns=feature_columns,
                numeric_columns=numeric_columns,
                categorical_columns=categorical_columns,
                target_column="RatingGroup",
                dataset_summary=summary,
                output_dir=output_dir,
                scale_numeric=model_spec["scale_numeric"],
            )
        )

    summary_output = sanitize_json_value({
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "source_data": str(data_path),
        "target_column": args.target_column,
        "dataset_summary": summary,
        "models": results,
    })

    summary_path = output_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary_output, indent=2), encoding="utf-8")

    # This last print gives a simple machine-readable summary after training finishes.
    print(json.dumps(summary_output, indent=2))


if __name__ == "__main__":
    main()

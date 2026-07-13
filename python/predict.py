from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from common import (
    CLASS_ORDER,
    MODEL_NAME_MAP,
    coerce_feature_types,
    load_artifact,
    load_tabular_file,
    standardize_columns,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CreditSight predictions on uploaded data.")
    parser.add_argument("--model", required=True, help="Model key such as decision_tree.")
    parser.add_argument("--input", required=True, help="Uploaded dataset path.")
    parser.add_argument("--output", required=True, help="CSV file to write predictions into.")
    return parser.parse_args()


def error_json(message: str, details: object | None = None) -> None:
    payload = {"error": message}
    if details is not None:
        payload["details"] = details
    print(json.dumps(payload))
    raise SystemExit(1)


def validate_model_key(model_key: str) -> str:
    allowed_models = {"decision_tree", "random_forest", "logistic_regression", "xgboost"}
    normalized = model_key.strip()
    if normalized not in allowed_models:
        error_json(
            "Invalid model name.",
            "Allowed model keys are: decision_tree, random_forest, logistic_regression, xgboost.",
        )
    return normalized


def build_prediction_rows(
    predicted_labels: list[str],
    confidence_scores: list[float | None],
    class_probabilities: list[dict[str, float] | None],
) -> list[dict]:
    # This turns each uploaded row into a small prediction record.
    rows = []
    for index, label in enumerate(predicted_labels, start=1):
        rows.append(
            {
                "row_index": index,
                "predicted_rating_group": label,
                "confidence_score": confidence_scores[index - 1],
                "class_probabilities": class_probabilities[index - 1],
            }
        )
    return rows


def get_predict_proba_output(pipeline, features: pd.DataFrame, class_labels: list[str]) -> tuple[list[float | None], list[dict[str, float] | None]]:
    if not hasattr(pipeline, "predict_proba"):
        count = len(features)
        return [None] * count, [None] * count

    probabilities = pipeline.predict_proba(features)
    confidence_scores: list[float | None] = []
    class_probability_rows: list[dict[str, float] | None] = []

    for row_probabilities in probabilities:
        max_probability = float(row_probabilities.max())
        confidence_scores.append(max_probability)
        class_probability_rows.append(
            {
                class_labels[index]: float(row_probabilities[index])
                for index in range(len(class_labels))
            }
        )

    return confidence_scores, class_probability_rows


def main() -> None:
    args = parse_args()
    model_key = validate_model_key(args.model)
    model_file = Path("models") / f"{model_key}.joblib"
    if not model_file.exists():
        error_json(
            f"Trained model file not found for '{args.model}'.",
            f"Expected file: {model_file}. Run `python train_models.py --data \"path\\to\\set A corporate_rating.csv\"` first.",
        )

    try:
        artifact = load_artifact(str(model_file))
        try:
            input_df = standardize_columns(load_tabular_file(args.input))
        except Exception as exc:
            error_json("The uploaded file is invalid or corrupted.", str(exc))

        if input_df.empty:
            error_json("The uploaded file is empty.", "Please upload a CSV or Excel file with at least one data row.")

        numeric_columns = artifact["numeric_columns"]
        categorical_columns = artifact["categorical_columns"]
        feature_columns = artifact["feature_columns"]
        label_encoder = artifact["label_encoder"]
        pipeline = artifact["pipeline"]
        probability_class_labels = artifact.get("encoder_class_labels") or list(label_encoder.classes_)

        missing_columns = [column for column in feature_columns if column not in input_df.columns]
        if missing_columns:
            error_json(
                "The uploaded dataset is missing required columns.",
                {"missing_columns": missing_columns},
            )

        raw_features = input_df[feature_columns].copy()

        invalid_columns = {}
        for column in numeric_columns:
            raw_series = raw_features[column]
            converted = pd.to_numeric(raw_series, errors="coerce")
            bad_mask = raw_series.notna() & converted.isna()
            if bad_mask.any():
                invalid_columns[column] = raw_series[bad_mask].head(5).tolist()

        if invalid_columns:
            error_json(
                "The uploaded dataset contains invalid numeric values.",
                {"invalid_columns": invalid_columns},
            )

        prepared_features = coerce_feature_types(raw_features, numeric_columns, categorical_columns)
        predicted_encoded = pipeline.predict(prepared_features)
        predicted_labels = label_encoder.inverse_transform(predicted_encoded.astype(int)).tolist()

        confidence_scores, class_probability_rows = get_predict_proba_output(
            pipeline,
            prepared_features,
            probability_class_labels,
        )

        output_df = input_df.copy()
        output_df["PredictedRatingGroup"] = predicted_labels
        output_df["ConfidenceScore"] = confidence_scores
        for class_label in probability_class_labels:
            column_name = f"Probability_{class_label}"
            probabilities_for_class = []
            for row_probability in class_probability_rows:
                probabilities_for_class.append(None if row_probability is None else row_probability.get(class_label))
            output_df[column_name] = probabilities_for_class

        output_df.to_csv(args.output, index=False)

        predictions = build_prediction_rows(predicted_labels, confidence_scores, class_probability_rows)

        payload = {
            "model_name": model_key,
            "model_display_name": artifact.get("model_display_name", MODEL_NAME_MAP.get(model_key, model_key)),
            "metrics": {
                "baseline_test_accuracy": artifact["metrics"]["accuracy"],
                "baseline_test_weighted_f1": artifact["metrics"]["weighted_f1"],
                "baseline_test_macro_f1": artifact["metrics"]["macro_f1"],
            },
            "metrics_note": "These metrics come from Dataset A's held-out test set, not the uploaded company file.",
            "confusion_matrix": artifact["metrics"]["confusion_matrix"],
            "classification_report_text": artifact["metrics"]["classification_report_text"],
            "class_labels": CLASS_ORDER,
            "prediction_count": len(predicted_labels),
            "predictions": predictions,
            "output_csv": str(Path(args.output)),
        }

        print(json.dumps(payload))
    except SystemExit:
        raise
    except Exception as exc:
        error_json("Prediction failed.", str(exc))


if __name__ == "__main__":
    main()

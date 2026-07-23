from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap

from common import (
    CLASS_ORDER,
    IDENTIFIER_ALIASES,
    MANUAL_OPTIONAL_FIELDS,
    MANUAL_REQUIRED_FIELDS,
    MODEL_NAME_MAP,
    TARGET_ALIASES,
    build_compatibility_report,
    build_direct_feature_frame,
    build_raw_feature_frame,
    build_manual_feature_row,
    coerce_feature_types,
    is_missing_value,
    load_artifact,
    load_tabular_file,
    normalize_column_name,
    standardize_columns,
    sanitize_json_value,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CreditSight predictions on uploaded data.")
    parser.add_argument("--model", required=True, help="Model key such as decision_tree.")
    parser.add_argument("--input", required=True, help="Uploaded dataset path.")
    parser.add_argument("--output", required=True, help="CSV file to write predictions into.")
    parser.add_argument("--mapping-file", default=None, help="Optional JSON file with column mappings.")
    parser.add_argument("--compatibility", action="store_true", help="Inspect the uploaded file without predicting.")
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


def parse_json_details(message: str) -> object | None:
    try:
        return json.loads(message)
    except Exception:
        return None


def load_mapping_file(mapping_file: str | None) -> dict[str, object]:
    if not mapping_file:
        return {}

    try:
        with open(mapping_file, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        error_json("The mapping file is invalid.", str(exc))


def extract_company_label(row: dict[str, object], fallback_index: int) -> str:
    for key in ["company_name", "Company Name", "Name", "symbol", "Symbol", "Issuer Name"]:
        value = row.get(key)
        if not is_missing_value(value):
            return str(value)
    return f"Row {fallback_index}"


def prettify_feature_name(feature_name: str) -> str:
    name = feature_name
    if "__" in name:
        name = name.split("__", 1)[1]
    if name.startswith("Sector_"):
        return f"Sector={name.split('Sector_', 1)[1]}"
    return name.replace("_", " ")


def normalize_shap_values(raw_values: object, class_index: int, feature_count: int) -> np.ndarray:
    if isinstance(raw_values, list):
        return np.asarray(raw_values[class_index])

    values = np.asarray(getattr(raw_values, "values", raw_values))
    if values.ndim == 2:
        return values
    if values.ndim == 3:
        if values.shape[1] == feature_count:
            return values[:, :, class_index]
        if values.shape[2] == feature_count:
            return values[class_index, :, :]

    raise ValueError(f"Unexpected SHAP value shape: {values.shape}")


def get_predict_proba_output(
    pipeline,
    features: pd.DataFrame,
    class_labels: list[str],
) -> tuple[list[float | None], list[dict[str, float] | None]]:
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


def classify_input_frame(input_df: pd.DataFrame, feature_columns: list[str]) -> str:
    if all(column in input_df.columns for column in feature_columns):
        return "batch"

    if all(column in input_df.columns for column in MANUAL_REQUIRED_FIELDS):
        return "manual"

    return "unknown"


def prepare_batch_features(
    input_df: pd.DataFrame,
    artifact: dict[str, object],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    feature_columns = artifact["feature_columns"]
    numeric_columns = artifact["numeric_columns"]
    categorical_columns = artifact["categorical_columns"]

    missing_columns = [column for column in feature_columns if column not in input_df.columns]
    if missing_columns:
        error_json(
            "The uploaded dataset is missing required columns.",
            {"missing_columns": missing_columns},
        )

    allowed_extra_columns = {
        normalize_column_name(column)
        for column in feature_columns
        + IDENTIFIER_ALIASES
        + TARGET_ALIASES
        + MANUAL_OPTIONAL_FIELDS
    }

    unsupported_columns = [
        column
        for column in input_df.columns
        if normalize_column_name(column) not in allowed_extra_columns
    ]

    raw_features = input_df[feature_columns].copy()

    invalid_columns: dict[str, list[object]] = {}
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
    return prepared_features, unsupported_columns, []


def prepare_manual_features(
    input_df: pd.DataFrame,
    artifact: dict[str, object],
) -> tuple[pd.DataFrame, dict[str, object], list[str]]:
    if len(input_df) != 1:
        error_json(
            "Manual company assessment expects exactly one company row.",
            {"rows_received": int(len(input_df))},
        )

    try:
        feature_row, metadata = build_manual_feature_row(input_df.iloc[0].to_dict())
    except ValueError as exc:
        details = parse_json_details(str(exc)) or str(exc)
        error_json("The manual company input is incomplete or invalid.", details)

    feature_columns = artifact["feature_columns"]
    numeric_columns = artifact["numeric_columns"]
    categorical_columns = artifact["categorical_columns"]

    prepared_features = pd.DataFrame([feature_row], columns=feature_columns)
    prepared_features = coerce_feature_types(prepared_features, numeric_columns, categorical_columns)
    return prepared_features, metadata, []


def prepare_dataset_features(
    input_df: pd.DataFrame,
    artifact: dict[str, object],
    mapping: dict[str, object] | None = None,
) -> tuple[pd.DataFrame, str, dict[str, object], list[str]]:
    feature_columns = artifact["feature_columns"]
    mapping = mapping or {}
    compatibility = build_compatibility_report(input_df, feature_columns)
    mode = str(mapping.get("mode") or compatibility.get("suggested_mode") or "direct").strip().lower()
    warnings: list[str] = []

    if mode in {"direct", "feature", "features"}:
        feature_mappings = mapping.get("feature_mappings") or compatibility.get("suggested_feature_mappings") or {}
        if not isinstance(feature_mappings, dict):
            error_json("Invalid feature mapping configuration.", "feature_mappings must be an object.")
        prepared_features = build_direct_feature_frame(input_df, feature_columns, feature_mappings)
        return prepared_features, "direct", compatibility, warnings

    if mode in {"raw", "manual", "calculated"}:
        raw_field_mappings = mapping.get("raw_field_mappings") or compatibility.get("suggested_raw_field_mappings") or {}
        if not isinstance(raw_field_mappings, dict):
            error_json("Invalid raw-field mapping configuration.", "raw_field_mappings must be an object.")
        prepared_features = build_raw_feature_frame(input_df, feature_columns, raw_field_mappings)
        return prepared_features, "raw", compatibility, warnings

    error_json(
        "The uploaded data does not match the selected model.",
        {
            "missing_required_features": compatibility.get("missing_required_features"),
            "different_domain": compatibility.get("different_domain"),
        },
    )


def compute_shap_contributions(
    pipeline,
    artifact: dict[str, object],
    prepared_features: pd.DataFrame,
    predicted_labels: list[str],
    top_n: int = 5,
) -> tuple[list[list[dict[str, object]] | None], list[str]]:
    warnings: list[str] = []
    reference_rows = artifact.get("shap_reference_rows") or []
    if not reference_rows:
        warnings.append("SHAP reference rows are unavailable; feature contributions were omitted.")
        return [None] * len(predicted_labels), warnings

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    feature_columns = artifact["feature_columns"]
    numeric_columns = artifact["numeric_columns"]
    categorical_columns = artifact["categorical_columns"]

    reference_df = pd.DataFrame(reference_rows).reindex(columns=feature_columns)
    reference_df = coerce_feature_types(reference_df, numeric_columns, categorical_columns)

    try:
        background_transformed = preprocessor.transform(reference_df)
        transformed_features = preprocessor.transform(prepared_features)
        feature_names = list(preprocessor.get_feature_names_out())

        if hasattr(model, "coef_"):
            explainer = shap.LinearExplainer(model, background_transformed)
            shap_values = explainer.shap_values(transformed_features)
        else:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(transformed_features)

        class_index_lookup = {
            label: index for index, label in enumerate(artifact.get("encoder_class_labels") or CLASS_ORDER)
        }

        contribution_rows: list[list[dict[str, object]] | None] = []
        for row_index, predicted_label in enumerate(predicted_labels):
            class_index = class_index_lookup.get(predicted_label, 0)
            class_values = normalize_shap_values(shap_values, class_index, len(feature_names))
            row_values = np.asarray(class_values[row_index], dtype=float)
            order = np.argsort(np.abs(row_values))[::-1][:top_n]
            contributions = []
            for feature_position in order:
                shap_value = float(row_values[feature_position])
                contributions.append(
                    {
                        "feature": prettify_feature_name(feature_names[feature_position]),
                        "shap_value": shap_value,
                        "direction": "positive" if shap_value >= 0 else "negative",
                    }
                )
            contribution_rows.append(contributions)

        return contribution_rows, warnings
    except Exception as exc:
        warnings.append(f"SHAP explanations could not be computed: {exc}")
        return [None] * len(predicted_labels), warnings


def format_contributions_text(contributions: list[dict[str, object]] | None) -> str:
    if not contributions:
        return "Unavailable"

    parts = []
    for item in contributions:
        shap_value = float(item["shap_value"])
        direction = "pushes toward" if shap_value >= 0 else "pushes away from"
        parts.append(f"{item['feature']} {direction} the prediction ({shap_value:+.4f})")
    return "; ".join(parts)


def summarize_model_parameters(model_key: str, artifact: dict[str, object]) -> dict[str, object]:
    parameters = artifact.get("model_parameters")
    if not isinstance(parameters, dict):
        parameters = {}

    if model_key == "xgboost":
        preferred_keys = [
            "n_estimators",
            "max_depth",
            "learning_rate",
            "subsample",
            "colsample_bytree",
            "random_state",
            "eval_metric",
        ]
        filtered = {key: parameters[key] for key in preferred_keys if key in parameters}
        if filtered:
            return sanitize_json_value(filtered)

        return sanitize_json_value({
            "n_estimators": 200,
            "max_depth": 5,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "eval_metric": "mlogloss",
        })

    return sanitize_json_value(parameters)


def validate_random_forest_pipeline(artifact: dict[str, object]) -> None:
    pipeline = artifact.get("pipeline")
    if not hasattr(pipeline, "named_steps"):
        raise ValueError("The saved Random Forest artifact does not contain a fitted sklearn Pipeline.")

    model = pipeline.named_steps.get("model")
    if model is None or model.__class__.__name__ != "RandomForestClassifier":
        raise ValueError("The saved Random Forest artifact does not contain a RandomForestClassifier model.")

    expected_parameters = {
        "n_estimators": 100,
        "criterion": "gini",
        "max_depth": 20,
        "min_samples_split": 5,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "bootstrap": False,
        "class_weight": None,
        "random_state": 42,
    }
    actual_parameters = model.get_params(deep=False)
    mismatches = {
        key: {"expected": expected, "actual": actual_parameters.get(key)}
        for key, expected in expected_parameters.items()
        if actual_parameters.get(key) != expected
    }
    if mismatches:
        raise ValueError(
            "The saved Random Forest artifact does not match the final tuned notebook parameters: "
            + json.dumps(mismatches)
        )

    scaler_names = {"MinMaxScaler", "StandardScaler", "MaxAbsScaler", "RobustScaler", "Normalizer"}
    for _name, step in pipeline.named_steps.items():
        if step.__class__.__name__ in scaler_names:
            raise ValueError("The saved Random Forest pipeline contains normalization.")
        if hasattr(step, "get_params"):
            for component in step.get_params(deep=True).values():
                if component.__class__.__name__ in scaler_names:
                    raise ValueError("The saved Random Forest pipeline contains normalization.")


def build_prediction_rows(
    input_df: pd.DataFrame,
    predicted_labels: list[str],
    confidence_scores: list[float | None],
    class_probabilities: list[dict[str, float] | None],
    contribution_rows: list[list[dict[str, object]] | None],
) -> list[dict]:
    rows = []
    for index, label in enumerate(predicted_labels, start=1):
        row_dict = input_df.iloc[index - 1].to_dict()
        rows.append(
            {
                "row_index": index,
                "company_name": extract_company_label(row_dict, index),
                "predicted_rating_group": label,
                "confidence_score": confidence_scores[index - 1],
                "class_probabilities": class_probabilities[index - 1],
                "top_contributions": contribution_rows[index - 1],
                "top_contributions_text": format_contributions_text(contribution_rows[index - 1]),
            }
        )
    return rows


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
        mapping = load_mapping_file(args.mapping_file)
        try:
            input_df = standardize_columns(load_tabular_file(args.input))
        except Exception as exc:
            error_json("The uploaded file is invalid or corrupted.", str(exc))

        if input_df.empty:
            error_json("The uploaded file is empty.", "Please upload a CSV or Excel file with at least one data row.")

        compatibility = build_compatibility_report(input_df, artifact["feature_columns"])
        if args.compatibility:
            payload = {
                "model_name": model_key,
                "model_display_name": artifact.get("model_display_name", MODEL_NAME_MAP.get(model_key, model_key)),
                "compatible": compatibility["compatible"],
                "different_domain": compatibility["different_domain"],
                "feature_columns": compatibility["feature_columns"],
                "required_features": compatibility["required_features"],
                "raw_source_fields": compatibility["raw_source_fields"],
                "optional_identifiers": compatibility["optional_identifiers"],
                "unsupported_columns": compatibility["unsupported_columns"],
                "missing_required_features": compatibility["missing_required_features"],
                "matched_columns": compatibility["matched_columns"],
                "ignored_columns": compatibility["ignored_columns"],
                "suggested_mode": compatibility["suggested_mode"],
                "suggested_feature_mappings": compatibility["suggested_feature_mappings"],
                "suggested_raw_field_mappings": compatibility["suggested_raw_field_mappings"],
            }
            print(json.dumps(payload))
            return

        numeric_columns = artifact["numeric_columns"]
        categorical_columns = artifact["categorical_columns"]
        feature_columns = artifact["feature_columns"]
        label_encoder = artifact["label_encoder"]
        pipeline = artifact["pipeline"]
        if model_key == "random_forest":
            validate_random_forest_pipeline(artifact)
        probability_class_labels = artifact.get("encoder_class_labels") or list(label_encoder.classes_)
        prediction_mode = classify_input_frame(input_df, feature_columns)

        unsupported_columns: list[str] = compatibility["unsupported_columns"]
        warnings: list[str] = []
        metadata: dict[str, object] = {}

        if prediction_mode == "manual":
            prepared_features, metadata, _ = prepare_manual_features(input_df, artifact)
            prediction_mode = "manual"
        else:
            prepared_features, dataset_mode, compatibility, compat_warnings = prepare_dataset_features(
                input_df,
                artifact,
                mapping,
            )
            prediction_mode = dataset_mode
            warnings.extend(compat_warnings)
            if unsupported_columns:
                warnings.append(
                    "Ignored unsupported columns: " + ", ".join(unsupported_columns)
                )

        predicted_encoded = pipeline.predict(prepared_features)
        predicted_labels = label_encoder.inverse_transform(predicted_encoded.astype(int)).tolist()

        confidence_scores, class_probability_rows = get_predict_proba_output(
            pipeline,
            prepared_features,
            probability_class_labels,
        )

        contribution_rows, shap_warnings = compute_shap_contributions(
            pipeline,
            artifact,
            prepared_features,
            predicted_labels,
        )
        warnings.extend(shap_warnings)

        output_df = input_df.copy()
        if prediction_mode == "manual":
            for feature_name in feature_columns:
                output_df[feature_name] = prepared_features.iloc[0][feature_name]

        output_df["PredictedRatingGroup"] = predicted_labels
        output_df["ConfidenceScore"] = confidence_scores
        for class_label in probability_class_labels:
            column_name = f"Probability_{class_label}"
            probabilities_for_class = []
            for row_probability in class_probability_rows:
                probabilities_for_class.append(None if row_probability is None else row_probability.get(class_label))
            output_df[column_name] = probabilities_for_class

        output_df["TopFeatureContributions"] = [
            format_contributions_text(row_contributions) for row_contributions in contribution_rows
        ]
        output_df.to_csv(args.output, index=False)

        predictions = build_prediction_rows(
            input_df=input_df,
            predicted_labels=predicted_labels,
            confidence_scores=confidence_scores,
            class_probabilities=class_probability_rows,
            contribution_rows=contribution_rows,
        )

        payload = {
            "model_name": model_key,
            "model_display_name": artifact.get("model_display_name", MODEL_NAME_MAP.get(model_key, model_key)),
            "model_parameters": summarize_model_parameters(model_key, artifact),
            "prediction_mode": prediction_mode,
            "compatibility_report": compatibility,
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
            "total_records": len(predicted_labels),
            "predictions": predictions,
            "warnings": warnings,
            "unsupported_columns": unsupported_columns,
            "manual_metadata": metadata,
            "output_csv": str(Path(args.output)),
        }
        if model_key == "random_forest":
            payload["metrics_note"] = (
                "Random Forest final tuned test accuracy is 71.26% on Dataset A's held-out test set, "
                "not the uploaded company file."
            )

        if prediction_mode == "manual" and predictions:
            payload["predicted_risk_category"] = predictions[0]["predicted_rating_group"]
            payload["confidence_score"] = predictions[0]["confidence_score"]
            payload["top_feature_contributions"] = predictions[0]["top_contributions"]

        print(json.dumps(payload))
    except SystemExit:
        raise
    except Exception as exc:
        error_json("Prediction failed.", str(exc))


if __name__ == "__main__":
    main()

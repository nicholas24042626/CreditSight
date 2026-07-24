from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder

RATING_GROUP_MAPPING = {
    "Investment-High": ["AAA", "AA", "A"],
    "Investment-Low": ["BBB"],
    "Speculative": ["BB", "B"],
    "Distressed": ["CCC", "CC", "C", "D"],
}

CLASS_ORDER = ["Investment-High", "Investment-Low", "Speculative", "Distressed"]
SECTOR_CATEGORIES = [
    "Basic Industries",
    "Capital Goods",
    "Consumer Durables",
    "Consumer Non-Durables",
    "Consumer Services",
    "Energy",
    "Finance",
    "Health Care",
    "Miscellaneous",
    "Public Utilities",
    "Technology",
    "Transportation",
]

MANUAL_REQUIRED_FIELDS = [
    "sector",
    "current_assets",
    "current_liabilities",
    "cash_and_equivalents",
    "inventory",
    "accounts_receivable",
    "revenue",
    "gross_profit",
    "operating_income",
    "ebit",
    "net_income",
    "pretax_income",
    "tax_expense",
    "total_assets",
    "net_fixed_assets",
    "total_debt",
    "shareholders_equity",
    "free_cash_flow",
    "operating_cash_flow",
    "shares_outstanding",
    "enterprise_value",
    "ebitda",
    "accounts_payable",
]

MANUAL_OPTIONAL_FIELDS = ["company_name", "symbol"]
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

MANUAL_RATIO_DEPENDENCIES = {
    "currentRatio": ["current_assets", "current_liabilities"],
    "quickRatio": ["current_assets", "current_liabilities", "inventory"],
    "cashRatio": ["cash_and_equivalents", "current_liabilities"],
    "daysOfSalesOutstanding": ["accounts_receivable", "revenue"],
    "netProfitMargin": ["net_income", "revenue"],
    "pretaxProfitMargin": ["pretax_income", "revenue"],
    "grossProfitMargin": ["gross_profit", "revenue"],
    "operatingProfitMargin": ["operating_income", "revenue"],
    "returnOnAssets": ["net_income", "total_assets"],
    "returnOnCapitalEmployed": ["ebit", "total_assets", "current_liabilities"],
    "returnOnEquity": ["net_income", "shareholders_equity"],
    "assetTurnover": ["revenue", "total_assets"],
    "fixedAssetTurnover": ["revenue", "net_fixed_assets"],
    "debtEquityRatio": ["total_debt", "shareholders_equity"],
    "debtRatio": ["total_debt", "total_assets"],
    "effectiveTaxRate": ["tax_expense", "pretax_income"],
    "freeCashFlowOperatingCashFlowRatio": ["free_cash_flow", "operating_cash_flow"],
    "freeCashFlowPerShare": ["free_cash_flow", "shares_outstanding"],
    "cashPerShare": ["cash_and_equivalents", "shares_outstanding"],
    "companyEquityMultiplier": ["total_assets", "shareholders_equity"],
    "ebitPerRevenue": ["ebit", "revenue"],
    "enterpriseValueMultiple": ["enterprise_value", "ebitda"],
    "operatingCashFlowPerShare": ["operating_cash_flow", "shares_outstanding"],
    "operatingCashFlowSalesRatio": ["operating_cash_flow", "revenue"],
    "payablesTurnover": ["accounts_payable", "revenue", "gross_profit"],
}

MANUAL_RATIO_FORMULAS = {
    "currentRatio": "current_assets / current_liabilities",
    "quickRatio": "(current_assets - inventory) / current_liabilities",
    "cashRatio": "cash_and_equivalents / current_liabilities",
    "daysOfSalesOutstanding": "(accounts_receivable / revenue) * 365",
    "netProfitMargin": "net_income / revenue",
    "pretaxProfitMargin": "pretax_income / revenue",
    "grossProfitMargin": "gross_profit / revenue",
    "operatingProfitMargin": "operating_income / revenue",
    "returnOnAssets": "net_income / total_assets",
    "returnOnCapitalEmployed": "ebit / (total_assets - current_liabilities)",
    "returnOnEquity": "net_income / shareholders_equity",
    "assetTurnover": "revenue / total_assets",
    "fixedAssetTurnover": "revenue / net_fixed_assets",
    "debtEquityRatio": "total_debt / shareholders_equity",
    "debtRatio": "total_debt / total_assets",
    "effectiveTaxRate": "tax_expense / pretax_income",
    "freeCashFlowOperatingCashFlowRatio": "free_cash_flow / operating_cash_flow",
    "freeCashFlowPerShare": "free_cash_flow / shares_outstanding",
    "cashPerShare": "cash_and_equivalents / shares_outstanding",
    "companyEquityMultiplier": "total_assets / shareholders_equity",
    "ebitPerRevenue": "ebit / revenue",
    "enterpriseValueMultiple": "enterprise_value / ebitda",
    "operatingCashFlowPerShare": "operating_cash_flow / shares_outstanding",
    "operatingCashFlowSalesRatio": "operating_cash_flow / revenue",
    "payablesTurnover": "(revenue - gross_profit) / accounts_payable",
}

# This file keeps the shared data rules in one place for training and prediction.

def normalize_text(value: object) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_column_name(value: object) -> str:
    return normalize_text(value).lower()


def compact_column_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", normalize_column_name(value))


def build_controlled_aliases(name: str) -> List[str]:
    base = normalize_text(name)
    if not base:
        return []

    snake = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", base).replace("-", "_")
    snake = re.sub(r"__+", "_", snake).strip("_")
    spaced = snake.replace("_", " ")
    compact = compact_column_name(base)
    title = spaced.title()

    aliases = {
        base,
        base.lower(),
        snake,
        snake.lower(),
        spaced,
        spaced.lower(),
        compact,
        title,
        title.lower(),
        snake.replace("_", ""),
        spaced.replace(" ", ""),
    }

    return [alias for alias in aliases if alias]


def build_alias_lookup(columns: List[str]) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for column in columns:
        for alias in build_controlled_aliases(column):
            lookup.setdefault(compact_column_name(alias), column)
    return lookup


def find_matching_column(columns: List[str], canonical_name: str) -> str | None:
    lookup = build_alias_lookup(columns)
    for alias in build_controlled_aliases(canonical_name):
        matched = lookup.get(compact_column_name(alias))
        if matched:
            return matched
    return None


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
    if suffix == ".json":
        return pd.read_json(path, orient="records")

    raise ValueError("Unsupported file type. Please upload a CSV or Excel file.")


def is_missing_value(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def coerce_float(value: object) -> float:
    if is_missing_value(value):
        raise ValueError("missing value")

    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number) or not np.isfinite(number):
        raise ValueError("invalid numeric value")
    return float(number)


def safe_divide(numerator: float, denominator: float, ratio_name: str, issues: List[str]) -> float:
    if denominator == 0:
        issues.append(f"{ratio_name} cannot be calculated because its denominator is 0.")
        return float("nan")
    return numerator / denominator


def build_manual_feature_row(record: Dict[str, object]) -> Tuple[Dict[str, object], Dict[str, object]]:
    cleaned: Dict[str, object] = {}
    errors: Dict[str, List[str]] = {"missing_fields": [], "invalid_fields": [], "division_by_zero": []}

    for field in MANUAL_REQUIRED_FIELDS + MANUAL_OPTIONAL_FIELDS:
        if field not in record or is_missing_value(record.get(field)):
            if field in MANUAL_REQUIRED_FIELDS:
                errors["missing_fields"].append(field)
            continue
        cleaned[field] = record[field]

    if errors["missing_fields"]:
        raise ValueError(json.dumps(errors))

    sector = normalize_text(cleaned["sector"])
    if sector not in SECTOR_CATEGORIES:
        raise ValueError(
            json.dumps(
                {
                    "invalid_fields": {
                        "sector": f"Unsupported sector '{sector}'. Choose one of the Dataset A sector labels."
                    }
                }
            )
        )

    numeric_values: Dict[str, float] = {}
    for field in MANUAL_REQUIRED_FIELDS:
        if field == "sector":
            continue
        try:
            numeric_values[field] = coerce_float(cleaned[field])
        except ValueError:
            errors["invalid_fields"].append(field)

    if errors["invalid_fields"]:
        raise ValueError(json.dumps(errors))

    current_assets = numeric_values["current_assets"]
    current_liabilities = numeric_values["current_liabilities"]
    cash_and_equivalents = numeric_values["cash_and_equivalents"]
    inventory = numeric_values["inventory"]
    accounts_receivable = numeric_values["accounts_receivable"]
    revenue = numeric_values["revenue"]
    gross_profit = numeric_values["gross_profit"]
    operating_income = numeric_values["operating_income"]
    ebit = numeric_values["ebit"]
    net_income = numeric_values["net_income"]
    pretax_income = numeric_values["pretax_income"]
    tax_expense = numeric_values["tax_expense"]
    total_assets = numeric_values["total_assets"]
    net_fixed_assets = numeric_values["net_fixed_assets"]
    total_debt = numeric_values["total_debt"]
    shareholders_equity = numeric_values["shareholders_equity"]
    free_cash_flow = numeric_values["free_cash_flow"]
    operating_cash_flow = numeric_values["operating_cash_flow"]
    shares_outstanding = numeric_values["shares_outstanding"]
    enterprise_value = numeric_values["enterprise_value"]
    ebitda = numeric_values["ebitda"]
    accounts_payable = numeric_values["accounts_payable"]

    cogs = revenue - gross_profit

    feature_row = {
        "Sector": sector,
        "currentRatio": safe_divide(current_assets, current_liabilities, "currentRatio", errors["division_by_zero"]),
        "quickRatio": safe_divide(current_assets - inventory, current_liabilities, "quickRatio", errors["division_by_zero"]),
        "cashRatio": safe_divide(cash_and_equivalents, current_liabilities, "cashRatio", errors["division_by_zero"]),
        "daysOfSalesOutstanding": safe_divide(accounts_receivable, revenue, "daysOfSalesOutstanding", errors["division_by_zero"]) * 365,
        "netProfitMargin": safe_divide(net_income, revenue, "netProfitMargin", errors["division_by_zero"]),
        "pretaxProfitMargin": safe_divide(pretax_income, revenue, "pretaxProfitMargin", errors["division_by_zero"]),
        "grossProfitMargin": safe_divide(gross_profit, revenue, "grossProfitMargin", errors["division_by_zero"]),
        "operatingProfitMargin": safe_divide(operating_income, revenue, "operatingProfitMargin", errors["division_by_zero"]),
        "returnOnAssets": safe_divide(net_income, total_assets, "returnOnAssets", errors["division_by_zero"]),
        "returnOnCapitalEmployed": safe_divide(ebit, total_assets - current_liabilities, "returnOnCapitalEmployed", errors["division_by_zero"]),
        "returnOnEquity": safe_divide(net_income, shareholders_equity, "returnOnEquity", errors["division_by_zero"]),
        "assetTurnover": safe_divide(revenue, total_assets, "assetTurnover", errors["division_by_zero"]),
        "fixedAssetTurnover": safe_divide(revenue, net_fixed_assets, "fixedAssetTurnover", errors["division_by_zero"]),
        "debtEquityRatio": safe_divide(total_debt, shareholders_equity, "debtEquityRatio", errors["division_by_zero"]),
        "debtRatio": safe_divide(total_debt, total_assets, "debtRatio", errors["division_by_zero"]),
        "effectiveTaxRate": safe_divide(tax_expense, pretax_income, "effectiveTaxRate", errors["division_by_zero"]),
        "freeCashFlowOperatingCashFlowRatio": safe_divide(free_cash_flow, operating_cash_flow, "freeCashFlowOperatingCashFlowRatio", errors["division_by_zero"]),
        "freeCashFlowPerShare": safe_divide(free_cash_flow, shares_outstanding, "freeCashFlowPerShare", errors["division_by_zero"]),
        "cashPerShare": safe_divide(cash_and_equivalents, shares_outstanding, "cashPerShare", errors["division_by_zero"]),
        "companyEquityMultiplier": safe_divide(total_assets, shareholders_equity, "companyEquityMultiplier", errors["division_by_zero"]),
        "ebitPerRevenue": safe_divide(ebit, revenue, "ebitPerRevenue", errors["division_by_zero"]),
        "enterpriseValueMultiple": safe_divide(enterprise_value, ebitda, "enterpriseValueMultiple", errors["division_by_zero"]),
        "operatingCashFlowPerShare": safe_divide(operating_cash_flow, shares_outstanding, "operatingCashFlowPerShare", errors["division_by_zero"]),
        "operatingCashFlowSalesRatio": safe_divide(operating_cash_flow, revenue, "operatingCashFlowSalesRatio", errors["division_by_zero"]),
        "payablesTurnover": safe_divide(cogs, accounts_payable, "payablesTurnover", errors["division_by_zero"]),
    }

    if errors["division_by_zero"]:
        raise ValueError(json.dumps(errors))

    metadata = {field: cleaned[field] for field in MANUAL_OPTIONAL_FIELDS if field in cleaned}
    return feature_row, metadata


def build_compatibility_report(
    input_df: pd.DataFrame,
    feature_columns: List[str],
) -> Dict[str, object]:
    columns = list(input_df.columns)
    lower_columns = [normalize_text(column) for column in columns]

    required_feature_rows: List[Dict[str, object]] = []
    raw_field_rows: List[Dict[str, object]] = []
    matched_columns: set[str] = set()
    supported_columns: set[str] = set()
    raw_domain_hits = 0
    feature_domain_hits = 0
    calculated_feature_count = 0

    identifier_rows = []
    for alias in IDENTIFIER_ALIASES + TARGET_ALIASES:
        matched_column = find_matching_column(columns, alias)
        if matched_column and matched_column not in matched_columns:
            matched_columns.add(matched_column)
            supported_columns.add(matched_column)
            identifier_rows.append(
                {
                    "input_column": matched_column,
                    "matched_as": alias,
                    "column_type": "identifier" if alias in IDENTIFIER_ALIASES else "target",
                }
            )

    for feature in feature_columns:
        direct_column = find_matching_column(columns, feature)
        if direct_column:
            matched_columns.add(direct_column)
            supported_columns.add(direct_column)
            feature_domain_hits += 1
            required_feature_rows.append(
                {
                    "feature": feature,
                    "status": "matched",
                    "mapping_type": "direct",
                    "source_column": direct_column,
                    "source_columns": [direct_column],
                    "formula": None,
                    "required": True,
                }
            )
            continue

        dependencies = MANUAL_RATIO_DEPENDENCIES.get(feature)
        source_columns: List[str] = []
        missing_dependencies: List[str] = []
        if dependencies:
            for dependency in dependencies:
                matched_dependency = find_matching_column(columns, dependency)
                if matched_dependency:
                    source_columns.append(matched_dependency)
                    matched_columns.add(matched_dependency)
                    supported_columns.add(matched_dependency)
                    raw_domain_hits += 1
                else:
                    missing_dependencies.append(dependency)

        if dependencies and not missing_dependencies:
            calculated_feature_count += 1
            required_feature_rows.append(
                {
                    "feature": feature,
                    "status": "calculated",
                    "mapping_type": "calculated",
                    "source_column": None,
                    "source_columns": source_columns,
                    "formula": MANUAL_RATIO_FORMULAS.get(feature),
                    "required": True,
                }
            )
            continue

        required_feature_rows.append(
            {
                "feature": feature,
                "status": "missing",
                "mapping_type": "missing",
                "source_column": None,
                "source_columns": source_columns,
                "formula": MANUAL_RATIO_FORMULAS.get(feature),
                "required": True,
                "missing_dependencies": missing_dependencies,
            }
        )

    for raw_field in MANUAL_REQUIRED_FIELDS:
        matched_column = find_matching_column(columns, raw_field)
        if matched_column:
            matched_columns.add(matched_column)
            supported_columns.add(matched_column)
            raw_domain_hits += 1
            raw_field_rows.append(
                {
                    "raw_field": raw_field,
                    "input_column": matched_column,
                    "status": "matched",
                    "required": raw_field != "sector",
                }
            )
        else:
            raw_field_rows.append(
                {
                    "raw_field": raw_field,
                    "input_column": None,
                    "status": "missing",
                    "required": raw_field != "sector",
                }
            )

    optional_rows = []
    for field in MANUAL_OPTIONAL_FIELDS:
        matched_column = find_matching_column(columns, field)
        if matched_column:
            matched_columns.add(matched_column)
            supported_columns.add(matched_column)
            optional_rows.append(
                {
                    "field": field,
                    "input_column": matched_column,
                    "status": "matched",
                }
            )
        else:
            optional_rows.append(
                {
                    "field": field,
                    "input_column": None,
                    "status": "missing",
                }
            )

    unsupported_columns = [
        column
        for column in columns
        if column not in matched_columns and compact_column_name(column) not in {compact_column_name(item) for item in IDENTIFIER_ALIASES + TARGET_ALIASES}
    ]

    suggested_feature_mappings = {
        row["feature"]: row["source_column"]
        for row in required_feature_rows
        if row["status"] == "matched" and row["source_column"]
    }
    suggested_raw_field_mappings = {
        row["raw_field"]: row["input_column"]
        for row in raw_field_rows
        if row["status"] == "matched" and row["input_column"]
    }

    missing_required_features = [row["feature"] for row in required_feature_rows if row["status"] == "missing"]
    compatible = len(missing_required_features) == 0
    domain_signal = feature_domain_hits + raw_domain_hits
    different_domain = domain_signal == 0

    if different_domain:
        compatible = False

    return {
        "compatible": compatible,
        "different_domain": different_domain,
        "feature_columns": feature_columns,
        "required_features": required_feature_rows,
        "raw_source_fields": raw_field_rows,
        "optional_identifiers": identifier_rows,
        "unsupported_columns": unsupported_columns,
        "missing_required_features": missing_required_features,
        "matched_columns": sorted(matched_columns),
        "recognized_columns": sorted(matched_columns),
        "ignored_columns": sorted(set(columns) - matched_columns - set(unsupported_columns)),
        "suggested_mode": "direct" if calculated_feature_count == 0 and feature_domain_hits > 0 else "raw",
        "suggested_feature_mappings": suggested_feature_mappings,
        "suggested_raw_field_mappings": suggested_raw_field_mappings,
    }


def build_direct_feature_frame(
    input_df: pd.DataFrame,
    feature_columns: List[str],
    mapping: Dict[str, str] | None = None,
) -> pd.DataFrame:
    direct_mapping = mapping or {}
    output = pd.DataFrame(index=input_df.index)
    for feature in feature_columns:
        source_column = direct_mapping.get(feature) or find_matching_column(list(input_df.columns), feature)
        if not source_column:
            raise ValueError(f"Missing required model feature: {feature}")
        output[feature] = input_df[source_column]
    return output[feature_columns]


def build_raw_feature_frame(
    input_df: pd.DataFrame,
    feature_columns: List[str],
    mapping: Dict[str, str] | None = None,
) -> pd.DataFrame:
    raw_mapping = mapping or {}
    records: List[Dict[str, object]] = []
    columns = list(input_df.columns)

    for index in range(len(input_df)):
        record: Dict[str, object] = {}
        for field in MANUAL_REQUIRED_FIELDS + MANUAL_OPTIONAL_FIELDS:
            source_column = raw_mapping.get(field) or find_matching_column(columns, field)
            if source_column and source_column in input_df.columns:
                record[field] = input_df.iloc[index][source_column]
        feature_row, _metadata = build_manual_feature_row(record)
        records.append(feature_row)

    return pd.DataFrame(records, columns=feature_columns)


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


def build_preprocessor(
    numeric_columns: List[str],
    categorical_columns: List[str],
    scale_numeric: bool = False,
) -> ColumnTransformer:
    try:
        one_hot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        one_hot = OneHotEncoder(handle_unknown="ignore", sparse=False)

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("minmax", MinMaxScaler()))

    numeric_pipe = Pipeline(steps=numeric_steps)

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


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clips each numeric column to percentile bounds learned on the fit data only.

    Ported from notebook/xgboost/xgboost.ipynb's "Improvement 3: Winsorization" cell - the
    single largest individual accuracy gain found across that notebook's XGBoost ablation work.
    """

    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lower_bounds_ = np.nanquantile(X, self.lower_quantile, axis=0)
        self.upper_bounds_ = np.nanquantile(X, self.upper_quantile, axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.lower_bounds_, self.upper_bounds_)

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        return np.asarray([f"x{i}" for i in range(self.lower_bounds_.shape[0])], dtype=object)


def build_winsorized_preprocessor(
    numeric_columns: List[str],
    categorical_columns: List[str],
) -> ColumnTransformer:
    """Same as build_preprocessor, but winsorizes numeric columns (1st/99th percentile clip,
    learned on the fit data only) instead of passing them through untouched. Matches
    notebook/xgboost/xgboost.ipynb's final selected XGBoost pipeline's preprocessing step."""
    try:
        one_hot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        one_hot = OneHotEncoder(handle_unknown="ignore", sparse=False)

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("winsorizer", Winsorizer(lower_quantile=0.01, upper_quantile=0.99)),
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
    model_parameters: Dict[str, object] | None = None,
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
        "model_parameters": model_parameters or {},
    }


def sanitize_json_value(value):
    if isinstance(value, dict):
        return {str(key): sanitize_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return sanitize_json_value(value.item())
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return value


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

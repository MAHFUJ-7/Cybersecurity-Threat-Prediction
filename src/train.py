from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a ransomware detection classifier from behavior telemetry."
    )
    parser.add_argument(
        "--input",
        default="data/ransomware_behavior.csv",
        help="Path to labeled CSV dataset.",
    )
    parser.add_argument(
        "--target-column",
        default="is_ransomware",
        help="Target column name (binary).",
    )
    parser.add_argument(
        "--drop-columns",
        nargs="*",
        default=["sample_id"],
        help="Columns to exclude from model training.",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Test split fraction (0-1)."
    )
    parser.add_argument(
        "--random-state", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--model-out",
        default="models/ransomware_model.joblib",
        help="Path to save model artifact.",
    )
    parser.add_argument(
        "--metrics-out",
        default="models/metrics.json",
        help="Path to save evaluation metrics JSON.",
    )
    parser.add_argument(
        "--positive-label",
        default=None,
        help=(
            "Optional explicit class label to treat as ransomware/positive class "
            "(for example: 0, 1, true, false, malicious)."
        ),
    )
    return parser.parse_args()


def infer_feature_groups(
    frame: pd.DataFrame, feature_columns: Sequence[str]
) -> tuple[list[str], list[str]]:
    numeric_features = [
        column
        for column in feature_columns
        if pd.api.types.is_numeric_dtype(frame[column])
    ]
    categorical_features = [
        column for column in feature_columns if column not in numeric_features
    ]
    return numeric_features, categorical_features


def build_pipeline(
    numeric_features: Sequence[str], categorical_features: Sequence[str], random_state: int
) -> Pipeline:
    transformers: list[tuple[str, Pipeline, Sequence[str]]] = []

    if numeric_features:
        numeric_pipeline = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median"))]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_features))

    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_features))

    if not transformers:
        raise ValueError("No features available after preprocessing configuration.")

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    classifier = RandomForestClassifier(
        n_estimators=350,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def _find_label_index(class_labels: Sequence[str], token: str) -> int | None:
    normalized_token = token.strip().lower()
    normalized_labels = [label.strip().lower() for label in class_labels]

    if normalized_token in normalized_labels:
        return normalized_labels.index(normalized_token)

    return None


def _choose_by_target_name(class_labels: Sequence[str], target_column: str) -> int | None:
    target_name = target_column.strip().lower()
    positive_indicators = ["malware", "ransomware", "attack", "threat", "infected"]
    negative_indicators = ["legitimate", "benign", "safe", "clean", "normal"]

    if any(token in target_name for token in positive_indicators):
        for token in ["1", "true", "yes", "malicious", "ransomware", "attack", "positive"]:
            index = _find_label_index(class_labels, token)
            if index is not None:
                return index

    if any(token in target_name for token in negative_indicators):
        for token in ["0", "false", "no"]:
            index = _find_label_index(class_labels, token)
            if index is not None:
                return index

    return None


def choose_positive_class_index(
    class_labels: Sequence[str], target_column: str, positive_label: str | None
) -> int:
    if positive_label is not None:
        explicit_index = _find_label_index(class_labels, positive_label)
        if explicit_index is None:
            raise ValueError(
                "positive-label was provided but does not match available classes. "
                f"Provided: {positive_label}, Available: {list(class_labels)}"
            )
        return explicit_index

    preferred_tokens = ["malicious", "ransomware", "attack", "positive", "1", "true", "yes"]
    for token in preferred_tokens:
        preferred_index = _find_label_index(class_labels, token)
        if preferred_index is not None:
            return preferred_index

    target_name_index = _choose_by_target_name(class_labels, target_column)
    if target_name_index is not None:
        return target_name_index

    if len(class_labels) == 2:
        return 1

    return 0


def main() -> None:
    args = parse_args()

    if not (0.05 <= args.test_size <= 0.5):
        raise ValueError("test-size must be between 0.05 and 0.5.")

    dataset = pd.read_csv(args.input)
    if args.target_column not in dataset.columns:
        raise ValueError(f"Target column '{args.target_column}' not found in dataset.")

    drop_columns = [column for column in args.drop_columns if column]
    unknown_drops = [column for column in drop_columns if column not in dataset.columns]
    if unknown_drops:
        raise ValueError(f"Drop columns not found in dataset: {unknown_drops}")

    excluded_columns = set(drop_columns + [args.target_column])
    feature_columns = [column for column in dataset.columns if column not in excluded_columns]
    if not feature_columns:
        raise ValueError("No usable feature columns were found.")

    target_series = dataset[args.target_column].astype(str).str.strip()
    if target_series.nunique() != 2:
        raise ValueError(
            f"Target column '{args.target_column}' must have exactly 2 classes. "
            f"Found: {sorted(target_series.unique().tolist())}"
        )

    label_encoder = LabelEncoder()
    encoded_target = label_encoder.fit_transform(target_series)
    class_labels = [str(label) for label in label_encoder.classes_]
    positive_class_index = choose_positive_class_index(
        class_labels=class_labels,
        target_column=args.target_column,
        positive_label=args.positive_label,
    )

    features = dataset[feature_columns].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        encoded_target,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=encoded_target,
    )

    numeric_features, categorical_features = infer_feature_groups(X_train, feature_columns)
    model_pipeline = build_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        random_state=args.random_state,
    )

    model_pipeline.fit(X_train, y_train)

    predictions = model_pipeline.predict(X_test)
    positive_mask = (y_test == positive_class_index).astype(int)

    probabilities = None
    if hasattr(model_pipeline, "predict_proba"):
        probabilities = model_pipeline.predict_proba(X_test)[:, positive_class_index]

    metrics: dict[str, object] = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(
                y_test,
                predictions,
                pos_label=positive_class_index,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_test,
                predictions,
                pos_label=positive_class_index,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_test,
                predictions,
                pos_label=positive_class_index,
                zero_division=0,
            )
        ),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "class_labels": class_labels,
        "positive_class_label": class_labels[positive_class_index],
        "negative_class_label": class_labels[1 - positive_class_index] if len(class_labels) == 2 else None,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "feature_count": int(len(feature_columns)),
        "classification_report": classification_report(
            y_test,
            predictions,
            target_names=class_labels,
            output_dict=True,
            zero_division=0,
        ),
    }

    if probabilities is not None:
        metrics["roc_auc"] = float(roc_auc_score(positive_mask, probabilities))

    class_distribution = {
        class_labels[index]: int((encoded_target == index).sum())
        for index in range(len(class_labels))
    }
    metrics["class_distribution"] = class_distribution

    model_output_path = Path(args.model_out)
    model_output_path.parent.mkdir(parents=True, exist_ok=True)

    metrics_output_path = Path(args.metrics_out)
    metrics_output_path.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "pipeline": model_pipeline,
        "feature_columns": feature_columns,
        "target_column": args.target_column,
        "drop_columns": drop_columns,
        "class_labels": class_labels,
        "positive_class_index": int(positive_class_index),
        "positive_class_label": class_labels[positive_class_index],
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    joblib.dump(artifact, model_output_path)

    with metrics_output_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)

    print(f"Model saved to: {model_output_path}")
    print(f"Metrics saved to: {metrics_output_path}")
    print(
        "Scores -> "
        f"Accuracy: {metrics['accuracy']:.4f}, "
        f"Precision: {metrics['precision']:.4f}, "
        f"Recall: {metrics['recall']:.4f}, "
        f"F1: {metrics['f1']:.4f}"
    )
    if "roc_auc" in metrics:
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")


if __name__ == "__main__":
    main()

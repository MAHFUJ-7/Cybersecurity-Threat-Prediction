from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ransomware predictions on telemetry records."
    )
    parser.add_argument(
        "--model",
        default="models/ransomware_model.joblib",
        help="Path to trained model artifact.",
    )
    parser.add_argument(
        "--input",
        default="data/ransomware_behavior.csv",
        help="Path to input CSV.",
    )
    parser.add_argument(
        "--output",
        default="models/predictions.csv",
        help="Path to save predictions CSV.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for positive ransomware class.",
    )
    parser.add_argument(
        "--positive-label",
        default=None,
        help=(
            "Optional class label override to use as ransomware/positive class. "
            "Useful when using older model artifacts without positive-class metadata."
        ),
    )
    return parser.parse_args()


def resolve_positive_class_index(
    class_labels: list[str],
    saved_index: int | None,
    positive_label_override: str | None,
) -> int:
    normalized_labels = [label.strip().lower() for label in class_labels]

    if positive_label_override is not None:
        token = positive_label_override.strip().lower()
        if token not in normalized_labels:
            raise ValueError(
                "positive-label override does not match model classes. "
                f"Provided: {positive_label_override}, Available: {class_labels}"
            )
        return normalized_labels.index(token)

    if saved_index is not None and 0 <= int(saved_index) < len(class_labels):
        return int(saved_index)

    for token in ["malicious", "ransomware", "attack", "positive", "1", "true", "yes"]:
        if token in normalized_labels:
            return normalized_labels.index(token)

    if len(class_labels) == 2:
        return 1

    return 0


def main() -> None:
    args = parse_args()

    if not (0.0 <= args.threshold <= 1.0):
        raise ValueError("threshold must be between 0 and 1.")

    artifact = joblib.load(args.model)
    if not isinstance(artifact, dict) or "pipeline" not in artifact:
        raise ValueError("Model artifact is invalid. Expected dictionary with 'pipeline'.")

    pipeline = artifact["pipeline"]
    feature_columns = artifact.get("feature_columns")
    class_labels = artifact.get("class_labels", ["0", "1"])
    if not isinstance(class_labels, list):
        class_labels = [str(label) for label in class_labels]
    else:
        class_labels = [str(label) for label in class_labels]

    saved_index = artifact.get("positive_class_index")
    positive_class_index = resolve_positive_class_index(
        class_labels=class_labels,
        saved_index=saved_index,
        positive_label_override=args.positive_label,
    )

    if not feature_columns:
        raise ValueError("Model artifact does not include feature_columns metadata.")
    if len(class_labels) != 2:
        raise ValueError("Prediction script expects a binary classification model.")

    input_frame = pd.read_csv(args.input)
    missing_columns = [column for column in feature_columns if column not in input_frame.columns]
    if missing_columns:
        raise ValueError(
            "Input file is missing required columns: "
            f"{missing_columns}. Expected at least: {feature_columns}"
        )

    features = input_frame[feature_columns].copy()

    if hasattr(pipeline, "predict_proba"):
        class_probabilities = pipeline.predict_proba(features)
        if class_probabilities.shape[1] != len(class_labels):
            raise ValueError("Model output classes do not match saved class labels.")

        ransomware_probability = class_probabilities[:, positive_class_index]
        negative_class_index = 1 - positive_class_index
        prediction_index = np.full(
            shape=ransomware_probability.shape[0],
            fill_value=negative_class_index,
            dtype=int,
        )
        prediction_index[ransomware_probability >= args.threshold] = positive_class_index
    else:
        prediction_index = pipeline.predict(features)
        ransomware_probability = np.where(
            prediction_index == positive_class_index,
            1.0,
            0.0,
        )

    predicted_labels = [class_labels[int(index)] for index in prediction_index]

    output_frame = input_frame.copy()
    output_frame["predicted_label"] = predicted_labels
    output_frame["ransomware_probability"] = ransomware_probability
    output_frame["is_positive_class_predicted"] = (
        (prediction_index == positive_class_index).astype(int)
    )
    output_frame["is_ransomware_predicted"] = output_frame["is_positive_class_predicted"]
    output_frame["positive_class_label"] = class_labels[positive_class_index]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_frame.to_csv(output_path, index=False)

    predicted_count = int((output_frame["is_positive_class_predicted"] == 1).sum())
    total_rows = int(len(output_frame))

    print(f"Predictions saved to: {output_path}")
    print(
        f"Predicted positive class ({class_labels[positive_class_index]}) rows: "
        f"{predicted_count}/{total_rows}"
    )


if __name__ == "__main__":
    main()

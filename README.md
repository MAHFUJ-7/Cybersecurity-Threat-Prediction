# Cybersecurity Threat Prediction — Ransomware Detection Using Machine Learning

This project provides a machine learning pipeline for ransomware detection, comparing multiple classifiers on ransomware behavior telemetry.

## What This Includes

- **Google Colab notebook** ([`ransomware_detection_colab.ipynb`](ransomware_detection_colab.ipynb)) — ransomware detection lab report with six ML models end-to-end.
- Six ML models compared: Random Forest, Decision Tree, Logistic Regression, SVM, KNN, and Naive Bayes.
- Classification reports, confusion matrices, and feature importance plots for each model.
- Synthetic ransomware behavior telemetry generation (no external dataset required).
- Model training with preprocessing for numeric features.
- Evaluation with accuracy, precision, recall, F1, confusion matrix, and model comparison chart.
- CLI pipeline for batch prediction with ransomware probability outputs.

## Feature Example

The baseline synthetic dataset includes features that are commonly useful in ransomware detection:

- `file_modifications_per_min`
- `files_encrypted_per_min`
- `entropy_delta`
- `ransom_note_created`
- `shadow_copy_delete_attempt`
- `backup_service_stop_attempt`
- `suspicious_extension_writes`
- `process_spawn_rate`
- `high_privilege_token_use`
- `outbound_unique_ips`
- `cpu_usage_percent`
- `disk_write_mb_s`
- `process_name`
- `parent_process`
- `signer_status`
- target: `is_ransomware`

## Google Colab Notebook (Lab Report)

The primary deliverable is [`ransomware_detection_colab.ipynb`](ransomware_detection_colab.ipynb). Open it in Google Colab:

1. Run all cells — the notebook generates its own synthetic ransomware telemetry dataset (no upload needed).
2. It trains and evaluates six models:
   - **Random Forest** — classification report, confusion matrix, feature importance
   - **Decision Tree** — classification report, confusion matrix, feature importance
   - **Logistic Regression** — classification report, confusion matrix
   - **SVM** — classification report, confusion matrix
   - **KNN** — classification report, confusion matrix
   - **Naive Bayes** — classification report, confusion matrix
3. A final comparison bar chart ranks all models by accuracy.

## Quick Start (CLI Pipeline)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate synthetic data (optional)

```bash
python src/generate_synthetic_data.py --output data/ransomware_behavior.csv --rows 5000
```

### 3. Train model

```bash
python src/train.py --input data/ransomware_behavior.csv --target-column is_ransomware --model-out models/ransomware_model.joblib --metrics-out models/metrics.json
```

If your dataset uses the opposite convention (for example `legitimate` where `0 = malware` and `1 = benign`), set `--positive-label` explicitly.

```bash
python src/train.py --input data/MalwareData.csv --target-column legitimate --drop-columns Name md5 --positive-label 0 --model-out models/malware_model.joblib --metrics-out models/malware_metrics.json
```

### 4. Run predictions

```bash
python src/predict.py --model models/ransomware_model.joblib --input data/ransomware_behavior.csv --output models/predictions.csv --threshold 0.5
```

For model artifacts trained with custom class meaning, you can override at inference time too:

```bash
python src/predict.py --model models/malware_model.joblib --input data/MalwareData.csv --output models/malware_predictions.csv --threshold 0.5 --positive-label 0
```

## Using a Real Dataset

- Replace the synthetic file with your telemetry CSV.
- Keep one binary target column (default: `is_ransomware`).
- Use `--drop-columns` in training to exclude identifiers like host IDs and timestamps if needed.
- Use `--positive-label` when malware is not encoded as `1`.

Example:

```bash
python src/train.py --input data/real_endpoint_telemetry.csv --target-column is_ransomware --drop-columns sample_id hostname event_time
```

## Notes

- This is a baseline model for learning and prototyping.
- For production deployment, add robust feature engineering, threshold tuning, drift monitoring, and adversarial resilience testing.

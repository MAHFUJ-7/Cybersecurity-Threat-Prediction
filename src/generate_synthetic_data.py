from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic telemetry dataset for ransomware detection."
    )
    parser.add_argument(
        "--output",
        default="data/ransomware_behavior.csv",
        help="Path to save the generated CSV file.",
    )
    parser.add_argument(
        "--rows", type=int, default=5000, help="Number of rows to generate."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def choose_process_name(rng: np.random.Generator, ransomware_flag: int) -> str:
    if ransomware_flag:
        choices = ["vssadmin.exe", "wbadmin.exe", "powershell.exe", "wscript.exe", "cmd.exe"]
        probs = [0.24, 0.18, 0.30, 0.16, 0.12]
    else:
        choices = ["explorer.exe", "chrome.exe", "msedge.exe", "outlook.exe", "teams.exe"]
        probs = [0.25, 0.25, 0.18, 0.17, 0.15]
    return str(rng.choice(choices, p=probs))


def choose_parent_process(rng: np.random.Generator, ransomware_flag: int) -> str:
    if ransomware_flag:
        choices = ["powershell.exe", "wscript.exe", "cmd.exe", "services.exe"]
        probs = [0.34, 0.22, 0.23, 0.21]
    else:
        choices = ["explorer.exe", "services.exe", "winlogon.exe"]
        probs = [0.59, 0.29, 0.12]
    return str(rng.choice(choices, p=probs))


def choose_signer_status(rng: np.random.Generator, ransomware_flag: int) -> str:
    if ransomware_flag:
        return str(rng.choice(["unsigned", "unknown", "signed"], p=[0.62, 0.24, 0.14]))
    return str(rng.choice(["signed", "unsigned", "unknown"], p=[0.78, 0.15, 0.07]))


def build_dataset(rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    is_ransomware = rng.binomial(1, 0.32, rows)

    file_modifications_per_min = rng.poisson(35 + (is_ransomware * 180))
    files_encrypted_per_min = rng.poisson(1 + (is_ransomware * 95))
    entropy_delta = np.clip(rng.normal(0.35 + (is_ransomware * 1.85), 0.35), 0.0, None)
    ransom_note_created = rng.binomial(1, 0.03 + (is_ransomware * 0.74))
    shadow_copy_delete_attempt = rng.binomial(1, 0.02 + (is_ransomware * 0.68))
    backup_service_stop_attempt = rng.binomial(1, 0.01 + (is_ransomware * 0.58))
    suspicious_extension_writes = rng.poisson(1 + (is_ransomware * 25))
    process_spawn_rate = rng.poisson(5 + (is_ransomware * 28))
    high_privilege_token_use = rng.binomial(1, 0.09 + (is_ransomware * 0.51))
    outbound_unique_ips = rng.poisson(2 + (is_ransomware * 8))
    cpu_usage_percent = np.clip(rng.normal(22 + (is_ransomware * 41), 10), 0.0, 100.0)
    disk_write_mb_s = np.clip(rng.normal(7 + (is_ransomware * 35), 8), 0.0, None)

    process_name = [choose_process_name(rng, int(flag)) for flag in is_ransomware]
    parent_process = [choose_parent_process(rng, int(flag)) for flag in is_ransomware]
    signer_status = [choose_signer_status(rng, int(flag)) for flag in is_ransomware]

    df = pd.DataFrame(
        {
            "sample_id": np.arange(1, rows + 1),
            "file_modifications_per_min": file_modifications_per_min,
            "files_encrypted_per_min": files_encrypted_per_min,
            "entropy_delta": entropy_delta,
            "ransom_note_created": ransom_note_created,
            "shadow_copy_delete_attempt": shadow_copy_delete_attempt,
            "backup_service_stop_attempt": backup_service_stop_attempt,
            "suspicious_extension_writes": suspicious_extension_writes,
            "process_spawn_rate": process_spawn_rate,
            "high_privilege_token_use": high_privilege_token_use,
            "outbound_unique_ips": outbound_unique_ips,
            "cpu_usage_percent": cpu_usage_percent,
            "disk_write_mb_s": disk_write_mb_s,
            "process_name": process_name,
            "parent_process": parent_process,
            "signer_status": signer_status,
            "is_ransomware": is_ransomware,
        }
    )

    return df


def main() -> None:
    args = parse_args()

    if args.rows < 100:
        raise ValueError("Please generate at least 100 rows for meaningful model training.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(rows=args.rows, seed=args.seed)
    dataset.to_csv(output_path, index=False)

    print(f"Synthetic dataset created: {output_path} ({len(dataset)} rows)")


if __name__ == "__main__":
    main()

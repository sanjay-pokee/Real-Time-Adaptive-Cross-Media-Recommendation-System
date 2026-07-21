from pathlib import Path
from typing import Any

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "datasets.yaml"


def load_dataset_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict) or "datasets" not in config:
        raise ValueError("Dataset config must contain a top-level 'datasets' key.")

    return config["datasets"]


def resolve_project_path(path: str) -> Path:
    return PROJECT_ROOT / path


def load_raw_dataset(dataset_name: str, config_path: Path = DEFAULT_CONFIG_PATH) -> pd.DataFrame:
    datasets = load_dataset_config(config_path)

    if dataset_name not in datasets:
        available = ", ".join(sorted(datasets))
        raise KeyError(f"Unknown dataset '{dataset_name}'. Available datasets: {available}")

    dataset_path = resolve_project_path(datasets[dataset_name]["path"])
    return pd.read_csv(dataset_path)


def load_all_raw_datasets(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, pd.DataFrame]:
    datasets = load_dataset_config(config_path)
    return {
        dataset_name: pd.read_csv(resolve_project_path(dataset_config["path"]))
        for dataset_name, dataset_config in datasets.items()
    }

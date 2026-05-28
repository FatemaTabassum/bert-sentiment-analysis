
# utils.py

# This file will contain general utility functions, such as metric computation that might be used 
# across different parts of your project (e.g., for evaluation).

from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import yaml
from pathlib import Path


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    return {"accuracy": accuracy, "f1": f1}


def load_config(config_dir: str = "config") -> dict:
    """Load and merge all YAML config files from config_dir."""
    base_path = Path(config_dir) / "config.yaml"
    with open(base_path) as f:
        config = yaml.safe_load(f) or {}

    for name in ["data_config", "model_config", "training_config"]:
        path = Path(config_dir) / f"{name}.yaml"
        with open(path) as f:
            config[name.replace("_config", "")] = yaml.safe_load(f)
    return config
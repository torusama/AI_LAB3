from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
DATA_CLEAN_DIR = PROJECT_ROOT / "data" / "clean"

ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
MODEL_DIR = ARTIFACT_DIR / "models"
REPORT_DIR = ARTIFACT_DIR / "reports"
FIGURE_DIR = ARTIFACT_DIR / "figures"


def configure_console_encoding() -> None:
    # Avoid UnicodeEncodeError on Windows terminals when the project path contains
    # non-ASCII characters such as Vietnamese folder names.
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None or not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass


def ensure_directories() -> None:
    for path in [DATA_CLEAN_DIR, MODEL_DIR, REPORT_DIR, FIGURE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def _load_target(path: Path) -> pd.Series:
    target_df = pd.read_csv(path)
    if target_df.shape[1] == 1:
        target = target_df.iloc[:, 0]
    else:
        target = target_df.squeeze("columns")
    return pd.Series(target).astype(int)


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    x_train_path = DATA_CLEAN_DIR / "X_train.csv"
    x_test_path = DATA_CLEAN_DIR / "X_test.csv"
    y_train_path = DATA_CLEAN_DIR / "y_train.csv"
    y_test_path = DATA_CLEAN_DIR / "y_test.csv"

    X_train = pd.read_csv(x_train_path)
    X_test = pd.read_csv(x_test_path)
    y_train = _load_target(y_train_path)
    y_test = _load_target(y_test_path)

    return X_train, X_test, y_train, y_test


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2, ensure_ascii=False)


def load_model(model_path: Path):
    import joblib

    return joblib.load(model_path)


def get_default_model_path() -> Path:
    return MODEL_DIR / "baseline_decision_tree.joblib"


configure_console_encoding()

from __future__ import annotations

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from common import (
    REPORT_DIR,
    ensure_directories,
    get_default_model_path,
    load_model,
    load_splits,
    save_json,
)


def compute_baseline_metrics() -> None:
    _, X_test, _, y_test = load_splits()

    model = load_model(get_default_model_path())

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = float(accuracy_score(y_test, y_pred))
    error_rate = float(1 - accuracy)
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, y_proba))
    conf_matrix = confusion_matrix(y_test, y_pred)

    metrics_df = pd.DataFrame(
        {
            "Metric": [
                "Accuracy",
                "Error Rate",
                "Precision",
                "Recall",
                "F1-score",
                "ROC-AUC",
            ],
            "Value": [accuracy, error_rate, precision, recall, f1, roc_auc],
        }
    )
    metrics_df.to_csv(REPORT_DIR / "baseline_metrics.csv", index=False)

    conf_matrix_df = pd.DataFrame(
        conf_matrix,
        index=["Actual_NoChurn", "Actual_Churn"],
        columns=["Pred_NoChurn", "Pred_Churn"],
    )
    conf_matrix_df.to_csv(REPORT_DIR / "baseline_confusion_matrix.csv")

    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=["No churn", "Churn"],
        output_dict=True,
        zero_division=0,
    )

    save_json(
        {
            "accuracy": accuracy,
            "error_rate": error_rate,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "confusion_matrix": conf_matrix.tolist(),
            "classification_report": report_dict,
        },
        REPORT_DIR / "baseline_metrics.json",
    )

    print("Baseline metrics computed.")
    print(metrics_df.to_string(index=False))
    print("\nConfusion Matrix:")
    print(conf_matrix)


if __name__ == "__main__":
    ensure_directories()
    compute_baseline_metrics()

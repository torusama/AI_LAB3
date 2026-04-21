from __future__ import annotations

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from common import MODEL_DIR, REPORT_DIR, ensure_directories, load_splits, save_json


def train_baseline_decision_tree() -> None:
    X_train, X_test, y_train, y_test = load_splits()

    baseline_tree = DecisionTreeClassifier(random_state=42)
    baseline_tree.fit(X_train, y_train)

    y_train_pred = baseline_tree.predict(X_train)
    y_test_pred = baseline_tree.predict(X_test)
    y_test_proba = baseline_tree.predict_proba(X_test)[:, 1]

    train_acc = float(accuracy_score(y_train, y_train_pred))
    test_acc = float(accuracy_score(y_test, y_test_pred))

    model_path = MODEL_DIR / "baseline_decision_tree.joblib"
    joblib.dump(baseline_tree, model_path)

    train_pred_df = pd.DataFrame(
        {
            "y_train": y_train.reset_index(drop=True),
            "y_train_pred": y_train_pred,
        }
    )
    train_pred_df.to_csv(REPORT_DIR / "baseline_train_predictions.csv", index=False)

    test_pred_df = pd.DataFrame(
        {
            "y_test": y_test.reset_index(drop=True),
            "y_test_pred": y_test_pred,
            "y_test_proba": y_test_proba,
        }
    )
    test_pred_df.to_csv(REPORT_DIR / "baseline_test_predictions.csv", index=False)

    save_json(
        {
            "model_path": str(model_path),
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
        },
        REPORT_DIR / "baseline_training_summary.json",
    )

    print("Baseline DecisionTreeClassifier training completed.")
    print(f"- Model saved to: {model_path}")
    print(f"- Train accuracy: {train_acc:.4f}")
    print(f"- Test accuracy : {test_acc:.4f}")


if __name__ == "__main__":
    ensure_directories()
    train_baseline_decision_tree()

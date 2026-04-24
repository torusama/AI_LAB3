from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from common import MODEL_DIR, REPORT_DIR, ensure_directories, load_model, load_splits, save_json


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _evaluate_holdout(model, X_train, X_test, y_train, y_test) -> dict:
    y_train_pred = model.predict(X_train)
    y_test_pred  = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    cm           = confusion_matrix(y_test, y_test_pred)

    train_acc = float(accuracy_score(y_train, y_train_pred))
    test_acc  = float(accuracy_score(y_test, y_test_pred))

    return {
        "train_acc":        train_acc,
        "test_acc":         test_acc,
        "error_rate":       float(1 - test_acc),
        "precision_churn":  float(precision_score(y_test, y_test_pred, zero_division=0)),
        "recall_churn":     float(recall_score(y_test, y_test_pred, zero_division=0)),
        "f1_churn":         float(f1_score(y_test, y_test_pred, zero_division=0)),
        "roc_auc":          float(roc_auc_score(y_test, y_test_proba)),
        "gap":              float(train_acc - test_acc),
        "confusion_matrix": cm.astype(int).tolist(),
    }


def _cross_validate_tree(X, y, params: dict) -> dict:
    splitter       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_f1:   list[float] = []
    fold_roc:  list[float] = []

    for train_idx, val_idx in splitter.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        m = DecisionTreeClassifier(**params, random_state=42)
        m.fit(X_tr, y_tr)

        fold_f1.append(float(f1_score(y_val, m.predict(X_val), zero_division=0)))
        fold_roc.append(float(roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])))

    return {
        "cv_mean_f1":      float(np.mean(fold_f1)),
        "cv_mean_roc_auc": float(np.mean(fold_roc)),
    }


def _is_better(candidate: dict, incumbent: dict | None) -> bool:
    if incumbent is None:
        return True
    if candidate["cv_mean_f1"] != incumbent["cv_mean_f1"]:
        return candidate["cv_mean_f1"] > incumbent["cv_mean_f1"]
    if candidate["cv_mean_roc_auc"] != incumbent["cv_mean_roc_auc"]:
        return candidate["cv_mean_roc_auc"] > incumbent["cv_mean_roc_auc"]
    return candidate["gap"] < incumbent["gap"]


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def run_improvement3() -> None:
    X_train, X_test, y_train, y_test = load_splits()

    # --- Load best params from Improvement 1 ---
    with (REPORT_DIR / "improvement1_summary.json").open("r", encoding="utf-8") as fp:
        imp1 = json.load(fp)
    best_params_imp1 = imp1["best_params"]

    base_params = {
        "max_depth":        best_params_imp1["max_depth"],
        "min_samples_split": best_params_imp1["min_samples_split"],
        "min_samples_leaf":  best_params_imp1["min_samples_leaf"],
    }

    # THAY bằng:
    unpruned_model = DecisionTreeClassifier(**base_params, random_state=42)
    unpruned_model.fit(X_train, y_train)

    baseline_cv      = _cross_validate_tree(X_train, y_train, base_params)
    baseline_holdout = _evaluate_holdout(unpruned_model, X_train, X_test, y_train, y_test)
    baseline_metrics = {**baseline_cv, **baseline_holdout}

    # --- Find candidate ccp_alpha values ---
    # Fit a full tree first to extract the pruning path
    full_tree = DecisionTreeClassifier(**base_params, random_state=42)
    full_tree.fit(X_train, y_train)
    path        = full_tree.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas  = path.ccp_alphas          # array of alpha values
    # Keep only alphas > 0 and up to a reasonable maximum
    ccp_alphas  = [a for a in ccp_alphas if 0 < a <= 0.05]
    # Sample at most 30 values to keep runtime reasonable
    if len(ccp_alphas) > 30:
        indices    = np.linspace(0, len(ccp_alphas) - 1, 30, dtype=int)
        ccp_alphas = [ccp_alphas[i] for i in indices]

    # --- Grid search over ccp_alpha ---
    grid_rows:   list[dict]               = []
    best_row:    dict | None              = None
    best_model:  DecisionTreeClassifier | None = None

    for alpha in ccp_alphas:
        params  = {**base_params, "ccp_alpha": alpha}
        cv_m    = _cross_validate_tree(X_train, y_train, params)

        model   = DecisionTreeClassifier(**params, random_state=42)
        model.fit(X_train, y_train)
        hold_m  = _evaluate_holdout(model, X_train, X_test, y_train, y_test)

        row = {"ccp_alpha": alpha, **cv_m, **hold_m,
               "n_leaves": int(model.get_n_leaves()),
               "tree_depth": int(model.get_depth())}
        grid_rows.append(row)

        if _is_better(row, best_row):
            best_row   = row
            best_model = model

    assert best_model is not None and best_row is not None

    improved_metrics = {k: best_row[k] for k in [
        "cv_mean_f1", "cv_mean_roc_auc", "train_acc", "test_acc",
        "error_rate", "precision_churn", "recall_churn",
        "f1_churn", "roc_auc", "gap", "confusion_matrix",
    ]}

    # --- Save artefacts ---
    joblib.dump(best_model, MODEL_DIR / "improvement3_pruned.joblib")

    grid_df = pd.DataFrame(grid_rows).sort_values(
        by=["cv_mean_f1", "cv_mean_roc_auc", "gap"],
        ascending=[False, False, True],
    )
    grid_df.to_csv(REPORT_DIR / "improvement3_grid_results.csv", index=False)

    comparison_df = pd.DataFrame({
        "Metric": [
            "Tree Depth", "N Leaves",
            "CV Mean F1", "CV Mean ROC-AUC", "Train Acc", "Test Acc",
            "Error Rate", "Precision (Churn)", "Recall (Churn)",
            "F1 (Churn)", "ROC-AUC", "Gap", "Confusion Matrix",
        ],
        "Baseline": [
            unpruned_model.get_depth(), unpruned_model.get_n_leaves(),
            baseline_metrics["cv_mean_f1"], baseline_metrics["cv_mean_roc_auc"],
            baseline_metrics["train_acc"],  baseline_metrics["test_acc"],
            baseline_metrics["error_rate"], baseline_metrics["precision_churn"],
            baseline_metrics["recall_churn"], baseline_metrics["f1_churn"],
            baseline_metrics["roc_auc"],    baseline_metrics["gap"],
            json.dumps(baseline_metrics["confusion_matrix"]),
        ],
        "Pruned": [
            best_row["tree_depth"], best_row["n_leaves"],
            improved_metrics["cv_mean_f1"], improved_metrics["cv_mean_roc_auc"],
            improved_metrics["train_acc"],  improved_metrics["test_acc"],
            improved_metrics["error_rate"], improved_metrics["precision_churn"],
            improved_metrics["recall_churn"], improved_metrics["f1_churn"],
            improved_metrics["roc_auc"],    improved_metrics["gap"],
            json.dumps(improved_metrics["confusion_matrix"]),
        ],
    })
    comparison_df.to_csv(REPORT_DIR / "improvement3_metrics_comparison.csv", index=False)

    # --- Build explanation text ---
    gap_delta  = baseline_metrics["gap"] - improved_metrics["gap"]
    f1_delta   = improved_metrics["f1_churn"] - baseline_metrics["f1_churn"]
    cv_f1_d    = improved_metrics["cv_mean_f1"] - baseline_metrics["cv_mean_f1"]
    cv_roc_d   = improved_metrics["cv_mean_roc_auc"] - baseline_metrics["cv_mean_roc_auc"]

    diagnosis = (
        f"Best ccp_alpha={best_row['ccp_alpha']:.6f} reduced tree to "
        f"{best_row['n_leaves']} leaves / depth {best_row['tree_depth']}. "
        f"CV F1 changed by {cv_f1_d:+.4f}, CV ROC-AUC changed by {cv_roc_d:+.4f}, "
        f"holdout gap changed by {gap_delta:+.4f} "
        f"(baseline {baseline_metrics['gap']:.4f} -> pruned {improved_metrics['gap']:.4f}), "
        f"holdout F1 changed by {f1_delta:+.4f}."
    )

    explanations = [
        "Cost-complexity pruning removes branches whose contribution to impurity reduction "
        "is smaller than the penalty alpha, resulting in a simpler, more interpretable tree.",
        "By eliminating low-value branches, the pruned tree reduces variance and overfitting "
        "while preserving the most discriminative splits.",
    ]
    if improved_metrics["gap"] < baseline_metrics["gap"]:
        explanations.append(
            "The reduced train-test accuracy gap confirms the pruned tree generalises better "
            "than the unpruned baseline."
        )
    else:
        explanations.append(
            "The train-test gap did not decrease, suggesting the chosen alpha may need "
            "further tuning or that pruning alone is insufficient for this dataset."
        )
    if improved_metrics["cv_mean_f1"] > baseline_metrics["cv_mean_f1"]:
        explanations.append(
            "Higher cross-validated F1 indicates the pruned tree detects churn more "
            "consistently across folds."
        )

    save_json(
        {
            "method": "Cost-Complexity Pruning (ccp_alpha)",
            "best_ccp_alpha": best_row["ccp_alpha"],
            "best_n_leaves":  best_row["n_leaves"],
            "best_tree_depth": best_row["tree_depth"],
            "selection_priority": [
                "highest_cv_mean_f1",
                "higher_cv_mean_roc_auc",
                "smaller_train_test_gap",
            ],
            "baseline": {k: baseline_metrics[k] for k in [
                "cv_mean_f1", "cv_mean_roc_auc", "train_acc", "test_acc",
                "error_rate", "precision_churn", "recall_churn",
                "f1_churn", "roc_auc", "gap", "confusion_matrix",
            ]},
            "improved": {k: improved_metrics[k] for k in [
                "cv_mean_f1", "cv_mean_roc_auc", "train_acc", "test_acc",
                "error_rate", "precision_churn", "recall_churn",
                "f1_churn", "roc_auc", "gap", "confusion_matrix",
            ]},
            "diagnosis":    diagnosis,
            "explanations": explanations,
        },
        REPORT_DIR / "improvement3_summary.json",
    )

    # --- Console output ---
    print("=== Improvement #3: Cost-Complexity Pruning ===")
    print(f"Best ccp_alpha : {best_row['ccp_alpha']:.6f}")
    print(f"Tree leaves    : {best_row['n_leaves']}  |  depth: {best_row['tree_depth']}")
    print()
    print(
        comparison_df.to_string(
            index=False,
            formatters={
                "Baseline": lambda v: f"{v:.4f}" if isinstance(v, float) else str(v),
                "Pruned":   lambda v: f"{v:.4f}" if isinstance(v, float) else str(v),
            },
        )
    )


if __name__ == "__main__":
    ensure_directories()
    run_improvement3()
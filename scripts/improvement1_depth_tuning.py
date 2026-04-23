from __future__ import annotations

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from common import MODEL_DIR, REPORT_DIR, ensure_directories, load_model, load_splits, save_json


def _evaluate_holdout(model, X_train, X_test, y_train, y_test) -> dict[str, float]:
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]

    train_acc = float(accuracy_score(y_train, y_train_pred))
    test_acc = float(accuracy_score(y_test, y_test_pred))
    f1_churn = float(f1_score(y_test, y_test_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, y_test_proba))

    return {
        "train_acc": train_acc,
        "test_acc": test_acc,
        "f1_churn": f1_churn,
        "roc_auc": roc_auc,
        "gap": float(train_acc - test_acc),
    }


def _cross_validate_tree(X, y, params: dict) -> dict[str, float]:
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_f1_scores: list[float] = []
    fold_roc_auc_scores: list[float] = []

    for train_idx, valid_idx in splitter.split(X, y):
        X_fold_train = X.iloc[train_idx]
        X_fold_valid = X.iloc[valid_idx]
        y_fold_train = y.iloc[train_idx]
        y_fold_valid = y.iloc[valid_idx]

        model = DecisionTreeClassifier(**params, random_state=42)
        model.fit(X_fold_train, y_fold_train)

        y_valid_pred = model.predict(X_fold_valid)
        y_valid_proba = model.predict_proba(X_fold_valid)[:, 1]

        fold_f1_scores.append(float(f1_score(y_fold_valid, y_valid_pred, zero_division=0)))
        fold_roc_auc_scores.append(float(roc_auc_score(y_fold_valid, y_valid_proba)))

    return {
        "cv_mean_f1": float(sum(fold_f1_scores) / len(fold_f1_scores)),
        "cv_mean_roc_auc": float(sum(fold_roc_auc_scores) / len(fold_roc_auc_scores)),
    }


def _fit_and_evaluate(params: dict, X_train, X_test, y_train, y_test) -> tuple[DecisionTreeClassifier, dict[str, float]]:
    cv_metrics = _cross_validate_tree(X_train, y_train, params)

    model = DecisionTreeClassifier(**params, random_state=42)
    model.fit(X_train, y_train)

    holdout_metrics = _evaluate_holdout(model, X_train, X_test, y_train, y_test)
    combined_metrics = {**cv_metrics, **holdout_metrics}

    return model, combined_metrics


def _is_better_candidate(candidate: dict, incumbent: dict | None) -> bool:
    if incumbent is None:
        return True

    if candidate["cv_mean_f1"] != incumbent["cv_mean_f1"]:
        return candidate["cv_mean_f1"] > incumbent["cv_mean_f1"]
    if candidate["cv_mean_roc_auc"] != incumbent["cv_mean_roc_auc"]:
        return candidate["cv_mean_roc_auc"] > incumbent["cv_mean_roc_auc"]
    return candidate["gap"] < incumbent["gap"]


def _build_explanations(best_params: dict, baseline_metrics: dict[str, float], improved_metrics: dict[str, float]) -> list[str]:
    explanations: list[str] = []

    if best_params["max_depth"] is None:
        explanations.append(
            "Allowing unlimited depth keeps the tree highly flexible, which can fit detailed patterns but also increases variance and overfitting risk."
        )
    elif best_params["max_depth"] <= 5:
        explanations.append(
            "A relatively shallow max_depth constrains tree growth, reducing variance and helping the model ignore noisy training-specific splits."
        )
    else:
        explanations.append(
            "A moderate max_depth preserves useful non-linear structure while still limiting some of the variance seen in an unconstrained tree."
        )

    if best_params["min_samples_split"] > 2:
        explanations.append(
            "A larger min_samples_split requires more evidence before creating new internal nodes, which reduces fragile branch creation and helps control overfitting."
        )
    else:
        explanations.append(
            "Keeping min_samples_split small allows flexible branching, so overfitting control depends more heavily on max_depth and min_samples_leaf."
        )

    if best_params["min_samples_leaf"] > 1:
        explanations.append(
            "Increasing min_samples_leaf prevents leaves built from very small groups, which lowers variance and usually produces more stable churn decisions."
        )
    else:
        explanations.append(
            "Using min_samples_leaf=1 leaves the model free to isolate tiny regions, which may improve fit on training data but can make generalization less stable."
        )

    if improved_metrics["gap"] < baseline_metrics["gap"]:
        explanations.append(
            "The smaller train-test accuracy gap indicates that the tuned tree generalizes more reliably than the baseline and is less overfit."
        )
    else:
        explanations.append(
            "The train-test accuracy gap did not shrink, so the tuned tree still shows notable variance and should be interpreted with caution."
        )

    if improved_metrics["cv_mean_f1"] > baseline_metrics["cv_mean_f1"]:
        explanations.append(
            "Higher cross-validated F1 suggests the tuned hyperparameters improve churn detection more consistently across folds, not only on one test split."
        )
    if improved_metrics["cv_mean_roc_auc"] > baseline_metrics["cv_mean_roc_auc"]:
        explanations.append(
            "Higher cross-validated ROC-AUC indicates the tuned tree separates churn and non-churn customers more effectively across decision thresholds."
        )

    return explanations


def _baseline_metrics(X_train, X_test, y_train, y_test) -> dict[str, float]:
    baseline_path = MODEL_DIR / "baseline_decision_tree.joblib"

    try:
        baseline_model = load_model(baseline_path)
    except Exception:
        baseline_model = DecisionTreeClassifier(random_state=42)
        baseline_model.fit(X_train, y_train)
        joblib.dump(baseline_model, baseline_path)

    baseline_cv = _cross_validate_tree(X_train, y_train, {})
    baseline_holdout = _evaluate_holdout(baseline_model, X_train, X_test, y_train, y_test)
    return {**baseline_cv, **baseline_holdout}


def run_improvement1() -> None:
    X_train, X_test, y_train, y_test = load_splits()
    baseline_metrics = _baseline_metrics(X_train, X_test, y_train, y_test)

    param_grid = {
        "max_depth": [3, 5, 7, 10, 15, None],
        "min_samples_split": [2, 10, 20, 50],
        "min_samples_leaf": [1, 5, 10, 20],
    }

    grid_rows: list[dict] = []
    best_model = None
    best_row = None

    for max_depth in param_grid["max_depth"]:
        for min_samples_split in param_grid["min_samples_split"]:
            for min_samples_leaf in param_grid["min_samples_leaf"]:
                params = {
                    "max_depth": max_depth,
                    "min_samples_split": min_samples_split,
                    "min_samples_leaf": min_samples_leaf,
                }
                model, metrics = _fit_and_evaluate(params, X_train, X_test, y_train, y_test)
                row = {**params, **metrics}
                grid_rows.append(row)

                if _is_better_candidate(row, best_row):
                    best_model = model
                    best_row = row

    assert best_model is not None and best_row is not None

    improved_metrics = {
        "cv_mean_f1": best_row["cv_mean_f1"],
        "cv_mean_roc_auc": best_row["cv_mean_roc_auc"],
        "train_acc": best_row["train_acc"],
        "test_acc": best_row["test_acc"],
        "f1_churn": best_row["f1_churn"],
        "roc_auc": best_row["roc_auc"],
        "gap": best_row["gap"],
    }

    comparison_df = pd.DataFrame(
        {
            "Metric": [
                "CV Mean F1",
                "CV Mean ROC-AUC",
                "Train Acc",
                "Test Acc",
                "F1 (Churn)",
                "ROC-AUC",
                "Gap",
            ],
            "Baseline": [
                baseline_metrics["cv_mean_f1"],
                baseline_metrics["cv_mean_roc_auc"],
                baseline_metrics["train_acc"],
                baseline_metrics["test_acc"],
                baseline_metrics["f1_churn"],
                baseline_metrics["roc_auc"],
                baseline_metrics["gap"],
            ],
            "Improved": [
                improved_metrics["cv_mean_f1"],
                improved_metrics["cv_mean_roc_auc"],
                improved_metrics["train_acc"],
                improved_metrics["test_acc"],
                improved_metrics["f1_churn"],
                improved_metrics["roc_auc"],
                improved_metrics["gap"],
            ],
        }
    )

    grid_df = pd.DataFrame(grid_rows).sort_values(
        by=["cv_mean_f1", "cv_mean_roc_auc", "gap"],
        ascending=[False, False, True],
    )

    model_path = MODEL_DIR / "improvement1_depth_tuned.joblib"
    joblib.dump(best_model, model_path)

    comparison_df.to_csv(REPORT_DIR / "improvement1_metrics_comparison.csv", index=False)
    grid_df.to_csv(REPORT_DIR / "improvement1_grid_results.csv", index=False)

    gap_delta = baseline_metrics["gap"] - improved_metrics["gap"]
    f1_delta = improved_metrics["f1_churn"] - baseline_metrics["f1_churn"]
    cv_f1_delta = improved_metrics["cv_mean_f1"] - baseline_metrics["cv_mean_f1"]
    cv_roc_delta = improved_metrics["cv_mean_roc_auc"] - baseline_metrics["cv_mean_roc_auc"]
    diagnosis = (
        f"Cross-validated F1 changed by {cv_f1_delta:+.4f} "
        f"and cross-validated ROC-AUC changed by {cv_roc_delta:+.4f}; "
        f"holdout gap changed by {gap_delta:+.4f} "
        f"(baseline {baseline_metrics['gap']:.4f} -> improved {improved_metrics['gap']:.4f}), "
        f"while holdout F1 changed by {f1_delta:+.4f}."
    )

    best_params = {
        "max_depth": best_row["max_depth"],
        "min_samples_split": int(best_row["min_samples_split"]),
        "min_samples_leaf": int(best_row["min_samples_leaf"]),
    }

    save_json(
        {
            "method": "Depth Tuning + Min Samples",
            "selection_priority": [
                "highest_cv_mean_f1",
                "higher_cv_mean_roc_auc",
                "smaller_train_test_gap",
            ],
            "best_params": best_params,
            "baseline": {
                "cv_mean_f1": baseline_metrics["cv_mean_f1"],
                "cv_mean_roc_auc": baseline_metrics["cv_mean_roc_auc"],
                "train_acc": baseline_metrics["train_acc"],
                "test_acc": baseline_metrics["test_acc"],
                "f1_churn": baseline_metrics["f1_churn"],
                "roc_auc": baseline_metrics["roc_auc"],
            },
            "improved": {
                "cv_mean_f1": improved_metrics["cv_mean_f1"],
                "cv_mean_roc_auc": improved_metrics["cv_mean_roc_auc"],
                "train_acc": improved_metrics["train_acc"],
                "test_acc": improved_metrics["test_acc"],
                "f1_churn": improved_metrics["f1_churn"],
                "roc_auc": improved_metrics["roc_auc"],
            },
            "accuracy_gap_baseline": baseline_metrics["gap"],
            "accuracy_gap_improved": improved_metrics["gap"],
            "diagnosis": diagnosis,
            "explanations": _build_explanations(best_params, baseline_metrics, improved_metrics),
        },
        REPORT_DIR / "improvement1_summary.json",
    )

    print("=== Improvement #1: Depth Tuning + Min Samples ===")
    print(
        "Best params: "
        f"max_depth={best_row['max_depth']}, "
        f"min_samples_split={int(best_row['min_samples_split'])}, "
        f"min_samples_leaf={int(best_row['min_samples_leaf'])}"
    )
    print()
    print(
        comparison_df.to_string(
            index=False,
            formatters={"Baseline": "{:.4f}".format, "Improved": "{:.4f}".format},
        )
    )


if __name__ == "__main__":
    ensure_directories()
    run_improvement1()

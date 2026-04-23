from __future__ import annotations

import json

import joblib
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

from common import MODEL_DIR, REPORT_DIR, ensure_directories, load_splits, save_json


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


def _evaluate_holdout(model, X_train, X_test, y_train, y_test) -> dict[str, float | list[list[int]]]:
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_test_pred)

    train_acc = float(accuracy_score(y_train, y_train_pred))
    test_acc = float(accuracy_score(y_test, y_test_pred))
    precision = float(precision_score(y_test, y_test_pred, zero_division=0))
    recall = float(recall_score(y_test, y_test_pred, zero_division=0))
    f1_churn = float(f1_score(y_test, y_test_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, y_test_proba))

    return {
        "train_acc": train_acc,
        "accuracy": test_acc,
        "test_acc": test_acc,
        "error_rate": float(1 - test_acc),
        "precision_churn": precision,
        "recall_churn": recall,
        "f1_churn": f1_churn,
        "roc_auc": roc_auc,
        "gap": float(train_acc - test_acc),
        "confusion_matrix": cm.astype(int).tolist(),
    }


def _fit_and_evaluate(name: str, params: dict, X_train, X_test, y_train, y_test) -> tuple[DecisionTreeClassifier, dict]:
    cv_metrics = _cross_validate_tree(X_train, y_train, params)

    model = DecisionTreeClassifier(**params, random_state=42)
    model.fit(X_train, y_train)

    holdout_metrics = _evaluate_holdout(model, X_train, X_test, y_train, y_test)
    metrics = {"model_name": name, **params, **cv_metrics, **holdout_metrics}
    return model, metrics


def _is_better_candidate(candidate: dict, incumbent: dict | None) -> bool:
    if incumbent is None:
        return True

    if candidate["cv_mean_f1"] != incumbent["cv_mean_f1"]:
        return candidate["cv_mean_f1"] > incumbent["cv_mean_f1"]
    if candidate["cv_mean_roc_auc"] != incumbent["cv_mean_roc_auc"]:
        return candidate["cv_mean_roc_auc"] > incumbent["cv_mean_roc_auc"]
    return candidate["gap"] < incumbent["gap"]


def _criterion_explanation(criterion: str) -> str:
    if criterion == "gini":
        return "Gini impurity measures class mixing efficiently and often produces strong baseline trees with fast, stable splits."
    if criterion == "entropy":
        return "Entropy is more sensitive to impurity changes, so it can emphasize minority-class separation and sometimes improve churn detection."
    return "Log-loss uses probabilistic impurity, which can improve probability ranking quality and therefore help ROC-AUC."


def _build_model_explanations(results_df: pd.DataFrame) -> list[str]:
    explanations: list[str] = []

    criterion_rows = results_df[results_df["experiment_group"] == "criterion"].copy()
    if not criterion_rows.empty:
        best_criterion_row = criterion_rows.sort_values(
            by=["cv_mean_f1", "cv_mean_roc_auc", "gap"],
            ascending=[False, False, True],
        ).iloc[0]
        explanations.append(
            f"{best_criterion_row['criterion']} ranked best among the criterion experiments because it achieved the strongest cross-validated F1/ROC-AUC trade-off with a gap of {best_criterion_row['gap']:.4f}."
        )

    if "entropy" in set(results_df["criterion"].astype(str)):
        explanations.append(
            "Compared with gini, entropy reacts more strongly to impurity differences, so it may recover more churn cases when the minority class needs cleaner separation."
        )
    if "log_loss" in set(results_df["criterion"].astype(str)):
        explanations.append(
            "Log-loss focuses on probabilistic quality, so improvements there are especially relevant when ROC-AUC matters more than raw accuracy."
        )
    if results_df["gap"].max() - results_df["gap"].min() > 0.01:
        explanations.append(
            "A smaller train-test gap indicates lower variance, while a larger gap suggests the tree is fitting training-specific structure too aggressively."
        )
    if results_df["class_weight"].astype(str).str.contains("balanced").any():
        explanations.append(
            "Balanced class weights increase the penalty for misclassifying churners, which can improve recall but may reduce overall accuracy through more false positives."
        )

    return explanations


def run_improvement2() -> None:
    X_train, X_test, y_train, y_test = load_splits()

    with (REPORT_DIR / "improvement1_summary.json").open("r", encoding="utf-8") as fp:
        imp1 = json.load(fp)
    best_params = imp1["best_params"]

    base_tree_params = {
        "max_depth": best_params["max_depth"],
        "min_samples_split": best_params["min_samples_split"],
        "min_samples_leaf": best_params["min_samples_leaf"],
    }

    experiment_configs = [
        {
            "name": "Imp2 - criterion=gini",
            "criterion": "gini",
            "class_weight": None,
            "experiment_group": "criterion",
        },
        {
            "name": "Imp2 - criterion=entropy",
            "criterion": "entropy",
            "class_weight": None,
            "experiment_group": "criterion",
        },
        {
            "name": "Imp2 - criterion=log_loss",
            "criterion": "log_loss",
            "class_weight": None,
            "experiment_group": "criterion",
        },
        {
            "name": "Bonus - gini + balanced",
            "criterion": "gini",
            "class_weight": "balanced",
            "experiment_group": "bonus",
        },
        {
            "name": "Bonus - gini + weight_1_to_2",
            "criterion": "gini",
            "class_weight": {0: 1, 1: 2},
            "experiment_group": "bonus",
        },
    ]

    fitted_models: dict[str, DecisionTreeClassifier] = {}
    result_rows: list[dict] = []
    best_criterion_result = None
    best_criterion_model = None

    for config in experiment_configs:
        params = {
            **base_tree_params,
            "criterion": config["criterion"],
            "class_weight": config["class_weight"],
        }
        model, metrics = _fit_and_evaluate(config["name"], params, X_train, X_test, y_train, y_test)

        row = {
            "model_name": config["name"],
            "experiment_group": config["experiment_group"],
            "criterion": config["criterion"],
            "class_weight": str(config["class_weight"]),
            "cv_mean_f1": metrics["cv_mean_f1"],
            "cv_mean_roc_auc": metrics["cv_mean_roc_auc"],
            "train_acc": metrics["train_acc"],
            "test_acc": metrics["test_acc"],
            "accuracy": metrics["accuracy"],
            "error_rate": metrics["error_rate"],
            "precision_churn": metrics["precision_churn"],
            "recall_churn": metrics["recall_churn"],
            "f1_churn": metrics["f1_churn"],
            "roc_auc": metrics["roc_auc"],
            "gap": metrics["gap"],
            "confusion_matrix": metrics["confusion_matrix"],
            "explanation": _criterion_explanation(config["criterion"]),
        }
        result_rows.append(row)
        fitted_models[config["name"]] = model

        if config["experiment_group"] == "criterion" and _is_better_candidate(row, best_criterion_result):
            best_criterion_result = row
            best_criterion_model = model

        if config["class_weight"] == "balanced":
            joblib.dump(model, MODEL_DIR / "improvement2_balanced.joblib")

    assert best_criterion_result is not None and best_criterion_model is not None

    results_df = pd.DataFrame(result_rows).sort_values(
        by=["experiment_group", "cv_mean_f1", "cv_mean_roc_auc", "gap"],
        ascending=[True, False, False, True],
    )

    criterion_df = results_df[results_df["experiment_group"] == "criterion"].copy()

    summary_df = results_df[
        [
            "model_name",
            "criterion",
            "class_weight",
            "cv_mean_f1",
            "cv_mean_roc_auc",
            "accuracy",
            "error_rate",
            "precision_churn",
            "recall_churn",
            "f1_churn",
            "roc_auc",
            "gap",
        ]
    ].copy()

    criterion_summary_df = criterion_df[
        [
            "model_name",
            "criterion",
            "cv_mean_f1",
            "cv_mean_roc_auc",
            "accuracy",
            "precision_churn",
            "recall_churn",
            "f1_churn",
            "roc_auc",
            "gap",
        ]
    ].copy()

    summary_df.to_csv(REPORT_DIR / "improvement2_metrics_comparison.csv", index=False)
    criterion_summary_df.to_csv(REPORT_DIR / "improvement2_criterion_comparison.csv", index=False)

    joblib.dump(best_criterion_model, MODEL_DIR / "improvement2_best_criterion.joblib")

    diagnosis = (
        f"Best criterion was {best_criterion_result['criterion']} with cross-validated F1={best_criterion_result['cv_mean_f1']:.4f}, "
        f"cross-validated ROC-AUC={best_criterion_result['cv_mean_roc_auc']:.4f}, and train-test gap={best_criterion_result['gap']:.4f}. "
        f"Holdout recall for churn was {best_criterion_result['recall_churn']:.4f} and holdout ROC-AUC was {best_criterion_result['roc_auc']:.4f}."
    )

    save_json(
        {
            "method": "Criterion Comparison with Tuned Tree Hyperparameters",
            "params_used": base_tree_params,
            "best_criterion": best_criterion_result["criterion"],
            "best_model_name": best_criterion_result["model_name"],
            "selection_priority": [
                "highest_cv_mean_f1",
                "higher_cv_mean_roc_auc",
                "smaller_train_test_gap",
            ],
            "criterion_results": {
                row["criterion"]: {
                    "cv_mean_f1": row["cv_mean_f1"],
                    "cv_mean_roc_auc": row["cv_mean_roc_auc"],
                    "accuracy": row["accuracy"],
                    "recall_churn": row["recall_churn"],
                    "f1_churn": row["f1_churn"],
                    "roc_auc": row["roc_auc"],
                    "gap": row["gap"],
                    "explanation": row["explanation"],
                }
                for row in result_rows
                if row["experiment_group"] == "criterion"
            },
            "all_models": {
                row["model_name"]: {
                    "criterion": row["criterion"],
                    "class_weight": row["class_weight"],
                    "cv_mean_f1": row["cv_mean_f1"],
                    "cv_mean_roc_auc": row["cv_mean_roc_auc"],
                    "accuracy": row["accuracy"],
                    "precision_churn": row["precision_churn"],
                    "recall_churn": row["recall_churn"],
                    "f1_churn": row["f1_churn"],
                    "roc_auc": row["roc_auc"],
                    "gap": row["gap"],
                }
                for row in result_rows
            },
            "confusion_matrices": {
                row["model_name"]: row["confusion_matrix"]
                for row in result_rows
            },
            "diagnosis": diagnosis,
            "explanations": _build_model_explanations(results_df),
        },
        REPORT_DIR / "improvement2_summary.json",
    )

    best_cm = best_criterion_result["confusion_matrix"]

    print("=== Improvement #2: Criterion Comparison ===")
    print()
    print(
        criterion_summary_df.to_string(
            index=False,
            formatters={
                "cv_mean_f1": "{:.4f}".format,
                "cv_mean_roc_auc": "{:.4f}".format,
                "accuracy": "{:.4f}".format,
                "precision_churn": "{:.4f}".format,
                "recall_churn": "{:.4f}".format,
                "f1_churn": "{:.4f}".format,
                "roc_auc": "{:.4f}".format,
                "gap": "{:.4f}".format,
            },
        )
    )
    print()
    print(
        f"Best criterion: {best_criterion_result['criterion']} "
        f"(CV F1={best_criterion_result['cv_mean_f1']:.4f}, "
        f"CV ROC-AUC={best_criterion_result['cv_mean_roc_auc']:.4f}, "
        f"Gap={best_criterion_result['gap']:.4f})"
    )
    print()
    print(f"Confusion Matrix - Best criterion ({best_criterion_result['criterion']}):")
    print("             Pred_NoChurn  Pred_Churn")
    print(f"Actual_NoChurn    {best_cm[0][0]:4d}        {best_cm[0][1]:4d}")
    print(f"Actual_Churn      {best_cm[1][0]:4d}        {best_cm[1][1]:4d}")


if __name__ == "__main__":
    ensure_directories()
    run_improvement2()

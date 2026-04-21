from __future__ import annotations

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import _tree

from common import (
    REPORT_DIR,
    ensure_directories,
    get_default_model_path,
    load_model,
    load_splits,
    save_json,
)


def format_rule(rule_parts: list[str]) -> str:
    return " and ".join(rule_parts) if rule_parts else "ROOT"


def extract_representative_rules(clf, feature_names: list[str], max_depth: int = 3) -> list[dict]:
    tree = clf.tree_
    rules: list[dict] = []

    def recurse(node_id: int, conditions: list[str], current_depth: int) -> None:
        left = tree.children_left[node_id]
        right = tree.children_right[node_id]

        stop_here = (
            left == _tree.TREE_LEAF
            or right == _tree.TREE_LEAF
            or current_depth == max_depth
        )

        if stop_here:
            class_counts = tree.value[node_id][0]
            total = int(tree.n_node_samples[node_id])
            pred_class = int(class_counts.argmax())
            count_sum = float(class_counts.sum())
            purity = float(class_counts[pred_class] / count_sum) if count_sum else 0.0

            rules.append(
                {
                    "rule": format_rule(conditions),
                    "samples": total,
                    "pred_class": pred_class,
                    "purity": purity,
                    "depth": current_depth,
                }
            )
            return

        feat_name = feature_names[tree.feature[node_id]]
        threshold = tree.threshold[node_id]

        recurse(
            left,
            conditions + [f"{feat_name} <= {threshold:.3f}"],
            current_depth + 1,
        )
        recurse(
            right,
            conditions + [f"{feat_name} > {threshold:.3f}"],
            current_depth + 1,
        )

    recurse(0, [], 0)
    return rules


def analyze_baseline_tree() -> None:
    X_train, X_test, y_train, y_test = load_splits()

    model = load_model(get_default_model_path())

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_acc = float(accuracy_score(y_train, y_train_pred))
    test_acc = float(accuracy_score(y_test, y_test_pred))

    depth = int(model.get_depth())
    node_count = int(model.tree_.node_count)
    leaf_count = int(model.get_n_leaves())

    importance_df = (
        pd.DataFrame(
            {
                "feature": X_train.columns,
                "importance": model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    top_splits = importance_df.head(10)
    top_splits.to_csv(REPORT_DIR / "baseline_top_splits.csv", index=False)

    tree_obj = model.tree_
    root_id = 0
    left_id = int(tree_obj.children_left[root_id])
    right_id = int(tree_obj.children_right[root_id])

    early_splits = {
        "root": {
            "feature": str(X_train.columns[tree_obj.feature[root_id]]),
            "threshold": float(tree_obj.threshold[root_id]),
        }
    }
    if left_id != _tree.TREE_LEAF:
        early_splits["left_child"] = {
            "feature": str(X_train.columns[tree_obj.feature[left_id]]),
            "threshold": float(tree_obj.threshold[left_id]),
        }
    if right_id != _tree.TREE_LEAF:
        early_splits["right_child"] = {
            "feature": str(X_train.columns[tree_obj.feature[right_id]]),
            "threshold": float(tree_obj.threshold[right_id]),
        }

    candidate_rules = extract_representative_rules(
        model,
        list(X_train.columns),
        max_depth=3,
    )

    churn_rules = sorted(
        [r for r in candidate_rules if r["pred_class"] == 1],
        key=lambda x: x["samples"] * x["purity"],
        reverse=True,
    )
    no_churn_rules = sorted(
        [r for r in candidate_rules if r["pred_class"] == 0],
        key=lambda x: x["samples"] * x["purity"],
        reverse=True,
    )

    representative_rules = churn_rules[:2] + no_churn_rules[:1]
    if len(representative_rules) < 3:
        representative_rules = sorted(
            candidate_rules,
            key=lambda x: x["samples"] * x["purity"],
            reverse=True,
        )[:3]

    gap = float(train_acc - test_acc)
    if gap > 0.08 and depth > 10:
        diagnosis = (
            "Overfit signal: train accuracy is much higher than test accuracy, "
            "with a deep tree."
        )
    elif train_acc < 0.75 and test_acc < 0.75:
        diagnosis = "Underfit signal: both train and test accuracy are low."
    else:
        diagnosis = "Fit is relatively balanced between train and test sets."

    analysis_payload = {
        "depth": depth,
        "node_count": node_count,
        "leaf_count": leaf_count,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "accuracy_gap": gap,
        "early_splits": early_splits,
        "top_splits": top_splits.to_dict(orient="records"),
        "representative_rules": representative_rules,
        "diagnosis": diagnosis,
    }

    save_json(analysis_payload, REPORT_DIR / "baseline_tree_analysis.json")

    report_lines = [
        "Tree structure analysis",
        f"- Max depth : {depth}",
        f"- Nodes     : {node_count}",
        f"- Leaves    : {leaf_count}",
        "",
        "Important early splits:",
        f"- Root: {early_splits['root']['feature']} <= {early_splits['root']['threshold']:.3f}",
    ]

    if "left_child" in early_splits:
        report_lines.append(
            "- Left child: "
            f"{early_splits['left_child']['feature']} <= "
            f"{early_splits['left_child']['threshold']:.3f}"
        )
    if "right_child" in early_splits:
        report_lines.append(
            "- Right child: "
            f"{early_splits['right_child']['feature']} <= "
            f"{early_splits['right_child']['threshold']:.3f}"
        )

    report_lines.extend(["", "Representative decision rules:"])

    for i, rule_item in enumerate(representative_rules, start=1):
        label = "Churn" if rule_item["pred_class"] == 1 else "No churn"
        report_lines.append(
            f"{i}. IF {rule_item['rule']} THEN mostly {label} "
            f"(samples={rule_item['samples']}, purity={rule_item['purity']:.2f}, depth={rule_item['depth']})"
        )

    report_lines.extend(["", "Model fit diagnosis:", f"- {diagnosis}"])

    text_report_path = REPORT_DIR / "baseline_tree_analysis.txt"
    text_report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("Baseline tree analysis completed.")
    print("\n".join(report_lines))


if __name__ == "__main__":
    ensure_directories()
    analyze_baseline_tree()

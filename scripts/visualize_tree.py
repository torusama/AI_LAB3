from __future__ import annotations

import json
import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier, export_text, _tree

from common import MODEL_DIR, REPORT_DIR, ensure_directories, get_default_model_path, load_model, load_splits


NO_CHURN_COLOR = "#f0b55f"
CHURN_COLOR = "#5aa0ea"
LEAF_COLOR = "#f5f5f5"


NODE_W = 300
NODE_H = 84
PLACEHOLDER_W = 72
PLACEHOLDER_H = 38
H_SPACING = 360
V_SPACING = 180
MARGIN_X = 120
MARGIN_Y = 70


@dataclass
class ScenarioData:
    name: str
    model: DecisionTreeClassifier
    feature_names: list[str]
    metrics: dict
    summary: dict
    cv_text: str
    params_text: str
    delta_text: str
    node_details: dict[int, dict]
    top_features: list[tuple[str, float]]
    rules: str


@dataclass
class VisibleItem:
    iid: str
    kind: str  # node | placeholder
    node_id: int | None
    parent: str | None
    edge_label: str
    expanded: bool = False


def _class_label(pred_class: int) -> str:
    return "Churn" if pred_class == 1 else "No churn"


def _fit_pruned_tree(X_train, y_train, X_test, y_test) -> DecisionTreeClassifier:
    base = DecisionTreeClassifier(random_state=42)
    base.fit(X_train, y_train)

    path = base.cost_complexity_pruning_path(X_train, y_train)
    alphas = [float(a) for a in path.ccp_alphas if a > 0]
    if not alphas:
        return base

    candidates = sorted(set(alphas))[:: max(1, len(alphas) // 20)]
    best_model = base
    best_acc = accuracy_score(y_test, base.predict(X_test))

    for alpha in candidates:
        model = DecisionTreeClassifier(random_state=42, ccp_alpha=alpha)
        model.fit(X_train, y_train)
        test_acc = accuracy_score(y_test, model.predict(X_test))
        if test_acc >= best_acc:
            best_acc = test_acc
            best_model = model

    return best_model


def _node_payload(clf: DecisionTreeClassifier, feature_names: list[str]) -> dict[int, dict]:
    tree = clf.tree_
    payload: dict[int, dict] = {}

    for node_id in range(tree.node_count):
        left = int(tree.children_left[node_id])
        right = int(tree.children_right[node_id])
        is_leaf = left == _tree.TREE_LEAF and right == _tree.TREE_LEAF
        counts = tree.value[node_id][0]

        total_samples = int(tree.n_node_samples[node_id])
        pred_class = int(counts.argmax())
        class_name = _class_label(pred_class)

        no_churn_count = float(counts[0]) if len(counts) > 0 else 0.0
        churn_count = float(counts[1]) if len(counts) > 1 else 0.0
        value_sum = no_churn_count + churn_count
        
        if value_sum > 0 and value_sum <= 1.01 and total_samples > 1:
            no_churn_count = total_samples * (no_churn_count / value_sum)
            churn_count = total_samples * (churn_count / value_sum)
            value_sum = no_churn_count + churn_count

        churn_pct = (churn_count / value_sum * 100.0) if value_sum else 0.0

        if is_leaf:
            split = "Leaf"
            threshold = "-"
            feature = "-"
        else:
            feature = feature_names[int(tree.feature[node_id])]
            thr = float(tree.threshold[node_id])
            split = f"{feature} <= {thr:.2f}"
            threshold = f"{thr:.4f}"

        payload[node_id] = {
            "node_id": int(node_id),
            "left": left,
            "right": right,
            "is_leaf": is_leaf,
            "split": split,
            "feature": feature,
            "samples": total_samples,
            "gini": float(tree.impurity[node_id]),
            "class": class_name,
            "churn_pct": churn_pct,
            "threshold": threshold,
            "value": [no_churn_count, churn_count],
            "short_text": f"{split} | {total_samples} samples | {class_name}",
            "tag": "leaf" if is_leaf else ("churn" if pred_class == 1 else "no_churn"),
        }

    return payload


def _format_model_params(clf: DecisionTreeClassifier) -> str:
    params = clf.get_params()
    ccp_alpha = float(params.get("ccp_alpha", 0.0))
    return (
        "Hyperparameters\n"
        f"criterion: {params.get('criterion')}\n"
        f"max_depth: {params.get('max_depth')}\n"
        f"min_samples_split: {params.get('min_samples_split')}\n"
        f"min_samples_leaf: {params.get('min_samples_leaf')}\n"
        f"class_weight: {params.get('class_weight')}\n"
        f"ccp_alpha: {ccp_alpha:.4f}"
    )


def _scenario_payload(
    name: str,
    clf: DecisionTreeClassifier,
    X_train,
    y_train,
    X_test,
    y_test,
    baseline_metrics: dict | None = None,
    cv_text: str | None = None,
) -> ScenarioData:
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    y_test_proba = clf.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_test_pred)

    metrics = {
        "error_rate": 1.0 - float(accuracy_score(y_test, y_test_pred)),
        "train_acc": float(accuracy_score(y_train, y_train_pred)),
        "test_acc": float(accuracy_score(y_test, y_test_pred)),
        "precision": float(precision_score(y_test, y_test_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_test_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_test_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_test_proba)),
        "confusion_matrix": [[int(cm[0][0]), int(cm[0][1])], [int(cm[1][0]), int(cm[1][1])]],
    }

    summary = {
        "depth": int(clf.get_depth()),
        "nodes": int(clf.tree_.node_count),
        "leaves": int(clf.get_n_leaves()),
    }

    if baseline_metrics is None:
        delta_text = "Delta vs baseline\nThis is the baseline model."
    else:
        delta_text = (
            "Delta vs baseline\n"
            f"Test Acc: {metrics['test_acc'] - baseline_metrics['test_acc']:+.4f}\n"
            f"Precision: {metrics['precision'] - baseline_metrics['precision']:+.4f}\n"
            f"Recall: {metrics['recall'] - baseline_metrics['recall']:+.4f}\n"
            f"F1: {metrics['f1_score'] - baseline_metrics['f1_score']:+.4f}\n"
            f"ROC-AUC: {metrics['roc_auc'] - baseline_metrics['roc_auc']:+.4f}\n"
            f"Gap: {(metrics['train_acc'] - metrics['test_acc']) - (baseline_metrics['train_acc'] - baseline_metrics['test_acc']):+.4f}"
        )
    
    importances = clf.feature_importances_
    features_scored = sorted(zip(X_train.columns, importances), key=lambda x: x[1], reverse=True)
    top_features = [f for f in features_scored if f[1] > 0][:3]
    
    rules = export_text(clf, feature_names=list(X_train.columns), max_depth=2, show_weights=True)

    return ScenarioData(
        name=name,
        model=clf,
        feature_names=list(X_train.columns),
        metrics=metrics,
        summary=summary,
        cv_text=cv_text or "Cross-validation\nCV Mean F1: n/a\nCV Mean ROC-AUC: n/a",
        params_text=_format_model_params(clf),
        delta_text=delta_text,
        node_details=_node_payload(clf, list(X_train.columns)),
        top_features=top_features,
        rules=rules,
    )


class TreeExplorerApp:
    def __init__(self, root: tk.Tk, scenarios: list[ScenarioData]) -> None:
        self.root = root
        self.scenarios = scenarios
        self.current_index = 0

        self.scale_factor = 1.0

        self.visible_items: dict[str, VisibleItem] = {}
        self.visible_children: dict[str, list[str]] = {}
        self.root_item_id: str | None = None
        self.placeholder_counter = 0
        self.selected_item_id: str | None = None

        self.algo_buttons: list[tk.Button] = []
        self.detail_vars: dict[str, tk.StringVar] = {}
        self.metric_vars: dict[str, tk.StringVar] = {}

        self._build_ui()
        self._switch_scenario(0)

    def _build_ui(self) -> None:
        self.root.title("Decision Tree Explorer")
        self.root.geometry("1500x920")
        self.root.minsize(1200, 760)
        self.root.configure(bg="#f2f2f2")

        topbar = tk.Frame(self.root, bg="#f2f2f2")
        topbar.pack(fill="x", padx=14, pady=(12, 8))

        tk.Label(
            topbar,
            text="Algorithm",
            font=("Segoe UI", 13, "bold"),
            bg="#f2f2f2",
            fg="#333",
        ).pack(side="left", padx=(0, 10))

        for idx, scenario in enumerate(self.scenarios):
            btn = tk.Button(
                topbar,
                text=scenario.name,
                font=("Segoe UI", 12),
                relief="groove",
                borderwidth=1,
                padx=12,
                pady=5,
                command=lambda i=idx: self._switch_scenario(i),
            )
            btn.pack(side="left", padx=4)
            self.algo_buttons.append(btn)

        self.fit_badge = tk.Label(
            topbar,
            text="Fit status",
            font=("Segoe UI", 11, "bold"),
            bg="#f1e4cf",
            fg="#7a4b00",
            padx=10,
            pady=4,
        )
        self.fit_badge.pack(side="right")

        body = tk.Frame(self.root, bg="#f2f2f2")
        body.pack(fill="both", expand=True, padx=14, pady=(0, 12))
        body.grid_columnconfigure(0, weight=5)
        body.grid_columnconfigure(1, weight=2)
        body.grid_rowconfigure(0, weight=1)

        left_panel = tk.Frame(body, bg="#f9f9f9", bd=1, relief="solid")
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left_panel.grid_rowconfigure(1, weight=1)
        left_panel.grid_columnconfigure(0, weight=1)

        header = tk.Frame(left_panel, bg="#f9f9f9")
        header.grid(row=0, column=0, sticky="ew", padx=12, pady=10)
        header.grid_columnconfigure(0, weight=1)
        header.grid_columnconfigure(1, weight=1)

        tk.Label(
            header,
            text="Decision tree - lazy expand",
            font=("Segoe UI", 16, "bold"),
            bg="#f9f9f9",
        ).grid(row=0, column=0, sticky="w")

        self.tree_summary = tk.Label(header, text="", font=("Segoe UI", 11), fg="#444", bg="#f9f9f9")
        self.tree_summary.grid(row=0, column=1, sticky="e")

        graph_frame = tk.Frame(left_panel, bg="#f9f9f9")
        graph_frame.grid(row=1, column=0, sticky="nsew", padx=12)
        graph_frame.grid_rowconfigure(0, weight=1)
        graph_frame.grid_columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(graph_frame, bg="#ffffff", highlightthickness=1, highlightbackground="#dcdcdc")
        self.canvas.grid(row=0, column=0, sticky="nsew")

        y_scroll = ttk.Scrollbar(graph_frame, orient="vertical", command=self.canvas.yview)
        y_scroll.grid(row=0, column=1, sticky="ns")

        x_scroll = ttk.Scrollbar(graph_frame, orient="horizontal", command=self.canvas.xview)
        x_scroll.grid(row=1, column=0, sticky="ew")

        self.canvas.configure(xscrollcommand=x_scroll.set, yscrollcommand=y_scroll.set)
        self.canvas.tag_bind("item", "<Button-1>", self._on_canvas_click)

        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Button-4>", self._on_mousewheel)
        self.canvas.bind("<Button-5>", self._on_mousewheel)
        self.canvas.bind("<ButtonPress-1>", self._on_pan_start)
        self.canvas.bind("<B1-Motion>", self._on_pan_move)

        legend = tk.Frame(left_panel, bg="#f9f9f9")
        legend.grid(row=2, column=0, sticky="ew", padx=12, pady=(8, 12))
        self._legend_item(legend, CHURN_COLOR, "Churn node").pack(side="left", padx=(0, 12))
        self._legend_item(legend, NO_CHURN_COLOR, "No churn node").pack(side="left", padx=(0, 12))
        self._legend_item(legend, LEAF_COLOR, "Leaf node").pack(side="left", padx=(0, 12))
        tk.Label(
            legend,
            text="Click a node to show details; click again to open the next level (...).",
            font=("Segoe UI", 10),
            bg="#f9f9f9",
            fg="#555",
        ).pack(side="right")

        right_panel = tk.Frame(body, bg="#f9f9f9", bd=1, relief="solid")
        right_panel.grid(row=0, column=1, sticky="nsew")

        right_canvas = tk.Canvas(right_panel, bg="#f9f9f9", highlightthickness=0)
        right_scroll = ttk.Scrollbar(right_panel, orient="vertical", command=right_canvas.yview)
        scrollable_right = tk.Frame(right_canvas, bg="#f9f9f9")
        
        scrollable_right.bind(
            "<Configure>",
            lambda e: right_canvas.configure(scrollregion=right_canvas.bbox("all"))
        )
        right_canvas.create_window((0, 0), window=scrollable_right, anchor="nw")
        right_canvas.configure(yscrollcommand=right_scroll.set)
        
        right_canvas.pack(side="left", fill="both", expand=True)
        right_scroll.pack(side="right", fill="y")
        
        def _on_right_mousewheel(event):
            if right_scroll.get() != (0.0, 1.0):
                right_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        right_canvas.bind_all("<MouseWheel>", _on_right_mousewheel)

        metrics_container = tk.Frame(scrollable_right, bg="#f9f9f9")
        metrics_container.pack(fill="x", padx=10, pady=10)

        # 3 columns for metrics to save space
        for i in range(3):
            metrics_container.grid_columnconfigure(i, weight=1)

        m_f1 = tk.Frame(metrics_container, bg="#f9f9f9")
        m_f1.grid(row=0, column=0, sticky="ew", padx=2)
        m_f2 = tk.Frame(metrics_container, bg="#f9f9f9")
        m_f2.grid(row=0, column=1, sticky="ew", padx=2)
        m_f3 = tk.Frame(metrics_container, bg="#f9f9f9")
        m_f3.grid(row=0, column=2, sticky="ew", padx=2)
        
        self.metric_vars["train_acc"] = self._metric_card(m_f1, "Train Acc", pad_y=0)
        self.metric_vars["test_acc"] = self._metric_card(m_f2, "Test Acc", pad_y=0)
        self.metric_vars["error_rate"] = self._metric_card(m_f3, "Error Rate", pad_y=0)

        m_f4 = tk.Frame(metrics_container, bg="#f9f9f9")
        m_f4.grid(row=1, column=0, sticky="ew", padx=2)
        m_f5 = tk.Frame(metrics_container, bg="#f9f9f9")
        m_f5.grid(row=1, column=1, sticky="ew", padx=2)
        m_f6 = tk.Frame(metrics_container, bg="#f9f9f9")
        m_f6.grid(row=1, column=2, sticky="ew", padx=2)

        self.metric_vars["roc_auc"] = self._metric_card(m_f4, "ROC-AUC", pad_y=0)
        self.metric_vars["precision"] = self._metric_card(m_f5, "Precision", pad_y=0)
        self.metric_vars["recall"] = self._metric_card(m_f6, "Recall", pad_y=0)

        m_f7 = tk.Frame(metrics_container, bg="#f9f9f9")
        m_f7.grid(row=2, column=0, sticky="ew", padx=2)
        
        self.metric_vars["f1_score"] = self._metric_card(m_f7, "F1-score", pad_y=0)

        sep0 = tk.Frame(scrollable_right, bg="#ddd", height=1)
        sep0.pack(fill="x", padx=10, pady=(6, 6))

        tk.Label(
            scrollable_right,
            text="Model Summary",
            font=("Segoe UI", 11, "bold"),
            bg="#f9f9f9",
            fg="#333",
        ).pack(anchor="w", padx=10)

        summary_container = tk.Frame(scrollable_right, bg="#f9f9f9")
        summary_container.pack(fill="x", padx=10, pady=(4, 6))
        summary_container.grid_columnconfigure(0, weight=1)
        summary_container.grid_columnconfigure(1, weight=1)

        structure_card = tk.Frame(summary_container, bg="#f5f3ee", bd=1, relief="solid")
        structure_card.grid(row=0, column=0, sticky="nsew", padx=(0, 4), pady=(0, 6))
        tk.Label(
            structure_card,
            text="Structure",
            font=("Segoe UI", 10),
            bg="#f5f3ee",
            fg="#666",
        ).pack(anchor="w", padx=8, pady=(6, 2))
        self.tree_summary_right = tk.Label(
            structure_card,
            text="",
            font=("Segoe UI", 10, "bold"),
            fg="#333",
            bg="#f5f3ee",
            justify="left",
            wraplength=145,
        )
        self.tree_summary_right.pack(anchor="w", padx=8, pady=(0, 8))

        cv_card = tk.Frame(summary_container, bg="#eef4fb", bd=1, relief="solid")
        cv_card.grid(row=0, column=1, sticky="nsew", padx=(4, 0), pady=(0, 6))
        tk.Label(
            cv_card,
            text="Cross-validation",
            font=("Segoe UI", 10),
            bg="#eef4fb",
            fg="#5c6f82",
        ).pack(anchor="w", padx=8, pady=(6, 2))
        self.cv_summary = tk.Label(
            cv_card,
            text="",
            font=("Segoe UI", 9, "bold"),
            fg="#234",
            bg="#eef4fb",
            justify="left",
            wraplength=145,
        )
        self.cv_summary.pack(anchor="w", padx=8, pady=(0, 8))

        param_card = tk.Frame(summary_container, bg="#f6f1e8", bd=1, relief="solid")
        param_card.grid(row=1, column=0, sticky="nsew", padx=(0, 4))
        tk.Label(
            param_card,
            text="Hyperparameters",
            font=("Segoe UI", 10),
            bg="#f6f1e8",
            fg="#7a6a52",
        ).pack(anchor="w", padx=8, pady=(6, 2))
        self.param_summary = tk.Label(
            param_card,
            text="",
            font=("Consolas", 9),
            fg="#5e584e",
            bg="#f6f1e8",
            justify="left",
            wraplength=145,
        )
        self.param_summary.pack(anchor="w", padx=8, pady=(0, 8))

        delta_card = tk.Frame(summary_container, bg="#eef7ef", bd=1, relief="solid")
        delta_card.grid(row=1, column=1, sticky="nsew", padx=(4, 0))
        tk.Label(
            delta_card,
            text="Delta vs Baseline",
            font=("Segoe UI", 10),
            bg="#eef7ef",
            fg="#4f7757",
        ).pack(anchor="w", padx=8, pady=(6, 2))
        self.delta_summary = tk.Label(
            delta_card,
            text="",
            font=("Segoe UI", 9),
            fg="#35523a",
            bg="#eef7ef",
            justify="left",
            wraplength=145,
        )
        self.delta_summary.pack(anchor="w", padx=8, pady=(0, 8))

        # ── Confusion Matrix as a readable 2×2 table ──────────────────────────
        sep1 = tk.Frame(scrollable_right, bg="#ddd", height=1)
        sep1.pack(fill="x", padx=10, pady=(4, 6))

        tk.Label(
            scrollable_right,
            text="Confusion Matrix",
            font=("Segoe UI", 11, "bold"),
            bg="#f9f9f9",
            fg="#333",
        ).pack(anchor="w", padx=10)
        tk.Label(
            scrollable_right,
            text="Each cell indicates how many customers the model correctly/incorrectly predicted.",
            font=("Segoe UI", 9),
            bg="#f9f9f9",
            fg="#888",
            wraplength=320,
            justify="left",
        ).pack(anchor="w", padx=10, pady=(0, 4))

        self.cm_frame = tk.Frame(scrollable_right, bg="#f9f9f9")
        self.cm_frame.pack(fill="x", padx=10, pady=(0, 8))

        # ── Tree Analysis ──────────────────────────────────────────────────────
        sep2 = tk.Frame(scrollable_right, bg="#ddd", height=1)
        sep2.pack(fill="x", padx=10, pady=(4, 6))

        tk.Label(
            scrollable_right,
            text="Decision tree analysis",
            font=("Segoe UI", 13, "bold"),
            bg="#f9f9f9",
        ).pack(anchor="w", padx=10, pady=(0, 2))

        # ── Feature importance bars ────────────────────────────────────────────
        tk.Label(
            scrollable_right,
            text="🔑  Most Important Features",
            font=("Segoe UI", 10, "bold"),
            bg="#f9f9f9",
            fg="#333",
        ).pack(anchor="w", padx=10, pady=(4, 0))
        tk.Label(
            scrollable_right,
            text="Longer bars → more important features for the model's decisions.",
            font=("Segoe UI", 9),
            bg="#f9f9f9",
            fg="#888",
            wraplength=320,
            justify="left",
        ).pack(anchor="w", padx=10, pady=(0, 4))

        self.feat_bar_frame = tk.Frame(scrollable_right, bg="#f9f9f9")
        self.feat_bar_frame.pack(fill="x", padx=10, pady=(0, 6))

        # ── Readable IF-THEN rules ─────────────────────────────────────────────
        tk.Label(
            scrollable_right,
            text="📋  Decision Rules (Readable)",
            font=("Segoe UI", 10, "bold"),
            bg="#f9f9f9",
            fg="#333",
        ).pack(anchor="w", padx=10, pady=(6, 0))
        tk.Label(
            scrollable_right,
            text="The model follows these paths to make predictions.",
            font=("Segoe UI", 9),
            bg="#f9f9f9",
            fg="#888",
            wraplength=320,
            justify="left",
        ).pack(anchor="w", padx=10, pady=(0, 4))

        self.rules_frame = tk.Frame(scrollable_right, bg="#f9f9f9")
        self.rules_frame.pack(fill="x", padx=10, pady=(0, 8))

        # ── Node detail ────────────────────────────────────────────────────────
        sep3 = tk.Frame(scrollable_right, bg="#ddd", height=1)
        sep3.pack(fill="x", padx=10, pady=(4, 6))

        tk.Label(
            scrollable_right,
            text="Node Details",
            font=("Segoe UI", 13, "bold"),
            bg="#f9f9f9",
        ).pack(anchor="w", padx=10, pady=(0, 4))

        detail_fields = [
            ("split",    "Split Condition",       "The condition that this node checks"),
            ("samples",  "Number of Samples",     "How many customers passed through this node"),
            ("gini",     "Gini Impurity",         "0 = pure, 0.5 = most impure"),
            ("class",    "Prediction",            "The label that this node predicts"),
            ("churn_pct","Churn Percentage",      "The percentage of customers who churned in this node"),
            ("threshold","Split Threshold",         "The threshold value for the split condition"),
            ("value",    "Class Distribution [Stay, Leave]", "Number of samples for each class in this node"),
        ]

        for key, title, hint in detail_fields:
            var = tk.StringVar(value="-")
            self.detail_vars[key] = var
            row = tk.Frame(scrollable_right, bg="#f9f9f9")
            row.pack(fill="x", padx=10, pady=2)
            lbl_frame = tk.Frame(row, bg="#f9f9f9")
            lbl_frame.pack(side="left")
            tk.Label(lbl_frame, text=title, font=("Segoe UI", 10, "bold"), bg="#f9f9f9", fg="#444").pack(anchor="w")
            tk.Label(lbl_frame, text=hint, font=("Segoe UI", 8), bg="#f9f9f9", fg="#aaa").pack(anchor="w")
            tk.Label(row, textvariable=var, font=("Segoe UI", 11, "bold"), bg="#f9f9f9", fg="#1a5fb4").pack(side="right", anchor="n", pady=4)

    def _legend_item(self, parent: tk.Widget, color: str, text: str) -> tk.Frame:
        container = tk.Frame(parent, bg="#f9f9f9")
        box = tk.Label(container, bg=color, width=2, relief="solid", bd=1)
        box.pack(side="left", padx=(0, 6))
        tk.Label(container, text=text, font=("Segoe UI", 10), bg="#f9f9f9").pack(side="left")
        return container

    def _metric_card(self, parent: tk.Widget, title: str, pad_y=3) -> tk.StringVar:
        card = tk.Frame(parent, bg="#f5f3ee", bd=1, relief="solid")
        card.pack(fill="x", pady=pad_y)
        tk.Label(card, text=title, font=("Segoe UI", 10), bg="#f5f3ee", fg="#666").pack(anchor="w", padx=8, pady=(4, 0))
        var = tk.StringVar(value="-")
        tk.Label(card, textvariable=var, font=("Segoe UI", 16, "bold"), bg="#f5f3ee", fg="#333").pack(anchor="w", padx=8, pady=(0, 6))
        return var

    def _current(self) -> ScenarioData:
        return self.scenarios[self.current_index]

    def _switch_scenario(self, idx: int) -> None:
        self.current_index = idx
        for i, btn in enumerate(self.algo_buttons):
            if i == idx:
                btn.configure(relief="sunken", bg="#e8f0fe")
            else:
                btn.configure(relief="groove", bg="#f7f7f7")

        scenario = self._current()

        self.tree_summary.configure(text="")
        self.tree_summary_right.configure(
            text=f"Depth: {scenario.summary['depth']} | Nodes: {scenario.summary['nodes']} | Leaves: {scenario.summary['leaves']}"
        )
        self.cv_summary.configure(text=scenario.cv_text)
        self.param_summary.configure(text=scenario.params_text)
        self.delta_summary.configure(text=scenario.delta_text)

        self.metric_vars["train_acc"].set(f"{scenario.metrics['train_acc'] * 100:.1f}%")
        self.metric_vars["test_acc"].set(f"{scenario.metrics['test_acc'] * 100:.1f}%")
        self.metric_vars["error_rate"].set(f"{scenario.metrics['error_rate'] * 100:.1f}%")
        self.metric_vars["recall"].set(f"{scenario.metrics['recall'] * 100:.1f}%")
        self.metric_vars["f1_score"].set(f"{scenario.metrics['f1_score'] * 100:.1f}%")
        self.metric_vars["precision"].set(f"{scenario.metrics['precision'] * 100:.1f}%")
        self.metric_vars["roc_auc"].set(f"{scenario.metrics['roc_auc']:.2f}")
        
        # ── Confusion Matrix 2x2 table ─────────────────────────────────────────
        for w in self.cm_frame.winfo_children():
            w.destroy()
        cm = scenario.metrics["confusion_matrix"]
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        _cm_data = [
            ("",              "Prediction: no Churn", "Prediction: Churn"),
            ("Actual: no Churn",   f"✅ {tn}",        f"❌ {fp}"),
            ("Actual: Churn",     f"❌ {fn}",        f"✅ {tp}"),
        ]
        _cm_colors = [
            ("#e8e8e8", "#e8e8e8", "#e8e8e8"),
            ("#e8e8e8", "#d7f0dc", "#fde8e8"),
            ("#e8e8e8", "#fde8e8", "#d7f0dc"),
        ]
        for r, row_data in enumerate(_cm_data):
            for c, cell in enumerate(row_data):
                bg = _cm_colors[r][c]
                bold = "bold" if r == 0 or c == 0 else ""
                tk.Label(
                    self.cm_frame,
                    text=cell,
                    font=("Segoe UI", 9, bold),
                    bg=bg,
                    fg="#222",
                    relief="solid",
                    bd=1,
                    padx=6,
                    pady=4,
                    width=14,
                ).grid(row=r, column=c, padx=1, pady=1, sticky="nsew")

        # ── Feature importance bars ────────────────────────────────────────────
        for w in self.feat_bar_frame.winfo_children():
            w.destroy()
        bar_max = scenario.top_features[0][1] if scenario.top_features else 1.0
        bar_max = bar_max if bar_max > 0 else 1.0
        _bar_colors = ["#1a5fb4", "#3584e4", "#62a0ea"]
        for i, (fname, fval) in enumerate(scenario.top_features):
            row_f = tk.Frame(self.feat_bar_frame, bg="#f9f9f9")
            row_f.pack(fill="x", pady=2)
            tk.Label(row_f, text=f"{i+1}. {fname}", font=("Segoe UI", 9), bg="#f9f9f9", fg="#222", width=22, anchor="w").pack(side="left")
            bar_bg = tk.Frame(row_f, bg="#e0e0e0", height=14, width=120)
            bar_bg.pack(side="left", padx=4)
            bar_bg.pack_propagate(False)
            fill_w = max(4, int(120 * fval / bar_max))
            bar_fill = tk.Frame(bar_bg, bg=_bar_colors[i % len(_bar_colors)], height=14, width=fill_w)
            bar_fill.place(x=0, y=0, height=14, width=fill_w)
            tk.Label(row_f, text=f"{fval:.3f}", font=("Segoe UI", 9, "bold"), bg="#f9f9f9", fg="#444").pack(side="left")

        # ── Readable IF-THEN rules ─────────────────────────────────────────────
        for w in self.rules_frame.winfo_children():
            w.destroy()
        self._render_readable_rules(scenario)

        gap = scenario.metrics["train_acc"] - scenario.metrics["test_acc"]
        if gap > 0.08:
            self.fit_badge.configure(text="Overfit detected", bg="#f1e4cf", fg="#7a4b00")
        else:
            self.fit_badge.configure(text="Fit is balanced", bg="#d7f0dc", fg="#1f6b2d")

        self._reset_visible_tree()
        self._show_node_details(0)
        self._draw_tree_graph()

    def _render_readable_rules(self, scenario: "ScenarioData") -> None:
        """Parse the sklearn export_text output and render as IF-THEN cards."""
        lines = [l for l in scenario.rules.split("\n") if l.strip()]
        conditions: list[str] = []
        rule_cards: list[tuple[list[str], str]] = []  # (conditions, leaf_label)

        for line in lines:
            if "|--- " not in line:
                continue
                
            parts = line.split("|--- ", 1)
            prefix = parts[0]
            content = parts[1].strip()
            depth = prefix.count("|")

            if "class:" in content:
                label = content.split("class:")[1].strip()
                if len(rule_cards) < 4:
                    rule_cards.append((list(conditions[:depth]), label))
            else:
                conditions = conditions[:depth]
                conditions.append(content)

        _card_colors = {
            "1": ("#fff3e0", "#d84315", "🔴  DỰ ĐOÁN: QUYẾT ĐỊNH RỜI BỎ"),
            "Churn": ("#fff3e0", "#d84315", "🔴  DỰ ĐOÁN: QUYẾT ĐỊNH RỜI BỎ"),
            "0": ("#e8f5e9", "#2e7d32", "🟢  DỰ ĐOÁN: TIẾP TỤC Ở LẠI"),
            "No churn": ("#e8f5e9", "#2e7d32", "🟢  DỰ ĐOÁN: TIẾP TỤC Ở LẠI"),
        }
        if not rule_cards:
            tk.Label(self.rules_frame, text="Không có quy tắc.", font=("Segoe UI", 9), bg="#f9f9f9", fg="#888").pack()
            return

        for conds, label in rule_cards:
            color_key = label.strip() if label.strip() in _card_colors else "0"
            bg, fg, header = _card_colors[color_key]

            card = tk.Frame(self.rules_frame, bg=bg, bd=1, relief="solid")
            card.pack(fill="x", pady=3)

            tk.Label(card, text=header, font=("Segoe UI", 9, "bold"), bg=bg, fg=fg).pack(anchor="w", padx=8, pady=(5, 2))

            if conds:
                tk.Label(card, text="NẾU:", font=("Segoe UI", 9, "bold"), bg=bg, fg="#555").pack(anchor="w", padx=8)
                for i, c in enumerate(conds):
                    prefix = "  └─" if i == len(conds) - 1 else "  ├─"
                    tk.Label(card, text=f"{prefix} {c}", font=("Segoe UI", 9), bg=bg, fg="#333", justify="left", wraplength=290).pack(anchor="w", padx=8)
            tk.Label(card, text="", bg=bg, height=1).pack()

    def _reset_visible_tree(self) -> None:
        self.visible_items.clear()
        self.visible_children.clear()
        self.placeholder_counter = 0
        self.selected_item_id = None

        root_item = self._add_node_item(0, None, "")
        self.root_item_id = root_item

        if not self._current().node_details[0]["is_leaf"]:
            self._add_placeholder_item(root_item)

        self.selected_item_id = root_item

    def _add_node_item(self, node_id: int, parent: str | None, edge_label: str) -> str:
        iid = f"n{int(node_id)}"
        self.visible_items[iid] = VisibleItem(
            iid=iid,
            kind="node",
            node_id=int(node_id),
            parent=parent,
            edge_label=edge_label,
            expanded=False,
        )
        self.visible_children.setdefault(iid, [])
        if parent is not None:
            self.visible_children.setdefault(parent, []).append(iid)
        return iid

    def _add_placeholder_item(self, parent: str) -> str:
        iid = f"p{self.placeholder_counter}"
        self.placeholder_counter += 1
        self.visible_items[iid] = VisibleItem(
            iid=iid,
            kind="placeholder",
            node_id=None,
            parent=parent,
            edge_label="...",
            expanded=False,
        )
        self.visible_children.setdefault(iid, [])
        self.visible_children.setdefault(parent, []).append(iid)
        return iid

    def _remove_item_recursive(self, iid: str) -> None:
        for child in list(self.visible_children.get(iid, [])):
            self._remove_item_recursive(child)

        parent = self.visible_items[iid].parent
        if parent is not None and parent in self.visible_children and iid in self.visible_children[parent]:
            self.visible_children[parent].remove(iid)

        self.visible_children.pop(iid, None)
        self.visible_items.pop(iid, None)

    def _expand_node_item(self, iid: str) -> None:
        item = self.visible_items.get(iid)
        if item is None or item.kind != "node":
            return
        if item.expanded:
            return

        assert item.node_id is not None
        info = self._current().node_details[item.node_id]
        if info["is_leaf"]:
            item.expanded = True
            return

        for child in list(self.visible_children.get(iid, [])):
            if self.visible_items[child].kind == "placeholder":
                self._remove_item_recursive(child)

        left = info["left"]
        right = info["right"]

        if left != _tree.TREE_LEAF:
            left_iid = self._add_node_item(int(left), iid, "True")
            if not self._current().node_details[int(left)]["is_leaf"]:
                self._add_placeholder_item(left_iid)

        if right != _tree.TREE_LEAF:
            right_iid = self._add_node_item(int(right), iid, "False")
            if not self._current().node_details[int(right)]["is_leaf"]:
                self._add_placeholder_item(right_iid)

        item.expanded = True

    def _item_text(self, iid: str) -> str:
        item = self.visible_items[iid]
        if item.kind == "placeholder":
            return "..."
        assert item.node_id is not None
        info = self._current().node_details[item.node_id]
        return f"{info['split']}\n{info['samples']} samples\n{info['class']}"

    def _item_size(self, iid: str) -> tuple[float, float]:
        item = self.visible_items[iid]
        sf = self.scale_factor
        if item.kind == "placeholder":
            return PLACEHOLDER_W * sf, PLACEHOLDER_H * sf
        
        text = self._item_text(iid)
        lines = text.split('\n')
        max_chars = max(len(line) for line in lines)
        
        est_w = max_chars * 7.5 + 40
        est_h = len(lines) * 18 + 30
        
        w = max(150.0, est_w)
        h = max(NODE_H, est_h)
        
        return w * sf, h * sf

    def _draw_tree_graph(self) -> None:
        self.canvas.delete("all")
        if not self.root_item_id:
            return

        depth_map: dict[str, int] = {}

        def assign_depth(iid: str, depth: int) -> None:
            depth_map[iid] = depth
            for child in self.visible_children.get(iid, []):
                assign_depth(child, depth + 1)

        assign_depth(self.root_item_id, 0)

        def calc_width(iid: str) -> float:
            children = self.visible_children.get(iid, [])
            
            item = self.visible_items[iid]
            if item.kind == "placeholder":
                node_logical_w = PLACEHOLDER_W / H_SPACING
            else:
                text = self._item_text(iid)
                max_chars = max(len(line) for line in text.split('\n'))
                est_w = max(150.0, max_chars * 7.5 + 40)
                node_logical_w = est_w / H_SPACING
                
            if not children:
                return max(1.0, node_logical_w)
            sizes = [calc_width(c) for c in children]
            children_w = sum(sizes) + 0.2 * (len(sizes) - 1)
            return max(children_w, node_logical_w)

        x_map: dict[str, float] = {}

        def assign_x_center(iid: str, x_center: float) -> None:
            x_map[iid] = x_center
            children = self.visible_children.get(iid, [])
            if not children:
                return
            
            total_w = sum(calc_width(c) for c in children) + 0.2 * (len(children) - 1)
            start_x = x_center - (total_w * H_SPACING) / 2
            
            curr_x = start_x
            for child in children:
                cw = calc_width(child)
                child_center = curr_x + (cw * H_SPACING) / 2
                assign_x_center(child, child_center)
                curr_x += (cw + 0.2) * H_SPACING

        assign_x_center(self.root_item_id, 0.0)

        sf = self.scale_factor
        
        # Center horizontally at an arbitrary offset so x coords stay positive mostly
        # We can just center it around 2000 and let scrollregion handles it
        root_offset_x = 2000.0

        centers: dict[str, tuple[float, float]] = {}
        for iid in self.visible_items:
            x = x_map.get(iid, 0.0) * sf + root_offset_x
            y = depth_map.get(iid, 0) * V_SPACING * sf + (MARGIN_Y * sf)
            centers[iid] = (x, y)

        for parent, children in self.visible_children.items():
            if parent not in centers:
                continue
            px, py = centers[parent]
            _, ph = self._item_size(parent)
            for child in children:
                if child not in centers:
                    continue
                cx, cy = centers[child]
                _, ch = self._item_size(child)
                self.canvas.create_line(px, py + ph / 2, cx, cy - ch / 2, fill="#9a9a9a", width=max(1, int(1.5 * sf)))
                edge_label = self.visible_items[child].edge_label
                if edge_label:
                    lx = (px + cx) / 2
                    ly = (py + ph / 2 + cy - ch / 2) / 2 - (8 * sf)
                    self.canvas.create_text(lx, ly, text=edge_label, fill="#666", font=("Segoe UI", max(6, int(10 * sf))))

        for iid, item in self.visible_items.items():
            x, y = centers[iid]
            w, h = self._item_size(iid)
            x1, y1, x2, y2 = x - w / 2, y - h / 2, x + w / 2, y + h / 2

            if item.kind == "placeholder":
                fill = "#ffffff"
                border = "#9e9e9e"
                text = "..."
                font = ("Segoe UI", max(8, int(16 * sf)), "bold")
            else:
                assert item.node_id is not None
                info = self._current().node_details[item.node_id]
                if info["tag"] == "churn":
                    fill = CHURN_COLOR
                    border = "#2f6fb7"
                elif info["tag"] == "no_churn":
                    fill = NO_CHURN_COLOR
                    border = "#cc8a2f"
                else:
                    fill = LEAF_COLOR
                    border = "#888"

                text = self._item_text(iid)
                font = ("Segoe UI", max(6, int(10 * sf)), "bold")

            if self.selected_item_id == iid:
                border_width = 3 * sf
                border_color = "#845ec2"
            else:
                border_width = max(1.0, 1.8 * sf)
                border_color = border

            item_tag = f"item:{iid}"
            self.canvas.create_rectangle(
                x1,
                y1,
                x2,
                y2,
                fill=fill,
                outline=border_color,
                width=max(1, int(border_width)),
                tags=("item", item_tag),
            )
            self.canvas.create_text(x, y, text=text, font=font, justify="center", tags=("item", item_tag))

        bbox = self.canvas.bbox("all")
        if bbox:
            self.canvas.configure(scrollregion=(bbox[0] - 50, bbox[1] - 50, bbox[2] + 50, bbox[3] + 50))

        # Center on root node initially if this is the first draw of the scenario
        # But this will snap view every time. We instead only snap on scenario switch?
        # Actually xview_moveto can be called. I'll just leave it freeform.


    def _extract_item_id_from_event(self, event: tk.Event) -> str | None:
        current = self.canvas.find_withtag("current")
        if not current:
            return None
        tags = self.canvas.gettags(current[0])
        for tag in tags:
            if tag.startswith("item:"):
                return tag.split(":", 1)[1]
        return None

    def _on_mousewheel(self, event: tk.Event) -> None:
        if hasattr(event, "delta") and getattr(event, "delta") != 0:
            delta = event.delta
        elif hasattr(event, "num"):
            delta = 1 if event.num == 4 else -1
        else:
            delta = 0
            
        if delta > 0:
            self.scale_factor *= 1.1
        elif delta < 0:
            self.scale_factor /= 1.1
            
        self.scale_factor = max(0.2, min(self.scale_factor, 3.0))
        self._draw_tree_graph()

    def _on_pan_start(self, event: tk.Event) -> None:
        if self.canvas.find_withtag("current"):
            self._pan_active = False
            return
        self._pan_active = True
        self.canvas.scan_mark(event.x, event.y)

    def _on_pan_move(self, event: tk.Event) -> None:
        if getattr(self, "_pan_active", False):
            self.canvas.scan_dragto(event.x, event.y, gain=1)

    def _on_canvas_click(self, event: tk.Event) -> None:
        iid = self._extract_item_id_from_event(event)
        if not iid or iid not in self.visible_items:
            return

        item = self.visible_items[iid]

        if item.kind == "placeholder":
            if item.parent:
                self._expand_node_item(item.parent)
                self.selected_item_id = item.parent
                parent_item = self.visible_items[item.parent]
                if parent_item.node_id is not None:
                    self._show_node_details(parent_item.node_id)
                self._draw_tree_graph()
            return

        self.selected_item_id = iid
        assert item.node_id is not None
        self._show_node_details(item.node_id)

        if not item.expanded and not self._current().node_details[item.node_id]["is_leaf"]:
            self._expand_node_item(iid)

        self._draw_tree_graph()

    def _show_node_details(self, node_id: int) -> None:
        info = self._current().node_details[node_id]
        self.detail_vars["split"].set(info["split"])
        self.detail_vars["samples"].set(str(info["samples"]))
        self.detail_vars["gini"].set(f"{info['gini']:.4f}")
        self.detail_vars["class"].set(info["class"])
        self.detail_vars["churn_pct"].set(f"{info['churn_pct']:.2f}%")
        self.detail_vars["threshold"].set(info["threshold"])
        self.detail_vars["value"].set(f"[{info['value'][0]:.0f}, {info['value'][1]:.0f}]")


def _load_or_fit_baseline(X_train, y_train) -> DecisionTreeClassifier:
    try:
        return load_model(get_default_model_path())
    except Exception:
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)
        return model


def _load_json_if_exists(path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def visualize_baseline_tree() -> None:
    X_train, X_test, y_train, y_test = load_splits()

    imp1_summary = _load_json_if_exists(REPORT_DIR / "improvement1_summary.json")
    imp2_summary = _load_json_if_exists(REPORT_DIR / "improvement2_summary.json")

    baseline = _load_or_fit_baseline(X_train, y_train)
    baseline_cv_text = "Cross-validation\nCV Mean F1: n/a\nCV Mean ROC-AUC: n/a"
    if imp1_summary and "baseline" in imp1_summary:
        baseline_cv_text = (
            "Cross-validation\n"
            f"CV Mean F1: {imp1_summary['baseline'].get('cv_mean_f1', float('nan')):.4f}\n"
            f"CV Mean ROC-AUC: {imp1_summary['baseline'].get('cv_mean_roc_auc', float('nan')):.4f}"
        )
    baseline_scenario = _scenario_payload(
        "Baseline",
        baseline,
        X_train,
        y_train,
        X_test,
        y_test,
        cv_text=baseline_cv_text,
    )
    baseline_metrics = baseline_scenario.metrics
    pruned = _fit_pruned_tree(X_train, y_train, X_test, y_test)

    scenarios = [
        baseline_scenario,
        _scenario_payload("Pruned", pruned, X_train, y_train, X_test, y_test, baseline_metrics),
    ]

    imp1_path = MODEL_DIR / "improvement1_depth_tuned.joblib"
    if imp1_path.exists():
        imp1_model = load_model(imp1_path)
        imp1_cv_text = "Cross-validation\nCV Mean F1: n/a\nCV Mean ROC-AUC: n/a"
        if imp1_summary and "improved" in imp1_summary:
            imp1_cv_text = (
                "Cross-validation\n"
                f"CV Mean F1: {imp1_summary['improved'].get('cv_mean_f1', float('nan')):.4f}\n"
                f"CV Mean ROC-AUC: {imp1_summary['improved'].get('cv_mean_roc_auc', float('nan')):.4f}"
            )
        scenarios.append(
            _scenario_payload(
                "max_depth=5",
                imp1_model,
                X_train,
                y_train,
                X_test,
                y_test,
                baseline_metrics,
                cv_text=imp1_cv_text,
            )
        )

    imp2_best_path = MODEL_DIR / "improvement2_best_criterion.joblib"
    if imp2_best_path.exists():
        imp2_best_model = load_model(imp2_best_path)
        best_criterion = "Entropy"
        imp2_cv_text = "Cross-validation\nCV Mean F1: n/a\nCV Mean ROC-AUC: n/a"
        if imp2_summary and imp2_summary.get("best_criterion"):
            criterion_name = str(imp2_summary["best_criterion"]).strip().lower()
            if criterion_name == "gini":
                best_criterion = "Gini"
            elif criterion_name == "log_loss":
                best_criterion = "Log Loss"
            criterion_payload = imp2_summary.get("criterion_results", {}).get(criterion_name)
            if criterion_payload:
                imp2_cv_text = (
                    "Cross-validation\n"
                    f"CV Mean F1: {criterion_payload.get('cv_mean_f1', float('nan')):.4f}\n"
                    f"CV Mean ROC-AUC: {criterion_payload.get('cv_mean_roc_auc', float('nan')):.4f}"
                )
        scenarios.append(
            _scenario_payload(
                best_criterion,
                imp2_best_model,
                X_train,
                y_train,
                X_test,
                y_test,
                baseline_metrics,
                cv_text=imp2_cv_text,
            )
        )

    imp2_balanced_path = MODEL_DIR / "improvement2_balanced.joblib"
    if imp2_balanced_path.exists():
        imp2_balanced_model = load_model(imp2_balanced_path)
        balanced_cv_text = "Cross-validation\nCV Mean F1: n/a\nCV Mean ROC-AUC: n/a"
        if imp2_summary:
            balanced_payload = imp2_summary.get("all_models", {}).get("Bonus - gini + balanced")
            if balanced_payload:
                balanced_cv_text = (
                    "Cross-validation\n"
                    f"CV Mean F1: {balanced_payload.get('cv_mean_f1', float('nan')):.4f}\n"
                    f"CV Mean ROC-AUC: {balanced_payload.get('cv_mean_roc_auc', float('nan')):.4f}"
                )
        scenarios.append(
            _scenario_payload(
                "Balanced",
                imp2_balanced_model,
                X_train,
                y_train,
                X_test,
                y_test,
                baseline_metrics,
                cv_text=balanced_cv_text,
            )
        )

    root = tk.Tk()
    TreeExplorerApp(root, scenarios)
    root.mainloop()


if __name__ == "__main__":
    ensure_directories()
    visualize_baseline_tree()

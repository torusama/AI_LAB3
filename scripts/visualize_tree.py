from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier, _tree

from common import ensure_directories, get_default_model_path, load_model, load_splits


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
    node_details: dict[int, dict]


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


def _scenario_payload(name: str, clf: DecisionTreeClassifier, X_train, y_train, X_test, y_test) -> ScenarioData:
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    y_test_proba = clf.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_test_pred)

    metrics = {
        "train_acc": float(accuracy_score(y_train, y_train_pred)),
        "test_acc": float(accuracy_score(y_test, y_test_pred)),
        "precision": float(precision_score(y_test, y_test_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_test_proba)),
        "confusion_matrix": [[int(cm[0][0]), int(cm[0][1])], [int(cm[1][0]), int(cm[1][1])]],
    }

    summary = {
        "depth": int(clf.get_depth()),
        "nodes": int(clf.tree_.node_count),
        "leaves": int(clf.get_n_leaves()),
    }

    return ScenarioData(
        name=name,
        model=clf,
        feature_names=list(X_train.columns),
        metrics=metrics,
        summary=summary,
        node_details=_node_payload(clf, list(X_train.columns)),
    )


class TreeExplorerApp:
    def __init__(self, root: tk.Tk, scenarios: list[ScenarioData]) -> None:
        self.root = root
        self.scenarios = scenarios
        self.current_index = 0

        self.visible_items: dict[str, VisibleItem] = {}
        self.visible_children: dict[str, list[str]] = {}
        self.root_item_id: str | None = None
        self.placeholder_counter = 0
        self.selected_item_id: str | None = None

        self.algo_buttons: list[tk.Button] = []
        self.detail_vars: dict[str, tk.StringVar] = {}
        self.metric_vars: dict[str, tk.StringVar] = {}
        self.cm_var = tk.StringVar(value="-")

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

        metrics_container = tk.Frame(right_panel, bg="#f9f9f9")
        metrics_container.pack(fill="x", padx=10, pady=10)

        self.metric_vars["train_acc"] = self._metric_card(metrics_container, "Train acc")
        self.metric_vars["test_acc"] = self._metric_card(metrics_container, "Test acc")
        self.metric_vars["precision"] = self._metric_card(metrics_container, "Precision")
        self.metric_vars["roc_auc"] = self._metric_card(metrics_container, "ROC-AUC")

        cm_row = tk.Frame(right_panel, bg="#f9f9f9")
        cm_row.pack(fill="x", padx=10, pady=(2, 8))
        tk.Label(cm_row, text="Confusion matrix", font=("Segoe UI", 10), bg="#f9f9f9", fg="#666").pack(side="left")
        tk.Label(cm_row, textvariable=self.cm_var, font=("Segoe UI", 10, "bold"), bg="#f9f9f9", fg="#333").pack(side="right")

        tk.Label(right_panel, text="Node detail", font=("Segoe UI", 14, "bold"), bg="#f9f9f9").pack(anchor="w", padx=10, pady=(4, 8))

        detail_fields = [
            ("split", "Split"),
            ("samples", "Samples"),
            ("gini", "Gini"),
            ("class", "Class"),
            ("churn_pct", "Churn %"),
            ("threshold", "Threshold"),
            ("value", "Value [No churn, Churn]"),
        ]

        for key, title in detail_fields:
            var = tk.StringVar(value="-")
            self.detail_vars[key] = var
            row = tk.Frame(right_panel, bg="#f9f9f9")
            row.pack(fill="x", padx=10, pady=2)
            tk.Label(row, text=title, font=("Segoe UI", 11), bg="#f9f9f9", fg="#555").pack(side="left")
            tk.Label(row, textvariable=var, font=("Segoe UI", 11, "bold"), bg="#f9f9f9", fg="#222").pack(side="right")

    def _legend_item(self, parent: tk.Widget, color: str, text: str) -> tk.Frame:
        container = tk.Frame(parent, bg="#f9f9f9")
        box = tk.Label(container, bg=color, width=2, relief="solid", bd=1)
        box.pack(side="left", padx=(0, 6))
        tk.Label(container, text=text, font=("Segoe UI", 10), bg="#f9f9f9").pack(side="left")
        return container

    def _metric_card(self, parent: tk.Widget, title: str) -> tk.StringVar:
        card = tk.Frame(parent, bg="#f5f3ee", bd=1, relief="solid")
        card.pack(fill="x", pady=3)
        tk.Label(card, text=title, font=("Segoe UI", 10), bg="#f5f3ee", fg="#666").pack(anchor="w", padx=8, pady=(4, 0))
        var = tk.StringVar(value="-")
        tk.Label(card, textvariable=var, font=("Segoe UI", 20, "bold"), bg="#f5f3ee", fg="#333").pack(anchor="w", padx=8, pady=(0, 6))
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

        self.tree_summary.configure(
            text=f"depth: {scenario.summary['depth']} - nodes: {scenario.summary['nodes']} - leaves: {scenario.summary['leaves']}"
        )

        self.metric_vars["train_acc"].set(f"{scenario.metrics['train_acc'] * 100:.1f}%")
        self.metric_vars["test_acc"].set(f"{scenario.metrics['test_acc'] * 100:.1f}%")
        self.metric_vars["precision"].set(f"{scenario.metrics['precision'] * 100:.1f}%")
        self.metric_vars["roc_auc"].set(f"{scenario.metrics['roc_auc']:.2f}")
        self.cm_var.set(str(scenario.metrics["confusion_matrix"]))

        gap = scenario.metrics["train_acc"] - scenario.metrics["test_acc"]
        if gap > 0.08:
            self.fit_badge.configure(text="Overfit detected", bg="#f1e4cf", fg="#7a4b00")
        else:
            self.fit_badge.configure(text="Fit is balanced", bg="#d7f0dc", fg="#1f6b2d")

        self._reset_visible_tree()
        self._show_node_details(0)
        self._draw_tree_graph()

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

    def _truncate(self, text: str, max_len: int = 34) -> str:
        if len(text) <= max_len:
            return text
        return text[: max_len - 3] + "..."

    def _item_size(self, iid: str) -> tuple[int, int]:
        item = self.visible_items[iid]
        if item.kind == "placeholder":
            return PLACEHOLDER_W, PLACEHOLDER_H
        return NODE_W, NODE_H

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

        x_map: dict[str, float] = {}
        next_leaf = [0]

        def assign_x(iid: str) -> float:
            children = self.visible_children.get(iid, [])
            if not children:
                x = float(next_leaf[0] * H_SPACING)
                next_leaf[0] += 1
            else:
                child_x = [assign_x(child) for child in children]
                x = sum(child_x) / len(child_x)
            x_map[iid] = x
            return x

        assign_x(self.root_item_id)

        centers: dict[str, tuple[float, float]] = {}
        for iid in self.visible_items:
            x = x_map.get(iid, 0.0) + MARGIN_X
            y = depth_map.get(iid, 0) * V_SPACING + MARGIN_Y
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
                self.canvas.create_line(px, py + ph / 2, cx, cy - ch / 2, fill="#9a9a9a", width=1.5)
                edge_label = self.visible_items[child].edge_label
                if edge_label:
                    lx = (px + cx) / 2
                    ly = (py + ph / 2 + cy - ch / 2) / 2 - 8
                    self.canvas.create_text(lx, ly, text=edge_label, fill="#666", font=("Segoe UI", 10))

        for iid, item in self.visible_items.items():
            x, y = centers[iid]
            w, h = self._item_size(iid)
            x1, y1, x2, y2 = x - w / 2, y - h / 2, x + w / 2, y + h / 2

            if item.kind == "placeholder":
                fill = "#ffffff"
                border = "#9e9e9e"
                text = "..."
                font = ("Segoe UI", 16, "bold")
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

                line1 = self._truncate(info["split"], 34)
                text = f"{line1}\n{info['samples']} samples\n{info['class']}"
                font = ("Segoe UI", 10, "bold")

            if self.selected_item_id == iid:
                border_width = 3
                border_color = "#845ec2"
            else:
                border_width = 1.8
                border_color = border

            item_tag = f"item:{iid}"
            self.canvas.create_rectangle(
                x1,
                y1,
                x2,
                y2,
                fill=fill,
                outline=border_color,
                width=border_width,
                tags=("item", item_tag),
            )
            self.canvas.create_text(x, y, text=text, font=font, justify="center", tags=("item", item_tag))

        bbox = self.canvas.bbox("all")
        if bbox:
            self.canvas.configure(scrollregion=(bbox[0] - 50, bbox[1] - 50, bbox[2] + 50, bbox[3] + 50))

    def _extract_item_id_from_event(self, event: tk.Event) -> str | None:
        current = self.canvas.find_withtag("current")
        if not current:
            return None
        tags = self.canvas.gettags(current[0])
        for tag in tags:
            if tag.startswith("item:"):
                return tag.split(":", 1)[1]
        return None

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


def visualize_baseline_tree() -> None:
    X_train, X_test, y_train, y_test = load_splits()

    baseline = _load_or_fit_baseline(X_train, y_train)

    max_depth_5 = DecisionTreeClassifier(max_depth=5, random_state=42)
    max_depth_5.fit(X_train, y_train)

    entropy_tree = DecisionTreeClassifier(criterion="entropy", random_state=42)
    entropy_tree.fit(X_train, y_train)

    pruned = _fit_pruned_tree(X_train, y_train, X_test, y_test)

    scenarios = [
        _scenario_payload("Baseline", baseline, X_train, y_train, X_test, y_test),
        _scenario_payload("max_depth=5", max_depth_5, X_train, y_train, X_test, y_test),
        _scenario_payload("Entropy", entropy_tree, X_train, y_train, X_test, y_test),
        _scenario_payload("Pruned", pruned, X_train, y_train, X_test, y_test),
    ]

    root = tk.Tk()
    TreeExplorerApp(root, scenarios)
    root.mainloop()


if __name__ == "__main__":
    ensure_directories()
    visualize_baseline_tree()

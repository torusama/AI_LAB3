from __future__ import annotations

import json
import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier, export_text, _tree

from common import MODEL_DIR, REPORT_DIR, ensure_directories, get_default_model_path, load_model, load_splits

import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


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


def _fit_pruned_tree(X_train, y_train) -> DecisionTreeClassifier:
    from sklearn.model_selection import GridSearchCV
    base = DecisionTreeClassifier(random_state=42)
    base.fit(X_train, y_train)

    path = base.cost_complexity_pruning_path(X_train, y_train)
    alphas = [float(a) for a in path.ccp_alphas if a > 0]
    if not alphas:
        return base

    candidates = sorted(set(alphas))[:: max(1, len(alphas) // 20)]
    
    grid_search = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid={'ccp_alpha': candidates},
        cv=5,
        scoring='accuracy'
    )
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_


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
        delta_text = "This is the baseline model."
    else:
        delta_text = (
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

    def _toggle_stats(self) -> None:
        if self.stats_frame.winfo_ismapped():
            self._hide_stats()
        else:
            self._show_stats()
 
    def _hide_stats(self) -> None:
        self.stats_frame.pack_forget()
        self.body.pack(fill="both", expand=True, padx=14, pady=(0, 12))
        self.stats_btn.configure(relief="groove", bg="#e8f4fd")
 
    def _show_stats(self) -> None:
        self.body.pack_forget()
        self._build_stats_content()
        self.stats_frame.pack(fill="both", expand=True, padx=14, pady=(0, 12))
        self.stats_btn.configure(relief="sunken", bg="#b3d9f5")
 
    def _build_stats_content(self) -> None:
        for w in self.stats_frame.winfo_children():
            w.destroy()
 
        scenarios = self.scenarios
        metrics  = ["Test Acc", "Train Acc", "Precision", "Recall", "F1-score", "ROC-AUC", "Error Rate"]
        keys     = ["test_acc", "train_acc", "precision", "recall", "f1_score", "roc_auc", "error_rate"]
        # roc_auc is included in pct_keys so it is multiplied by 100 for display consistency
        pct_keys = {"test_acc", "train_acc", "precision", "recall", "f1_score", "error_rate", "roc_auc"}
 
        def val(s, k):
            v = s.metrics[k]
            return round(v * 100, 1) if k in pct_keys else round(v, 4)
 
        PALETTE = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2"]
        colors  = PALETTE[:len(scenarios)]
 
        title_bar = tk.Frame(self.stats_frame, bg="#1a5f8a", pady=8)
        title_bar.pack(fill="x")
        tk.Label(
            title_bar,
            text="Model Comparison — Statistics Dashboard",
            font=("Segoe UI", 15, "bold"),
            bg="#1a5f8a", fg="white",
        ).pack(side="left", padx=16)
 
        outer = tk.Frame(self.stats_frame, bg="#f2f2f2")
        outer.pack(fill="both", expand=True)
 
        vscroll = ttk.Scrollbar(outer, orient="vertical")
        vscroll.pack(side="right", fill="y")
 
        scroll_canvas = tk.Canvas(outer, bg="#f2f2f2", highlightthickness=0, yscrollcommand=vscroll.set)
        scroll_canvas.pack(side="left", fill="both", expand=True)
 
        inner = tk.Frame(scroll_canvas, bg="#f2f2f2")
        win_id = scroll_canvas.create_window((0, 0), window=inner, anchor="nw")
 
        def _on_inner_configure(e):
            scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all"))
        def _on_canvas_resize(e):
            scroll_canvas.itemconfig(win_id, width=e.width)
 
        inner.bind("<Configure>", _on_inner_configure)
        scroll_canvas.bind("<Configure>", lambda e: scroll_canvas.itemconfig(win_id, width=e.width))

        def on_mouse_wheel(event):
            if hasattr(event, 'delta') and getattr(event, 'delta') != 0:
                delta = int(-1 * (event.delta / 120))
            elif hasattr(event, 'num'):
                delta = -1 if event.num == 4 else 1
            else:
                delta = 0
            scroll_canvas.yview_scroll(delta, "units")

        def _bind_mousewheel(widget):
            widget.unbind("<MouseWheel>")
            widget.unbind("<Button-4>")
            widget.unbind("<Button-5>")
            widget.bind("<MouseWheel>", on_mouse_wheel)
            widget.bind("<Button-4>", on_mouse_wheel)
            widget.bind("<Button-5>", on_mouse_wheel)
            for child in widget.winfo_children():
                _bind_mousewheel(child)

        vscroll.config(command=scroll_canvas.yview)

        # ═══════════════════════════════════════════════════════════
        # SECTION A — Comparison Table
        # ═══════════════════════════════════════════════════════════
        sec_a = tk.Frame(inner, bg="#ffffff", bd=1, relief="solid")
        sec_a.pack(fill="x", padx=16, pady=(14, 8))
 
        tk.Label(
            sec_a,
            text="Metrics Comparison Table",
            font=("Segoe UI", 13, "bold"),
            bg="#f0f4ff", fg="#1a5f8a", anchor="w", pady=6, padx=12
        ).pack(fill="x")
 
        tbl = tk.Frame(sec_a, bg="#ffffff")
        tbl.pack(fill="x", padx=12, pady=(4, 10))
 
        header_texts = ["Model", "Test Acc %", "Train Acc %", "Precision %",
                        "Recall %", "F1-score %", "ROC-AUC", "Error Rate %"]
        
        for c in range(len(header_texts)):
            tbl.grid_columnconfigure(c, weight=1, minsize=100)
 
        def _th(parent, text, col, bg="#1a5f8a", fg="white"):
            tk.Label(parent, text=text, font=("Segoe UI", 10, "bold"),
                     bg=bg, fg=fg, relief="flat", padx=6, pady=5, anchor="center"
            ).grid(row=0, column=col, padx=1, pady=1, sticky="nsew")
 
        for c, h in enumerate(header_texts):
            _th(tbl, h, c)
 
        for r, sc in enumerate(scenarios):
            row_bg = "#f7f9ff" if r % 2 == 0 else "#ffffff"
            dot_lbl = tk.Label(tbl, text=f"●  {sc.name}",
                               font=("Segoe UI", 10, "bold"),
                               bg=row_bg, fg=colors[r], padx=6, pady=4, anchor="w")
            dot_lbl.grid(row=r + 1, column=0, padx=1, pady=1, sticky="nsew")
            for c, k in enumerate(keys):
                v = val(sc, k)
                tk.Label(tbl, text=str(v), font=("Segoe UI", 10),
                         bg=row_bg, fg="#222", padx=6, pady=4, anchor="center",
                ).grid(row=r + 1, column=c + 1, padx=1, pady=1, sticky="nsew")
 
        legend_row = tk.Frame(sec_a, bg="#f8f8f8", pady=4)
        legend_row.pack(fill="x", padx=12, pady=(0, 8))
        tk.Label(legend_row, text="Legend:", font=("Segoe UI", 9, "bold"), bg="#f8f8f8").pack(side="left", padx=(4, 8))
        for i, sc in enumerate(scenarios):
            chip = tk.Frame(legend_row, bg=colors[i], width=14, height=14)
            chip.pack(side="left", padx=(0, 2))
            chip.pack_propagate(False)
            tk.Label(legend_row, text=sc.name, font=("Segoe UI", 9), bg="#f8f8f8").pack(side="left", padx=(0, 12))

        DISPLAY_METRICS = ["Test Acc", "Precision", "Recall", "F1-score", "ROC-AUC"]
        DISPLAY_KEYS    = ["test_acc", "precision", "recall", "f1_score", "roc_auc"]

        def _save_chart(fig, chart_name: str):
            try:
                from tkinter import filedialog
                path = filedialog.asksaveasfilename(
                    defaultextension=".png", filetypes=[("PNG image", "*.png"), ("All files", "*.*")],
                    initialfile=f"{chart_name}.png", title="Save chart as…"
                )
                if path: fig.savefig(path, dpi=150, bbox_inches="tight")
            except Exception as exc: print(f"Save error: {exc}")
 
        def _embed_chart(parent, fig, chart_name: str):
            canvas = FigureCanvasTkAgg(fig, master=parent)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=4)
            btn_row = tk.Frame(parent, bg="#f9f9f9")
            btn_row.pack(fill="x", padx=8, pady=(0, 8))
            tk.Button(
                btn_row, text="Download PNG", font=("Segoe UI", 9),
                bg="#1a5f8a", fg="white", relief="flat", padx=8, pady=3,
                command=lambda f=fig, n=chart_name: _save_chart(f, n),
            ).pack(side="right")
            _bind_mousewheel(parent)

        # ═══════════════════════════════════════════════════════════
        # SECTION B — Bar Chart
        # ═══════════════════════════════════════════════════════════
        sec_b = tk.Frame(inner, bg="#ffffff", bd=1, relief="solid")
        sec_b.pack(fill="x", padx=16, pady=8)
        
        tk.Label(sec_b, text="Bar Chart — Grouped Metrics per Model (Hover to see details)", font=("Segoe UI", 13, "bold"),
                 bg="#fff8e7", fg="#7a4b00", anchor="w", pady=6, padx=12).pack(fill="x")
        
        # Increased figure height for better readability
        fig_bar, ax_bar = plt.subplots(figsize=(13, 5.5))
        fig_bar.patch.set_facecolor("#fafafa")
        ax_bar.set_facecolor("#fafafa")
        n_models, n_metrics = len(scenarios), len(DISPLAY_METRICS)
        x, bar_width = range(n_metrics), 0.72 / n_models
        
        bars_list = []
        for i, (sc, col) in enumerate(zip(scenarios, colors)):
            offsets = [xi + i * bar_width - (n_models - 1) * bar_width / 2 for xi in x]
            vals = [val(sc, k) for k in DISPLAY_KEYS]
            bars = ax_bar.bar(offsets, vals, bar_width * 0.88, color=col, alpha=0.88, label=sc.name, edgecolor="white", linewidth=0.6)
            bars_list.append((bars, vals, sc.name, col))
            # Draw labels immediately
            for bar in bars:
                h = bar.get_height()
                ax_bar.text(bar.get_x() + bar.get_width() / 2, h + 0.8, f"{h:.1f}", ha="center", va="bottom", fontsize=7.5, color="#333")
            
        ax_bar.set_xticks(list(x))
        ax_bar.set_xticklabels(DISPLAY_METRICS, fontsize=10)
        ax_bar.set_ylabel("Score (%  or  value)", fontsize=9)
        ax_bar.set_title("Grouped Bar Chart — Baseline vs Improvements", fontsize=11, fontweight="bold", pad=12)
        
        # Y-axis zoomed in from 30 to 100 for better readability
        ax_bar.set_yticks(range(30, 101, 10))
        ax_bar.set_ylim(30, 105) 
        
        # Legend placed outside the chart to avoid overlapping bars
        ax_bar.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=8.5, framealpha=0.9)
        ax_bar.grid(axis="y", linestyle="--", alpha=0.4)
        ax_bar.spines[["top", "right"]].set_visible(False)
        fig_bar.tight_layout()

        annot_bar = ax_bar.annotate("", xy=(0,0), xytext=(0, 15), textcoords="offset points",
                                 bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", lw=1.5, alpha=0.95),
                                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
        annot_bar.set_visible(False)

        all_patches = []
        for bars, targets, name, col in bars_list:
            for bar, val_b, metric in zip(bars, targets, DISPLAY_METRICS):
                all_patches.append((bar, val_b, name, metric, col))

        def hover_bar(event):
            is_visible = annot_bar.get_visible()
            if event.inaxes == ax_bar:
                for bar, val_b, name, metric, col in all_patches:
                    cont, _ = bar.contains(event)
                    if cont:
                        annot_bar.xy = (bar.get_x() + bar.get_width() / 2, bar.get_height())
                        annot_bar.set_text(f"{name}\n{metric}: {val_b:.1f}")
                        annot_bar.get_bbox_patch().set_edgecolor(col)
                        annot_bar.set_visible(True)
                        fig_bar.canvas.draw_idle()
                        return
            if is_visible:
                annot_bar.set_visible(False)
                fig_bar.canvas.draw_idle()

        fig_bar.canvas.mpl_connect("motion_notify_event", hover_bar)
        _embed_chart(sec_b, fig_bar, "bar_chart_comparison")

        # ═══════════════════════════════════════════════════════════
        # SECTION C — Line Chart
        # ═══════════════════════════════════════════════════════════
        sec_c = tk.Frame(inner, bg="#ffffff", bd=1, relief="solid")
        sec_c.pack(fill="x", padx=16, pady=8)
        
        tk.Label(sec_c, text="Line Chart — Performance Trend Across Metrics (Hover to see details)", font=("Segoe UI", 13, "bold"),
                 bg="#edfbf3", fg="#1f6b2d", anchor="w", pady=6, padx=12).pack(fill="x")

        # Increased figure height for better visibility
        fig_line, ax_line = plt.subplots(figsize=(13, 5.5))
        fig_line.patch.set_facecolor("#fafafa")
        ax_line.set_facecolor("#fafafa")

        lines_list = []
        x_positions = range(len(DISPLAY_METRICS))

        # Collect all points to handle overlapping labels
        all_points = {x: [] for x in x_positions}

        for i, (sc, col) in enumerate(zip(scenarios, colors)):
            vals = [val(sc, k) for k in DISPLAY_KEYS]
            line, = ax_line.plot(x_positions, vals, marker="o", markersize=7, linewidth=2.2, color=col, label=sc.name, alpha=0.92, picker=5)
            lines_list.append((line, vals, col, sc.name))
            
            for x_val, y_val in zip(x_positions, vals):
                all_points[x_val].append((y_val, i))  # store value and model index

        # Stagger text labels to avoid overlap
        for x_idx, points in all_points.items():
            # Sort points at this x position by y value (ascending)
            points.sort(key=lambda item: item[0])
            
            for rank, (y_val, model_idx) in enumerate(points):
                col = colors[model_idx]
                
                # Stagger positions when multiple points cluster together
                if len(points) > 1:
                    # Lowest point → text below, highest point → text above
                    if rank == 0: 
                        y_offset = -1.5
                        va = "top"
                    elif rank == len(points) - 1:
                        y_offset = 1.5
                        va = "bottom"
                    else:
                        # Middle points → slight left/right offset
                        y_offset = 0
                        va = "center"
                        x_offset = 0.05 if rank % 2 == 0 else -0.05
                        ax_line.text(x_idx + x_offset, y_val, f"{y_val:.1f}", ha="left" if rank%2==0 else "right", va=va, fontsize=7.5, color="#333", zorder=4, 
                                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.5))
                        continue 
                else:
                    y_offset = 1.5
                    va = "bottom"
                
                ax_line.text(x_idx, y_val + y_offset, f"{y_val:.1f}", ha="center", va=va, fontsize=7.5, color="#333", zorder=4,
                             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.5))

        ax_line.set_xticks(list(x_positions))
        ax_line.set_xticklabels(DISPLAY_METRICS, fontsize=10)
        ax_line.set_title("Line Chart — Metric Trend per Model", fontsize=11, fontweight="bold", pad=12)
        ax_line.set_ylabel("Score (%  or  value)", fontsize=9)
        
        # ZOOM IN TRỤC Y TỪ 30 ĐẾN 100
        # Y-axis zoomed in from 30 to 100
        ax_line.set_yticks(range(30, 101, 10))
        ax_line.set_ylim(30, 105)
        
        # Legend placed outside the chart
        ax_line.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=8.5, framealpha=0.9)
        ax_line.grid(linestyle="--", alpha=0.35)
        ax_line.spines[["top", "right"]].set_visible(False)
        fig_line.tight_layout()

        annot_line = ax_line.annotate("", xy=(0,0), xytext=(0, 15), textcoords="offset points",
                                 bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", lw=1.5, alpha=0.95),
                                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
        annot_line.set_visible(False)

        def hover_line(event):
            is_visible = annot_line.get_visible()
            if event.inaxes == ax_line:
                for line, vals_l, col, name in lines_list:
                    cont, ind = line.contains(event)
                    if cont:
                        idx = ind["ind"][0]
                        x_pos = x_positions[idx]
                        y_pos = line.get_ydata()[idx]
                        annot_line.xy = (x_pos, y_pos)
                        annot_line.set_text(f"{name}\n{DISPLAY_METRICS[idx]}: {y_pos:.1f}")
                        annot_line.get_bbox_patch().set_edgecolor(col)
                        
                        if idx == len(DISPLAY_METRICS) - 1:
                            annot_line.set_position((-10, 15))
                            annot_line.set_ha("right")
                        else:
                            annot_line.set_position((10, 15))
                            annot_line.set_ha("left")
                            
                        annot_line.set_visible(True)
                        fig_line.canvas.draw_idle()
                        return
            if is_visible:
                annot_line.set_visible(False)
                fig_line.canvas.draw_idle()

        fig_line.canvas.mpl_connect("motion_notify_event", hover_line)
        _embed_chart(sec_c, fig_line, "line_chart_comparison")

        # ═══════════════════════════════════════════════════════════
        # SECTION D — Pie Chart
        # ═══════════════════════════════════════════════════════════
        sec_d = tk.Frame(inner, bg="#ffffff", bd=1, relief="solid")
        sec_d.pack(fill="x", padx=16, pady=(8, 16))
        
        tk.Label(sec_d, text="Pie Charts — Relative Share of Score per Metric", font=("Segoe UI", 13, "bold"),
                 bg="#fde8e8", fg="#8b1a1a", anchor="w", pady=6, padx=12).pack(fill="x")
        
        PIE_METRICS, PIE_KEYS = ["Test Acc", "Precision", "Recall", "F1-score"], ["test_acc", "precision", "recall", "f1_score"]
        fig_pie, axes = plt.subplots(1, len(PIE_METRICS), figsize=(13, 5.8))
        fig_pie.patch.set_facecolor("#fafafa")
        
        # Render Pie immediately without animation
        for ax, metric, key in zip(axes, PIE_METRICS, PIE_KEYS):
            vals = [max(val(sc, key), 0.01) for sc in scenarios]
            wedges, texts, autotexts = ax.pie(
                vals, labels=None, colors=colors[:len(scenarios)],
                autopct="%1.1f%%",
                startangle=90, pctdistance=0.72,
                wedgeprops={"edgecolor": "white", "linewidth": 1.5}
            )
            for at in autotexts:
                at.set_fontsize(7.5)
                at.set_color("white")
                at.set_fontweight("bold")
            ax.set_title(metric, fontsize=10, fontweight="bold", pad=8)
            
        patches = [mpatches.Patch(color=colors[i], label=sc.name) for i, sc in enumerate(scenarios)]
        fig_pie.legend(handles=patches, loc="lower center", ncol=len(scenarios), fontsize=8.5, framealpha=0.9, bbox_to_anchor=(0.5, 0.01))
        fig_pie.suptitle("Pie Charts — Score Share per Metric (all models)", fontsize=11, fontweight="bold", y=0.96)
        fig_pie.tight_layout(rect=[0, 0.08, 1, 0.93])

        _embed_chart(sec_d, fig_pie, "pie_charts_comparison")

        _bind_mousewheel(outer)

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

        self.stats_btn = tk.Button(
            topbar, text="Statistics",
            font=("Segoe UI", 12), relief="groove", borderwidth=1,
            padx=12, pady=5, bg="#e8f4fd", fg="#1a5f8a",
            command=self._toggle_stats,
        )
        self.stats_btn.pack(side="left", padx=(16, 4))

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

        self.body = tk.Frame(self.root, bg="#f2f2f2")
        self.body.pack(fill="both", expand=True, padx=14, pady=(0, 12))
        body = self.body 

        self.stats_frame = tk.Frame(self.root, bg="#f2f2f2")
        
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
        self.canvas.bind("<Configure>", lambda e: self._draw_tree_graph())

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
        _right_win_id = right_canvas.create_window((0, 0), window=scrollable_right, anchor="nw")
        right_canvas.configure(yscrollcommand=right_scroll.set)
        right_canvas.bind(
            "<Configure>",
            lambda e: right_canvas.itemconfig(_right_win_id, width=e.width)
        )
        
        right_canvas.pack(side="left", fill="both", expand=True)
        right_scroll.pack(side="right", fill="y")
        
        def _on_right_mousewheel(event):
            if right_scroll.get() != (0.0, 1.0):
                right_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        right_canvas.bind_all("<MouseWheel>", _on_right_mousewheel)

        metrics_container = tk.Frame(scrollable_right, bg="#e5e7eb", bd=1, relief="solid")
        metrics_container.pack(fill="x", padx=10, pady=(10, 4))

        for i in range(3):
            metrics_container.grid_columnconfigure(i, weight=1, uniform="mcol")
        for i in range(3):
            metrics_container.grid_rowconfigure(i, weight=1)

        m_f1 = tk.Frame(metrics_container, bg="#e5e7eb")
        m_f1.grid(row=0, column=0, sticky="ew", padx=1, pady=1)
        m_f2 = tk.Frame(metrics_container, bg="#e5e7eb")
        m_f2.grid(row=0, column=1, sticky="ew", padx=1, pady=1)
        m_f3 = tk.Frame(metrics_container, bg="#e5e7eb")
        m_f3.grid(row=0, column=2, sticky="ew", padx=1, pady=1)
        
        self.metric_vars["train_acc"] = self._metric_card(
            m_f1, "Train Acc", pad_y=0, accent="#3b82f6")
        self.metric_vars["test_acc"] = self._metric_card(
            m_f2, "Test Acc", pad_y=0, accent="#0ea5e9")
        self.metric_vars["error_rate"] = self._metric_card(
            m_f3, "Error Rate", pad_y=0, accent="#ef4444")

        m_f4 = tk.Frame(metrics_container, bg="#e5e7eb")
        m_f4.grid(row=1, column=0, sticky="ew", padx=1, pady=1)
        m_f5 = tk.Frame(metrics_container, bg="#e5e7eb")
        m_f5.grid(row=1, column=1, sticky="ew", padx=1, pady=1)
        m_f6 = tk.Frame(metrics_container, bg="#e5e7eb")
        m_f6.grid(row=1, column=2, sticky="ew", padx=1, pady=1)

        self.metric_vars["roc_auc"] = self._metric_card(
            m_f4, "ROC-AUC", pad_y=0, accent="#8b5cf6")
        self.metric_vars["precision"] = self._metric_card(
            m_f5, "Precision", pad_y=0, accent="#10b981")
        self.metric_vars["recall"] = self._metric_card(
            m_f6, "Recall", pad_y=0, accent="#f59e0b")

        m_f7 = tk.Frame(metrics_container, bg="#e5e7eb")
        m_f7.grid(row=2, column=0, columnspan=3, sticky="ew", padx=1, pady=1)

        self.metric_vars["f1_score"] = self._metric_card(
            m_f7, "F1-score", pad_y=0, accent="#ec4899", centered=True)

        sep0 = tk.Frame(scrollable_right, bg="#ddd", height=1)
        sep0.pack(fill="x", padx=10, pady=(6, 0))
        hdr_summary = tk.Frame(scrollable_right, bg="#1a3a5c")
        hdr_summary.pack(fill="x", padx=10, pady=(0, 4))
        tk.Label(hdr_summary, text="  Model Summary", font=("Segoe UI", 11, "bold"),
                 bg="#1a3a5c", fg="white", pady=6).pack(anchor="w")

        summary_container = tk.Frame(scrollable_right, bg="#f9f9f9")
        summary_container.pack(fill="x", padx=10, pady=(4, 6))
        summary_container.grid_columnconfigure(0, weight=1, uniform="sumcol")
        summary_container.grid_columnconfigure(1, weight=1, uniform="sumcol")
        summary_container.grid_rowconfigure(0, weight=1, uniform="sumrow")
        summary_container.grid_rowconfigure(1, weight=1, uniform="sumrow")

        # ── helper: one label+value row inside a summary card ──────────────────
        def _kv_row(parent, label_text, value_text, accent, bg, bold_val=True):
            row = tk.Frame(parent, bg=bg)
            row.pack(fill="x", padx=8, pady=1)
            tk.Label(row, text=label_text,
                     font=("Segoe UI", 8), bg=bg, fg="#888",
                     anchor="w").pack(side="left")
            vf = tk.Label(row, text=value_text,
                          font=("Segoe UI", 10, "bold" if bold_val else "normal"),
                          bg=bg, fg=accent, anchor="e")
            vf.pack(side="right")
            return vf

        # ── Structure card ───────────────────────────────────────────────────
        structure_outer = tk.Frame(summary_container, bg="#3d5a80", bd=0)
        structure_outer.grid(row=0, column=0, sticky="nsew", padx=(0, 3), pady=(0, 3))
        tk.Frame(structure_outer, bg="#3d5a80", height=3).pack(fill="x")
        structure_card = tk.Frame(structure_outer, bg="#eef2f8", bd=1, relief="solid")
        structure_card.pack(fill="both", expand=True)

        hdr_s = tk.Frame(structure_card, bg="#dde5f0")
        hdr_s.pack(fill="x")
        tk.Label(hdr_s, text="Structure", font=("Segoe UI", 9, "bold"),
                 bg="#dde5f0", fg="#3d5a80", pady=5, padx=8).pack(anchor="w")

        sep_s = tk.Frame(structure_card, bg="#c8d5e8", height=1)
        sep_s.pack(fill="x")

        struct_body = tk.Frame(structure_card, bg="#eef2f8")
        struct_body.pack(fill="both", expand=True, pady=(4, 6))

        self._depth_val  = _kv_row(struct_body, "Depth",  "–", "#3d5a80", "#eef2f8")
        self._nodes_val  = _kv_row(struct_body, "Nodes",  "–", "#3d5a80", "#eef2f8")
        self._leaves_val = _kv_row(struct_body, "Leaves", "–", "#3d5a80", "#eef2f8")

        # ── Cross-validation card ────────────────────────────────────────────
        cv_outer = tk.Frame(summary_container, bg="#1a5f8a", bd=0)
        cv_outer.grid(row=0, column=1, sticky="nsew", padx=(3, 0), pady=(0, 3))
        tk.Frame(cv_outer, bg="#1a5f8a", height=3).pack(fill="x")
        cv_card = tk.Frame(cv_outer, bg="#eef4fb", bd=1, relief="solid")
        cv_card.pack(fill="both", expand=True)

        hdr_cv = tk.Frame(cv_card, bg="#d9eaf6")
        hdr_cv.pack(fill="x")
        tk.Label(hdr_cv, text="Cross-validation", font=("Segoe UI", 9, "bold"),
                 bg="#d9eaf6", fg="#1a5f8a", pady=5, padx=8).pack(anchor="w")

        sep_cv = tk.Frame(cv_card, bg="#b8d4ec", height=1)
        sep_cv.pack(fill="x")

        cv_body = tk.Frame(cv_card, bg="#eef4fb")
        cv_body.pack(fill="both", expand=True, pady=(4, 6))
        self._cv_f1_val   = _kv_row(cv_body, "CV Mean F1",      "–", "#1a5f8a", "#eef4fb")
        self._cv_auc_val  = _kv_row(cv_body, "CV Mean ROC-AUC", "–", "#1a5f8a", "#eef4fb")

        # ── Hyperparameters card ─────────────────────────────────────────────
        param_outer = tk.Frame(summary_container, bg="#5a4a3a", bd=0)
        param_outer.grid(row=1, column=0, sticky="nsew", padx=(0, 3), pady=(3, 0))
        tk.Frame(param_outer, bg="#7a6a52", height=3).pack(fill="x")
        param_card = tk.Frame(param_outer, bg="#f6f1e8", bd=1, relief="solid")
        param_card.pack(fill="both", expand=True)

        hdr_p = tk.Frame(param_card, bg="#ece4d6")
        hdr_p.pack(fill="x")
        tk.Label(hdr_p, text="Hyperparameters", font=("Segoe UI", 9, "bold"),
                 bg="#ece4d6", fg="#5a4a3a", pady=5, padx=8).pack(anchor="w")

        sep_p = tk.Frame(param_card, bg="#d9cdb8", height=1)
        sep_p.pack(fill="x")

        self.param_body = tk.Frame(param_card, bg="#f6f1e8")
        self.param_body.pack(fill="both", expand=True, pady=(4, 6))

        # ── Delta vs Baseline card ───────────────────────────────────────────
        delta_outer = tk.Frame(summary_container, bg="#1b5e20", bd=0)
        delta_outer.grid(row=1, column=1, sticky="nsew", padx=(3, 0), pady=(3, 0))
        tk.Frame(delta_outer, bg="#27ae60", height=3).pack(fill="x")
        delta_card = tk.Frame(delta_outer, bg="#eef7ef", bd=1, relief="solid")
        delta_card.pack(fill="both", expand=True)

        hdr_d = tk.Frame(delta_card, bg="#d6eeda")
        hdr_d.pack(fill="x")
        tk.Label(hdr_d, text="Delta vs Baseline", font=("Segoe UI", 9, "bold"),
                 bg="#d6eeda", fg="#1b5e20", pady=5, padx=8).pack(anchor="w")

        sep_d = tk.Frame(delta_card, bg="#afd6b4", height=1)
        sep_d.pack(fill="x")

        self.delta_frame = tk.Frame(delta_card, bg="#eef7ef")
        self.delta_frame.pack(anchor="w", fill="x", pady=(4, 6))

        sep1 = tk.Frame(scrollable_right, bg="#ddd", height=1)
        sep1.pack(fill="x", padx=10, pady=(4, 0))
        hdr_cm = tk.Frame(scrollable_right, bg="#4a235a")
        hdr_cm.pack(fill="x", padx=10, pady=(0, 4))
        tk.Label(hdr_cm, text="  Confusion Matrix", font=("Segoe UI", 11, "bold"),
                 bg="#4a235a", fg="white", pady=6).pack(anchor="w")
        tk.Label(scrollable_right,
                 text="Each cell shows how many customers the model predicted correctly / incorrectly.",
                 font=("Segoe UI", 9), bg="#f9f9f9", fg="#888",
                 wraplength=320, justify="left").pack(anchor="w", padx=10, pady=(0, 4))

        self.cm_frame = tk.Frame(scrollable_right, bg="#f9f9f9")
        self.cm_frame.pack(fill="x", padx=10, pady=(0, 8))

        sep2 = tk.Frame(scrollable_right, bg="#ddd", height=1)
        sep2.pack(fill="x", padx=10, pady=(4, 0))
        hdr_tree = tk.Frame(scrollable_right, bg="#7a4b00")
        hdr_tree.pack(fill="x", padx=10, pady=(0, 4))
        tk.Label(hdr_tree, text="  Decision Tree Analysis",
                 font=("Segoe UI", 11, "bold"),
                 bg="#7a4b00", fg="white", pady=6).pack(anchor="w")

        tk.Label(
            scrollable_right,
            text="Most Important Features",
            font=("Segoe UI", 10, "bold"),
            bg="#f9f9f9",
            fg="#7a4b00",
        ).pack(anchor="w", padx=10, pady=(8, 0))
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

        tk.Label(
            scrollable_right,
            text="Decision Rules (Readable)",
            font=("Segoe UI", 10, "bold"),
            bg="#f9f9f9",
            fg="#7a4b00",
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

        sep3 = tk.Frame(scrollable_right, bg="#ddd", height=1)
        sep3.pack(fill="x", padx=10, pady=(4, 0))
        hdr_node = tk.Frame(scrollable_right, bg="#1a5f8a")
        hdr_node.pack(fill="x", padx=10, pady=(0, 4))
        tk.Label(hdr_node, text="  Node Details",
                 font=("Segoe UI", 11, "bold"),
                 bg="#1a5f8a", fg="white", pady=6).pack(anchor="w")

        _detail_meta = [
            ("split",     "Split Condition",                  "The condition this node checks",           "#1565c0", "#e8f4fd"),
            ("samples",   "Number of Samples",                "Customers that passed through this node",  "#2e7d32", "#e8f5e9"),
            ("gini",      "Gini Impurity",                    "0 = pure,  0.5 = most impure",             "#e65100", "#fff3e0"),
            ("class",     "Prediction",                       "The label this node predicts",             "#7b1fa2", "#f3e5f5"),
            ("churn_pct", "Churn Percentage",                 "% of customers who churned at this node",  "#b71c1c", "#fdecea"),
            ("threshold", "Split Threshold",                  "Threshold value used for splitting",       "#0277bd", "#e1f5fe"),
            ("value",     "Class Distribution [Stay, Leave]", "Sample count per class in this node",      "#37474f", "#eceff1"),
        ]

        def _make_hover(widget_list, bg_normal, bg_hover):
            def on_enter(e):
                for w in widget_list:
                    try:
                        w.configure(bg=bg_hover)
                    except Exception:
                        pass
            def on_leave(e):
                for w in widget_list:
                    try:
                        w.configure(bg=bg_normal)
                    except Exception:
                        pass
            for w in widget_list:
                w.bind("<Enter>", on_enter)
                w.bind("<Leave>", on_leave)

        for key, title, hint, accent, bg in _detail_meta:
            # Darken bg slightly for hover
            import colorsys
            var = tk.StringVar(value="-")
            self.detail_vars[key] = var
            row_card = tk.Frame(scrollable_right, bg=accent, bd=0)
            row_card.pack(fill="x", padx=10, pady=2)
            tk.Frame(row_card, bg=accent, height=2).pack(fill="x")
            row_inner = tk.Frame(row_card, bg=bg, bd=0)
            row_inner.pack(fill="x")
            left_col = tk.Frame(row_inner, bg=bg)
            left_col.pack(side="left", fill="both", expand=True, padx=(8, 4), pady=(5, 5))
            title_lbl = tk.Label(left_col, text=title, font=("Segoe UI", 9, "bold"),
                                 bg=bg, fg="#333")
            title_lbl.pack(anchor="w")
            hint_lbl = tk.Label(left_col, text=hint, font=("Segoe UI", 8),
                                bg=bg, fg="#888")
            hint_lbl.pack(anchor="w")
            val_lbl = tk.Label(row_inner, textvariable=var, font=("Segoe UI", 11, "bold"),
                               bg=bg, fg=accent, anchor="e")
            val_lbl.pack(side="right", padx=(4, 12), pady=5)
            # Hover: slightly brighter background
            bg_hover = bg.replace("f", "e") if "f" in bg else bg
            _make_hover([row_inner, left_col, title_lbl, hint_lbl, val_lbl], bg, bg_hover)

    def _legend_item(self, parent: tk.Widget, color: str, text: str) -> tk.Frame:
        container = tk.Frame(parent, bg="#f9f9f9")
        box = tk.Label(container, bg=color, width=2, relief="solid", bd=1)
        box.pack(side="left", padx=(0, 6))
        tk.Label(container, text=text, font=("Segoe UI", 10), bg="#f9f9f9").pack(side="left")
        return container

    def _metric_card(self, parent: tk.Widget, title: str, pad_y=3,
                     accent="#1a5fb4", bg="#f0f4ff", centered=False) -> tk.StringVar:
        # Always use clean white background, accent only as top border
        outer = tk.Frame(parent, bg=accent, bd=0)
        outer.pack(fill="both", expand=True, pady=pad_y)
        # Thin coloured top strip
        tk.Frame(outer, bg=accent, height=3).pack(fill="x")
        inner = tk.Frame(outer, bg="#ffffff", bd=0)
        inner.pack(fill="both", expand=True)
        # Title — small, grey, always centered
        tk.Label(inner, text=title,
                 font=("Segoe UI", 8),
                 bg="#ffffff", fg="#999999").pack(anchor="center", pady=(7, 0))
        var = tk.StringVar(value="–")
        # Value — large, bold, accent colour, always centered
        tk.Label(inner, textvariable=var,
                 font=("Segoe UI", 16, "bold"),
                 bg="#ffffff", fg=accent).pack(anchor="center", pady=(0, 8))
        return var

    def _current(self) -> ScenarioData:
        return self.scenarios[self.current_index]

    def _switch_scenario(self, idx: int) -> None:
        # If the statistics panel is open, close it first to show the algorithm panel
        if hasattr(self, 'stats_frame') and self.stats_frame.winfo_ismapped():
            self._hide_stats()

        self.current_index = idx
        for i, btn in enumerate(self.algo_buttons):
            if i == idx:
                btn.configure(relief="sunken", bg="#e8f0fe")
            else:
                btn.configure(relief="groove", bg="#f7f7f7")

        scenario = self._current()

        self.tree_summary.configure(text="")

        # Structure card
        self._depth_val.configure( text=str(scenario.summary['depth']))
        self._nodes_val.configure( text=str(scenario.summary['nodes']))
        self._leaves_val.configure(text=str(scenario.summary['leaves']))

        # CV card — parse "Cross-validation\nCV Mean F1: X\nCV Mean ROC-AUC: Y"
        cv_lines = scenario.cv_text.strip().split("\n")
        def _parse_cv_val(lines, keyword):
            for ln in lines:
                if keyword.lower() in ln.lower() and ":" in ln:
                    return ln.split(":", 1)[1].strip()
            return "n/a"
        self._cv_f1_val.configure( text=_parse_cv_val(cv_lines, "F1"))
        self._cv_auc_val.configure(text=_parse_cv_val(cv_lines, "ROC"))

        # Hyperparameters card — rebuild rows from params_text
        for w in self.param_body.winfo_children():
            w.destroy()
        _PARAM_LABELS = {
            "criterion":          "Criterion",
            "max_depth":          "Max Depth",
            "min_samples_split":  "Min Split",
            "min_samples_leaf":   "Min Leaf",
            "class_weight":       "Class Weight",
            "ccp_alpha":          "CCP Alpha",
        }
        for raw_line in scenario.params_text.strip().split("\n"):
            if ":" not in raw_line:
                continue
            k, v = raw_line.split(":", 1)
            k, v = k.strip(), v.strip()
            display_k = _PARAM_LABELS.get(k, k)
            row = tk.Frame(self.param_body, bg="#f6f1e8")
            row.pack(fill="x", padx=8, pady=1)
            tk.Label(row, text=display_k,
                     font=("Segoe UI", 8), bg="#f6f1e8", fg="#888",
                     anchor="w").pack(side="left")
            tk.Label(row, text=v,
                     font=("Segoe UI", 9, "bold"), bg="#f6f1e8", fg="#5a4a3a",
                     anchor="e").pack(side="right")

        for w in self.delta_frame.winfo_children():
            w.destroy()
        delta_lines = scenario.delta_text.strip().split("\n")
        if len(delta_lines) == 1:
            tk.Label(self.delta_frame, text=delta_lines[0],
                     font=("Segoe UI", 9), bg="#eef7ef", fg="#35523a",
                     wraplength=140, justify="left").pack(anchor="w", padx=8)
        else:
            for dl in delta_lines:
                parts = dl.split(":", 1)
                if len(parts) == 2:
                    lbl_f = tk.Frame(self.delta_frame, bg="#eef7ef")
                    lbl_f.pack(anchor="w", fill="x", padx=8)
                    tk.Label(lbl_f, text=parts[0] + ":",
                             font=("Segoe UI", 8), bg="#eef7ef", fg="#888",
                             width=10, anchor="w").pack(side="left")
                    val_str = parts[1].strip()
                    try:
                        val_num = float(val_str)
                        val_color = "#1b7a2e" if val_num > 0 else ("#b71c1c" if val_num < 0 else "#555")
                        val_prefix = "▲ " if val_num > 0 else ("▼ " if val_num < 0 else "")
                    except ValueError:
                        val_color, val_prefix = "#555", ""
                    tk.Label(lbl_f, text=val_prefix + val_str,
                             font=("Segoe UI", 9, "bold"), bg="#eef7ef",
                             fg=val_color).pack(side="right")
                else:
                    tk.Label(self.delta_frame, text=dl, font=("Segoe UI", 8),
                             bg="#eef7ef", fg="#35523a", wraplength=140,
                             padx=8).pack(anchor="w")

        self.metric_vars["train_acc"].set(f"{scenario.metrics['train_acc'] * 100:.1f}%")
        self.metric_vars["test_acc"].set(f"{scenario.metrics['test_acc'] * 100:.1f}%")
        self.metric_vars["error_rate"].set(f"{scenario.metrics['error_rate'] * 100:.1f}%")
        self.metric_vars["recall"].set(f"{scenario.metrics['recall'] * 100:.1f}%")
        self.metric_vars["f1_score"].set(f"{scenario.metrics['f1_score'] * 100:.1f}%")
        self.metric_vars["precision"].set(f"{scenario.metrics['precision'] * 100:.1f}%")
        self.metric_vars["roc_auc"].set(f"{scenario.metrics['roc_auc']:.2f}")
        
        for w in self.cm_frame.winfo_children():
            w.destroy()
        cm = scenario.metrics["confusion_matrix"]
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

        # Column weights so cells fill the full panel width
        self.cm_frame.grid_columnconfigure(0, weight=2)
        self.cm_frame.grid_columnconfigure(1, weight=3)
        self.cm_frame.grid_columnconfigure(2, weight=3)

        _cm_data = [
            ("",                 "Prediction: No Churn", "Prediction: Churn"),
            ("Actual: No Churn", f"{tn}",                f"{fp}"),
            ("Actual: Churn",    f"{fn}",                f"{tp}"),
        ]
        # Color scheme: header row, correct predictions, wrong predictions
        _cm_colors = [
            ("#2c3e50",  "#2c3e50",  "#2c3e50"),
            ("#f9f9f9",  "#d7f0dc",  "#fde8e8"),
            ("#f9f9f9",  "#fde8e8",  "#d7f0dc"),
        ]
        _cm_fg = [
            ("white",   "white",   "white"),
            ("#333",    "#1f6b2d", "#b71c1c"),
            ("#333",    "#b71c1c", "#1f6b2d"),
        ]
        for r, row_data in enumerate(_cm_data):
            for c, cell in enumerate(row_data):
                is_header = (r == 0 or c == 0)
                is_value = r > 0 and c > 0
                tk.Label(
                    self.cm_frame,
                    text=cell,
                    font=("Segoe UI", 10 if is_value else 9, "bold" if is_header else "normal"),
                    bg=_cm_colors[r][c],
                    fg=_cm_fg[r][c],
                    relief="flat",
                    bd=1,
                    padx=6,
                    pady=7,
                ).grid(row=r, column=c, padx=1, pady=1, sticky="nsew")

        for w in self.feat_bar_frame.winfo_children():
            w.destroy()
        bar_max = scenario.top_features[0][1] if scenario.top_features else 1.0
        bar_max = bar_max if bar_max > 0 else 1.0
        _bar_colors = ["#1565c0", "#2e7d32", "#6a1b9a"]
        for i, (fname, fval) in enumerate(scenario.top_features):
            row_f = tk.Frame(self.feat_bar_frame, bg="#f9f9f9")
            row_f.pack(fill="x", pady=(2, 4))
            # Top line: rank + full name + score
            top_row = tk.Frame(row_f, bg="#f9f9f9")
            top_row.pack(fill="x")
            tk.Label(top_row, text=f"{i+1}. {fname}",
                     font=("Segoe UI", 9, "bold"),
                     bg="#f9f9f9", fg="#222", anchor="w").pack(side="left")
            tk.Label(top_row, text=f"{fval:.3f}",
                     font=("Segoe UI", 9, "bold"),
                     bg="#f9f9f9", fg=_bar_colors[i % len(_bar_colors)],
                     anchor="e").pack(side="right")
            # Bottom line: full-width bar
            bar_bg = tk.Frame(row_f, bg="#e4e8ee", height=10)
            bar_bg.pack(fill="x", padx=0)
            bar_bg.pack_propagate(False)
            bar_fill = tk.Frame(bar_bg, bg=_bar_colors[i % len(_bar_colors)], height=10)
            bar_fill.place(x=0, y=0, height=10, relwidth=fval / bar_max)

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
        lines = [l for l in scenario.rules.split("\n") if l.strip()]
        conditions: list[str] = []
        rule_cards: list[tuple[list[str], str]] = []

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

        _styles = {
            "churn": {"tag_bg": "#fee2e2", "tag_fg": "#b91c1c",
                      "line_col": "#f87171", "label": "Prediction: Will Churn"},
            "stay":  {"tag_bg": "#dcfce7", "tag_fg": "#166534",
                      "line_col": "#4ade80", "label": "Prediction: Will Stay"},
        }

        def _style_for(lbl):
            return _styles["churn"] if lbl.strip() in ("1", "Churn") else _styles["stay"]

        if not rule_cards:
            tk.Label(self.rules_frame, text="No rules found.", font=("Segoe UI", 9),
                     bg="#f9f9f9", fg="#888").pack()
            return

        ROW_H  = 30   # px per condition row (fixed for canvas drawing)
        CONN_W = 20   # width of connector canvas

        for conds, label in rule_cards:
            st = _style_for(label)

            # ── Outer card: white, subtle shadow via offset frame ──────────
            shadow = tk.Frame(self.rules_frame, bg="#d1d5db")
            shadow.pack(fill="x", pady=(0, 6))
            card = tk.Frame(shadow, bg="#ffffff")
            card.pack(fill="both", padx=(0, 1), pady=(0, 1))

            # ── Header: thin tag pill + label text ─────────────────────────
            hdr = tk.Frame(card, bg="#ffffff")
            hdr.pack(fill="x", padx=10, pady=(8, 0))

            tag = tk.Label(hdr, text=st["label"],
                           font=("Segoe UI", 8, "bold"),
                           bg=st["tag_bg"], fg=st["tag_fg"],
                           padx=8, pady=2)
            tag.pack(side="left")

            # ── Conditions with drawn connector lines ──────────────────────
            if conds:
                n = len(conds)
                outer_cond = tk.Frame(card, bg="#ffffff")
                outer_cond.pack(fill="x", padx=10, pady=(6, 10))

                # Left canvas for connector lines
                conn_h = n * ROW_H
                conn_cv = tk.Canvas(outer_cond, bg="#ffffff",
                                    width=CONN_W, height=conn_h,
                                    highlightthickness=0)
                conn_cv.pack(side="left", anchor="n")

                labels_col = tk.Frame(outer_cond, bg="#ffffff")
                labels_col.pack(side="left", fill="both", expand=True)

                lx = 8   # x of the vertical trunk line

                for idx, c in enumerate(conds):
                    is_last = idx == n - 1
                    cy_top  = idx * ROW_H
                    cy_mid  = cy_top + ROW_H // 2

                    # Vertical trunk: from top of row to mid (always)
                    conn_cv.create_line(lx, cy_top, lx, cy_mid,
                                       fill="#9ca3af", width=1)
                    if not is_last:
                        # Continue trunk down to next row
                        conn_cv.create_line(lx, cy_mid, lx, cy_top + ROW_H,
                                           fill="#9ca3af", width=1)
                        # Horizontal branch right
                        conn_cv.create_line(lx, cy_mid, CONN_W, cy_mid,
                                           fill="#9ca3af", width=1)
                    else:
                        # Last: L-shaped corner — trunk stops, elbow curves right
                        conn_cv.create_line(lx, cy_mid, CONN_W, cy_mid,
                                           fill="#9ca3af", width=1)

                    # Condition label
                    bg_lbl = "#f9fafb" if idx % 2 == 0 else "#ffffff"
                    tk.Label(labels_col, text=c,
                             font=("Segoe UI", 9),
                             bg=bg_lbl, fg="#374151",
                             anchor="w", padx=8,
                             height=1,
                             justify="left", wraplength=240,
                             ).pack(fill="x", pady=1)
            else:
                tk.Frame(card, bg="#ffffff", height=6).pack()

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

        # Centre the root node in the visible canvas width
        canvas_w = self.canvas.winfo_width()
        if canvas_w <= 1:
            canvas_w = 700
        root_offset_x = float(canvas_w) / 2.0

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
                x1, y1, x2, y2, fill=fill, outline=border_color, width=max(1, int(border_width)), tags=("item", item_tag),
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
    imp3_summary = _load_json_if_exists(REPORT_DIR / "improvement3_summary.json")

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

    imp3_path = MODEL_DIR / "improvement3_pruned.joblib"
    if imp3_path.exists():
        pruned = load_model(imp3_path)
    else:
        pruned = _fit_pruned_tree(X_train, y_train)

    imp3_cv_text = "Cross-validation\nCV Mean F1: n/a\nCV Mean ROC-AUC: n/a"
    if imp3_summary and "improved" in imp3_summary:
        imp3_cv_text = (
            "Cross-validation\n"
            f"CV Mean F1: {imp3_summary['improved'].get('cv_mean_f1', float('nan')):.4f}\n"
            f"CV Mean ROC-AUC: {imp3_summary['improved'].get('cv_mean_roc_auc', float('nan')):.4f}"
        )

    scenarios = [
        baseline_scenario,
        _scenario_payload("Pruned", pruned, X_train, y_train, X_test, y_test, baseline_metrics, cv_text=imp3_cv_text),
    ]

    imp1_path = MODEL_DIR / "improvement1_depth_tuned.joblib"
    if imp1_path.exists():
        imp1_model = load_model(imp1_path)
        imp1_cv_text = "Cross-validation\nCV Mean F1: n/a\nCV Mean ROC-AUC: n/a"
        imp1_tab_name = "max_depth=5"
        if imp1_summary and "improved" in imp1_summary:
            imp1_cv_text = (
                "Cross-validation\n"
                f"CV Mean F1: {imp1_summary['improved'].get('cv_mean_f1', float('nan')):.4f}\n"
                f"CV Mean ROC-AUC: {imp1_summary['improved'].get('cv_mean_roc_auc', float('nan')):.4f}"
            )
        if imp1_summary and "best_params" in imp1_summary:
            best_depth = imp1_summary["best_params"].get("max_depth")
            imp1_tab_name = f"max_depth={best_depth}"
        scenarios.append(
            _scenario_payload(
                imp1_tab_name,
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
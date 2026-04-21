from __future__ import annotations

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.tree import plot_tree

from common import FIGURE_DIR, ensure_directories, get_default_model_path, load_model, load_splits


def visualize_baseline_tree() -> None:
    X_train, X_test, _, y_test = load_splits()

    model = load_model(get_default_model_path())
    y_pred = model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(24, 8))

    ConfusionMatrixDisplay(
        confusion_matrix=conf_matrix,
        display_labels=["No churn", "Churn"],
    ).plot(ax=axes[0], cmap="Blues", colorbar=False)
    axes[0].set_title("Confusion Matrix - Baseline")

    plot_tree(
        model,
        feature_names=list(X_train.columns),
        class_names=["No churn", "Churn"],
        filled=True,
        rounded=True,
        max_depth=3,
        fontsize=8,
        ax=axes[1],
    )
    axes[1].set_title("Decision Tree (top 3 levels)")

    fig.tight_layout()

    output_path = FIGURE_DIR / "baseline_tree_and_confusion_matrix.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Baseline visualization saved.")
    print(f"- Figure path: {output_path}")


if __name__ == "__main__":
    ensure_directories()
    visualize_baseline_tree()

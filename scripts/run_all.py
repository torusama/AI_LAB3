from __future__ import annotations

from analyze_tree import analyze_baseline_tree
from compute_metrics import compute_baseline_metrics
from data_cleaning import clean_and_prepare_data
from train_baseline import train_baseline_decision_tree
from visualize_tree import visualize_baseline_tree
from common import ensure_directories


def run_pipeline() -> None:
    ensure_directories()

    clean_and_prepare_data()
    train_baseline_decision_tree()
    compute_baseline_metrics()
    analyze_baseline_tree()
    
    print("\nFull baseline pipeline finished successfully.")
    print("Starting visualizer...")
    visualize_baseline_tree()


if __name__ == "__main__":
    run_pipeline()

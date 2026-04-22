# Decision Tree Pipeline Scripts

This folder contains the notebook logic extracted into standalone Python scripts.

## Scripts and responsibilities

- `data_cleaning.py`: clean raw data, encode features, split train/test, save to `data/clean`.
- `train_baseline.py`: train baseline `DecisionTreeClassifier`, save model and predictions.
- `compute_metrics.py`: compute Accuracy, Error Rate, Precision, Recall, F1, ROC-AUC, Confusion Matrix.
- `visualize_tree.py`: open a desktop interactive Decision Tree Explorer window (Tkinter, lazy node expand).
- `analyze_tree.py`: analyze depth, nodes/leaves, important splits, representative rules, and fit diagnosis.
- `run_all.py`: execute all steps in order.

## Run order

From project root:

```powershell
python scripts/data_cleaning.py
python scripts/train_baseline.py
python scripts/compute_metrics.py
python scripts/visualize_tree.py
python scripts/analyze_tree.py
```

Or run everything at once:

```powershell
python scripts/run_all.py
```

## Environment notes

- Run commands from project root so relative paths resolve correctly.
- If you use a virtual environment, activate it first, then run the commands above.

## Output locations

- Model: `artifacts/models/baseline_decision_tree.joblib`
- Metrics: `artifacts/reports/baseline_metrics.csv`, `baseline_metrics.json`
- Confusion matrix table: `artifacts/reports/baseline_confusion_matrix.csv`
- Analysis: `artifacts/reports/baseline_tree_analysis.json`, `baseline_tree_analysis.txt`
- Top feature splits: `artifacts/reports/baseline_top_splits.csv`
- Interactive UI: opens in a desktop window when running `python scripts/visualize_tree.py`

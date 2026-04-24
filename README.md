<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1xj-RD3bsIaZr59FddJsGqaoTeAQI5u53" alt="Image 1" width="45%" />
  <img src="https://drive.google.com/uc?export=view&id=1Axkz54BBeZ_BblH9EMBVN-6oZG6301JN" alt="Image 2" width="45%" />
</p>
<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1nH0jzzAKsaC5BB3dRa09hC4Tss8MPVnE" alt="Image 3" width="45%" />
  <img src="https://drive.google.com/uc?export=view&id=16k3HfcEHoFjlbEJQWXQOfpkHpv7qvH8j" alt="Image 4" width="45%" />
</p>

# Lab 3 — Decision Tree Modeling and Improvement
**Course:** Introduction to Artificial Intelligence — 24C08

---

## Team Members

| Name | Student ID |
|---|---|
| Võ Tấn An | 24127318 |
| Mai Khánh Băng | 24127147 |
| Nguyễn Ngọc Minh | 24127204 |
| Trần Minh Hiển | 24127037 |
| Đoàn Võ Ngọc Lâm | 24127435 |

---

## Project Overview

This project applies a **Decision Tree classifier** to the **IBM Telco Customer Churn** dataset. The goal is to build a baseline model, analyze its structure and performance, and propose multiple improvement methods to increase predictive quality.

---

## Repository Structure

```
AI_LAB3/
├── data/
│   ├── raw/                  # Original dataset
│   └── clean/                # Cleaned and encoded splits (X_train, X_test, y_train, y_test)
├── notebooks/
│   └── eda_preprocessing_fixed.ipynb   # EDA, preprocessing, baseline model (notebook)
├── scripts/
│   ├── common.py                        # Shared paths and helper utilities
│   ├── data_cleaning.py                 # Data cleaning, encoding, train/test split
│   ├── train_baseline.py                # Train baseline DecisionTreeClassifier
│   ├── compute_metrics.py               # Compute all evaluation metrics for baseline
│   ├── analyze_tree.py                  # Analyze tree structure, splits, rules, fit diagnosis
│   ├── improvement1_depth_tuning.py     # Improvement 1: Depth tuning + min_samples
│   ├── improvement2_class_weight.py     # Improvement 2: Criterion comparison (Gini / Entropy / Log-loss)
│   ├── improvement3_pruning.py          # Improvement 3: Cost-complexity pruning (ccp_alpha)
│   ├── visualize_tree.py                # Interactive Decision Tree Explorer (Tkinter UI)
│   └── run_all.py                       # Run full pipeline end-to-end
├── artifacts/
│   ├── models/               # Saved .joblib model files
│   └── reports/              # CSV and JSON metric reports
└── requirements.txt
```

---

## Setup

Install dependencies from project root:

```powershell
pip install -r requirements.txt
```

> **Note:** Run all commands from the **project root** folder so relative paths resolve correctly.  
> If using a virtual environment, activate it first.

---

## Run Order

### Step-by-step

```powershell
python scripts/data_cleaning.py
python scripts/train_baseline.py
python scripts/compute_metrics.py
python scripts/analyze_tree.py
python scripts/improvement1_depth_tuning.py
python scripts/improvement2_class_weight.py
python scripts/improvement3_pruning.py
python scripts/visualize_tree.py
```

### Or run everything at once

```powershell
python scripts/run_all.py
```

---

## Scripts and Responsibilities

| Script | Description |
|---|---|
| `data_cleaning.py` | Clean raw data, encode features, split train/test, save to `data/clean` |
| `train_baseline.py` | Train baseline `DecisionTreeClassifier`, save model and predictions |
| `compute_metrics.py` | Compute Accuracy, Error Rate, Precision, Recall, F1, ROC-AUC, Confusion Matrix |
| `analyze_tree.py` | Analyze depth, nodes/leaves, important splits, representative rules, fit diagnosis |
| `improvement1_depth_tuning.py` | Improvement 1 — grid search over `max_depth` and `min_samples` |
| `improvement2_class_weight.py` | Improvement 2 — compare splitting criteria (Gini, Entropy, Log-loss) |
| `improvement3_pruning.py` | Improvement 3 — cost-complexity pruning via `ccp_alpha` grid search |
| `visualize_tree.py` | Interactive Decision Tree Explorer desktop window (Tkinter, lazy node expand) |
| `run_all.py` | Execute all steps above in order |

---

## Output Locations

| Output | Path |
|---|---|
| Baseline model | `artifacts/models/baseline_decision_tree.joblib` |
| Improvement 1 model | `artifacts/models/improvement1_depth_tuned.joblib` |
| Improvement 2 best criterion model | `artifacts/models/improvement2_best_criterion.joblib` |
| Improvement 2 balanced model | `artifacts/models/improvement2_balanced.joblib` |
| Improvement 3 pruned model | `artifacts/models/improvement3_pruned.joblib` |
| Baseline metrics | `artifacts/reports/baseline_metrics.csv`, `baseline_metrics.json` |
| Baseline confusion matrix | `artifacts/reports/baseline_confusion_matrix.csv` |
| Baseline tree analysis | `artifacts/reports/baseline_tree_analysis.json`, `baseline_tree_analysis.txt` |
| Improvement 1 report | `artifacts/reports/improvement1_summary.json`, `improvement1_metrics_comparison.csv` |
| Improvement 2 report | `artifacts/reports/improvement2_summary.json`, `improvement2_metrics_comparison.csv` |
| Improvement 3 report | `artifacts/reports/improvement3_summary.json`, `improvement3_metrics_comparison.csv` |
| Interactive UI | Opens in a desktop window — `python scripts/visualize_tree.py` |

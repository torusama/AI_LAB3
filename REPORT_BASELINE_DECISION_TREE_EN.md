# Baseline Decision Tree Report - Telco Churn

## 1. Tree Figure and Baseline Metrics Table

### Tree figure (with confusion matrix)

![Baseline tree and confusion matrix](artifacts/figures/baseline_tree_and_confusion_matrix.png)

### Baseline metrics table

| Metric Name | Short Formula | Achieved Value |
|---|---|---|
| Accuracy | (TP + TN) / (TP + TN + FP + FN) | 71.86% |
| Error Rate | 1 - Accuracy | 28.14% |
| Precision (Churn class) | TP / (TP + FP) | 47.01% |
| Recall (Churn class) | TP / (TP + FN) | 46.26% |
| F1-score (Churn class) | 2 x Precision x Recall / (Precision + Recall) | 46.63% |
| ROC-AUC | Area under the ROC curve | 0.6366 |

Confusion Matrix:

|  | Pred_NoChurn | Pred_Churn |
|---|---:|---:|
| Actual_NoChurn | 838 | 195 |
| Actual_Churn | 201 | 173 |

The baseline model reaches an Accuracy of 71.86%, with an Error Rate of 28.14%.
The model performs better on the No Churn class, while Churn detection is still limited (Precision and Recall for Churn are both around 46-47%).

## 2. Tree Analysis

### Structural evaluation

- Maximum tree depth: 23
- Total nodes: 2223
- Leaf nodes: 1112

Comment: This tree is structurally complex (very deep and large). With this size, it is more likely to memorize training patterns rather than generalize well to unseen data.

### Root node and important splits

- Root node: tenure <= 16.5
- Early split on both first branches: InternetService_Fiber optic <= 0.5
- Top feature importances:

| Feature | Importance |
|---|---:|
| tenure | 0.2110 |
| TotalCharges | 0.1990 |
| MonthlyCharges | 0.1872 |

Why root matters: tenure is selected at the root because it provides the strongest impurity reduction at the first split. From a business perspective, customer tenure is a strong churn signal because newer customers generally have higher churn risk.

### Representative decision rules

1. IF tenure <= 16.500 AND InternetService_Fiber optic > 0.500 AND TotalCharges > 120.000 THEN Predicted class = Churn (purity about 63.68%).
2. IF tenure <= 16.500 AND InternetService_Fiber optic > 0.500 AND TotalCharges <= 120.000 THEN Predicted class = Churn (purity about 85.94%).
3. IF tenure > 16.500 AND InternetService_Fiber optic <= 0.500 AND Contract_Two year <= 0.500 THEN Predicted class = No Churn (purity about 90.11%).

### Overfit/Underfit diagnosis

- Train Accuracy: 99.88%
- Test Accuracy: 71.86%
- Train-test gap: 28.02 percentage points

Assessment: There is a clear overfitting signal. The model learns the training set extremely well, but its generalization performance on test data drops significantly. With depth 23, the tree is likely over-specialized.

## 3. Technical Report

### Implemented programming techniques for baseline DecisionTree training

- Organized the workflow into standalone scripts for reproducibility and reuse.
- Trained DecisionTreeClassifier(random_state=42) for deterministic results.
- Used cleaned train/test splits as inputs for baseline training.
- Saved the trained model with joblib (.joblib) for later inference and analysis.
- Exported train/test predictions to CSV for traceability and verification.

### Implemented programming techniques for tree visualization

- Used matplotlib to produce a two-panel figure (confusion matrix + tree view).
- Used ConfusionMatrixDisplay for readable classification diagnostics.
- Used plot_tree with max_depth=3 to keep the visualization interpretable.
- Saved the figure at high resolution (dpi=300) for report usage.

### Implemented programming techniques for tree analysis

- Accessed model.tree_ directly to extract depth, node_count, and leaf_count.
- Extracted early splits (root, left child, right child) and feature importances.
- Implemented recursive rule extraction with depth constraints for representative rules.
- Used n_node_samples for stable node sample counts.
- Diagnosed fit quality with train/test accuracy gap and tree depth criteria.

## 4. Short conclusion

The baseline Decision Tree provides a complete baseline package (metrics, visualization, rules, and diagnosis), but it shows strong overfitting. The next step should be pruning and regularization (max_depth, min_samples_leaf, ccp_alpha), followed by re-evaluation with cross-validation.

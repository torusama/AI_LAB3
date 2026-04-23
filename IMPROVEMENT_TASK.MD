# Improvement #1 và #2 — Hướng dẫn cho Codex

## Bối cảnh dự án

Pipeline hiện tại đã có:
- `data_cleaning.py` — làm sạch, encode, split train/test (80/20, stratify, random_state=42)
- `train_baseline.py` — train `DecisionTreeClassifier(random_state=42)` không giới hạn, lưu model
- `compute_metrics.py` — tính Accuracy, Error Rate, Precision, Recall, F1, ROC-AUC, Confusion Matrix
- `analyze_tree.py` — phân tích depth, nodes/leaves, top splits, representative rules, fit diagnosis
- `visualize_tree.py` — Tkinter explorer có sẵn 4 scenario: Baseline / max_depth=5 / Entropy / Pruned
- `common.py` — shared paths & helpers (`load_splits`, `load_model`, `save_json`, ...)
- `run_all.py` — chạy toàn bộ pipeline

Dataset: **Telco Customer Churn** (WA_Fn-UseC_-Telco-Customer-Churn.csv)  
Target: `Churn_Yes` (binary classification, class imbalance ~26% churn)

---

## Chẩn đoán baseline (để Codex hiểu lý do cải tiến)

Baseline tree **không giới hạn** thường có:
- Train accuracy rất cao (~98–100%), test accuracy thấp hơn đáng kể (~78–80%)
- Depth rất lớn (20+), node count hàng trăm → **overfit rõ ràng**
- F1 cho class Churn thấp (~50–55%) vì class imbalance chưa được xử lý

---

## Improvement #1 — Tuning Depth + Min Samples (Chống Overfit)

### Mục tiêu
Giảm overfitting bằng cách giới hạn độ phức tạp của cây.

### File cần tạo: `scripts/improvement1_depth_tuning.py`

#### Việc cần làm:

**1. Grid search thủ công** (không dùng GridSearchCV, viết vòng lặp để dễ hiểu):

```python
param_grid = {
    "max_depth": [3, 5, 7, 10, 15, None],
    "min_samples_split": [2, 10, 20, 50],
    "min_samples_leaf": [1, 5, 10, 20],
}
```

Với mỗi combo:
- Fit `DecisionTreeClassifier(**params, random_state=42)` trên X_train
- Tính train_acc và test_acc
- Tính F1 (churn class) và ROC-AUC trên test set

**2. Chọn best model** theo tiêu chí: test F1-score cao nhất (ưu tiên F1 vì class imbalance)

**3. So sánh** best model vs baseline: tạo DataFrame gồm cả hai, in ra

**4. Lưu outputs:**
- Model: `artifacts/models/improvement1_depth_tuned.joblib`
- Metrics table (baseline vs improved): `artifacts/reports/improvement1_metrics_comparison.csv`
- Full grid results: `artifacts/reports/improvement1_grid_results.csv`
- Summary JSON: `artifacts/reports/improvement1_summary.json`

Format `improvement1_summary.json`:
```json
{
  "method": "Depth Tuning + Min Samples",
  "best_params": { "max_depth": ..., "min_samples_split": ..., "min_samples_leaf": ... },
  "baseline": { "train_acc": ..., "test_acc": ..., "f1_churn": ..., "roc_auc": ... },
  "improved": { "train_acc": ..., "test_acc": ..., "f1_churn": ..., "roc_auc": ... },
  "accuracy_gap_baseline": ...,
  "accuracy_gap_improved": ...,
  "diagnosis": "..."
}
```

`diagnosis` là string giải thích ngắn: giảm gap train-test bao nhiêu, F1 thay đổi ra sao.

**5. In ra terminal** đủ để copy vào report:
```
=== Improvement #1: Depth Tuning + Min Samples ===
Best params: max_depth=X, min_samples_split=Y, min_samples_leaf=Z

             Baseline   Improved
Train Acc    0.xxxx     0.xxxx
Test Acc     0.xxxx     0.xxxx
F1 (Churn)   0.xxxx     0.xxxx
ROC-AUC      0.xxxx     0.xxxx
Gap          0.xxxx     0.xxxx
```

#### Lý thuyết cần viết trong report (Codex KHÔNG cần code phần này, chỉ cần in gợi ý):
- `max_depth` giới hạn chiều sâu → cây không học noise của training data
- `min_samples_split` / `min_samples_leaf` → tránh tạo node từ quá ít mẫu → giảm variance
- Trade-off: depth quá nhỏ → underfit; depth vừa phải → bias-variance balance tốt hơn

---

## Improvement #2 — Class Imbalance Handling với `class_weight='balanced'`

### Mục tiêu
Dataset bị imbalance (~26% churn vs ~74% no-churn). Baseline bias về class 0 (No Churn) → Recall cho Churn thấp. Cải tiến bằng cách điều chỉnh trọng số class.

### File cần tạo: `scripts/improvement2_class_weight.py`

#### Việc cần làm:

**1. Dùng best params từ Improvement #1** (load từ `improvement1_summary.json`):
```python
import json
with open(REPORT_DIR / "improvement1_summary.json") as f:
    imp1 = json.load(f)
best_params = imp1["best_params"]
```

**2. Train 3 variants** để so sánh:
```python
models = {
    "Baseline (no weight)": DecisionTreeClassifier(random_state=42),
    "Imp1 (depth tuned)":   DecisionTreeClassifier(**best_params, random_state=42),
    "Imp2 (balanced)":      DecisionTreeClassifier(**best_params, class_weight="balanced", random_state=42),
}
```

**3. Với mỗi model**, tính đầy đủ:
- Accuracy, Error Rate, Precision, Recall, F1 (churn), ROC-AUC
- Confusion Matrix (quan trọng: xem False Negative giảm không)

**4. Lưu outputs:**
- Model: `artifacts/models/improvement2_balanced.joblib`
- So sánh 3 models: `artifacts/reports/improvement2_metrics_comparison.csv`
- Summary JSON: `artifacts/reports/improvement2_summary.json`

Format `improvement2_summary.json`:
```json
{
  "method": "class_weight=balanced",
  "params_used": { ... },
  "models": {
    "baseline":   { "accuracy": ..., "recall_churn": ..., "f1_churn": ..., "roc_auc": ... },
    "imp1":       { "accuracy": ..., "recall_churn": ..., "f1_churn": ..., "roc_auc": ... },
    "imp2":       { "accuracy": ..., "recall_churn": ..., "f1_churn": ..., "roc_auc": ... }
  },
  "confusion_matrices": {
    "baseline": [[TN, FP], [FN, TP]],
    "imp1":     [[TN, FP], [FN, TP]],
    "imp2":     [[TN, FP], [FN, TP]]
  },
  "diagnosis": "..."
}
```

**5. In ra terminal:**
```
=== Improvement #2: class_weight='balanced' ===

                    Baseline   Imp1(depth)  Imp2(balanced)
Accuracy            0.xxxx     0.xxxx       0.xxxx
Recall (Churn)      0.xxxx     0.xxxx       0.xxxx
Precision (Churn)   0.xxxx     0.xxxx       0.xxxx
F1 (Churn)          0.xxxx     0.xxxx       0.xxxx
ROC-AUC             0.xxxx     0.xxxx       0.xxxx

Confusion Matrix - Imp2 (balanced):
             Pred_NoChurn  Pred_Churn
Actual_NoChurn    xxxx        xxx
Actual_Churn       xxx        xxx
```

#### Lý thuyết cần viết trong report (Codex in gợi ý):
- `class_weight='balanced'` tự tính weight = `n_samples / (n_classes * n_samples_per_class)`
- Kết quả: penalty cho misclassifying minority class (Churn) nặng hơn → cây ưu tiên đúng Churn hơn
- Trade-off: Accuracy tổng có thể giảm nhẹ, nhưng Recall và F1 cho Churn tăng → phù hợp bài toán kinh doanh (cost of missing churner > cost of false alarm)

---

## Cập nhật `run_all.py`

Sau khi tạo 2 file improvement, thêm vào `run_all.py`:

```python
from improvement1_depth_tuning import run_improvement1
from improvement2_class_weight import run_improvement2

# Trong run_pipeline(), sau analyze_baseline_tree():
run_improvement1()
run_improvement2()
```

Mỗi improvement script cần có hàm `run_improvement1()` / `run_improvement2()` bao bọc logic chính (tương tự pattern của các script hiện tại).

---

## Cập nhật `visualize_tree.py` (tùy chọn, nếu có thời gian)

`visualize_baseline_tree()` hiện đã có 4 scenario. Có thể thêm scenario từ Improvement #2:

```python
imp2_model = load_model(MODEL_DIR / "improvement2_balanced.joblib")
scenarios.append(_scenario_payload("Balanced", imp2_model, X_train, y_train, X_test, y_test))
```

---

## Checklist output cuối cùng

```
artifacts/
├── models/
│   ├── baseline_decision_tree.joblib          ✅ có sẵn
│   ├── improvement1_depth_tuned.joblib         ← Imp1 tạo
│   └── improvement2_balanced.joblib            ← Imp2 tạo
└── reports/
    ├── baseline_metrics.csv                    ✅ có sẵn
    ├── improvement1_grid_results.csv           ← Imp1 tạo
    ├── improvement1_metrics_comparison.csv     ← Imp1 tạo
    ├── improvement1_summary.json               ← Imp1 tạo
    ├── improvement2_metrics_comparison.csv     ← Imp2 tạo
    └── improvement2_summary.json              ← Imp2 tạo
```

---

## Lưu ý cho Codex

- Giữ nguyên import pattern từ `common.py`: dùng `load_splits()`, `save_json()`, `REPORT_DIR`, `MODEL_DIR`
- Tất cả script chạy từ thư mục gốc project (không phải từ `scripts/`)
- Dùng `random_state=42` cho tất cả model để reproducible
- Improvement #2 **phụ thuộc** Improvement #1 (cần file `improvement1_summary.json`), nên chạy theo thứ tự
- Không cần vẽ hình (matplotlib/graphviz) — phần visualization đã có `visualize_tree.py`
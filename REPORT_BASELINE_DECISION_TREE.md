# Bao cao Baseline Decision Tree - Telco Churn

## 1. Hinh cay va Bang Metrics Baseline

### Hinh cay (kem confusion matrix)

![Baseline tree and confusion matrix](artifacts/figures/baseline_tree_and_confusion_matrix.png)

### Bang Metrics Baseline

| Ten Metric | Cong thuc rut gon | Gia tri dat duoc |
|---|---|---|
| Accuracy | (TP + TN) / (TP + TN + FP + FN) | 71.86% |
| Error Rate | 1 - Accuracy | 28.14% |
| Precision (lop Churn) | TP / (TP + FP) | 47.01% |
| Recall (lop Churn) | TP / (TP + FN) | 46.26% |
| F1-score (lop Churn) | 2 x Precision x Recall / (Precision + Recall) | 46.63% |
| ROC-AUC | Dien tich duoi duong ROC (TPR theo FPR) | 0.6366 |

Ma tran nham lan (Confusion Matrix):

|  | Pred_NoChurn | Pred_Churn |
|---|---:|---:|
| Actual_NoChurn | 838 | 195 |
| Actual_Churn | 201 | 173 |

Mo hinh co so dat do chinh xac Accuracy la 71.86%, voi ty le loi Error Rate la 28.14%.
Ket qua cho thay mo hinh da phan loai tot lop No Churn, nhung kha nang bat lop Churn con han che (Precision va Recall cua lop Churn deu quanh 46-47%).

## 2. Phan tich cay (Tree Analysis)

### Danh gia cau truc

- Do sau toi da cua cay: 23
- So node: 2223
- So node la: 1112

Nhan xet: Cay co do sau lon va so node/la rat nhieu, cho thay cau truc phuc tap. Voi kich thuoc nay, cay de hoc qua sat du lieu huan luyen va giam kha nang tong quat hoa tren du lieu moi.

### Phan tich nut goc va split quan trong

- Nut goc (root node): tenure <= 16.5
- Split quan trong tiep theo o hai nhanh dau: InternetService_Fiber optic <= 0.5
- Top feature importance:
  - tenure: 0.2110
  - TotalCharges: 0.1990
  - MonthlyCharges: 0.1872

Giai thich: tenure duoc chon lam nut goc vi no giam impurity (Gini) nhieu nhat ngay tu lan tach dau tien. Ve mat nghiep vu, so thang gan bo voi dich vu la tin hieu manh cho hanh vi churn: khach hang moi (tenure thap) thuong co rui ro roi bo cao hon.

### Quy tac quyet dinh tieu bieu (Decision Rules)

1. IF tenure <= 16.500 AND InternetService_Fiber optic > 0.500 AND TotalCharges > 120.000 THEN du doan = Churn (muc do thuan nhat ~ 63.68%).
2. IF tenure <= 16.500 AND InternetService_Fiber optic > 0.500 AND TotalCharges <= 120.000 THEN du doan = Churn (muc do thuan nhat ~ 85.94%).
3. IF tenure > 16.500 AND InternetService_Fiber optic <= 0.500 AND Contract_Two year <= 0.500 THEN du doan = No Churn (muc do thuan nhat ~ 90.11%).

### Chan doan Overfit/Underfit

- Train Accuracy: 99.88%
- Test Accuracy: 71.86%
- Chenh lech train-test: 28.02 diem %

Danh gia: Co dau hieu overfit ro rang. Diem manh la mo hinh hoc rat ky mau huan luyen, nhung diem yeu la kha nang tong quat hoa tren tap test giam manh. Voi do sau hien tai (23), cay co xu huong hoc vet cac mau cu the thay vi hoc quy luat on dinh.

## 3. Bao cao ky thuat (Technical Report)

### Ky thuat lap trinh da ap dung cho Train baseline DecisionTreeClassifier

- To chuc pipeline thanh script rieng de de tai su dung va de test.
- Su dung DecisionTreeClassifier(random_state=42) de dam bao tinh lap lai.
- Tach train/test bang du lieu da clean va train mo hinh baseline khong tuning.
- Luu mo hinh bang joblib (.joblib) de phuc vu suy luan va phan tich ve sau.
- Luu du doan train/test ra CSV de doi chieu va kiem chung.

### Ky thuat lap trinh da ap dung cho Visualize tree

- Dung matplotlib de tao figure gom 2 panel: confusion matrix va decision tree.
- Dung ConfusionMatrixDisplay de hien thi ket qua phan loai truc quan.
- Dung plot_tree voi max_depth=3 de tranh hinh qua roi, de doc duoc nhanh.
- Luu hinh do phan giai cao (dpi=300) phuc vu bao cao.

### Ky thuat lap trinh da ap dung cho Phan tich cay

- Truy cap truc tiep cau truc cay qua model.tree_ de lay depth, node_count, leaf_count.
- Trich xuat split som (root, left child, right child) va feature importance.
- Viet ham de quy trich xuat decision rules dai dien voi gioi han do sau de tranh qua chi tiet.
- Su dung n_node_samples de lay so mau tai tung node mot cach on dinh.
- Chan doan fit quality dua tren train/test accuracy gap va do sau cay de ket luan overfit/underfit.

## 4. Ket luan ngan

Baseline Decision Tree da tao duoc bo ket qua day du (metrics, visualization, rules, diagnosis), nhung mo hinh hien tai overfit ro net. Buoc tiep theo nen thu pruning (max_depth, min_samples_leaf, ccp_alpha) va danh gia lai bang cross-validation de cai thien kha nang tong quat hoa.

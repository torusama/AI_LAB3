# Báo cáo Baseline Decision Tree - Telco Churn

## 1. Hình cây và Bảng Metrics Baseline

### Hình cây (kèm confusion matrix)

![Baseline tree and confusion matrix](artifacts/figures/baseline_tree_and_confusion_matrix.png)

### Bảng Metrics Baseline

| Tên Metric | Công thức rút gọn | Giá trị đạt được |
|---|---|---|
| Accuracy | (TP + TN) / (TP + TN + FP + FN) | 71.86% |
| Error Rate | 1 - Accuracy | 28.14% |
| Precision (lớp Churn) | TP / (TP + FP) | 47.01% |
| Recall (lớp Churn) | TP / (TP + FN) | 46.26% |
| F1-score (lớp Churn) | 2 x Precision x Recall / (Precision + Recall) | 46.63% |
| ROC-AUC | Diện tích dưới đường ROC | 0.6366 |

Ma trận nhầm lẫn (Confusion Matrix):

|  | Pred_NoChurn | Pred_Churn |
|---|---:|---:|
| Actual_NoChurn | 838 | 195 |
| Actual_Churn | 201 | 173 |

Mô hình cơ sở đạt độ chính xác Accuracy là 71.86%, với tỷ lệ lỗi Error Rate là 28.14%.
Kết quả cho thấy mô hình phân loại tốt lớp No Churn, nhưng khả năng phát hiện lớp Churn còn hạn chế (Precision và Recall cho lớp Churn chỉ quanh mức 46-47%).

## 2. Phần phân tích cây (Tree Analysis)

### Đánh giá cấu trúc

- Độ sâu tối đa của cây: 23
- Tổng số node: 2223
- Số node lá: 1112

Nhận xét: Cây có cấu trúc khá phức tạp do độ sâu lớn và số node rất nhiều. Với kích thước này, cây dễ học quá sát dữ liệu huấn luyện và giảm khả năng tổng quát hóa trên dữ liệu mới.

### Phân tích nút gốc và split quan trọng

- Nút gốc (root node): tenure <= 16.5
- Split quan trọng ở hai nhánh đầu: InternetService_Fiber optic <= 0.5
- Top feature importance:

| Đặc trưng | Độ quan trọng |
|---|---:|
| tenure | 0.2110 |
| TotalCharges | 0.1990 |
| MonthlyCharges | 0.1872 |

Giải thích: tenure được chọn làm nút gốc vì nó làm giảm impurity mạnh nhất ngay tại lần tách đầu tiên. Về mặt nghiệp vụ, số tháng gắn bó dịch vụ là tín hiệu quan trọng cho churn, vì nhóm khách hàng mới thường có rủi ro rời bỏ cao hơn.

### Quy tắc quyết định tiêu biểu (Decision Rules)

1. IF tenure <= 16.500 AND InternetService_Fiber optic > 0.500 AND TotalCharges > 120.000 THEN Dự đoán = Churn (độ thuần nhất khoảng 63.68%).
2. IF tenure <= 16.500 AND InternetService_Fiber optic > 0.500 AND TotalCharges <= 120.000 THEN Dự đoán = Churn (độ thuần nhất khoảng 85.94%).
3. IF tenure > 16.500 AND InternetService_Fiber optic <= 0.500 AND Contract_Two year <= 0.500 THEN Dự đoán = No Churn (độ thuần nhất khoảng 90.11%).

### Chẩn đoán Overfit/Underfit

- Train Accuracy: 99.88%
- Test Accuracy: 71.86%
- Chênh lệch train-test: 28.02 điểm %

Đánh giá: Có dấu hiệu overfit rất rõ. Mô hình học gần như hoàn hảo trên tập huấn luyện nhưng giảm đáng kể hiệu năng trên tập kiểm tra. Với độ sâu 23, cây có xu hướng học vẹt các mẫu cụ thể.

## 3. Báo cáo kỹ thuật (Technical Report)

### Kỹ thuật lập trình đã áp dụng cho Train baseline DecisionTreeClassifier

- Tổ chức quy trình thành các script độc lập để tái sử dụng và dễ kiểm thử.
- Huấn luyện DecisionTreeClassifier(random_state=42) để đảm bảo tính lặp lại.
- Dùng dữ liệu đã làm sạch và chia train/test làm đầu vào baseline.
- Lưu mô hình bằng joblib (.joblib) để suy luận và phân tích về sau.
- Xuất dự đoán train/test ra CSV để đối chiếu và kiểm chứng.

### Kỹ thuật lập trình đã áp dụng cho Visualize tree

- Dùng matplotlib để tạo hình gồm 2 panel (confusion matrix và cây quyết định).
- Dùng ConfusionMatrixDisplay để hiển thị kết quả phân loại trực quan.
- Dùng plot_tree với max_depth=3 để giữ hình dễ đọc, dễ diễn giải.
- Lưu ảnh độ phân giải cao (dpi=300) phục vụ báo cáo.

### Kỹ thuật lập trình đã áp dụng cho Phân tích cây

- Truy cập trực tiếp model.tree_ để lấy depth, node_count và leaf_count.
- Trích xuất các split sớm (root, left child, right child) và feature importance.
- Cài đặt hàm đệ quy trích xuất decision rules đại diện với giới hạn độ sâu.
- Dùng n_node_samples để lấy số mẫu tại node ổn định và nhất quán.
- Chẩn đoán chất lượng fit dựa trên chênh lệch train/test accuracy và độ sâu cây.

## 4. Kết luận ngắn

Baseline Decision Tree đã tạo đầy đủ bộ kết quả (metrics, visualization, rules, diagnosis), nhưng đang overfit rõ rệt. Bước tiếp theo nên áp dụng pruning và regularization (max_depth, min_samples_leaf, ccp_alpha), sau đó đánh giá lại bằng cross-validation.

# Guide Viết Báo Cáo Improvement #1 và #2

File này là checklist và khung nội dung để bạn viết phần báo cáo cho 2 cải tiến của mô hình Decision Tree trong bài Lab 3.

---

## 1. Mục tiêu phần báo cáo

Trong phần này, bạn cần chứng minh 3 ý chính:

1. Baseline Decision Tree đang bị overfitting.
2. Improvement #1 giúp giảm overfitting và cải thiện khả năng tổng quát hóa.
3. Improvement #2 so sánh các tiêu chí chia nhánh (`gini`, `entropy`, `log_loss`) để chọn cách tách dữ liệu phù hợp hơn.

---

## 2. Các file cần mở khi viết báo cáo

### Improvement #1

- `artifacts/reports/improvement1_grid_results.csv`
- `artifacts/reports/improvement1_metrics_comparison.csv`
- `artifacts/reports/improvement1_summary.json`
- `artifacts/models/improvement1_depth_tuned.joblib`

### Improvement #2

- `artifacts/reports/improvement2_criterion_comparison.csv`
- `artifacts/reports/improvement2_metrics_comparison.csv`
- `artifacts/reports/improvement2_summary.json`
- `artifacts/models/improvement2_best_criterion.joblib`

### Baseline để đối chiếu

- `artifacts/reports/baseline_metrics.csv`
- `artifacts/reports/baseline_tree_analysis.json`
- `artifacts/reports/baseline_tree_analysis.txt`

---

## 3. Cấu trúc nên viết trong báo cáo

Bạn nên viết theo 5 phần:

1. Bài toán và dữ liệu
2. Baseline model và vấn đề hiện tại
3. Improvement #1: Depth Tuning + Min Samples
4. Improvement #2: Criterion Comparison
5. Kết luận cuối cùng

---

## 4. Phần mở đầu cho báo cáo

### Nội dung cần có

- Tên bài toán: dự đoán khách hàng rời bỏ dịch vụ (`Churn_Yes`)
- Dataset: Telco Customer Churn
- Loại bài toán: binary classification
- Đặc điểm dữ liệu: class imbalance, churn chiếm tỷ lệ thấp hơn no-churn
- Mô hình baseline: `DecisionTreeClassifier(random_state=42)`

### Mẫu viết ngắn

> Bài toán của đồ án là dự đoán khả năng khách hàng rời bỏ dịch vụ viễn thông dựa trên bộ dữ liệu Telco Customer Churn. Đây là bài toán phân loại nhị phân với nhãn mục tiêu là `Churn_Yes`. Mô hình baseline sử dụng Decision Tree không giới hạn độ sâu, do đó có nguy cơ học quá mức trên tập huấn luyện và giảm khả năng tổng quát hóa trên tập kiểm tra.

---

## 5. Phần Baseline: cần viết gì

### Mục tiêu

Chỉ ra baseline bị overfitting.

### Các ý phải nêu

- Train accuracy rất cao nhưng test accuracy thấp hơn đáng kể
- Khoảng cách train-test lớn là dấu hiệu overfitting
- F1 cho lớp churn chưa tốt
- ROC-AUC chưa cao

### Câu nên phân tích

- Cây không giới hạn độ sâu nên dễ học cả nhiễu của dữ liệu train
- Khi cây quá sâu, số node/lá tăng mạnh và variance tăng
- Điều này làm mô hình hoạt động tốt trên train nhưng kém ổn định trên test

### Mẫu viết

> Kết quả baseline cho thấy mô hình đạt độ chính xác rất cao trên tập huấn luyện nhưng giảm đáng kể trên tập kiểm tra. Điều này phản ánh hiện tượng overfitting, tức là mô hình học quá sát dữ liệu train và không tổng quát hóa tốt. Ngoài ra, F1-score cho lớp churn còn hạn chế, cho thấy khả năng nhận diện khách hàng rời bỏ chưa thực sự tốt.

---

## 6. Improvement #1: cần viết gì

## 6.1. Mục tiêu

Giảm overfitting bằng cách kiểm soát độ phức tạp của cây.

## 6.2. Phương pháp cần mô tả

Bạn cần ghi rõ:

- Dùng grid search thủ công trên:
  - `max_depth = [3, 5, 7, 10, 15, None]`
  - `min_samples_split = [2, 10, 20, 50]`
  - `min_samples_leaf = [1, 5, 10, 20]`
- Dùng `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- Tiêu chí chọn mô hình:
  1. `cv_mean_f1` cao nhất
  2. nếu bằng nhau thì chọn `cv_mean_roc_auc` cao hơn
  3. nếu vẫn bằng nhau thì chọn `gap` nhỏ hơn

## 6.3. Ý nghĩa ML cần giải thích

- `max_depth` nhỏ hơn giúp cây bớt phức tạp, giảm variance
- `min_samples_leaf` lớn hơn giúp tránh tạo lá từ quá ít mẫu
- `min_samples_split` kiểm soát việc có nên tách node tiếp hay không
- Cross-validation giúp đánh giá ổn định hơn so với chỉ nhìn 1 train/test split

## 6.4. Số liệu hiện tại của project

Theo `improvement1_summary.json`, mô hình tốt nhất đang là:

- `max_depth = 5`
- `min_samples_split = 2`
- `min_samples_leaf = 10`

So sánh baseline và improved:

| Metric | Baseline | Improved |
|---|---:|---:|
| CV Mean F1 | 0.5119 | 0.5528 |
| CV Mean ROC-AUC | 0.6677 | 0.8296 |
| Train Accuracy | 0.9988 | 0.8011 |
| Test Accuracy | 0.7186 | 0.7783 |
| F1 (Churn) | 0.4663 | 0.5884 |
| ROC-AUC | 0.6366 | 0.8197 |
| Gap | 0.2802 | 0.0228 |

## 6.5. Cách diễn giải kết quả

Bạn nên viết theo các ý sau:

- Gap giảm rất mạnh từ `0.2802` xuống `0.0228`
- Test accuracy tăng từ `0.7186` lên `0.7783`
- F1 cho churn tăng từ `0.4663` lên `0.5884`
- ROC-AUC tăng mạnh từ `0.6366` lên `0.8197`
- CV Mean F1 và CV Mean ROC-AUC đều tăng, chứng tỏ cải thiện ổn định chứ không chỉ tốt trên riêng test set

## 6.6. Mẫu đoạn viết cho Improvement #1

> Improvement #1 tập trung vào việc kiểm soát độ phức tạp của cây quyết định thông qua ba siêu tham số: `max_depth`, `min_samples_split` và `min_samples_leaf`. Thay vì chỉ dựa vào một train/test split, mô hình được đánh giá bằng 5-fold Stratified Cross-Validation để tăng độ tin cậy của kết quả. Mô hình tốt nhất được chọn theo thứ tự ưu tiên: F1-score trung bình cao nhất, ROC-AUC trung bình cao hơn, và khoảng cách train-test nhỏ hơn.

> Kết quả cho thấy bộ tham số tốt nhất là `max_depth=5`, `min_samples_split=2`, `min_samples_leaf=10`. So với baseline, mô hình mới giúp giảm mạnh khoảng cách train-test từ `0.2802` xuống `0.0228`, đồng thời cải thiện test accuracy từ `0.7186` lên `0.7783`. Đặc biệt, F1-score cho lớp churn tăng từ `0.4663` lên `0.5884`, và ROC-AUC tăng từ `0.6366` lên `0.8197`. Điều này cho thấy việc giới hạn độ sâu và tăng số mẫu tối thiểu ở leaf đã giúp giảm overfitting và cải thiện khả năng tổng quát hóa.

---

## 7. Improvement #2: cần viết gì

## 7.1. Mục tiêu

So sánh các tiêu chí chia nhánh của Decision Tree:

- `gini`
- `entropy`
- `log_loss`

Tất cả đều dùng lại best params từ Improvement #1.

## 7.2. Phương pháp cần mô tả

Bạn cần nêu:

- Giữ nguyên:
  - `max_depth = 5`
  - `min_samples_split = 2`
  - `min_samples_leaf = 10`
- Thay đổi `criterion`
- Đánh giá bằng:
  - `cv_mean_f1`
  - `cv_mean_roc_auc`
  - accuracy
  - recall churn
  - F1 churn
  - ROC-AUC
  - gap
- Chọn best criterion bằng cùng thứ tự:
  1. `cv_mean_f1`
  2. `cv_mean_roc_auc`
  3. `gap` nhỏ hơn

## 7.3. Lý thuyết cần giải thích

### `gini`

- Đo mức độ impurity theo xác suất phân lớp sai
- Tính toán nhanh
- Thường là baseline tốt và ổn định

### `entropy`

- Nhạy hơn với sự pha trộn class
- Có thể giúp phân biệt minority class tốt hơn trong một số trường hợp

### `log_loss`

- Gắn với xác suất dự đoán
- Có thể hữu ích khi quan tâm đến chất lượng ranking/probability và ROC-AUC

## 7.4. Số liệu criterion hiện tại của project

Theo `improvement2_summary.json`, kết quả 3 criterion là:

| Criterion | CV Mean F1 | CV Mean ROC-AUC | Accuracy | Recall Churn | F1 Churn | ROC-AUC | Gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| gini | 0.5528 | 0.8296 | 0.7783 | 0.5963 | 0.5884 | 0.8197 | 0.0228 |
| entropy | 0.5554 | 0.8312 | 0.7754 | 0.4733 | 0.5284 | 0.8159 | 0.0173 |
| log_loss | 0.5554 | 0.8312 | 0.7754 | 0.4733 | 0.5284 | 0.8159 | 0.0173 |

Best criterion hiện tại:

- `entropy`

## 7.5. Cách phân tích kết quả

Bạn nên phân tích cẩn thận như sau:

- `entropy` và `log_loss` có CV Mean F1 cùng cao hơn nhẹ so với `gini`
- `entropy` có CV Mean ROC-AUC cao hơn và gap nhỏ hơn
- Vì tiêu chí chọn ưu tiên CV rồi mới xét gap, nên `entropy` được chọn là best criterion
- Tuy nhiên trên holdout test set, `gini` lại có `F1 churn` và `recall churn` cao hơn `entropy`

## 7.6. Cách viết phần thảo luận cho đúng

Đây là điểm rất quan trọng. Bạn không nên viết rằng `entropy` tốt hơn tuyệt đối trên mọi metric. Cách viết đúng là:

- `entropy` được chọn vì kết quả cross-validation ổn định hơn
- `gini` vẫn tốt hơn trên một số chỉ số holdout
- Điều này cho thấy lựa chọn mô hình nên dựa trên độ ổn định tổng thể, không chỉ một test split duy nhất

## 7.7. Mẫu đoạn viết cho Improvement #2

> Sau khi xác định được cấu hình độ sâu phù hợp ở Improvement #1, bước tiếp theo là so sánh các tiêu chí chia nhánh của Decision Tree gồm `gini`, `entropy` và `log_loss`. Tất cả các mô hình đều sử dụng cùng một bộ tham số tối ưu để đảm bảo việc so sánh chỉ tập trung vào ảnh hưởng của `criterion`.

> Kết quả cross-validation cho thấy `entropy` và `log_loss` đạt `CV Mean F1 = 0.5554`, cao hơn nhẹ so với `gini = 0.5528`, đồng thời có `CV Mean ROC-AUC = 0.8312`, nhỉnh hơn so với `0.8296` của `gini`. Ngoài ra, khoảng cách train-test của `entropy` nhỏ hơn (`0.0173` so với `0.0228`), nên `entropy` được chọn là tiêu chí tốt nhất theo quy tắc lựa chọn mô hình đã đặt ra. Tuy nhiên, trên holdout test set, `gini` lại cho `F1 churn` và `recall churn` tốt hơn. Điều này cho thấy `entropy` có lợi thế về độ ổn định tổng thể qua nhiều fold, trong khi `gini` vẫn hoạt động rất cạnh tranh trên tập kiểm tra cụ thể.

---

## 8. Nếu muốn viết thêm phần bonus

Trong code hiện tại còn có thêm 2 thí nghiệm bonus:

- `gini + class_weight="balanced"`
- `gini + class_weight={0:1, 1:2}`

Bạn có thể thêm 1 đoạn ngắn để nhận xét:

- `balanced` tăng recall churn mạnh nhưng accuracy giảm
- `{0:1, 1:2}` cho kết quả khá cân bằng giữa recall, F1 và gap

### Số liệu bonus hiện tại

| Model | CV Mean F1 | Accuracy | Recall Churn | F1 Churn | Gap |
|---|---:|---:|---:|---:|---:|
| gini + balanced | 0.6066 | 0.7043 | 0.7834 | 0.5848 | 0.0286 |
| gini + {0:1, 1:2} | 0.6119 | 0.7783 | 0.6578 | 0.6119 | 0.0118 |

### Cách viết ngắn

> Ngoài phần so sánh criterion theo yêu cầu chính, mô hình còn được thử nghiệm thêm với cơ chế class weighting. Kết quả cho thấy việc tăng trọng số cho lớp churn giúp cải thiện recall đáng kể, đặc biệt trong bối cảnh dữ liệu mất cân bằng. Tuy nhiên, trade-off là số lượng false positive có thể tăng lên. Đây là hướng mở rộng phù hợp nếu mục tiêu nghiệp vụ ưu tiên giảm bỏ sót khách hàng có nguy cơ churn.

---

## 9. Kết luận cuối bài nên viết gì

Bạn nên chốt lại theo logic sau:

1. Baseline bị overfitting rõ rệt.
2. Improvement #1 là cải tiến quan trọng nhất vì giảm mạnh overfitting và tăng mạnh F1/ROC-AUC.
3. Improvement #2 cho thấy việc thay đổi `criterion` có ảnh hưởng, trong đó `entropy` được chọn tốt nhất theo tiêu chí CV.
4. Nếu xét mục tiêu business ưu tiên bắt churn, các cấu hình có class weight cũng rất đáng cân nhắc.

### Mẫu kết luận

> Tóm lại, mô hình baseline Decision Tree ban đầu gặp hiện tượng overfitting rõ rệt. Improvement #1 đã giải quyết vấn đề này hiệu quả bằng cách điều chỉnh độ sâu cây và số lượng mẫu tối thiểu khi chia nhánh hoặc tạo leaf, từ đó giúp giảm mạnh khoảng cách train-test và cải thiện rõ rệt F1-score cho lớp churn. Improvement #2 tiếp tục khảo sát ảnh hưởng của tiêu chí chia nhánh, trong đó `entropy` được chọn là phương án tối ưu nhất theo kết quả cross-validation. Nhìn chung, quá trình cải tiến đã giúp mô hình ổn định hơn, tin cậy hơn và phù hợp hơn cho bài toán dự đoán churn.

---

## 10. Checklist trước khi nộp báo cáo

- Có mô tả bài toán và dataset
- Có nêu baseline và vấn đề overfitting
- Có giải thích vì sao chọn Improvement #1
- Có nêu rõ grid search và cross-validation
- Có bảng so sánh baseline vs improvement #1
- Có giải thích ý nghĩa của `max_depth`, `min_samples_split`, `min_samples_leaf`
- Có mô tả Improvement #2 là so sánh `gini`, `entropy`, `log_loss`
- Có bảng so sánh các criterion
- Có giải thích vì sao chọn `entropy`
- Có kết luận tổng quát

---

## 11. Gợi ý chèn hình hoặc bảng

Nếu bạn muốn báo cáo đẹp hơn, có thể thêm:

- Bảng baseline metrics
- Bảng comparison của Improvement #1
- Bảng criterion comparison của Improvement #2
- Ảnh chụp visualizer cho:
  - Baseline
  - Improved tree
  - Best criterion tree

---

## 12. Gợi ý câu cuối nếu thầy/cô hỏi "improvement nào quan trọng nhất?"

Bạn có thể trả lời:

> Improvement #1 là cải tiến quan trọng nhất vì nó trực tiếp giải quyết hiện tượng overfitting của baseline, giúp tăng cả độ ổn định lẫn chất lượng dự đoán trên lớp churn. Improvement #2 chủ yếu là bước tinh chỉnh thêm để so sánh cách chia nhánh và lựa chọn tiêu chí phù hợp hơn.

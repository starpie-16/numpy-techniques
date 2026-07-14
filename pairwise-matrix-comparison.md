#  NumPy Deep Dive: Kỹ thuật Vector hóa với Kỹ thuật So sánh Chéo (Cross-Comparison)

Tài liệu này giải thích cách kết hợp 4 kỹ thuật đỉnh cao của NumPy: **Slicing nâng cao (`None`)**, **Broadcasting**, **So sánh Boolean (`==`)** và **Giảm chiều (làm "sập chiều" bằng `np.sum`)** để giải quyết bài toán Bag of Words (BoW) trong NLP mà không cần dùng bất kỳ vòng lặp `for` nào.

---

## 1. Kỹ thuật Tạo Ma Trận Vuông Góc & Cơ Chế Broadcasting

Để so sánh **mọi từ trong từ điển** với **mọi từ trong câu** cùng một lúc, chúng ta không thể để hai mảng nằm song song dạng 1 chiều dọc hay 1 chiều ngang. Chúng ta phải ép chúng nằm **vuông góc** với nhau để tạo thành một lưới tọa độ 2 chiều (Pairwise Matrix).

### Cách thiết lập trục:
* **Từ điển (Vocab - kích thước \(V\)):** Dùng `vocab[:, None]` để bẻ mảng thành một **Mảng Cột đứng** kích thước `(V, 1)`.
* **Câu (Sentence - kích thước \(T\)):** Giữ nguyên làm **Mảng Hàng ngang** kích thước `(1, T)`.

### Bản chất tại dấu `==`:
Khi ta viết `vocab[:, None] == sent`, NumPy phát hiện sự lệch kích thước giữa `(V, 1)` và `(1, T)`. Nó kích hoạt cơ chế **Broadcasting (Lan truyền)** để tự động nhân bản dữ liệu ở tầng bộ nhớ C:
* Mảng cột `(V, 1)` tự lặp lại theo chiều ngang \(T\) lần -> `(V, T)`.
* Mảng hàng `(1, T)` tự lặp lại theo chiều dọc \(V\) lần -> `(V, T)`.
* Phép so sánh `==` diễn ra đồng thời trên toàn bộ lưới tọa độ `(V, T)` này.

---

## 2. Quá trình Dry Run Chi Tiết

### Bước 1: Khởi tạo dữ liệu mẫu
* **Vocab (\(V=4\)):** `['anh', 'yêu', 'em', 'nhiều']` -> Mã hóa ID: `[10, 20, 30, 40]`
* **Padded Sentence (\(T=5\)):** `['anh', 'yêu', 'em', '<pad>', '<pad>']` -> Mã hóa ID: `[10, 20, 30, 0, 0]`

### Bước 2: Ép vuông góc và chuẩn bị gộp dữ liệu

Định hình lại `vocab` thành vector cột kích thước `(4, 1)` bằng cách sử dụng `vocab[:, None]`, đặt vuông góc với vector hàng `sent` có kích thước `(1, 5)` để chuẩn bị cho quá trình Broadcasting:

```math
\text{vocab} (4, 1) = \begin{bmatrix} 10 \\ 20 \\ 30 \\ 40 \end{bmatrix} 
\quad , \quad 
\text{sent} (1, 5) = \begin{bmatrix} 10 & 20 & 30 & 0 & 0 \end{bmatrix}
```

### Bước 3: Broadcasting âm thầm diễn ra tại dấu `==` 

Cả hai ma trận được kéo giãn ra cùng kích thước `(4, 5)` để đối đầu trực tiếp từng phần tử: 

```math
\begin{bmatrix} 
10 & 10 & 10 & 10 & 10 \\ 
20 & 20 & 20 & 20 & 20 \\ 
30 & 30 & 30 & 30 & 30 \\ 
40 & 40 & 40 & 40 & 40 
\end{bmatrix} 
== 
\begin{bmatrix} 
10 & 20 & 30 & 0 & 0 \\ 
10 & 20 & 30 & 0 & 0 \\ 
10 & 20 & 30 & 0 & 0 \\ 
10 & 20 & 30 & 0 & 0 
\end{bmatrix}
```

### Bước 4: Kết quả ma trận Boolean (Comparison Matrix)

Mỗi hàng đại diện cho 1 từ trong Vocab, mỗi cột đại diện cho vị trí từ đó xuất hiện trong câu:

```text
[[ True, False, False, False, False], # Từ 'anh' khớp ở vị trí đầu câu
 [False,  True, False, False, False], # Từ 'yêu' khớp ở vị trí thứ hai
 [False, False,  True, False, False], # Từ 'em' khớp ở vị trí thứ ba
 [False, False, False, False, False]] # Từ 'nhiều' không xuất hiện (toàn False)
```

---

## 3. Kỹ thuật "Sập Chiều" Bằng `np.sum(axis=1)` Để Tạo Bản Đồ Tần Số (BoW)

Khi ta thực hiện hàm toán học như `np.sum()` lên ma trận Boolean ở trên, NumPy sẽ tự động ép kiểu ngầm định: `True` thành `1` và `False` thành `0`.

### Quy tắc sập chiều:
* Ma trận Boolean có kích thước `(4, 5)` tương ứng là `(chiều_Vocab, chiều_Câu)`.
* Khi ta chọn **`axis=1`** (tính tổng theo hàng ngang), ta đang ra lệnh: *"Hãy cộng dồn toàn bộ các cột trên từng hàng lại với nhau"*.
* Phép toán này làm **"sập" hoàn toàn chiều dọc của câu (chiều 5 biến mất)**, gộp dữ liệu lại và chỉ giữ lại chiều của từ điển (chiều 4).

```math
\begin{bmatrix} 
1 & 0 & 0 & 0 & 0 \\ 
0 & 1 & 0 & 0 & 0 \\ 
0 & 0 & 1 & 0 & 0 \\ 
0 & 0 & 0 & 0 & 0 
\end{bmatrix} 
\rightarrow 
\begin{bmatrix} 
1+0+0+0+0 \\ 
0+1+0+0+0 \\ 
0+0+1+0+0 \\ 
0+0+0+0+0 
\end{bmatrix} 
= 
\begin{bmatrix} 
1 \\ 
1 \\ 
1 \\ 
0 
\end{bmatrix}
```


Kết quả trả về một Vector Bag of Words 1 chiều phẳng: `[1, 1, 1, 0]`.

---

## 4. Code Mẫu Hoàn Chỉnh (One-liner)

```python
import numpy as np

# Khai báo dữ liệu mã hóa số
vocab = np.array([10, 20, 30, 40])
sent = np.array([10, 20, 30, 0, 0])

# Dòng code tinh túy kết hợp cả 4 kỹ thuật
bow_vector = np.sum(vocab[:, None] == sent, axis=1)

print("Vector Bag of Words:", bow_vector)
# Output: [1 1 1 0]
```

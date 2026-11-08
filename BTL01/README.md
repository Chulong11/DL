# Xấp xỉ hàm Gaussian 2D bằng mạng MLP

Dự án xây dựng và huấn luyện mạng perceptron đa tầng (MLP) bằng PyTorch để xấp xỉ hàm mật độ Gaussian hai chiều. Dự án đồng thời khảo sát ảnh hưởng của:

- Số lượng mẫu huấn luyện — thí nghiệm A.
- Số neuron trong tầng ẩn — thí nghiệm B.
- Số tầng ẩn — thí nghiệm C.

Mỗi cấu hình được chạy với ba random seed `11`, `22`, `33`. Mô hình tốt nhất của mỗi lượt chạy được chọn theo Validation MSE; tập test độc lập chỉ được sử dụng để đánh giá cuối cùng.

## Thành viên

| Họ và tên | MSSV |
|---|---|
| Nhữ Dương Luân | 3123411184 |
| Trần Lê Tuấn Kiệt | 3123411172 |
| Chu Thế Long | 3123411178 |
| Nguyễn Mai Hiển | 3123411093 |

## Bài toán

Hàm mục tiêu là Gaussian hai chiều:

```text
f(x) = 1 / (2π√det(C)) × exp[-1/2 × (x - μ)ᵀ C⁻¹ (x - μ)]
```

Với:

```text
μ = [0, 0]ᵀ
C = [[ 0.3, -0.5],
     [-0.5,  2.0]]
x₁ ∈ [-2, 2]
x₂ ∈ [-4, 4]
```

Dữ liệu được tạo với seed `42` và chia thành:

| Tập dữ liệu | Số mẫu | Mục đích |
|---|---:|---|
| Train | 2.000 | Huấn luyện mô hình |
| Validation | 400 | Chọn checkpoint và cấu hình |
| Test | 100 | Đánh giá cuối cùng |

## Cấu hình huấn luyện

| Tham số | Giá trị |
|---|---|
| Framework | PyTorch |
| Activation | Tanh |
| Loss | Mean Squared Error |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 32 |
| Epochs | 100 |
| Seeds | 11, 22, 33 |

Mô hình được cài đặt bằng `torch.nn.Module`, `nn.Linear` và `nn.Tanh`. Vòng lặp huấn luyện sử dụng `DataLoader`, `model.train()`, `zero_grad()`, `backward()`, `optimizer.step()`, `model.eval()` và `torch.no_grad()`.

## Cấu trúc thư mục

```text
BTL01/
├── data/                         # train.csv, validation.csv, test.csv
├── models/                       # định nghĩa MLP và checkpoint .pth
│   └── model.py
├── plots/                        # biểu đồ khảo sát dữ liệu
├── results/
│   ├── runs/                     # loss theo epoch của từng lượt chạy
│   ├── plots/                    # biểu đồ thí nghiệm A/B/C và test
│   ├── experiment_A.csv
│   ├── experiment_B.csv
│   ├── experiment_C.csv
│   └── test_metrics.csv
├── src/
│   ├── generate_data.py          # sinh và chia dữ liệu
│   ├── visualize_data.py         # trực quan hóa dữ liệu Gaussian
│   ├── train.py                  # vòng lặp huấn luyện
│   ├── evaluate.py               # tính MSE trên DataLoader
│   ├── experiments.py            # chạy thí nghiệm A/B/C
│   ├── plot_experiment_A.py      # tổng hợp và vẽ kết quả A
│   ├── plot_experiment_B.py      # tổng hợp và vẽ kết quả B
│   ├── plot_experiment_C.py      # tổng hợp và vẽ kết quả C
│   └── evaluate_best_model.py    # đánh giá checkpoint đã chọn trên test
├── build_presentation.py         # tạo baocao.pptx
├── baocao.docx                   # báo cáo viết
├── baocao.pptx                   # slide thuyết trình
├── requirements.txt
└── README.md
```

## Cài đặt

Yêu cầu Python 3.10 trở lên. Khuyến nghị sử dụng môi trường ảo.

### Windows PowerShell

```powershell
cd BTL01
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Nếu PowerShell không cho phép kích hoạt script, có thể dùng Command Prompt:

```bat
cd BTL01
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Linux hoặc macOS

```bash
cd BTL01
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> Với máy có GPU, có thể cài phiên bản PyTorch/CUDA phù hợp trước, sau đó cài các thư viện còn lại từ `requirements.txt`.

## Cách chạy

Tất cả lệnh dưới đây cần được chạy tại thư mục gốc `BTL01`.

### 1. Sinh lại dữ liệu

```powershell
python src\generate_data.py
```

Lệnh tạo các file:

```text
data/train.csv
data/validation.csv
data/test.csv
```

### 2. Khảo sát và trực quan hóa dữ liệu

```powershell
python src\visualize_data.py
```

Các biểu đồ được lưu trong `plots/`, gồm scatter 3D, bề mặt Gaussian, phân bố dữ liệu và scatter đầu vào.

### 3. Chọn và chạy một thí nghiệm

Mở `src/experiments.py`, thay giá trị:

```python
EXPERIMENT = "A"  # hoặc "B", "C"
```

Sau đó chạy:

```powershell
python src\experiments.py
```

Phạm vi các thí nghiệm:

| Thí nghiệm | Biến thay đổi | Các cấu hình |
|---|---|---|
| A | Số mẫu train | 200, 400, ..., 2.000 |
| B | Số neuron của một tầng ẩn | 1, 2, ..., 10 |
| C | Số tầng ẩn, mỗi tầng 10 neuron | 1, 2, 3, 4 |

Kết quả tổng hợp được lưu tại `results/experiment_A.csv`, `results/experiment_B.csv` hoặc `results/experiment_C.csv`. Loss theo từng epoch được lưu trong `results/runs/`; checkpoint được lưu trong `models/`.

### 4. Tạo biểu đồ kết quả

Sau khi hoàn tất các thí nghiệm, chạy:

```powershell
python src\plot_experiment_A.py
python src\plot_experiment_B.py
python src\plot_experiment_C.py
```

Biểu đồ và các file summary được lưu trong `results/plots/` và `results/`.

### 5. Đánh giá mô hình đã chọn trên test

```powershell
python src\evaluate_best_model.py
```

Script hiện đánh giá checkpoint:

```text
models/C_n2000_h10-10-10-10_seed11.pth
```

Đầu ra:

- `results/test_metrics.csv`: MSE, MAE, RMSE và R².
- `results/plots/test/test_true_vs_pred.png`: giá trị thật so với dự đoán.
- `results/plots/test/true_vs_approximated_surface.png`: bề mặt thật và bề mặt xấp xỉ.

Kết quả hiện tại trên 100 mẫu test:

| Chỉ số | Giá trị |
|---|---:|
| MSE | 0.0000032613 |
| MAE | 0.0014236 |
| RMSE | 0.0018059 |
| R² | 0.9990791 |



## Thứ tự tái lập toàn bộ

```powershell
python src\generate_data.py
python src\visualize_data.py

# Chạy experiments.py ba lần, lần lượt với EXPERIMENT = "A", "B", "C"
python src\experiments.py

python src\plot_experiment_A.py
python src\plot_experiment_B.py
python src\plot_experiment_C.py
python src\evaluate_best_model.py
python build_presentation.py
```

## Kết quả chính

- Khi tăng số mẫu train từ 200 lên 2.000, Mean Validation MSE giảm khoảng 73,5%.
- Trong thí nghiệm độ rộng, cấu hình 9 neuron đạt Mean Validation MSE thấp nhất, khoảng `0.000374`.
- Tăng từ một lên hai tầng ẩn tạo mức cải thiện lớn nhất trong thí nghiệm độ sâu.
- Kiến trúc bốn tầng ẩn, mỗi tầng 10 neuron đạt Test MSE khoảng `3.26 × 10⁻⁶` trên tập test độc lập.

## Ghi chú

- Không dùng tập test để lựa chọn mô hình hoặc siêu tham số.
- Kết quả có thể chênh lệch nhẹ giữa phiên bản PyTorch, hệ điều hành và phần cứng.
- Các thí nghiệm đầy đủ gồm 72 lượt huấn luyện và có thể cần vài phút trên CPU.
- Khi thay checkpoint trong `evaluate_best_model.py`, cần đặt `hidden_layers` đúng với kiến trúc của checkpoint đó.


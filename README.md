# BTL01 — Xấp xỉ hàm Gaussian 2D bằng MLP

Mã nguồn và toàn bộ sản phẩm của bài tập nằm trong thư mục [`BTL01`](BTL01/).

Tài liệu cài đặt, cấu trúc dự án, lệnh chạy thí nghiệm A/B/C, đánh giá test và tạo slide được trình bày đầy đủ tại:

- [`BTL01/README.md`](BTL01/README.md)
- [`BTL01/requirements.txt`](BTL01/requirements.txt)

## Cài đặt nhanh

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cd BTL01
```

## Chạy nhanh

```powershell
python src\generate_data.py
python src\visualize_data.py
python src\experiments.py
python src\plot_experiment_A.py
python src\plot_experiment_B.py
python src\plot_experiment_C.py
python src\evaluate_best_model.py
python build_presentation.py
```

Trước khi chạy `experiments.py`, chọn `EXPERIMENT = "A"`, `"B"` hoặc `"C"` trong file tương ứng. Xem hướng dẫn chi tiết trong [`BTL01/README.md`](BTL01/README.md).

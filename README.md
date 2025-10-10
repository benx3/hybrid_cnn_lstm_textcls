
# Hybrid CNN + LSTM Text Classification

Mô hình phân loại văn bản kết hợp **BiLSTM** (bắt tín hiệu tuần tự dài hạn) và **1D CNN đa-kernel** (mạnh về pattern n-gram cục bộ) chạy **song song** trên cùng embedding của token. Đầu ra đã pool từ hai nhánh được nối lại và đưa qua classifier.

## Nội dung repo

```
.
├── agnews/      # dữ liệu + script cho tập AG News
├── amazon/      # dữ liệu + script cho Amazon reviews
├── imdb/        # dữ liệu + script cho IMDB sentiment
└── more_cnn_lstm.py  # script mô hình Hybrid CNN + BiLSTM
```

## Kiến trúc mô hình (tóm tắt)

- **Embedding** →  
  Nhánh 1: **1D CNN** đa kernel (ví dụ: 3/4/5) + max-pool → vector đặc trưng cục bộ (n-gram).  
  Nhánh 2: **BiLSTM** (+ optional dropout) → pool/last-hidden → vector ngữ cảnh dài hạn.  
- **Concatenate** hai vector → **Classifier (Dense + Softmax/Sigmoid)**.

## Yêu cầu hệ thống

- Python 3.8+
- Thư viện:
  - torch, torchvision, torchaudio
  - numpy, pandas, scikit-learn, tqdm
  - (tùy chọn) torchtext hoặc transformers

Cài đặt:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Cách chạy nhanh

```bash
python more_cnn_lstm.py --dataset agnews --epochs 10 --batch-size 64 --lr 1e-3
```

## Kết quả

- In loss/accuracy mỗi epoch.
- Có thể thêm checkpoint, tensorboard nếu muốn.

## Giấy phép

MIT License (cập nhật nếu cần).

## Đóng góp

PR/Issue luôn hoan nghênh!

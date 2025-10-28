import numpy as np
from src.model_training import train_model
import os
import pickle

DATA_PATH = "data/history_data.pkl"
MODEL_PATH = "models/trained_model.pkl"

def load_initial_data():
    """
    Tải dữ liệu ban đầu (initial data) với 12 cột cố định.
    Đây là nguồn dữ liệu ban đầu cho mô hình 12 features.
    """
    # Dữ liệu giả lập (5 mẫu) - 12 cột (đã bao gồm Streak)
    X_init = np.array([
        # D_N1 | Dlt N-1/N-2 | Dlt N-2/N-3 | Ttl Dlt | Res N-2 | Streak
        # (3)  | (3)         | (3)         | (1)     | (1)     | (1) = 12
        [3, 4, 2,  1, 0, -1,   0, 0, 0,      0,        0,        -1],
        [4, 2, 5,  1, -2, 3,   1, 0, -1,     2,        1,        -2],
        [2, 5, 3, -2, 3, -2,   1, -2, 3,    -1,        0,         1],
        [5, 3, 4,  3, -2, 1,   -2, 3, -2,    2,        1,         2],
        [1, 1, 5, -4, -2, 1,   3, -2, 1,    -5,        0,        -1]
    ])

    y_init = np.array([10, 12, 9, 11, 7])

    if not os.path.exists("data"):
        os.makedirs("data")

    with open(DATA_PATH, "wb") as f:
        pickle.dump({"X": X_init, "y": y_init}, f)

    return X_init, y_init

def load_data_and_retrain(new_features, predicted, actual_total):
    """Tải toàn bộ lịch sử, thêm mẫu dữ liệu mới, và tái huấn luyện mô hình."""
    X = np.array([])
    y = np.array([])
    try:
        with open(DATA_PATH, "rb") as f:
            data = pickle.load(f)
            X = data["X"]
            y = data["y"]
    except (FileNotFoundError, EOFError):
        # Nếu không tìm thấy lịch sử, TẠO MỚI (buộc sử dụng 12 features)
        X, y = load_initial_data() 
        
    # KIỂM TRA ĐỘ CHÍNH XÁC CỦA FEATURE MỚI
    if len(new_features) != 12:
        print(f"Lỗi SHAPE CẤP TÍNH: Features cần 12 phần tử, nhận {len(new_features)}.")
        return # Ngăn chặn huấn luyện với dữ liệu sai kích thước

    X_new = np.array([new_features])
    y_new = np.array([actual_total])

    X_full = np.vstack([X, X_new])
    y_full = np.concatenate([y, y_new])

    with open(DATA_PATH, "wb") as f:
        pickle.dump({"X": X_full, "y": y_full}, f)

    # Huấn luyện mô hình
    train_model(X_full, y_full, model_path=MODEL_PATH)

# --- KHỐI BẢO VỆ HUẤN LUYỆN LẦN ĐẦU ---
if not os.path.exists(MODEL_PATH):
    if not os.path.exists("models"):
        os.makedirs("models")
        
    # Khởi tạo dữ liệu và huấn luyện
    X_init, y_init = load_initial_data()
    train_model(X_init, y_init, model_path=MODEL_PATH)
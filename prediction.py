import pickle
import numpy as np
import os

def predict_next(features, model_path="models/trained_model.pkl"):
    """
    Dự đoán tổng điểm phiên tiếp theo và phân loại Tài/Xỉu/HOLD bằng Soft Thresholding.
    Args:
        features (list/array): Vector 12 đặc trưng.
    Returns:
        dict: Chứa tổng điểm dự đoán ('total') và kết quả phân loại ('result').
    """
    if not os.path.exists(model_path):
        return {"total": np.float64(10.5), "result": "Xỉu"}

    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    except (EOFError, Exception) as e:
        print(f"Lỗi tải mô hình: {e}. Trả về dự đoán mặc định.")
        return {"total": np.float64(10.5), "result": "Xỉu"}

    # Kiểm tra số lượng features
    expected_features = 12
    if len(features) != expected_features:
        if hasattr(model, 'n_features_in_') and model.n_features_in_ != len(features):
             expected_features = model.n_features_in_
        print(f"Lỗi: Mô hình cần {expected_features} features, nhận {len(features)}. Cần tái huấn luyện.")
        return {"total": np.float64(10.5), "result": "Xỉu"}

    X_new = np.array([features])
    predicted_total = model.predict(X_new)[0]
    predicted_total = np.float64(predicted_total)

    # --- CÔNG THỨC MỚI: TỐI ƯU RANH GIỚI VÀ ÁP DỤNG VÙNG CẤM ---
    threshold_low = 10.2
    threshold_high = 10.8

    if predicted_total < threshold_low:
        result = "Xỉu"
    elif predicted_total > threshold_high:
        result = "Tài"
    else:
        result = "HOLD"
        predicted_total = 10.50 # Giá trị đại diện cho HOLD

    return {"total": predicted_total, "result": result}
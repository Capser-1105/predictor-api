import pickle
import numpy as np

def update_model(new_data, model_path="models/trained_model.pkl"):
    """
    Cập nhật mô hình với dữ liệu mới (online learning).
    new_data = {"features": [x1, x2, x3, d1, d2, d3], "label": y}
    """
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Kiểm tra nếu mô hình hỗ trợ partial_fit
    if hasattr(model, "partial_fit"):
        X_new = np.array([new_data["features"]])
        y_new = np.array([new_data["label"]])
        model.partial_fit(X_new, y_new)
        print("✅ Mô hình đã được cập nhật.")
    else:
        print("⚠ Mô hình hiện tại không hỗ trợ online learning. Cần tái huấn luyện.")

    # Lưu lại mô hình
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
``
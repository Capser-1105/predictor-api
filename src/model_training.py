import pickle
import numpy as np
import os
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error

# --- Định nghĩa Custom Scorer ---
def custom_mse_scorer(y_true, y_pred):
    """
    Tính Mean Squared Error (MSE), nhưng tăng cường phạt lỗi gần ranh giới 10.5.
    """
    mse = mean_squared_error(y_true, y_pred)
    
    # Tính toán lỗi bổ sung cho các dự đoán sai gần 10.5
    penalty = 0
    for true_val, pred_val in zip(y_true, y_pred):
        is_true_tai = true_val >= 11
        is_pred_tai = pred_val >= 11
        
        # Phạt nặng nếu dự đoán sai Tài/Xỉu và dự đoán nằm trong vùng nguy hiểm [10.0, 11.0]
        if is_true_tai != is_pred_tai and (10.0 < pred_val < 11.0):
             # Trọng số phạt tăng cường (ví dụ: nhân 5 lần)
             penalty += (pred_val - 10.5)**2 * 5 
             
    # GridSearchCV tìm cách TỐI THIỂU HÓA điểm số
    return mse + penalty 

# Tạo scorer object từ hàm custom
custom_scorer = make_scorer(custom_mse_scorer, greater_is_better=False)

def train_model(X, y, model_path="models/trained_model.pkl"):
    """
    Huấn luyện mô hình XGBoost Regressor sử dụng GridSearchCV để tìm tham số tốt nhất, 
    tập trung vào L1/L2 Regularization để giảm lỗi tuyệt đối.
    """
    # --- Định nghĩa lưới tham số (Parameter Grid) ---
    param_grid = {
        'n_estimators': [150, 200],          
        'max_depth': [3, 4],                 
        'learning_rate': [0.05, 0.08],     
        'subsample': [0.8, 0.9],            
        'colsample_bytree': [0.8, 0.9],
        
        # CẢI THIỆN LỖI TUYỆT ĐỐI: Tinh chỉnh L1 (alpha) và L2 (lambda)
        'reg_alpha': [0.005, 0.01, 0.05],   
        'reg_lambda': [0.5, 1, 2]           
    }

    # Tạo mô hình XGBoost cơ bản
    # Sử dụng objective='reg:squarederror' (MSE) làm nền tảng
    xgb = XGBRegressor(objective='reg:squarederror', random_state=42)

    # --- Thiết lập GridSearchCV ---
    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring=custom_scorer, # Sử dụng custom scorer để ưu tiên ranh giới 10/11
        cv=3,
        n_jobs=-1, # Sử dụng tất cả các CPU cores để tăng tốc
        verbose=1 
    )

    print("Bắt đầu tìm kiếm tham số tốt nhất bằng GridSearchCV...")
    
    # Thực hiện tìm kiếm
    grid_search.fit(X, y)

    print(f"Tham số tốt nhất tìm được: {grid_search.best_params_}")
    print(f"Điểm số tốt nhất (Custom MSE): {grid_search.best_score_}")

    # Lấy mô hình tốt nhất đã được huấn luyện
    best_model = grid_search.best_estimator_

    # --- Lưu mô hình tốt nhất ---
    if not os.path.exists("models"):
        os.makedirs("models")
        
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)

    print(f"✅ Mô hình XGBoost tốt nhất đã được huấn luyện và lưu tại {model_path}")
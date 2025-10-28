import os
from flask import Flask, request, jsonify, render_template
import numpy as np
import base64
import tempfile
import sys
import logging # Thêm logging để gỡ lỗi khởi động

# ----------------------------------------------------
# KHỞI TẠO ỨNG DỤNG FLASK NGAY LẬP TỨC (RẤT QUAN TRỌNG)
# Việc này đảm bảo biến 'app' được định nghĩa dù có lỗi import sau đó
# ----------------------------------------------------
app = Flask(__name__)

# Thiết lập Logger để hiển thị lỗi khởi động rõ ràng hơn
app.logger.setLevel(logging.INFO)

# --- SỬ DỤNG TRY/EXCEPT CHO CÁC IMPORT CỤC BỘ ---
# Nếu bất kỳ file nào trong thư mục src/ hoặc thư mục gốc bị lỗi, 
# code sẽ không dừng lại mà hiển thị lỗi rõ ràng.
try:
    # Thêm thư mục gốc vào path để import các module con
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    from prediction import predict_next
    from train import load_data_and_retrain
    from src.image_processing import extract_dice_values
    from history_manager import load_data, save_data, get_latest_dice_values, calculate_streak
    
except ImportError as e:
    app.logger.error(f"LỖI KHỞI TẠO MODULE CỤC BỘ: {e}")
    # Nếu lỗi, chúng ta có thể dừng server hoặc trả về lỗi
    raise RuntimeError(f"Lỗi khởi động server: Không tìm thấy module cần thiết: {e}") from e

# --- KHỞI TẠO DỮ LIỆU CHUNG (GLOBAL DATA) ---
try:
    dice_history_init, stats_init = load_data()
    app.config["dice_history"] = dice_history_init
    app.config["stats"] = stats_init
    app.logger.info("Dữ liệu lịch sử và thống kê đã được tải thành công.")
except Exception as e:
    app.logger.error(f"LỖI TẢI DỮ LIỆU BAN ĐẦU: {e}")
    raise RuntimeError(f"Lỗi tải dữ liệu: {e}") from e


@app.route("/", methods=["GET"])
def index():
    """Hiển thị giao diện web (Frontend)."""
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Endpoint nhận ảnh Base64, tính 12 đặc trưng và trả về dự đoán Phiên N."""
    data = request.json
    image_base64 = data.get("image")
    
    if not image_base64:
        return jsonify({"error": "Không tìm thấy dữ liệu ảnh."}), 400

    try:
        # 1. Xử lý ảnh
        image_data = base64.b64decode(image_base64.split(",")[1])
        
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            tmp_file.write(image_data)
            image_path = tmp_file.name
        dice_values_curr = extract_dice_values(image_path)
        os.unlink(image_path)

        if len(dice_values_curr) != 3:
            return jsonify({"error": f"Không trích xuất được 3 giá trị xí ngầu (tìm thấy {len(dice_values_curr)}).", "extracted": dice_values_curr}), 400

        # 3. TẠO 12 ĐẶC TRƯNG MỞ RỘNG
        dice_prev_N2, dice_prev_N3 = get_latest_dice_values(app.config["dice_history"])
        deltas_N2_to_N1 = [dice_values_curr[j] - dice_prev_N2[j] for j in range(3)]
        deltas_N3_to_N2 = [dice_prev_N2[j] - dice_prev_N3[j] for j in range(3)]
        total_delta = sum(deltas_N2_to_N1)
        dice_total_N2 = sum(dice_prev_N2)
        result_prev_encoded = 1 if dice_total_N2 >= 11 else 0

        last_streak_type, last_streak_count = calculate_streak(app.config["dice_history"])
        streak_feature = last_streak_count if last_streak_type == "Tài" else -last_streak_count

        features = dice_values_curr + deltas_N2_to_N1 + deltas_N3_to_N2 + [total_delta, result_prev_encoded, streak_feature]

        # 4. Dự đoán Phiên N
        prediction = predict_next(features)
        predicted_total = prediction["total"]
        predicted_result = prediction["result"]
        
        # 5. Xử lý lỗi đặc biệt từ prediction.py
        if predicted_result == "LỖI HỆ THỐNG":
            return jsonify({"error": "LỖI: Mô hình bị lỗi shape (kích thước feature không khớp). Vui lòng xóa file mô hình cũ và khởi động lại server."}), 500

        # 6. Trả về kết quả
        return jsonify({
            "success": True,
            "prediction": {"total": float(predicted_total), "result": predicted_result},
            "history": {
                "dice_prev": dice_prev_N2,
                "dice_curr": dice_values_curr,
                "deltas": deltas_N2_to_N1,
                "features": features
            }
        })

    except Exception as e:
        app.logger.error(f"Prediction error: {e}")
        return jsonify({"error": f"Lỗi xử lý backend: {e}"}), 500


@app.route("/api/update_model", methods=["POST"])
def api_update_model():
    data = request.json
    features = data.get("features")
    actual_total = data.get("actual_total")
    dice_values_curr = data.get("dice_values_curr")
    predicted_result = data.get("predicted_result")

    if not all([features, actual_total, dice_values_curr, predicted_result]):
        return jsonify({"error": "Dữ liệu cập nhật không đầy đủ."}), 400
    if len(features) != 12:
        return jsonify({"error": f"Lỗi: Features cần 12 phần tử, nhận được {len(features)}"}), 400

    try:
        # 1. Tái huấn luyện
        load_data_and_retrain(new_features=features, predicted=0, actual_total=actual_total)

        # 2. Cập nhật thống kê
        actual_result = "Tài" if actual_total >= 11 else "Xỉu"
        is_correct = (predicted_result == actual_result) if predicted_result != "HOLD" else False 
        app.config["stats"].append({
            "features": features, "predicted": predicted_result,
            "actual": actual_result, "is_correct": is_correct
        })

        # 3. Cập nhật lịch sử xí ngầu
        new_dice_entry = {"dice": dice_values_curr, "total": actual_total}
        app.config["dice_history"].append(new_dice_entry)

        # 4. Lưu dữ liệu
        save_data(app.config["dice_history"], app.config["stats"])

        return jsonify({"success": True, "message": "Mô hình và lịch sử đã được cập nhật thành công!"})

    except Exception as e:
        app.logger.error(f"Update error: {e}")
        return jsonify({"error": f"Lỗi cập nhật mô hình: {e}"}), 500


@app.route("/api/stats", methods=["GET"])
def api_stats():
    total_predictions = len(app.config["stats"])
    valid_stats = [s for s in app.config["stats"] if s["predicted"] != "HOLD"]
    correct_predictions = sum(1 for stat in valid_stats if stat["is_correct"])
    total_valid_predictions = len(valid_stats)

    accuracy = (correct_predictions / total_valid_predictions) * 100 if total_valid_predictions > 0 else 0

    return jsonify({
        "total": total_predictions,
        "correct": correct_predictions,
        "incorrect": total_valid_predictions - correct_predictions,
        "accuracy": round(accuracy, 2)
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)

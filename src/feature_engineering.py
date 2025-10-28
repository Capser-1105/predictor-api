import numpy as np
def create_features(history):
    """
    Tạo đặc trưng từ lịch sử:
    - Giá trị 3 viên xí ngầu
    - Độ lệch so với phiên trước
    """
    features = []
    for i in range(1, len(history)):
        prev = history[i-1]["dice"]
        curr = history[i]["dice"]
        deltas = [curr[j] - prev[j] for j in range(3)]
        features.append(prev + deltas)
    return np.array(features)

def create_labels(history):
    """
    Tạo nhãn (tổng điểm phiên tiếp theo).
    """
    labels = [sum(history[i]["dice"]) for i in range(1, len(history))]
    return np.array(labels)
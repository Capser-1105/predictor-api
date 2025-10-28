import pickle
import os

HISTORY_PATH = "data/history.pkl"
DICE_HISTORY_KEY = "dice_history"
STATS_KEY = "stats"

def load_data():
    """Tải lịch sử xí ngầu và thống kê từ file hoặc trả về khởi tạo."""
    if not os.path.exists("data"):
        os.makedirs("data")

    try:
        with open(HISTORY_PATH, "rb") as f:
            data = pickle.load(f)
            # Đảm bảo lịch sử ban đầu có ít nhất 2 phiên
            history = data.get(DICE_HISTORY_KEY, [])
            if len(history) < 2:
                history = [{"dice": [3, 3, 3], "total": 9}, {"dice": [3, 3, 3], "total": 9}]
            return history, data.get(STATS_KEY, [])
    except (FileNotFoundError, EOFError):
        initial_history = [{"dice": [3, 3, 3], "total": 9}, {"dice": [3, 3, 3], "total": 9}] # Bắt đầu với 2 phiên giống nhau
        initial_stats = []
        save_data(initial_history, initial_stats)
        return initial_history, initial_stats

def save_data(history, stats):
    """Lưu toàn bộ dữ liệu (lịch sử xí ngầu và thống kê)."""
    if not os.path.exists("data"):
        os.makedirs("data")

    with open(HISTORY_PATH, "wb") as f:
        pickle.dump({DICE_HISTORY_KEY: history, STATS_KEY: stats}, f)

def get_latest_dice_values(history):
    """Lấy giá trị xí ngầu 2 phiên gần nhất (N-2 và N-3)."""
    if len(history) >= 2:
        return history[-1]["dice"], history[-2]["dice"] # N-2, N-3
    elif len(history) == 1:
        return history[-1]["dice"], [3, 3, 3] # N-2, N-3 mặc định
    return [3, 3, 3], [3, 3, 3] # Mặc định N-2, N-3

# --- HÀM MỚI: Tính chuỗi Tài/Xỉu ---
def calculate_streak(history):
    """
    Tính chuỗi Tài/Xỉu liên tiếp gần nhất.
    Trả về: (result_type, count) ví dụ: ('Tài', 3) hoặc ('Xỉu', 2)
    """
    if not history:
        return ('Xỉu', 0) # Mặc định nếu không có lịch sử

    last_result = "Tài" if history[-1]["total"] >= 11 else "Xỉu"
    streak_count = 0

    # Lặp ngược qua lịch sử
    for i in range(len(history) - 1, -1, -1):
        current_result = "Tài" if history[i]["total"] >= 11 else "Xỉu"
        if current_result == last_result:
            streak_count += 1
        else:
            break # Chuỗi bị ngắt

    return last_result, streak_count
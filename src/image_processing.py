import cv2
import pytesseract
import os
import re # Thêm thư viện regex để lọc số hiệu quả hơn

# Cấu hình đường dẫn Tesseract (Chỉ cần thiết trên môi trường Windows local)
try:
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
except:
    # Bỏ qua lỗi nếu không tìm thấy đường dẫn Tesseract
    pass


def extract_dice_values(image_path):
    """
    Trích xuất giá trị xí ngầu từ ảnh bằng OCR.
    Trả về danh sách [dice1, dice2, dice3].
    """
    if not os.path.exists(image_path):
        print(f"Lỗi: Không tìm thấy ảnh tại {image_path}")
        return []

    img = cv2.imread(image_path)
    if img is None:
        print("Lỗi: Không thể đọc file ảnh.")
        return []
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- KỸ THUẬT CẢI TIẾN TIỀN XỬ LÝ (Để xử lý nền tối/chữ trắng) ---
    # 1. Tăng độ tương phản (CLAHE) - Giúp nổi bật số
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(gray)
    
    # 2. Threshold đơn giản để tập trung vào số
    # Thử THRESH_BINARY_INV vì nền tối và chữ sáng
    _, thresh = cv2.threshold(enhanced_img, 150, 255, cv2.THRESH_BINARY)
    
    # Dùng Tesseract OCR để nhận diện số
    # Sử dụng PSM 3 (Phân tích layout ảnh tự động) và loại bỏ config 'digits' 
    # để Tesseract ít bị giới hạn hơn
    config = "--psm 3" 
    
    try:
        # Truyền ảnh đã tiền xử lý để nhận dạng
        text = pytesseract.image_to_string(thresh, config=config)
    except Exception as e:
        print(f"Lỗi Tesseract OCR: {e}. Vui lòng kiểm tra cài đặt Tesseract.")
        return []

    # --- LỌC KÝ TỰ MẠNH MẼ HƠN ---
    # Tìm tất cả các số có 1 chữ số (có thể là xí ngầu)
    raw_numbers = re.findall(r'\d', text)
    
    numbers = []
    for n_str in raw_numbers:
        num = int(n_str)
        # Chỉ chấp nhận giá trị xí ngầu hợp lệ (1-6)
        if 1 <= num <= 6:
            numbers.append(num)

    # Giả định chỉ lấy 3 số đầu tiên
    return numbers[:3]
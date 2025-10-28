# Sử dụng base image của Python
FROM python:3.10-slim

# Cài đặt Tesseract OCR và các phụ thuộc cần thiết
# Đây là bước quan trọng nhất để chạy image_processing.py
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    libleptonica-dev \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# Thiết lập thư mục làm việc trong container
WORKDIR /app

# Sao chép các file cấu hình và cài đặt phụ thuộc
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép toàn bộ dự án vào thư mục /app trong container
COPY . /app

# Thiết lập biến môi trường cho Flask và cổng (Render tự động cung cấp PORT)
ENV FLASK_APP=api_app.py

# Lệnh khởi động ứng dụng bằng Gunicorn
# Gunicorn sẽ chạy biến 'app' trong file 'api_app.py'
CMD exec gunicorn --bind 0.0.0.0:$PORT api_app:app
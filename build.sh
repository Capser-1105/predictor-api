#!/usr/bin/env bash
# Đảm bảo thư mục models/ tồn tại
mkdir -p models
# Chạy script Python để huấn luyện mô hình lần đầu nếu nó chưa tồn tại
python train.py
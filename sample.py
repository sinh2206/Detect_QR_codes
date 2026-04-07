#!/usr/bin/env python3
"""
sample.py - Sử dụng cv2.QRCodeDetector để phát hiện QR code từ danh sách ảnh trong public.csv.
Tạo file sample.csv làm cơ sở đối chiếu với kết quả từ main.py (phương pháp từ đầu).
Chạy: python sample.py --data public.csv --output sample.csv
"""

import argparse
import cv2
import numpy as np
import pandas as pd
import os

def process_image_qrcode_detector(image_path):
    """
    Dùng cv2.QRCodeDetector để tìm tất cả QR code trong ảnh.
    Trả về: (số_lượng, danh_sách_các_bộ_4_góc)
    Mỗi bộ góc là list 4 điểm (x,y).
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Không thể đọc ảnh: {image_path}")
        return 0, []

    detector = cv2.QRCodeDetector()
    # detectAndDecodeMulti trả về (ok, decoded_info, points, straight_qrcode)
    # points là list các numpy array shape (4,2) - tọa độ 4 góc theo thứ tự.
    ok, decoded_info, points, _ = detector.detectAndDecodeMulti(img)
    if not ok or points is None:
        return 0, []

    num_qr = len(points)
    corners_list = []
    for pts in points:
        # pts shape (4,2) -> chuyển thành list tuple (x,y)
        quad = [tuple(map(int, pt)) for pt in pts]
        # Đảm bảo thứ tự: top-left, top-right, bottom-right, bottom-left?
        # points trả về thường theo thứ tự, nhưng để đồng bộ với main.py, ta giữ nguyên.
        corners_list.append(quad)
    return num_qr, corners_list

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="File CSV chứa cột 'image_path'")
    parser.add_argument("--output", default="sample.csv", help="File CSV kết quả")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"Lỗi: không tìm thấy {args.data}")
        return

    df = pd.read_csv(args.data)
    if 'image_path' not in df.columns:
        print("File CSV phải có cột 'image_path'")
        return

    results = []
    for idx, row in df.iterrows():
        img_path = row['image_path']
        print(f"Đang xử lý [{idx+1}/{len(df)}]: {img_path}")
        qty, corners = process_image_qrcode_detector(img_path)

        # Chuyển corners thành chuỗi
        if qty > 0:
            corner_strs = []
            for quad in corners:
                # Mỗi quad: list 4 tuple (x,y)
                quad_str = "[" + ",".join([f"({x},{y})" for (x,y) in quad]) + "]"
                corner_strs.append(quad_str)
            corners_str = ";".join(corner_strs)
        else:
            corners_str = ""

        results.append({
            "image_path": img_path,
            "qr_count": qty,
            "corners": corners_str
        })

    df_out = pd.DataFrame(results)
    df_out.to_csv(args.output, index=False, encoding='utf-8')
    print(f"Hoàn tất! Kết quả đã lưu vào {args.output}")

if __name__ == "__main__":
    main()
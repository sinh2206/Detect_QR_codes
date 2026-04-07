#!/usr/bin/env python3
"""
Phát hiện và định vị QR code trong ảnh bằng phương pháp xử lý ảnh truyền thống.
Chỉ sử dụng numpy và OpenCV cơ bản, không dùng các thư viện giải mã QR hay deep learning.
"""

import argparse
import cv2
import numpy as np
import argparse
import cv2
import numpy as np
import csv
from typing import List, Tuple, Optional

# Định nghĩa kiểu dữ liệu cho dễ đọc
Point = Tuple[int, int]
Contour = np.ndarray
Hierarchy = np.ndarray


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Tiền xử lý ảnh:
    - Chuyển sang ảnh xám
    - Làm mờ Gaussian để giảm nhiễu
    - Adaptive threshold để tách biệt đối tượng khỏi nền
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Dùng adaptive threshold với phương pháp Gaussian, ngưỡng lấy từ trung bình vùng lân cận
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    return binary


def is_square_contour(contour: Contour, eps_factor: float = 0.02) -> bool:
    """
    Kiểm tra xem contour có xấp xỉ hình vuông không (4 đỉnh và gần đều).
    Dùng xấp xỉ đa giác với epsilon = eps_factor * chu vi.
    """
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, eps_factor * peri, True)
    if len(approx) != 4:
        return False
    # Có thể kiểm tra thêm các góc gần 90 độ nếu cần, nhưng tạm thời chỉ cần 4 đỉnh
    return True


def get_contour_center(contour: Contour) -> Point:
    """Tính tâm của contour dùng moment."""
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return (0, 0)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)


def find_finder_patterns(
    binary: np.ndarray,
) -> List[Tuple[Contour, Point, Contour, Contour, Contour]]:
    """
    Tìm tất cả các finder patterns (hình vuông lồng nhau 3 lớp) trong ảnh nhị phân.
    Trả về danh sách các tuple chứa:
        - contour ngoài cùng (finder pattern)
        - tâm của nó
        - contour lớp giữa
        - contour lớp trong
    """
    # Tìm contours với hierarchy đầy đủ
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    if hierarchy is None:
        return []
    hierarchy = hierarchy[0]  # hierarchy là mảng 3D, lấy phần tử đầu

    patterns = []

    for i, cnt in enumerate(contours):
        # Chỉ xét contour không có parent (outermost)
        if hierarchy[i][3] != -1:
            continue

        # Kiểm tra có con không
        child_idx = hierarchy[i][2]
        if child_idx == -1:
            continue

        # Lấy con đầu tiên (giả sử chỉ có một con)
        cnt_child = contours[child_idx]
        # Kiểm tra con có con không
        grandchild_idx = hierarchy[child_idx][2]
        if grandchild_idx == -1:
            continue

        cnt_grandchild = contours[grandchild_idx]
        # Kiểm tra cháu có con không (không nên có)
        if hierarchy[grandchild_idx][2] != -1:
            continue

        # Lúc này ta có 3 cấp: cnt (ngoài), cnt_child (giữa), cnt_grandchild (trong)
        # Kiểm tra hình dạng: cả ba nên xấp xỉ hình vuông
        if not (
            is_square_contour(cnt)
            and is_square_contour(cnt_child)
            and is_square_contour(cnt_grandchild)
        ):
            continue

        # Tính diện tích
        area_outer = cv2.contourArea(cnt)
        area_middle = cv2.contourArea(cnt_child)
        area_inner = cv2.contourArea(cnt_grandchild)

        # Kiểm tra thứ tự diện tích giảm dần
        if not (area_outer > area_middle > area_inner):
            continue

        # Kiểm tra tỷ lệ diện tích so với lý thuyết (49:25:9)
        # Cho phép sai số ± 40%
        if not (1.5 < area_outer / area_middle < 2.5):
            continue
        if not (2.0 < area_middle / area_inner < 3.5):
            continue

        # Kiểm độ đồng tâm: các tâm gần nhau
        center_outer = get_contour_center(cnt)
        center_middle = get_contour_center(cnt_child)
        center_inner = get_contour_center(cnt_grandchild)

        # Tính khoảng cách giữa các tâm, lấy ngưỡng bằng 10% kích thước contour ngoài
        x, y, w, h = cv2.boundingRect(cnt)
        max_dim = max(w, h)
        threshold = max_dim * 0.1

        dist_outer_middle = np.linalg.norm(np.array(center_outer) - np.array(center_middle))
        dist_outer_inner = np.linalg.norm(np.array(center_outer) - np.array(center_inner))
        dist_middle_inner = np.linalg.norm(np.array(center_middle) - np.array(center_inner))

        if max(dist_outer_middle, dist_outer_inner, dist_middle_inner) > threshold:
            continue

        # Nếu tất cả điều kiện thỏa mãn, coi đây là một finder pattern
        patterns.append((cnt, center_outer, cnt_child, cnt_grandchild))

    return patterns


def get_corner_from_pattern(
    pattern_contour: Contour, corner_type: str
) -> Optional[Point]:
    """
    Từ contour của một finder pattern (lớp ngoài), lấy đỉnh tương ứng với góc của QR.
    corner_type: 'tl' (top-left), 'tr' (top-right), 'bl' (bottom-left)
    """
    # Xấp xỉ contour để lấy 4 đỉnh
    peri = cv2.arcLength(pattern_contour, True)
    approx = cv2.approxPolyDP(pattern_contour, 0.02 * peri, True)
    if len(approx) != 4:
        # Nếu xấp xỉ không ra 4 đỉnh, dùng bounding rect tạm
        x, y, w, h = cv2.boundingRect(pattern_contour)
        if corner_type == 'tl':
            return (x, y)
        elif corner_type == 'tr':
            return (x + w, y)
        elif corner_type == 'bl':
            return (x, y + h)
        else:
            return None

    points = approx.reshape(4, 2)

    # Tính các đặc trưng để tìm góc
    sums = points.sum(axis=1)  # x+y
    diffs = points[:, 0] - points[:, 1]  # x-y

    if corner_type == 'tl':
        # Góc trên trái: tổng nhỏ nhất
        idx = np.argmin(sums)
    elif corner_type == 'tr':
        # Góc trên phải: hiệu lớn nhất (x lớn, y nhỏ)
        idx = np.argmax(diffs)
    elif corner_type == 'bl':
        # Góc dưới trái: hiệu âm lớn nhất? thực tế dùng y-x lớn nhất
        # Ta có thể dùng -diffs (vì y-x = -(x-y)), nên y-x lớn nhất ứng với x-y nhỏ nhất
        idx = np.argmin(diffs)
    else:
        return None

    return tuple(map(int, points[idx]))


def group_and_find_corners(
    patterns: List[Tuple[Contour, Point, Contour, Contour]]
) -> List[List[Point]]:
    """
    Nhóm các finder patterns thành từng bộ 3 thuộc cùng một QR code,
    xác định 4 góc của mỗi QR và trả về danh sách các bộ 4 điểm.
    """
    if len(patterns) < 3:
        return []

    # Lấy danh sách các tâm và contour ngoài
    centers = [p[1] for p in patterns]
    outer_contours = [p[0] for p in patterns]

    qr_corners_list = []
    used = set()  # đánh dấu các pattern đã dùng

    # Duyệt tất cả các tổ hợp 3 pattern
    n = len(patterns)
    for i in range(n):
        if i in used:
            continue
        for j in range(i + 1, n):
            if j in used:
                continue
            for k in range(j + 1, n):
                if k in used:
                    continue

                A = np.array(centers[i])
                B = np.array(centers[j])
                C = np.array(centers[k])

                # Tính các vector
                AB = B - A
                AC = C - A
                BC = C - B

                # Kiểm tra xem bộ ba có tạo thành tam giác vuông cân không
                # Thử từng điểm làm đỉnh góc vuông
                candidates = []
                # Tại A
                dot_AB_AC = np.dot(AB, AC)
                len_AB = np.linalg.norm(AB)
                len_AC = np.linalg.norm(AC)
                if abs(dot_AB_AC) < 0.1 * len_AB * len_AC and abs(len_AB - len_AC) < 0.2 * max(len_AB, len_AC):
                    candidates.append((i, j, k, 'A'))
                # Tại B
                BA = -AB
                BC_vec = C - B
                dot_BA_BC = np.dot(BA, BC_vec)
                len_BA = len_AB
                len_BC = np.linalg.norm(BC)
                if abs(dot_BA_BC) < 0.1 * len_BA * len_BC and abs(len_BA - len_BC) < 0.2 * max(len_BA, len_BC):
                    candidates.append((j, i, k, 'B'))
                # Tại C
                CA = -AC
                CB = -BC
                dot_CA_CB = np.dot(CA, CB)
                len_CA = len_AC
                len_CB = len_BC
                if abs(dot_CA_CB) < 0.1 * len_CA * len_CB and abs(len_CA - len_CB) < 0.2 * max(len_CA, len_CB):
                    candidates.append((k, i, j, 'C'))

                if not candidates:
                    continue

                # Lấy candidate đầu tiên (giả sử chỉ có một)
                # candidate chứa: idx_vertex (đỉnh góc vuông), idx1, idx2 (hai đỉnh còn lại)
                idx_v, idx_a, idx_b, _ = candidates[0]

                # Xác định đỉnh góc vuông là top-left (theo định nghĩa)
                tl_idx = idx_v
                # Hai đỉnh kia: cần phân biệt top-right và bottom-left
                p1 = centers[idx_a]
                p2 = centers[idx_b]
                v1 = np.array(p1) - np.array(centers[tl_idx])
                v2 = np.array(p2) - np.array(centers[tl_idx])

                # So sánh thành phần x để quyết định: top-right có x lớn hơn
                if v1[0] > v2[0]:
                    tr_idx, bl_idx = idx_a, idx_b
                else:
                    tr_idx, bl_idx = idx_b, idx_a

                # Lấy các góc từ contour
                tl_contour = outer_contours[tl_idx]
                tr_contour = outer_contours[tr_idx]
                bl_contour = outer_contours[bl_idx]

                tl_pt = get_corner_from_pattern(tl_contour, 'tl')
                tr_pt = get_corner_from_pattern(tr_contour, 'tr')
                bl_pt = get_corner_from_pattern(bl_contour, 'bl')

                if tl_pt is None or tr_pt is None or bl_pt is None:
                    continue

                # Tính góc thứ 4 bằng tổng vector
                tl_arr = np.array(tl_pt)
                tr_arr = np.array(tr_pt)
                bl_arr = np.array(bl_pt)
                br_arr = tl_arr + (tr_arr - tl_arr) + (bl_arr - tl_arr)
                br_pt = tuple(map(int, br_arr))

                qr_corners_list.append([tl_pt, tr_pt, br_pt, bl_pt])  # có thể xoay vòng

                # Đánh dấu các pattern đã dùng
                used.update([tl_idx, tr_idx, bl_idx])
                break  # thoát vòng k sau khi tìm được bộ cho i, j
        # Có thể cần break tiếp, nhưng dùng used nên không cần

    return qr_corners_list


def process_image(image_path: str) -> Tuple[int, List[List[Point]]]:
    """
    Xử lý một ảnh: tìm QR code và trả về số lượng cùng danh sách các bộ 4 góc.
    """
    # Đọc ảnh
    img = cv2.imread(image_path)
    if img is None:
        print(f"Không thể đọc ảnh: {image_path}")
        return 0, []

    # Tiền xử lý
    binary = preprocess_image(img)

    # Tìm finder patterns
    patterns = find_finder_patterns(binary)

    # Nhóm và tìm góc
    qr_corners = group_and_find_corners(patterns)

    return len(qr_corners), qr_corners


def main():
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description="Phát hiện QR code trong ảnh")
    parser.add_argument("--data", required=True, help="Đường dẫn đến file CSV chứa danh sách ảnh")
    args = parser.parse_args()

    # Đọc file CSV
    try:
        with open(args.data, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if 'image_path' not in reader.fieldnames:
                print("File CSV phải có cột 'image_path'")
                return
            
            # Duyệt từng ảnh
            for row in reader:
                img_path = row['image_path']
                print(f"Xử lý ảnh: {img_path}")
                num_qr, corners_list = process_image(img_path)
                print(f"Số lượng QR: {num_qr}")
                for i, corners in enumerate(corners_list):
                    print(f"  QR {i+1}: {corners}")
                print("-" * 50)
    except Exception as e:
        print(f"Lỗi đọc file CSV: {e}")
        return


if __name__ == "__main__":
    main()
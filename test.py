#!/usr/bin/env python3
"""
test.py - Kiểm tra phát hiện QR Code từ một ảnh đơn (hỗ trợ bóng đè, méo phối cảnh)
Cách chạy: python test.py --image <đường_dẫn_ảnh>
Ví dụ: python test.py --image qr_with_shadow.jpg
"""

import argparse
import cv2
import numpy as np
from typing import List, Tuple, Optional

# ------------------- CẤU HÌNH -------------------
MORPH_KERNEL = 3          # kernel size cho morphology
ADAPTIVE_BLOCK = 15       # block size cho adaptive threshold
ADAPTIVE_C = 4
MIN_AREA_RATIO = 1.2      # tỷ lệ diện tích ngoài/giữa tối thiểu
MAX_AREA_RATIO = 4.0      # tỷ lệ tối đa
MIN_SOLIDITY = 0.7        # độ đặc tối thiểu của contour ngoài
MAX_ANGLE_DEVIATION = 30  # độ lệch góc cho phép (độ)
DOT_TOLERANCE = 0.2       # tích vô hướng (cos) cho góc 90°
LEN_TOLERANCE = 0.4       # chênh lệch cạnh góc vuông cho phép


# ------------------- HÀM TIỀN XỬ LÝ -------------------
def preprocess(image: np.ndarray) -> np.ndarray:
    """Chuyển ảnh thành nhị phân, nối các vùng bị đứt do bóng."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, ADAPTIVE_BLOCK, ADAPTIVE_C
    )
    kernel = np.ones((MORPH_KERNEL, MORPH_KERNEL), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    return closed


# ------------------- KIỂM TRA TỨ GIÁC -------------------
def is_quadrilateral(contour: np.ndarray) -> bool:
    """Kiểm tra contour có xấp xỉ tứ giác lồi và góc gần 90° không."""
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    if len(approx) != 4:
        return False
    if not cv2.isContourConvex(approx):
        return False
    pts = approx.reshape(4, 2)
    for i in range(4):
        p1 = pts[i]
        p2 = pts[(i+1)%4]
        p3 = pts[(i+2)%4]
        v1 = p1 - p2
        v2 = p3 - p2
        dot = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            continue
        cos_angle = dot / (norm1 * norm2)
        if abs(cos_angle) > np.cos(np.radians(MAX_ANGLE_DEVIATION)):
            return False
    return True


def contour_center(contour: np.ndarray) -> Tuple[int, int]:
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return (0, 0)
    return (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))


def is_nested(outer: np.ndarray, inner: np.ndarray) -> bool:
    """Kiểm tra inner có nằm gọn trong outer không."""
    x1, y1, w1, h1 = cv2.boundingRect(outer)
    x2, y2, w2, h2 = cv2.boundingRect(inner)
    if not (x1 < x2 and y1 < y2 and x1+w1 > x2+w2 and y1+h1 > y2+h2):
        return False
    cx, cy = contour_center(inner)
    return cv2.pointPolygonTest(outer, (float(cx), float(cy)), False) >= 0


# ------------------- TÌM FINDER PATTERNS -------------------
def find_finder_patterns(binary: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
    """Trả về danh sách (contour_ngoài, tâm) của các finder pattern."""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    quads = []
    for cnt in contours:
        if is_quadrilateral(cnt):
            quads.append((cnt, cv2.contourArea(cnt)))
    quads.sort(key=lambda x: x[1], reverse=True)

    n = len(quads)
    used = [False] * n
    patterns = []

    for i in range(n):
        if used[i]:
            continue
        outer_cnt, area_outer = quads[i]
        # Tìm hai tứ giác nhỏ hơn nằm trong outer_cnt
        inner_candidates = []
        for j in range(i+1, n):
            if used[j]:
                continue
            inner_cnt, area_inner = quads[j]
            if area_inner >= area_outer:
                continue
            if is_nested(outer_cnt, inner_cnt):
                inner_candidates.append((j, inner_cnt, area_inner))
        if len(inner_candidates) < 2:
            continue
        inner_candidates.sort(key=lambda x: x[2], reverse=True)
        mid_idx, mid_cnt, area_mid = inner_candidates[0]
        inner_idx, inner_cnt, area_inner = inner_candidates[-1]
        if not is_nested(mid_cnt, inner_cnt):
            continue
        ratio_om = area_outer / area_mid
        ratio_mi = area_mid / area_inner
        if not (MIN_AREA_RATIO < ratio_om < MAX_AREA_RATIO and
                MIN_AREA_RATIO < ratio_mi < MAX_AREA_RATIO):
            continue
        hull = cv2.convexHull(outer_cnt)
        solidity = area_outer / cv2.contourArea(hull)
        if solidity < MIN_SOLIDITY:
            continue
        patterns.append((outer_cnt, contour_center(outer_cnt)))
        used[i] = used[mid_idx] = used[inner_idx] = True
    return patterns


# ------------------- LẤY GÓC TỪ PATTERN -------------------
def get_corner(contour: np.ndarray, corner_type: str) -> Optional[Tuple[int, int]]:
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    if len(approx) != 4:
        x, y, w, h = cv2.boundingRect(contour)
        if corner_type == 'tl':
            return (x, y)
        elif corner_type == 'tr':
            return (x+w, y)
        elif corner_type == 'bl':
            return (x, y+h)
        return None
    pts = approx.reshape(4, 2)
    sums = pts.sum(axis=1)
    diffs = pts[:, 0] - pts[:, 1]
    if corner_type == 'tl':
        idx = np.argmin(sums)
    elif corner_type == 'tr':
        idx = np.argmax(diffs)
    elif corner_type == 'bl':
        idx = np.argmin(diffs)
    else:
        return None
    return tuple(pts[idx])


# ------------------- NHÓM 3 PATTERN THÀNH QR -------------------
def group_to_qr(patterns: List[Tuple[np.ndarray, Tuple[int, int]]]) -> List[List[Tuple[int, int]]]:
    if len(patterns) < 3:
        return []
    centers = [p[1] for p in patterns]
    contours = [p[0] for p in patterns]
    n = len(patterns)
    used = [False] * n
    qr_list = []

    for i in range(n):
        if used[i]:
            continue
        for j in range(i+1, n):
            if used[j]:
                continue
            for k in range(j+1, n):
                if used[k]:
                    continue
                A = np.array(centers[i])
                B = np.array(centers[j])
                C = np.array(centers[k])
                AB = B - A
                AC = C - A
                BC = C - B

                # Đỉnh A
                dot_AB_AC = np.dot(AB, AC)
                len_AB = np.linalg.norm(AB)
                len_AC = np.linalg.norm(AC)
                if abs(dot_AB_AC) < DOT_TOLERANCE * len_AB * len_AC and \
                   abs(len_AB - len_AC) < LEN_TOLERANCE * max(len_AB, len_AC):
                    tl, a1, a2 = i, j, k
                    if B[0] > C[0]:
                        tr, bl = j, k
                    else:
                        tr, bl = k, j
                    tl_pt = get_corner(contours[tl], 'tl')
                    tr_pt = get_corner(contours[tr], 'tr')
                    bl_pt = get_corner(contours[bl], 'bl')
                    if None in (tl_pt, tr_pt, bl_pt):
                        continue
                    tl_arr = np.array(tl_pt)
                    tr_arr = np.array(tr_pt)
                    bl_arr = np.array(bl_pt)
                    br_arr = tl_arr + (tr_arr - tl_arr) + (bl_arr - tl_arr)
                    qr_list.append([tl_pt, tr_pt, tuple(map(int, br_arr)), bl_pt])
                    used[tl] = used[tr] = used[bl] = True
                    break

                # Đỉnh B
                BA = -AB
                BC_vec = C - B
                dot_BA_BC = np.dot(BA, BC_vec)
                len_BA = len_AB
                len_BC = np.linalg.norm(BC_vec)
                if abs(dot_BA_BC) < DOT_TOLERANCE * len_BA * len_BC and \
                   abs(len_BA - len_BC) < LEN_TOLERANCE * max(len_BA, len_BC):
                    tl, a1, a2 = j, i, k
                    if centers[i][0] > centers[k][0]:
                        tr, bl = i, k
                    else:
                        tr, bl = k, i
                    tl_pt = get_corner(contours[tl], 'tl')
                    tr_pt = get_corner(contours[tr], 'tr')
                    bl_pt = get_corner(contours[bl], 'bl')
                    if None in (tl_pt, tr_pt, bl_pt):
                        continue
                    tl_arr = np.array(tl_pt)
                    tr_arr = np.array(tr_pt)
                    bl_arr = np.array(bl_pt)
                    br_arr = tl_arr + (tr_arr - tl_arr) + (bl_arr - tl_arr)
                    qr_list.append([tl_pt, tr_pt, tuple(map(int, br_arr)), bl_pt])
                    used[tl] = used[tr] = used[bl] = True
                    break

                # Đỉnh C
                CA = -AC
                CB = -BC
                dot_CA_CB = np.dot(CA, CB)
                len_CA = len_AC
                len_CB = len_BC
                if abs(dot_CA_CB) < DOT_TOLERANCE * len_CA * len_CB and \
                   abs(len_CA - len_CB) < LEN_TOLERANCE * max(len_CA, len_CB):
                    tl, a1, a2 = k, i, j
                    if centers[i][0] > centers[j][0]:
                        tr, bl = i, j
                    else:
                        tr, bl = j, i
                    tl_pt = get_corner(contours[tl], 'tl')
                    tr_pt = get_corner(contours[tr], 'tr')
                    bl_pt = get_corner(contours[bl], 'bl')
                    if None in (tl_pt, tr_pt, bl_pt):
                        continue
                    tl_arr = np.array(tl_pt)
                    tr_arr = np.array(tr_pt)
                    bl_arr = np.array(bl_pt)
                    br_arr = tl_arr + (tr_arr - tl_arr) + (bl_arr - tl_arr)
                    qr_list.append([tl_pt, tr_pt, tuple(map(int, br_arr)), bl_pt])
                    used[tl] = used[tr] = used[bl] = True
                    break
            if used[i]:
                break
    return qr_list


# ------------------- XỬ LÝ ẢNH -------------------
def detect_qr(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Không thể đọc ảnh: {image_path}")
        return
    binary = preprocess(img)
    patterns = find_finder_patterns(binary)
    qr_corners = group_to_qr(patterns)
    print(f"📷 Ảnh: {image_path}")
    print(f"🔢 Số lượng QR code phát hiện: {len(qr_corners)}")
    for idx, quad in enumerate(qr_corners, 1):
        print(f"  QR {idx}: {quad}")
    if len(qr_corners) == 0:
        print("⚠️ Không tìm thấy QR code nào (có thể do bóng đè quá nặng hoặc méo quá mức).")


# ------------------- MAIN -------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phát hiện QR Code từ ảnh (chịu bóng, méo)")
    parser.add_argument("--image", required=True, help="Đường dẫn đến file ảnh cần kiểm tra")
    args = parser.parse_args()
    detect_qr(args.image)
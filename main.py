#!/usr/bin/env python3
"""
Phat hien va dinh vi QR code bang xu ly anh truyen thong.
Khong dung QRCodeDetector/deep learning.
CAI TIEN V2:
- Them refine_to_axis_aligned_bbox() vi toan bo GT la axis-aligned rectangles
- Them detect_qr_by_outline() phat hien QR qua contour truc tiep
- Tinh chinh toa do chinh xac hon bang cach quet vung binary
"""

import argparse  # Xu li tham so command-line
import csv  # Doc/ghi file CSV
import os  # Lam viec voi duong dan file
import time  # Do thoi gian chay chuong trinh (requirement 5.5)
from typing import Dict, List, Optional, Tuple  # Type hints

import cv2  # OpenCV - xu ly hinh anh
import numpy as np  # NumPy - tinh toan ma tran

Point = Tuple[int, int]  # Diem 2D (x, y)
Contour = np.ndarray  # Duong contour tu OpenCV


def to_console_safe(text: str) -> str:
    return str(text).encode("ascii", errors="backslashreplace").decode("ascii")


def read_image_any_path(image_path: str) -> Optional[np.ndarray]:
    has_non_ascii = any(ord(ch) > 127 for ch in str(image_path))
    if has_non_ascii:
        try:
            buffer = np.fromfile(image_path, dtype=np.uint8)
            if buffer.size > 0:
                decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
                if decoded is not None:
                    return decoded
        except Exception:
            pass
    img = cv2.imread(image_path)
    if img is not None:
        return img
    try:
        buffer = np.fromfile(image_path, dtype=np.uint8)
        if buffer.size == 0:
            return None
        return cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    except Exception:
        return None


def order_quad_clockwise_start_top_left(quad: np.ndarray) -> np.ndarray:
    q = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    center = np.mean(q, axis=0)
    ang = np.arctan2(q[:, 1] - center[1], q[:, 0] - center[0])
    q = q[np.argsort(ang)]
    start = int(np.argmin(q[:, 0] + q[:, 1]))
    return np.roll(q, -start, axis=0)


def bbox_iou_xyxy(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = float(iw * ih)
    area_a = float(max(0, ax2 - ax1) * max(0, ay2 - ay1))
    area_b = float(max(0, bx2 - bx1) * max(0, by2 - by1))
    union = area_a + area_b - inter
    if union <= 1e-9:
        return 0.0
    return inter / union


def preprocess_image(image: np.ndarray, variant: int = 0) -> np.ndarray:
    if image is None or image.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    h, w = gray.shape

    if variant == 1:
        gray = cv2.equalizeHist(gray)
        gray = cv2.GaussianBlur(gray, (0, 0), 0.8)
        sharpening_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        gray = cv2.filter2D(gray, -1, sharpening_kernel)
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    sigma = max(3.0, min(h, w) / 32.0)
    gray_f = gray.astype(np.float32)
    background = cv2.GaussianBlur(gray_f, (0, 0), sigmaX=sigma, sigmaY=sigma)
    normalized = cv2.divide(gray_f, background + 1.0, scale=255.0)
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)

    clip_limit = 3.2 if variant != 2 else 2.5
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8 if min(h, w) >= 480 else 6,) * 2)
    enhanced = clahe.apply(normalized)
    denoised = cv2.medianBlur(enhanced, 3)

    sharp_weight = 1.35 if variant != 2 else 1.25
    sharpened = cv2.addWeighted(
        denoised, sharp_weight,
        cv2.GaussianBlur(denoised, (0, 0), 1.2), -(sharp_weight - 1.0), 0,
    )

    block = max(15, min(51, ((min(h, w) // 24) | 1)))
    if variant == 1:
        block = max(11, block - 4)

    binary_g = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, 3)
    binary_m = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block, 5)
    binary_g_sharp = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, 2)
    _, binary_o = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, binary_o_sharp = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary = cv2.bitwise_or(cv2.bitwise_or(binary_g, binary_m), cv2.bitwise_or(binary_o, binary_o_sharp))
    binary = cv2.bitwise_or(binary, binary_g_sharp)

    if variant == 1:
        _, binary_enh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        binary = cv2.bitwise_or(binary, binary_enh)

    k = max(3, min(7, ((min(h, w) // 220) | 1)))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    min_area = max(3000, int(h * w * 0.00008))
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    for lab in range(1, num_labels):
        if stats[lab, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == lab] = 255
    return cleaned


def find_finder_patterns(binary: np.ndarray) -> List[Tuple[Contour, Point, float]]:
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    hierarchy = hierarchy[0]

    patterns: List[Tuple[Contour, Point, float]] = []

    for i, outer in enumerate(contours):
        first_child = hierarchy[i][2]
        if first_child == -1:
            continue

        cur_child = first_child
        while cur_child != -1:
            middle = contours[cur_child]
            first_grandchild = hierarchy[cur_child][2]
            cur_grandchild = first_grandchild

            while cur_grandchild != -1:
                inner = contours[cur_grandchild]

                ok_shape = True
                centers = []
                areas = []
                for cnt in (outer, middle, inner):
                    area = abs(float(cv2.contourArea(cnt)))
                    peri = cv2.arcLength(cnt, True)
                    if area < 30 or peri < 1e-6:
                        ok_shape = False
                        break

                    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                    if len(approx) < 4 or len(approx) > 8:
                        ok_shape = False
                        break

                    rect = cv2.minAreaRect(cnt)
                    (_, _), (rw, rh), _ = rect
                    if rw < 2 or rh < 2:
                        ok_shape = False
                        break
                    wh_ratio = max(rw, rh) / max(1e-6, min(rw, rh))
                    fill = area / max(1e-6, rw * rh)
                    if wh_ratio > 2.8 or fill < 0.50:
                        ok_shape = False
                        break

                    m = cv2.moments(cnt)
                    if abs(m["m00"]) > 1e-6:
                        cx = int(round(m["m10"] / m["m00"]))
                        cy = int(round(m["m01"] / m["m00"]))
                    else:
                        x, y, w, h = cv2.boundingRect(cnt)
                        cx, cy = x + w // 2, y + h // 2
                    centers.append((cx, cy))
                    areas.append(area)

                if ok_shape:
                    area_o, area_m, area_i = areas
                    if area_o > area_m > area_i and area_i > 18:
                        ratio_om = area_o / max(1e-6, area_m)
                        ratio_mi = area_m / max(1e-6, area_i)
                        if 1.15 < ratio_om < 6.5 and 1.15 < ratio_mi < 8.5:
                            co = np.array(centers[0], dtype=np.float32)
                            cm = np.array(centers[1], dtype=np.float32)
                            ci = np.array(centers[2], dtype=np.float32)
                            _, _, bw, bh = cv2.boundingRect(outer)
                            threshold = max(bw, bh) * 0.35
                            if max(
                                np.linalg.norm(co - cm),
                                np.linalg.norm(co - ci),
                                np.linalg.norm(cm - ci),
                            ) <= threshold:
                                patterns.append((outer, centers[0], area_o))

                cur_grandchild = hierarchy[cur_grandchild][0]
            cur_child = hierarchy[cur_child][0]

    if len(patterns) < 3:
        img_h, img_w = binary.shape[:2]
        img_area = float(img_h * img_w)
        for i, cnt in enumerate(contours):
            child_idx = hierarchy[i][2]
            if child_idx == -1:
                continue

            area = abs(float(cv2.contourArea(cnt)))
            if area < max(500.0, img_area * 0.008) or area > img_area * 0.30:
                continue

            child_area = abs(float(cv2.contourArea(contours[child_idx])))
            child_ratio = child_area / max(1e-6, area)
            if child_ratio < 0.015 or child_ratio > 0.92:
                continue

            peri = cv2.arcLength(cnt, True)
            if peri < 1e-6:
                continue
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) < 4 or len(approx) > 10:
                continue

            rect = cv2.minAreaRect(cnt)
            (_, _), (rw, rh), _ = rect
            if rw < 4 or rh < 4:
                continue
            wh_ratio = max(rw, rh) / max(1e-6, min(rw, rh))
            fill = area / max(1e-6, rw * rh)
            if wh_ratio > 2.5 or fill < 0.35:
                continue

            m = cv2.moments(cnt)
            if abs(m["m00"]) > 1e-6:
                cx = int(round(m["m10"] / m["m00"]))
                cy = int(round(m["m01"] / m["m00"]))
            else:
                x, y, w, h = cv2.boundingRect(cnt)
                cx, cy = x + w // 2, y + h // 2
            patterns.append((cnt, (cx, cy), area))

    unique: List[Tuple[Contour, Point, float]] = []
    for cand in patterns:
        _, c, area = cand
        dup = False
        for _, old_c, old_area in unique:
            dist = np.linalg.norm(np.array(c, dtype=np.float32) - np.array(old_c, dtype=np.float32))
            area_ratio = min(area, old_area) / max(1e-6, max(area, old_area))
            if dist < 10 and area_ratio > 0.78:
                dup = True
                break
        if not dup:
            unique.append(cand)
    return unique


def build_qr_quads(patterns: List[Tuple[Contour, Point, float]], image_shape: Tuple[int, ...]) -> List[np.ndarray]:
    """
    Từ danh sách `patterns` (mỗi phần tử là (contour, center, area)) và kích thước ảnh,
    xây các tứ giác (quads) ứng viên cho QR code.

    Tham số:
      - patterns: List[Tuple[Contour, Point, float]]
          Danh sách finder patterns: (contour, (cx, cy), area).
      - image_shape: Tuple[int,...]
          Kích thước ảnh (H, W, ...), dùng để loại các cấu hình bất hợp lý.

    Trả về:
      - List[np.ndarray]: Mỗi phần tử là mảng shape (4,2) dtype float32 biểu diễn 4 góc
        của tứ giác ứng viên (toạ độ ảnh). Thứ tự điểm có thể chưa sắp xếp theo quy ước,
        các hàm tiếp theo sẽ chuẩn hoá thứ tự nếu cần.
    """
    if len(patterns) < 3:
        return []

    centers = [np.array(p[1], dtype=np.float32) for p in patterns]
    contours = [p[0] for p in patterns]
    areas = [p[2] for p in patterns]
    h, w = image_shape[:2]
    quads: List[np.ndarray] = []

    for i in range(len(patterns)):
        for j in range(i + 1, len(patterns)):
            for k in range(j + 1, len(patterns)):
                idxs = [i, j, k]
                pts = [centers[i], centers[j], centers[k]]

                d01 = float(np.linalg.norm(pts[1] - pts[0]))
                d02 = float(np.linalg.norm(pts[2] - pts[0]))
                d12 = float(np.linalg.norm(pts[2] - pts[1]))
                longest_pair = max([(d01, (0, 1)), (d02, (0, 2)), (d12, (1, 2))], key=lambda x: x[0])[1]
                right_local = ({0, 1, 2} - set(longest_pair)).pop()

                tl_idx = idxs[right_local]
                others = [idx for idx in idxs if idx != tl_idx]
                a_idx, b_idx = others[0], others[1]

                va = centers[a_idx] - centers[tl_idx]
                vb = centers[b_idx] - centers[tl_idx]
                la = float(np.linalg.norm(va))
                lb = float(np.linalg.norm(vb))
                if la < 1e-6 or lb < 1e-6:
                    continue

                cos_val = abs(float(np.dot(va, vb) / (la * lb)))
                len_ratio = min(la, lb) / max(la, lb)
                if cos_val > 0.45 or len_ratio < 0.42:
                    continue

                for tr_idx, bl_idx, ex, ey in (
                    (a_idx, b_idx, va / la, vb / lb),
                    (b_idx, a_idx, vb / lb, va / la),
                ):
                    contour_boxes = []
                    for cidx in (tl_idx, tr_idx, bl_idx):
                        cnt = contours[cidx]
                        peri = cv2.arcLength(cnt, True)
                        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                        if len(approx) == 4:
                            box = approx.reshape(4, 2).astype(np.float32)
                        else:
                            box = cv2.boxPoints(cv2.minAreaRect(cnt)).astype(np.float32)
                        contour_boxes.append(box)

                    corner_dirs = (-ex - ey, ex - ey, -ex + ey)
                    chosen = []
                    for box, d in zip(contour_boxes, corner_dirs):
                        c = np.mean(box, axis=0)
                        dn = d / max(1e-6, np.linalg.norm(d))
                        scores = np.dot(box - c, dn)
                        chosen.append(box[int(np.argmax(scores))])

                    tl_pt, tr_pt, bl_pt = chosen

                    # ---------------------------------------------------
                    # CAI TIEN: Tinh BR dung homography tu 3 tam finder pattern
                    # Thay vi dung cong thuc hinh binh hanh (bi sai khi co perspective distortion)
                    # ---------------------------------------------------
                    # TL_center, TR_center, BL_center trong khong gian anh
                    tl_c = centers[tl_idx]
                    tr_c = centers[tr_idx]
                    bl_c = centers[bl_idx]

                    # Uoc luong kich thuoc module tu finder pattern
                    # Finder pattern = 7 modules, area ~ (7*module)^2
                    fp_area_tl = areas[tl_idx]
                    fp_area_tr = areas[tr_idx]
                    fp_area_bl = areas[bl_idx]
                    avg_fp_area = (fp_area_tl + fp_area_tr + fp_area_bl) / 3.0
                    module_size_px = max(1.0, np.sqrt(avg_fp_area) / 7.0)

                    # Khoang cach giua cac tam finder pattern
                    d_tl_tr = float(np.linalg.norm(tr_c - tl_c))
                    d_tl_bl = float(np.linalg.norm(bl_c - tl_c))

                    # Uoc luong version QR: dist_between_centers ~ (4V+10) * module
                    # => QR total size = (4V+17) * module
                    # Extend from center: (4V+17 - 7) / 2 = (4V+10)/2 modules beyond finder center
                    # Finder outer edge la QR outer edge
                    # BR center = TL_center + (TR_center - TL_center) + (BL_center - TL_center)
                    br_c = tr_c + bl_c - tl_c

                    # Tinh goc BR cua QR: di tu BR center theo huong ra ngoai QR
                    # Huong ra ngoai tai BR: doi lap voi huong tu BR den TL center
                    dir_to_tl = tl_c - br_c
                    dist_br_tl = float(np.linalg.norm(dir_to_tl))
                    if dist_br_tl > 1e-6:
                        dir_out_br = -dir_to_tl / dist_br_tl
                        # Di ra ngoai khoang 3.5 modules
                        br_pt = br_c + dir_out_br * module_size_px * 3.5
                    else:
                        br_pt = tr_pt + bl_pt - tl_pt

                    quad = np.array([tl_pt, tr_pt, br_pt, bl_pt], dtype=np.float32)

                    area_q = abs(float(cv2.contourArea(quad)))
                    if area_q < 2000:
                        continue

                    inside = np.sum(
                        (quad[:, 0] >= -0.05 * w)
                        & (quad[:, 0] <= 1.05 * w)
                        & (quad[:, 1] >= -0.05 * h)
                        & (quad[:, 1] <= 1.05 * h)
                    )
                    if inside < 4:
                        continue

                    center_q = np.mean(quad, axis=0)
                    ang = np.arctan2(quad[:, 1] - center_q[1], quad[:, 0] - center_q[0])
                    quad = quad[np.argsort(ang)]
                    start = int(np.argmin(quad[:, 0] + quad[:, 1]))
                    quad = np.roll(quad, -start, axis=0)
                    quads.append(quad)

    return quads


def suppress_overlapping_quads(quads: List[np.ndarray], iou_threshold: float = 0.30) -> List[np.ndarray]:
    """
    Loại bỏ các tứ giác (quad) chồng lấp bằng cách sắp xếp theo điểm số
    và giữ những quad có độ ưu tiên cao hơn.

    Tham số:
      - quads: List[np.ndarray]
          Danh sách các tứ giác ứng viên, mỗi phần tử là mảng (4,2) float32.
      - iou_threshold: float
          Ngưỡng IoU (axis-aligned intersection over union trên bounding boxes) để
          coi hai vùng là chồng lấp (mặc định 0.30). Nếu IoU >= ngưỡng thì sẽ loại quad thấp hơn.

    Trả về:
      - List[np.ndarray]: Danh sách các quads đã được lọc (không chồng lấp nhiều theo ngưỡng).
    """
    if not quads:
        return []

    scored = []
    for quad in quads:
        q = np.asarray(quad, dtype=np.float32).reshape(4, 2)
        area = abs(float(cv2.contourArea(q)))
        if area < 1e-6:
            continue
        x1, y1 = np.min(q[:, 0]), np.min(q[:, 1])
        x2, y2 = np.max(q[:, 0]), np.max(q[:, 1])
        bw, bh = max(1e-6, x2 - x1), max(1e-6, y2 - y1)
        fill = min(1.2, area / (bw * bh))
        edges = np.array([np.linalg.norm(q[(i + 1) % 4] - q[i]) for i in range(4)], dtype=np.float32)
        balance = float(np.min(edges) / max(1e-6, np.max(edges)))
        score = area * (0.4 + 0.6 * fill) * (0.4 + 0.6 * balance)
        scored.append((score, q, (x1, y1, x2, y2)))

    scored.sort(key=lambda x: x[0], reverse=True)

    selected: List[np.ndarray] = []
    selected_boxes: List[Tuple[float, float, float, float]] = []
    for _, q, b in scored:
        bx1, by1, bx2, by2 = b
        barea = max(1e-6, (bx2 - bx1) * (by2 - by1))
        keep = True
        for sb in selected_boxes:
            sx1, sy1, sx2, sy2 = sb
            sarea = max(1e-6, (sx2 - sx1) * (sy2 - sy1))
            ix1, iy1 = max(bx1, sx1), max(by1, sy1)
            ix2, iy2 = min(bx2, sx2), min(by2, sy2)
            iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
            inter = iw * ih
            union = barea + sarea - inter
            iou = inter / max(1e-6, union)
            overlap_min = inter / max(1e-6, min(barea, sarea))
            if iou >= iou_threshold or overlap_min >= 0.58:
                keep = False
                break
        if keep:
            selected.append(q)
            selected_boxes.append(b)
    return selected


# ===========================================================================
# CAI TIEN CHINH: Tinh chinh toa do ve axis-aligned bounding box
# Toan bo ground truth la axis-aligned rectangles
# ===========================================================================

def binarize_patch(gray: np.ndarray) -> np.ndarray:
    """Nhi phan hoa patch de phat hien vung QR."""
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bin_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = gray.shape
    block = max(11, min(51, ((min(h, w) // 8) | 1)))
    bin_adapt = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, 4
    )
    return cv2.bitwise_or(bin_otsu, bin_adapt)


def refine_to_axis_aligned_bbox(
    img: np.ndarray, rough_quad: np.ndarray
) -> Optional[np.ndarray]:
    """
    Tu rough quad (tu finder pattern), tinh chinh lai thanh axis-aligned bbox
    chinh xac bang cach phan tich vung anh.

    Chien luoc:
    1. Lay vung anh quanh rough quad (co padding)
    2. Nhi phan hoa
    3. Ket noi cac module QR bang morphology
    4. Tim contour lon nhat trung voi rough quad
    5. Lay axis-aligned bounding box cua contour do
    6. Quet row/col de tinh chinh bien chinh xac hon
    """
    q = np.asarray(rough_quad, dtype=np.float32).reshape(4, 2)
    x_min, y_min = np.min(q, axis=0)
    x_max, y_max = np.max(q, axis=0)
    bw = x_max - x_min
    bh = y_max - y_min

    if bw < 6 or bh < 6:
        return None

    H_img, W_img = img.shape[:2]

    # Padding de dam bao chung ta co du vung de tim bien
    pad = max(int(max(bw, bh) * 0.25), 15)
    rx1 = max(0, int(x_min) - pad)
    ry1 = max(0, int(y_min) - pad)
    rx2 = min(W_img, int(x_max) + pad)
    ry2 = min(H_img, int(y_max) + pad)

    patch = img[ry1:ry2, rx1:rx2]
    ph, pw = patch.shape[:2]
    if ph < 8 or pw < 8:
        return None

    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch.copy()
    binary = binarize_patch(gray)

    # Ket noi module QR bang morphology
    k_size = max(3, int(min(bw, bh) / 15))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    binary_closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Tim contour ngoai
    contours, _ = cv2.findContours(binary_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Tam cua rough quad trong toa do patch
    rough_cx = (x_min + x_max) / 2.0 - rx1
    rough_cy = (y_min + y_max) / 2.0 - ry1
    rough_area = bw * bh

    best_cnt = None
    best_score = -1.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < rough_area * 0.06:
            continue

        m = cv2.moments(cnt)
        if m["m00"] < 1:
            continue
        cx = m["m10"] / m["m00"]
        cy = m["m01"] / m["m00"]

        dist = np.sqrt((cx - rough_cx) ** 2 + (cy - rough_cy) ** 2)
        diag = np.sqrt(bw ** 2 + bh ** 2)

        if dist > diag * 0.75:
            continue

        # Uu tien contour lon, gan tam rough quad
        score = area / (1.0 + dist * 0.5)
        if score > best_score:
            best_score = score
            best_cnt = cnt

    if best_cnt is None:
        return None

    # Lay axis-aligned bounding box cua contour
    bx, by, bw2, bh2 = cv2.boundingRect(best_cnt)

    # Chuyen ve toa do anh goc
    abs_x1 = bx + rx1
    abs_y1 = by + ry1
    abs_x2 = abs_x1 + bw2
    abs_y2 = abs_y1 + bh2

    # Tinh chinh them bang cach quet row/col de tim bien chinh xac
    abs_x1, abs_y1, abs_x2, abs_y2 = refine_bbox_by_scanning(
        img, abs_x1, abs_y1, abs_x2, abs_y2, W_img, H_img
    )

    # Kiem tra hop le
    new_w = abs_x2 - abs_x1
    new_h = abs_y2 - abs_y1
    if new_w < 5 or new_h < 5:
        return None

    new_area = new_w * new_h
    if not (0.08 * rough_area < new_area < 15.0 * rough_area):
        return None

    new_cx = (abs_x1 + abs_x2) / 2.0
    new_cy = (abs_y1 + abs_y2) / 2.0
    old_cx = (x_min + x_max) / 2.0
    old_cy = (y_min + y_max) / 2.0
    if np.sqrt((new_cx - old_cx) ** 2 + (new_cy - old_cy) ** 2) > max(bw, bh) * 0.7:
        return None

    return np.array([
        [abs_x1, abs_y1],
        [abs_x2, abs_y1],
        [abs_x2, abs_y2],
        [abs_x1, abs_y2],
    ], dtype=np.float32)


def refine_bbox_by_scanning(
    img: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    W: int, H: int,
    max_expand: int = 30,
) -> Tuple[int, int, int, int]:
    """
    Tinh chinh bien axis-aligned bbox bang cach quet row/col.
    Mo rong toi da max_expand pixels de tim bien chinh xac.
    """
    # Them padding de quet
    sx1 = max(0, x1 - max_expand)
    sy1 = max(0, y1 - max_expand)
    sx2 = min(W, x2 + max_expand)
    sy2 = min(H, y2 + max_expand)

    patch = img[sy1:sy2, sx1:sx2]
    if patch.size == 0:
        return x1, y1, x2, y2

    ph, pw = patch.shape[:2]
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch.copy()
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Roi cua rough bbox trong patch coords
    roi_x1 = x1 - sx1
    roi_y1 = y1 - sy1
    roi_x2 = x2 - sx1
    roi_y2 = y2 - sy1

    # Quet hang (rows) de tim bien tren/duoi
    def find_edge_row(start, end, step, threshold_ratio=0.08):
        for row in range(start, end, step):
            if 0 <= row < ph:
                dark_ratio = np.mean(binary[row, max(0, roi_x1):min(pw, roi_x2)] > 0)
                if dark_ratio >= threshold_ratio:
                    return row
        return None

    def find_edge_col(start, end, step, threshold_ratio=0.08):
        for col in range(start, end, step):
            if 0 <= col < pw:
                dark_ratio = np.mean(binary[max(0, roi_y1):min(ph, roi_y2), col] > 0)
                if dark_ratio >= threshold_ratio:
                    return col
        return None

    # Tim bien tren: quet len tu roi_y1
    top_row = find_edge_row(roi_y1, max(0, roi_y1 - max_expand) - 1, -1)
    if top_row is not None:
        new_y1 = top_row + sy1
    else:
        new_y1 = y1

    # Tim bien duoi: quet xuong tu roi_y2
    bot_row = find_edge_row(roi_y2, min(ph, roi_y2 + max_expand) + 1, 1)
    if bot_row is not None:
        new_y2 = bot_row + sy1
    else:
        new_y2 = y2

    # Tim bien trai
    left_col = find_edge_col(roi_x1, max(0, roi_x1 - max_expand) - 1, -1)
    if left_col is not None:
        new_x1 = left_col + sx1
    else:
        new_x1 = x1

    # Tim bien phai
    right_col = find_edge_col(roi_x2, min(pw, roi_x2 + max_expand) + 1, 1)
    if right_col is not None:
        new_x2 = right_col + sx1
    else:
        new_x2 = x2

    # Dam bao thu tu dung
    if new_x1 >= new_x2:
        new_x1, new_x2 = x1, x2
    if new_y1 >= new_y2:
        new_y1, new_y2 = y1, y2

    return int(new_x1), int(new_y1), int(new_x2), int(new_y2)


def detect_qr_by_outline(img: np.ndarray, max_candidates: int = 4) -> List[np.ndarray]:
    """
    Phat hien QR code bang phan tich contour truc tiep.
    Tim cac vung co:
    - Dang gan vuong (ratio < 3.5)
    - Do phuc tap texture cao (QR co nhieu chuyen doi den/trang)
    - Kich thuoc hop ly

    Day la phuong phap bo sung cho finder-pattern detection.
    """
    if img is None or img.size == 0:
        return []

    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()

    results: List[np.ndarray] = []
    seen_boxes: List[Tuple[int, int, int, int]] = []

    # Thu nhieu muc do nhi phan
    variants = []

    # Variant 1: Otsu
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bin1 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    variants.append(bin1)

    # Variant 2: adaptive
    block = max(21, min(71, ((min(H, W) // 12) | 1)))
    bin2 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, block, 5)
    variants.append(bin2)

    # Variant 3: normalised
    bg = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigmaX=max(5, min(H, W) // 20))
    norm = cv2.divide(gray.astype(np.float32), bg + 1.0, scale=255.0)
    norm = np.clip(norm, 0, 255).astype(np.uint8)
    _, bin3 = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    variants.append(bin3)

    for binary_raw in variants:
        # Ket noi cac module QR
        k_sizes = [max(3, int(min(H, W) / 80)), max(3, int(min(H, W) / 40))]
        for k in k_sizes:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
            binary = cv2.morphologyEx(binary_raw, cv2.MORPH_CLOSE, kernel, iterations=2)

            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 200 or area > H * W * 0.5:
                    continue

                x, y, cw, ch = cv2.boundingRect(cnt)
                if cw < 10 or ch < 10:
                    continue

                # Ty le canh: khong qua bien dang
                ratio = max(cw, ch) / max(1, min(cw, ch))
                if ratio > 4.0:
                    continue

                # Kiem tra texture (chuyen doi den/trang)
                patch_gray = gray[y:y + ch, x:x + cw]
                if patch_gray.size < 100:
                    continue
                patch_small = cv2.resize(patch_gray, (48, 48), interpolation=cv2.INTER_AREA)
                _, patch_bin = cv2.threshold(patch_small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                row_trans = np.mean(np.sum(patch_bin[:, 1:] != patch_bin[:, :-1], axis=1) / 47.0)
                col_trans = np.mean(np.sum(patch_bin[1:, :] != patch_bin[:-1, :], axis=0) / 47.0)
                transition_score = 0.5 * (row_trans + col_trans)

                if transition_score < 0.10:
                    continue

                # Kiem tra fill ratio
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                if hull_area < 1:
                    continue
                convexity = area / hull_area
                if convexity < 0.5:
                    continue

                # Tinh chinh bbox
                abs_x1 = x
                abs_y1 = y
                abs_x2 = x + cw
                abs_y2 = y + ch

                # Kiem tra trung lap
                box = (abs_x1, abs_y1, abs_x2, abs_y2)
                if any(bbox_iou_xyxy(box, old) > 0.4 for old in seen_boxes):
                    continue

                quad = np.array([
                    [abs_x1, abs_y1],
                    [abs_x2, abs_y1],
                    [abs_x2, abs_y2],
                    [abs_x1, abs_y2],
                ], dtype=np.float32)

                results.append(quad)
                seen_boxes.append(box)

                if len(results) >= max_candidates * 3:
                    break

    # Sap xep theo diem texture
    return results[:max_candidates * 2]


def fallback_corner_cluster_quads(image: np.ndarray, max_candidates: int = 2) -> List[np.ndarray]:
    """
    Phương pháp dự phòng: tìm cụm góc (corner clusters) để đoán vùng chứa QR khi
    phương pháp finder-pattern không đủ hiệu quả.

    Tham số:
      - image: np.ndarray
          Ảnh BGR hoặc grayscale.
      - max_candidates: int
          Số lượng ứng viên tối đa để trả về (mặc định: 2).

    Trả về:
      - List[np.ndarray]: Danh sách các tứ giác axis-aligned (4 góc float32) ứng viên.
    """
    if image is None or image.size == 0:
        return []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=1600,
        qualityLevel=0.008,
        minDistance=2,
        blockSize=3,
        useHarrisDetector=False,
    )
    if corners is None or len(corners) < 24:
        return []

    pts = np.round(corners.reshape(-1, 2)).astype(np.int32)
    h, w = gray.shape[:2]
    if h < 40 or w < 40:
        return []

    grid_size = max(20, min(36, min(h, w) // 24))
    cell_w = max(1e-6, float(w) / float(grid_size))
    cell_h = max(1e-6, float(h) / float(grid_size))

    grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    for x, y in pts:
        if x < 0 or y < 0 or x >= w or y >= h:
            continue
        gx = min(grid_size - 1, int(x / cell_w))
        gy = min(grid_size - 1, int(y / cell_h))
        grid[gy, gx] += 1.0

    smooth = cv2.GaussianBlur(grid, (5, 5), 0)
    ranked = np.argsort(smooth.reshape(-1))[::-1]

    selected_quads: List[np.ndarray] = []
    selected_boxes: List[Tuple[int, int, int, int]] = []

    for flat_idx in ranked[: grid_size * 4]:
        score = float(smooth.reshape(-1)[flat_idx])
        if score < 1.5:
            break

        gy, gx = divmod(int(flat_idx), grid_size)
        radius = 3
        x0 = max(0, int((gx - radius) * cell_w))
        y0 = max(0, int((gy - radius) * cell_h))
        x1 = min(w, int((gx + radius + 1) * cell_w))
        y1 = min(h, int((gy + radius + 1) * cell_h))
        if x1 - x0 < 16 or y1 - y0 < 16:
            continue

        mask = (pts[:, 0] >= x0) & (pts[:, 0] < x1) & (pts[:, 1] >= y0) & (pts[:, 1] < y1)
        local_pts = pts[mask]
        if len(local_pts) < 24:
            continue

        qx1 = float(np.percentile(local_pts[:, 0], 2))
        qx2 = float(np.percentile(local_pts[:, 0], 98))
        qy1 = float(np.percentile(local_pts[:, 1], 2))
        qy2 = float(np.percentile(local_pts[:, 1], 98))
        qbw = max(1.0, qx2 - qx1)
        qbh = max(1.0, qy2 - qy1)

        bx1 = max(0, int(np.floor(qx1 - 0.15 * qbw - 2)))
        bx2 = min(w, int(np.ceil(qx2 + 0.15 * qbw + 2)))
        by1 = max(0, int(np.floor(qy1 - 0.05 * qbh - 2)))
        by2 = min(h, int(np.ceil(qy2 + 0.18 * qbh + 2)))
        bw2, bh2 = bx2 - bx1, by2 - by1
        if bw2 < 16 or bh2 < 16:
            continue

        ratio = max(bw2, bh2) / max(1.0, min(bw2, bh2))
        area = float(bw2 * bh2)
        if ratio > 3.5 or area < 600 or area > 0.32 * h * w:
            continue

        local = local_pts - np.array([bx1, by1], dtype=np.int32)
        occ = np.zeros((3, 3), dtype=np.uint8)
        for px, py in local:
            cx2 = min(2, max(0, int((px * 3) / max(1, bw2))))
            cy2 = min(2, max(0, int((py * 3) / max(1, bh2))))
            occ[cy2, cx2] = 1
        if int(np.sum(occ)) < 4:
            continue

        patch = gray[by1:by2, bx1:bx2]
        if patch.size == 0:
            continue
        patch = cv2.resize(patch, (72, 72), interpolation=cv2.INTER_CUBIC)
        patch_bin = cv2.adaptiveThreshold(patch, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 21, 3)
        row_trans = np.mean(np.sum(patch_bin[:, 1:] != patch_bin[:, :-1], axis=1) / 71.0)
        col_trans = np.mean(np.sum(patch_bin[1:, :] != patch_bin[:-1, :], axis=0) / 71.0)
        transition_score = 0.5 * (row_trans + col_trans)
        if transition_score < 0.12:
            continue

        candidate_box = (bx1, by1, bx2, by2)
        if any(bbox_iou_xyxy(candidate_box, old) > 0.55 for old in selected_boxes):
            continue

        quad = np.array([[bx1, by1], [bx2, by1], [bx2, by2], [bx1, by2]], dtype=np.float32)
        selected_quads.append(order_quad_clockwise_start_top_left(quad))
        selected_boxes.append(candidate_box)

        if len(selected_quads) >= max_candidates:
            break

    return selected_quads


def process_image(image_path: str) -> Tuple[int, List[List[Point]]]:
    """
    Xu ly mot anh va tra ve (so_luong_qr, danh_sach_4_goc axis-aligned).

    Flow:
    1. Phat hien QR bang finder-pattern (nhieu bien the preprocessing)
    2. Bo sung phat hien bang contour truc tiep
    3. Tinh chinh toa do ve axis-aligned bbox chinh xac
    4. NMS cuoi cung
    """
    img = read_image_any_path(image_path)
    if img is None:
        print(f"Khong the doc anh: {to_console_safe(image_path)}")
        return 0, []

    H_img, W_img = img.shape[:2]

    # -----------------------------------------------------------------------
    # BUOC 1: Thu tim bang finder-pattern
    # -----------------------------------------------------------------------
    binaries: List[np.ndarray] = []
    binaries.append(preprocess_image(img, variant=0))
    binaries.append(preprocess_image(img, variant=1))
    binaries.append(preprocess_image(img, variant=2))

    blur = cv2.GaussianBlur(img, (0, 0), 1.3)
    sharpened_bgr = cv2.addWeighted(img, 1.45, blur, -0.45, 0)
    binaries.append(preprocess_image(sharpened_bgr, variant=0))
    binaries.append(preprocess_image(sharpened_bgr, variant=1))

    h, w = img.shape[:2]
    scales = []
    if min(h, w) < 600:
        scales = [1.4, 1.25]
    elif min(h, w) < 800:
        scales = [1.3, 1.15]
    else:
        scales = [1.2, 1.1]

    for scale in scales:
        up = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        binaries.append(preprocess_image(up, variant=0))
        binaries.append(preprocess_image(up, variant=1))

    patterns: List[Tuple[Contour, Point, float]] = []
    for b_idx, binary in enumerate(binaries):
        local_patterns = find_finder_patterns(binary)
        inv_patterns = find_finder_patterns(cv2.bitwise_not(binary))
        local_patterns.extend(inv_patterns)

        if b_idx >= 5:
            scale_idx = (b_idx - 5) // 2
            scale = scales[scale_idx]
            mapped_patterns: List[Tuple[Contour, Point, float]] = []
            for cnt, center, area in local_patterns:
                cnt_scaled = (cnt.astype(np.float32) / scale).astype(np.int32)
                cx = int(round(center[0] / scale))
                cy = int(round(center[1] / scale))
                mapped_patterns.append((cnt_scaled, (cx, cy), float(area) / (scale * scale)))
            local_patterns = mapped_patterns
        patterns.extend(local_patterns)

    unique_patterns: List[Tuple[Contour, Point, float]] = []
    for cand in patterns:
        _, c, area = cand
        duplicated = False
        for _, old_c, old_area in unique_patterns:
            dist = np.linalg.norm(np.array(c, dtype=np.float32) - np.array(old_c, dtype=np.float32))
            ar = min(area, old_area) / max(1e-6, max(area, old_area))
            if dist < 11 and ar > 0.72:
                duplicated = True
                break
        if not duplicated:
            unique_patterns.append(cand)

    rough_quads = build_qr_quads(unique_patterns, img.shape)
    rough_quads = suppress_overlapping_quads(rough_quads, iou_threshold=0.28)

    # -----------------------------------------------------------------------
    # BUOC 2: Tinh chinh toa do sang axis-aligned bbox
    # -----------------------------------------------------------------------
    refined_quads: List[np.ndarray] = []
    refined_boxes: List[Tuple[int, int, int, int]] = []

    for rq in rough_quads:
        refined = refine_to_axis_aligned_bbox(img, rq)
        if refined is None:
            # Neu khong refine duoc, dung axis-aligned bbox cua rough quad
            q2 = np.asarray(rq, dtype=np.float32).reshape(4, 2)
            x1, y1 = int(np.min(q2[:, 0])), int(np.min(q2[:, 1]))
            x2, y2 = int(np.max(q2[:, 0])), int(np.max(q2[:, 1]))
            refined = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)

        r = refined.reshape(4, 2)
        x1 = int(np.min(r[:, 0]))
        y1 = int(np.min(r[:, 1]))
        x2 = int(np.max(r[:, 0]))
        y2 = int(np.max(r[:, 1]))

        box = (x1, y1, x2, y2)
        if not any(bbox_iou_xyxy(box, old) > 0.4 for old in refined_boxes):
            refined_quads.append(refined)
            refined_boxes.append(box)

    # -----------------------------------------------------------------------
    # BUOC 3: Neu chua co du QR, thu phat hien bang contour outline
    # -----------------------------------------------------------------------
    # Chi them neu chua phat hien duoc gi, hoac thu them vung co the bo qua
    outline_quads = detect_qr_by_outline(img, max_candidates=4)

    for oq in outline_quads:
        r = oq.reshape(4, 2)
        x1 = int(np.min(r[:, 0]))
        y1 = int(np.min(r[:, 1]))
        x2 = int(np.max(r[:, 0]))
        y2 = int(np.max(r[:, 1]))

        box = (x1, y1, x2, y2)
        if any(bbox_iou_xyxy(box, old) > 0.3 for old in refined_boxes):
            continue

        # Kiem tra texture de dam bao day la QR
        bw_q = x2 - x1
        bh_q = y2 - y1
        if bw_q < 10 or bh_q < 10:
            continue

        # Tinh chinh outline quad
        refined = refine_to_axis_aligned_bbox(img, oq)
        if refined is None:
            refined = oq

        r2 = refined.reshape(4, 2)
        rx1 = int(np.min(r2[:, 0]))
        ry1 = int(np.min(r2[:, 1]))
        rx2 = int(np.max(r2[:, 0]))
        ry2 = int(np.max(r2[:, 1]))
        rbox = (rx1, ry1, rx2, ry2)

        if not any(bbox_iou_xyxy(rbox, old) > 0.35 for old in refined_boxes):
            refined_quads.append(refined)
            refined_boxes.append(rbox)

    # -----------------------------------------------------------------------
    # BUOC 4: Fallback neu van chua co gi
    # -----------------------------------------------------------------------
    if not refined_quads:
        fallback = fallback_corner_cluster_quads(img, max_candidates=3)
        for fq in fallback:
            refined = refine_to_axis_aligned_bbox(img, fq)
            if refined is None:
                q2 = np.asarray(fq, dtype=np.float32).reshape(4, 2)
                x1, y1 = int(np.min(q2[:, 0])), int(np.min(q2[:, 1]))
                x2, y2 = int(np.max(q2[:, 0])), int(np.max(q2[:, 1]))
                refined = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
            refined_quads.append(refined)

    # -----------------------------------------------------------------------
    # BUOC 5: NMS cuoi cung va sap xep ket qua
    # -----------------------------------------------------------------------
    # Convert all quads to proper axis-aligned form for final NMS
    final_quads: List[np.ndarray] = []
    final_boxes: List[Tuple[int, int, int, int]] = []

    for q in refined_quads:
        r = np.asarray(q, dtype=np.float32).reshape(4, 2)
        x1 = int(np.min(r[:, 0]))
        y1 = int(np.min(r[:, 1]))
        x2 = int(np.max(r[:, 0]))
        y2 = int(np.max(r[:, 1]))

        if x2 - x1 < 5 or y2 - y1 < 5:
            continue

        box = (x1, y1, x2, y2)
        if any(bbox_iou_xyxy(box, old) > 0.35 for old in final_boxes):
            continue

        aa_quad = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        final_quads.append(aa_quad)
        final_boxes.append(box)

    # Sap xep theo vi tri (tren->duoi, trai->phai)
    final_quads.sort(key=lambda q: (np.min(q[:, 1]), np.min(q[:, 0])))

    corners = []
    for q in final_quads:
        r = q.reshape(4, 2)
        x1 = int(np.min(r[:, 0]))
        y1 = int(np.min(r[:, 1]))
        x2 = int(np.max(r[:, 0]))
        y2 = int(np.max(r[:, 1]))
        corners.append([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

    return len(corners), corners


# ===========================================================================
# Cac ham danh gia (giu nguyen)
# ===========================================================================

def polygon_signed_area(poly: np.ndarray) -> float:
        """
        Tính diện tích có hướng (signed area) của đa giác `poly`.

        Tham số:
            - poly: np.ndarray (N,2) hoặc tương tự
        Trả về:
            - float: diện tích (có dấu). Giá trị âm nếu thứ tự điểm là clockwise.
        """
        if poly is None or len(poly) < 3:
                return 0.0
        x = poly[:, 0]
        y = poly[:, 1]
        return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def polygon_area(poly: np.ndarray) -> float:
        """
        Diện tích (không dấu) của đa giác `poly`.

        Tham số:
            - poly: np.ndarray (N,2)
        Trả về:
            - float: diện tích dương của đa giác.
        """
        return abs(polygon_signed_area(poly))


def ensure_ccw(poly: np.ndarray) -> np.ndarray:
        """
        Đảm bảo thứ tự điểm của đa giác là counter-clockwise (CCW).

        Tham số:
            - poly: np.ndarray (N,2)
        Trả về:
            - np.ndarray: đa giác đã được đảo thứ tự nếu cần, dtype float32
        """
        p = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
        if len(p) >= 3 and polygon_signed_area(p) < 0:
                p = p[::-1]
        return p


def line_intersection(p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """
        Tinh giao diem cua 2 doan thang (p1-p2) va (q1-q2).

        Tham so:
            - p1, p2, q1, q2: np.ndarray co 2 phan tu (x,y)
        Tra ve:
            - np.ndarray (2,) la diem giao tuyen. Neu 2 doan song song, tra ve p2 copy.
        """
        r = p2 - p1
        s = q2 - q1
        denom = float(r[0] * s[1] - r[1] * s[0])
        if abs(denom) < 1e-9:
                return p2.copy()
        qp = q1 - p1
        t = float((qp[0] * s[1] - qp[1] * s[0]) / denom)
        return p1 + t * r


def clip_polygon_sutherland_hodgman(subject: np.ndarray, clipper: np.ndarray) -> np.ndarray:
    """
    Thuật toán Sutherland–Hodgman để clip (cắt) một đa giác `subject` bởi đa giác `clipper`.

    Tham số:
      - subject: np.ndarray (N,2) - đa giác cần cắt
      - clipper: np.ndarray (M,2) - đa giác cắt (cửa sổ clip)
    Trả về:
      - np.ndarray (K,2): đa giác giao (có thể rỗng)
    """
    out = ensure_ccw(subject)
    clip = ensure_ccw(clipper)
    if len(out) < 3 or len(clip) < 3:
        return np.zeros((0, 2), dtype=np.float32)

    for i in range(len(clip)):
        a = clip[i]
        b = clip[(i + 1) % len(clip)]
        inp = out
        if len(inp) == 0:
            break
        out_pts = []
        for j in range(len(inp)):
            cur = inp[j]
            prev = inp[j - 1]
            cur_inside = ((b[0] - a[0]) * (cur[1] - a[1]) - (b[1] - a[1]) * (cur[0] - a[0])) >= 0
            prev_inside = ((b[0] - a[0]) * (prev[1] - a[1]) - (b[1] - a[1]) * (prev[0] - a[0])) >= 0
            if cur_inside:
                if not prev_inside:
                    out_pts.append(line_intersection(prev, cur, a, b))
                out_pts.append(cur)
            elif prev_inside:
                out_pts.append(line_intersection(prev, cur, a, b))
        out = np.array(out_pts, dtype=np.float32) if out_pts else np.zeros((0, 2), dtype=np.float32)

    return out


def quad_iou(quad_a: np.ndarray, quad_b: np.ndarray) -> float:
    """
    Tính IoU giữa hai tứ giác (không phải bounding box hướng trục).

    Tham số:
      - quad_a, quad_b: np.ndarray dạng (4,2) hoặc tương tự (các góc của tứ giác)
    Trả về:
      - float: IoU (giao / hợp) trên diện tích hình chữ nhật/quadrilateral.
    """
    qa = ensure_ccw(np.asarray(quad_a, dtype=np.float32).reshape(4, 2))
    qb = ensure_ccw(np.asarray(quad_b, dtype=np.float32).reshape(4, 2))
    area_a = polygon_area(qa)
    area_b = polygon_area(qb)
    if area_a <= 1e-9 or area_b <= 1e-9:
        return 0.0

    inter_poly = clip_polygon_sutherland_hodgman(qa, qb)
    inter = polygon_area(inter_poly) if len(inter_poly) >= 3 else 0.0
    union = area_a + area_b - inter
    if union <= 1e-9:
        return 0.0
    return float(inter / union)


def load_quads_by_image_id(csv_path: str) -> Dict[str, List[np.ndarray]]:
    grouped: Dict[str, List[np.ndarray]] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        required = {"image_id", "qr_index", "x0", "y0", "x1", "y1", "x2", "y2", "x3", "y3"}
        if not required.issubset(fieldnames):
            missing = sorted(required - fieldnames)
            raise ValueError(f"CSV '{csv_path}' thieu cot: {', '.join(missing)}")

        for row in reader:
            image_id = (row.get("image_id") or "").strip()
            if image_id == "":
                continue
            grouped.setdefault(image_id, [])

            qr_index = (row.get("qr_index") or "").strip()
            if qr_index == "":
                continue

            coords = []
            for key in ("x0", "y0", "x1", "y1", "x2", "y2", "x3", "y3"):
                val = (row.get(key) or "").strip()
                if val == "":
                    coords = []
                    break
                coords.append(float(val))
            if len(coords) != 8:
                continue

            quad = np.array([
                [coords[0], coords[1]], [coords[2], coords[3]],
                [coords[4], coords[5]], [coords[6], coords[7]],
            ], dtype=np.float32)
            quad = order_quad_clockwise_start_top_left(quad)
            grouped[image_id].append(quad)
    return grouped


def greedy_iou_match(
    pred_by_image: Dict[str, List[np.ndarray]],
    gt_by_image: Dict[str, List[np.ndarray]],
    iou_threshold: float = 0.5,
) -> Dict[str, Dict[str, float]]:
    tp_iou = 0; fp_iou = 0; fn_iou = 0
    tp_count = 0; fp_count = 0; fn_count = 0

    all_ids = set(pred_by_image.keys()) | set(gt_by_image.keys())
    for image_id in all_ids:
        preds = pred_by_image.get(image_id, [])
        gts = gt_by_image.get(image_id, [])

        p_len = len(preds); g_len = len(gts)
        tp_count += min(p_len, g_len)
        fp_count += max(0, p_len - g_len)
        fn_count += max(0, g_len - p_len)

        matched = [False] * g_len
        for p in preds:
            best_iou = 0.0; best_j = -1
            for j, g in enumerate(gts):
                if matched[j]:
                    continue
                iou = quad_iou(p, g)
                if iou > best_iou:
                    best_iou = iou; best_j = j
            if best_j >= 0 and best_iou >= iou_threshold:
                matched[best_j] = True; tp_iou += 1
            else:
                fp_iou += 1
        fn_iou += sum(1 for used in matched if not used)

    def f1(tp, fp, fn):
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0, prec, rec

    f1c, pc, rc = f1(tp_count, fp_count, fn_count)
    f1i, pi, ri = f1(tp_iou, fp_iou, fn_iou)

    return {
        "count": {"tp": float(tp_count), "fp": float(fp_count), "fn": float(fn_count),
                  "precision": pc, "recall": rc, "f1": f1c},
        "iou": {"tp": float(tp_iou), "fp": float(fp_iou), "fn": float(fn_iou),
                "precision": pi, "recall": ri, "f1": f1i},
    }


def evaluate_csvs(pred_csv: str, gt_csv: str, iou_threshold: float = 0.5,
                  filter_ids: Optional[set] = None) -> None:
    print(f"\n{'='*70}")
    print("DANH GIA KET QUA DETECTION")
    print(f"{'='*70}")
    print(f"  Prediction : {to_console_safe(pred_csv)}")
    print(f"  Ground Truth: {to_console_safe(gt_csv)}")
    print(f"  IoU threshold: {iou_threshold:.3f}")
    print(f"{'='*70}\n")

    pred_quads = load_quads_by_image_id(pred_csv)
    gt_quads = load_quads_by_image_id(gt_csv)

    if filter_ids:
        pred_quads = {k: v for k, v in pred_quads.items() if k in filter_ids}
        gt_quads = {k: v for k, v in gt_quads.items() if k in filter_ids}

    stats = greedy_iou_match(pred_quads, gt_quads, iou_threshold=iou_threshold)

    sc = stats["count"]
    print(f"{'='*70}")
    print(f"PHAN 1: SO SANH THEO SO LUONG QR (qr_index)")
    print(f"{'='*70}")
    print(f"  True  Positives (TP): {int(sc['tp']):>6}")
    print(f"  False Positives (FP): {int(sc['fp']):>6}")
    print(f"  False Negatives (FN): {int(sc['fn']):>6}")
    print(f"  Precision : {sc['precision']:.6f}  ({sc['precision']*100:.2f}%)")
    print(f"  Recall    : {sc['recall']:.6f}  ({sc['recall']*100:.2f}%)")
    print(f"  F1 SCORE  : {sc['f1']:.6f}  ({sc['f1']*100:.2f}%)")
    print(f"{'='*70}\n")

    si = stats["iou"]
    print(f"{'='*70}")
    print(f"PHAN 2: SO SANH THEO TOA DO DUONG BAO (IoU >= {iou_threshold:.2f})")
    print(f"{'='*70}")
    print(f"  True  Positives (TP): {int(si['tp']):>6}")
    print(f"  False Positives (FP): {int(si['fp']):>6}")
    print(f"  False Negatives (FN): {int(si['fn']):>6}")
    print(f"  Precision : {si['precision']:.6f}  ({si['precision']*100:.2f}%)")
    print(f"  Recall    : {si['recall']:.6f}  ({si['recall']*100:.2f}%)")
    print(f"  F1 SCORE  : {si['f1']:.6f}  ({si['f1']*100:.2f}%)")
    print(f"{'='*70}\n")


def main() -> None:
    """
    Hàm chính xử lý phát hiện QR code theo yêu cầu specification 5.5.
    - Đo tổng thời gian chạy (wall-clock time) từ lúc bắt đầu đến lúc kết thúc
    - Hiển thị thống kê tốc độ: giây/ảnh (chuẩn hóa trên số lượng ảnh)
    - Máy chấm sử dụng CPU (không có GPU)
    """
    # Ghi lại thời gian bắt đầu chương trình (wall-clock time)
    start_time = time.time()
    
    # Khởi tạo argument parser để xử lý các tham số command-line
    parser = argparse.ArgumentParser(description="Phat hien QR code trong anh (khong dung detector co san)")
    # --data: Đường dẫn file CSV chứa danh sách ảnh cần xử lý (bắt buộc khi không dùng --eval-only)
    parser.add_argument("--data", default="", help="Duong dan file CSV co cot image_path")
    # --gt: Đường dẫn file ground-truth CSV để so sánh kết quả (tùy chọn)
    parser.add_argument("--gt", default="", help="File ground-truth CSV de cham (tuy chon)")
    # --iou-threshold: Ngưỡng IoU để xác định match giữa dự đoán và ground-truth (giá trị mặc định: 0.5)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    # --eval-only: Hoạt động đánh giá chỉ (không xử lý ảnh, chỉ so sánh prediction với ground-truth)
    parser.add_argument("--eval-only", action="store_true")
    # --pred: Đường dẫn file prediction khi dùng mode --eval-only
    parser.add_argument("--pred", default="")
    # --valid: Đường dẫn file ground-truth khi dùng mode --eval-only
    parser.add_argument("--valid", default="")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    if args.eval_only:
        pred_path = os.path.abspath(args.pred) if args.pred else os.path.join(script_dir, "output.csv")
        valid_path = os.path.abspath(args.valid) if args.valid else os.path.join(script_dir, "output_valid.csv")
        if not os.path.isfile(pred_path):
            print(f"Loi: khong tim thay file prediction: {pred_path}")
            return
        if not os.path.isfile(valid_path):
            print(f"Loi: khong tim thay file ground-truth: {valid_path}")
            return
        try:
            evaluate_csvs(pred_path, valid_path, iou_threshold=float(args.iou_threshold))
        except Exception as e:
            print(f"\nLOI KHI DANH GIA: {e}")
            import traceback; traceback.print_exc()
        return

    if not args.data:
        parser.error("Can --data <csv_path> hoac --eval-only")

    try:
        data_csv_path = os.path.abspath(args.data)
        csv_dir = os.path.dirname(data_csv_path)
        output_path = os.path.join(script_dir, "output.csv")

        with open(data_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            if not {"image_id", "image_path"}.issubset(fieldnames):
                print("File CSV phai co day du 2 cot: 'image_id' va 'image_path'")
                return
            rows = list(reader)
        data_image_ids = {(row.get("image_id") or "").strip() for row in rows
                          if (row.get("image_id") or "").strip()}

        results = []
        total_images = len(rows)
        print(f"\n{'='*70}")
        print(f"BAT DAU XU LY {total_images} ANH")
        print(f"{'='*70}\n")

        for idx, row in enumerate(rows, start=1):
            image_id = (row.get("image_id") or "").strip()
            image_path_raw = (row.get("image_path") or "").strip()
            img_path = image_path_raw
            if not os.path.isabs(img_path):
                img_path = os.path.normpath(os.path.join(csv_dir, img_path))

            print(f"[{idx}/{total_images}] Xu ly: {to_console_safe(image_id)}", end=" ... ")
            num_qr, corners_list = process_image(img_path)
            print(f"Tim thay {num_qr} QR code")

            if num_qr > 0:
                for qr_index, quad in enumerate(corners_list):
                    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = quad
                    results.append({
                        "image_id": image_id, "qr_index": qr_index,
                        "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                        "x2": x2, "y2": y2, "x3": x3, "y3": y3, "content": "",
                    })
            else:
                results.append({
                    "image_id": image_id, "qr_index": "",
                    "x0": "", "y0": "", "x1": "", "y1": "",
                    "x2": "", "y2": "", "x3": "", "y3": "", "content": "",
                })

        with open(output_path, "w", newline="", encoding="utf-8") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=[
                "image_id", "qr_index", "x0", "y0", "x1", "y1",
                "x2", "y2", "x3", "y3", "content"])
            writer.writeheader()
            writer.writerows(results)

        print(f"\n{'='*70}")
        print(f"DA GHI KET QUA VAO: {output_path}")
        print(f"{'='*70}\n")

        gt_path = (args.gt or "").strip()
        if gt_path:
            evaluate_csvs(output_path, os.path.abspath(gt_path),
                          iou_threshold=float(args.iou_threshold),
                          filter_ids=data_image_ids if data_image_ids else None)

        auto_valid = os.path.join(script_dir, "output_valid.csv")
        if not gt_path and os.path.isfile(auto_valid):
            evaluate_csvs(output_path, auto_valid,
                          iou_threshold=float(args.iou_threshold),
                          filter_ids=data_image_ids if data_image_ids else None)
        
        # ===== DANH GIA TOC DO (theo requirement 5.5) =====
        # Tính toán tổng thời gian chạy (wall-clock time) từ lúc bắt đầu đến hiện tại
        end_time = time.time()
        total_time = end_time - start_time
        
        # Chuẩn hóa tốc độ: giây/ảnh (seconds per image)
        speed_per_image = total_time / total_images if total_images > 0 else 0.0
        
        # Hiển thị thống kê tốc độ ra terminal
        print(f"{'='*70}")
        print(f"THONG KE TOC DO (requirement 5.5)")
        print(f"{'='*70}")
        print(f"Tong so anh: {total_images}")
        print(f"Tong thoi gian chay (wall-clock time): {total_time:.2f} giay")
        print(f"Toc do trung binh: {speed_per_image:.4f} giay/anh")
        print(f"{'='*70}\n")

    except Exception as e:
        print(f"\nLOI XU LY DU LIEU: {e}")
        import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
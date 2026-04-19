#!/usr/bin/env python3
"""
Phat hien va dinh vi QR code bang xu ly anh truyen thong.
Khong dung QRCodeDetector/deep learning.

Modun xu ly anh va phat hien QR code tren cac hinh anh.
Su dung phuong phap finder-pattern detection, contour analysis,
va connected-components de phat hien QR codes.
"""

import argparse  # Xu li tham so command-line
import csv  # Doc/ghi file CSV
import os  # Lam viec voi duong dan file
import time  # Do thoi gian chay chuong trinh (requirement 5.5)
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple, Union  # Type hints

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


def write_image_any_path(image_path: str, image: np.ndarray) -> bool:
    if image is None or image.size == 0:
        return False
    ext = os.path.splitext(image_path)[1]
    if ext == "":
        ext = ".png"
    ok = cv2.imwrite(image_path, image)
    if ok:
        return True
    try:
        success, encoded = cv2.imencode(ext, image)
        if success:
            encoded.tofile(image_path)
            return True
    except Exception:
        pass
    return False


def ensure_utf8_text(value: Union[str, bytes, None]) -> str:
    """
    Chuan hoa chuoi an toan UTF-8 de ghi CSV.
    Neu loi decode/encode thi thay the ky tu loi, khong nem exception.
    """
    if value is None:
        return ""
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    else:
        text = str(value)
    text = text.replace("\x00", "")
    return text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")


def decode_qr_content_from_quad(
    image: Optional[np.ndarray],
    quad: List[Point],
    detector: Optional[cv2.QRCodeDetector] = None,
) -> str:
    """
    Giai ma noi dung QR tu vung quad.
    Neu khong giai ma duoc hoac loi -> tra ve chuoi rong.
    """
    try:
        if image is None or image.size == 0 or not quad or len(quad) != 4:
            return ""

        local_detector = detector if detector is not None else cv2.QRCodeDetector()
        pts = np.array(quad, dtype=np.float32).reshape(4, 2)
        h, w = image.shape[:2]
        x1 = max(0, int(np.floor(np.min(pts[:, 0]))))
        y1 = max(0, int(np.floor(np.min(pts[:, 1]))))
        x2 = min(w, int(np.ceil(np.max(pts[:, 0]))))
        y2 = min(h, int(np.ceil(np.max(pts[:, 1]))))
        if x2 - x1 < 6 or y2 - y1 < 6:
            return ""

        pad = max(2, int(0.12 * max(x2 - x1, y2 - y1)))
        rx1 = max(0, x1 - pad)
        ry1 = max(0, y1 - pad)
        rx2 = min(w, x2 + pad)
        ry2 = min(h, y2 + pad)
        roi = image[ry1:ry2, rx1:rx2]
        if roi is None or roi.size == 0:
            return ""

        txt, _, _ = local_detector.detectAndDecode(roi)
        txt = ensure_utf8_text(txt)
        if txt:
            return txt

        # Thu them voi ROI upscale de tang kha nang decode QR nho/mo.
        roi_up = cv2.resize(roi, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        txt_up, _, _ = local_detector.detectAndDecode(roi_up)
        txt_up = ensure_utf8_text(txt_up)
        if txt_up:
            return txt_up

        return ""
    except Exception:
        return ""


def decode_qr_contents_for_quads(
    image: Optional[np.ndarray],
    corners_list: List[List[Point]],
) -> List[str]:
    if not corners_list:
        return []
    detector = cv2.QRCodeDetector()
    decoded: List[str] = []
    for quad in corners_list:
        decoded.append(decode_qr_content_from_quad(image, quad, detector=detector))
    return decoded


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


def boxes_overlap_or_touch(
    a: Tuple[int, int, int, int],
    b: Tuple[int, int, int, int],
    margin: int = 0,
) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    if ax2 + margin < bx1:
        return False
    if bx2 + margin < ax1:
        return False
    if ay2 + margin < by1:
        return False
    if by2 + margin < ay1:
        return False
    return True


def quad_to_box_int(quad: np.ndarray) -> Tuple[int, int, int, int]:
    q = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    return (
        int(np.floor(np.min(q[:, 0]))),
        int(np.floor(np.min(q[:, 1]))),
        int(np.ceil(np.max(q[:, 0]))),
        int(np.ceil(np.max(q[:, 1]))),
    )


def expand_axis_aligned_quad(
    quad: np.ndarray,
    image_shape: Tuple[int, int],
    pad_ratio: float = 0.12,
    min_pad: int = 4,
    max_pad: int = 44,
) -> np.ndarray:
    q = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    h, w = int(image_shape[0]), int(image_shape[1])
    x1, y1, x2, y2 = quad_to_box_int(q)
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    base = int(round(pad_ratio * float(min(bw, bh))))
    pad = max(min_pad, min(max_pad, base))
    pad_x = max(pad, int(round(0.04 * bw)))
    pad_y = max(pad, int(round(0.04 * bh)))

    nx1 = max(0, x1 - pad_x)
    ny1 = max(0, y1 - pad_y)
    nx2 = min(w, x2 + pad_x)
    ny2 = min(h, y2 + pad_y)

    return np.array([[nx1, ny1], [nx2, ny1], [nx2, ny2], [nx1, ny2]], dtype=np.float32)


def add_quad_to_mask(mask: np.ndarray, quad: np.ndarray, dilate_px: int = 0) -> None:
    if mask is None or mask.size == 0:
        return
    q = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    h, w = mask.shape[:2]
    q_int = np.round(q).astype(np.int32)
    q_int[:, 0] = np.clip(q_int[:, 0], 0, w - 1)
    q_int[:, 1] = np.clip(q_int[:, 1], 0, h - 1)
    cv2.fillPoly(mask, [q_int.reshape(-1, 1, 2)], 255)
    if dilate_px > 0:
        k = 2 * dilate_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        cv2.dilate(mask, kernel, dst=mask, iterations=1)


def masked_overlap_ratio(quad: np.ndarray, mask: np.ndarray) -> float:
    if mask is None or mask.size == 0:
        return 0.0
    q = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    h, w = mask.shape[:2]
    q_int = np.round(q).astype(np.int32)
    q_int[:, 0] = np.clip(q_int[:, 0], 0, w - 1)
    q_int[:, 1] = np.clip(q_int[:, 1], 0, h - 1)
    tmp = np.zeros_like(mask)
    cv2.fillPoly(tmp, [q_int.reshape(-1, 1, 2)], 255)
    area = cv2.countNonZero(tmp)
    if area <= 0:
        return 0.0
    inter = cv2.countNonZero(cv2.bitwise_and(tmp, mask))
    return float(inter) / float(area)


def preprocess_image(
    image: np.ndarray,
    variant: int = 0,
    cache: Optional[Dict[Tuple[str, int], np.ndarray]] = None,
) -> np.ndarray:
    """
    Tien xu ly nhanh:
    - Tu dong gamma correction khi anh toi
    - Bilateral filter de giam noise va giu canh module QR
    - Chi giu 2 binary variants: Otsu + 1 adaptive
    - Morphology toi gian: 1 lan close kernel 3x3
    """
    if image is None or image.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    h, w = gray.shape
    mean_val = float(np.mean(gray))

    # Tu dong nang sang anh toi bang gamma correction.
    if mean_val < 90.0:
        # mean cang thap -> gamma cang cao trong khoang 1.8 - 2.2
        t = (90.0 - mean_val) / 90.0
        gamma = 1.8 + 0.4 * float(np.clip(t, 0.0, 1.0))
        inv_gamma = 1.0 / max(1e-6, gamma)
        lut = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)
        gray = cv2.LUT(gray, lut)

    # Bilateral thay cho Gaussian trong tien xu ly de giu canh module.
    denoised = cv2.bilateralFilter(gray, d=5, sigmaColor=35, sigmaSpace=35)

    # Cache CLAHE khi cung mot anh duoc goi nhieu variant.
    img_key = int(denoised.__array_interface__["data"][0])
    cache_key = ("clahe", img_key)
    enhanced: np.ndarray
    if cache is not None and cache_key in cache:
        enhanced = cache[cache_key]
    else:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8 if min(h, w) >= 480 else 6,) * 2)
        enhanced = clahe.apply(denoised)
        if cache is not None:
            cache[cache_key] = enhanced

    # Chi giu 1 adaptive threshold theo yeu cau.
    block = max(15, min(71, min(h, w) // 18))
    if block % 2 == 0:
        block += 1
    c_val = 2 if variant == 0 else 3
    adaptive = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block,
        c_val,
    )
    _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Anh toi uu tien adaptive vi Otsu de lech.
    if mean_val < 90.0:
        binary = adaptive
    else:
        binary = cv2.bitwise_or(adaptive, otsu)

    # Morphology toi gian, 1 lan close 3x3.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Loc noise nho.
    min_area = max(300, int(h * w * 0.00010))
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    for lab in range(1, num_labels):
        if stats[lab, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == lab] = 255
    return cleaned


def find_finder_patterns(binary: np.ndarray) -> List[Tuple[Contour, Point, float]]:
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 5000:
        return []
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


def _extract_patterns_bidirectional(binary: np.ndarray) -> List[Tuple[Contour, Point, float]]:
    local_patterns = find_finder_patterns(binary)
    inv_patterns = find_finder_patterns(cv2.bitwise_not(binary))
    local_patterns.extend(inv_patterns)
    return local_patterns


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
                    # Sanity theo kich thuoc suy ra tu 3 finder centers.
                    side_est = 0.5 * (d_tl_tr + d_tl_bl) + 7.0 * module_size_px
                    area_est = side_est * side_est
                    if area_est > 1.0 and not (0.22 * area_est <= area_q <= 2.8 * area_est):
                        continue
                    if area_q > float(h * w) * 0.38:
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


def suppress_overlapping_quads(
    quads: List[np.ndarray],
    iou_threshold: float = 0.30,
    image_shape: Optional[Tuple[int, int]] = None,
    return_regions: bool = False,
    touch_margin: int = 1,
) -> Union[List[np.ndarray], Tuple[List[np.ndarray], List[np.ndarray]]]:
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
        return ([], []) if return_regions else []

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

    if image_shape is not None:
        h_mask, w_mask = int(image_shape[0]), int(image_shape[1])
    else:
        max_x = int(max(v[2][2] for v in scored) + 3)
        max_y = int(max(v[2][3] for v in scored) + 3)
        h_mask = max(8, max_y)
        w_mask = max(8, max_x)
    occupied_mask = np.zeros((h_mask, w_mask), dtype=np.uint8)

    selected: List[np.ndarray] = []
    selected_boxes: List[Tuple[float, float, float, float]] = []
    selected_regions: List[np.ndarray] = []
    for _, q, b in scored:
        bx1, by1, bx2, by2 = b
        barea = max(1e-6, (bx2 - bx1) * (by2 - by1))
        keep = True

        overlap_region = masked_overlap_ratio(q, occupied_mask)
        if overlap_region > 0.001:
            keep = False

        for sb in selected_boxes:
            if not keep:
                break
            sx1, sy1, sx2, sy2 = sb
            sarea = max(1e-6, (sx2 - sx1) * (sy2 - sy1))
            ix1, iy1 = max(bx1, sx1), max(by1, sy1)
            ix2, iy2 = min(bx2, sx2), min(by2, sy2)
            iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
            inter = iw * ih
            union = barea + sarea - inter
            iou = inter / max(1e-6, union)
            overlap_min = inter / max(1e-6, min(barea, sarea))
            touch_conflict = False
            if touch_margin >= 0:
                touch_conflict = boxes_overlap_or_touch(
                    (int(bx1), int(by1), int(bx2), int(by2)),
                    (int(sx1), int(sy1), int(sx2), int(sy2)),
                    margin=touch_margin,
                )
            if iou >= iou_threshold or overlap_min >= 0.58 or touch_conflict:
                keep = False
                break
        if keep:
            selected.append(q)
            selected_boxes.append(b)
            selected_regions.append(q.copy())
            add_quad_to_mask(occupied_mask, q, dilate_px=max(1, touch_margin))
    if return_regions:
        return selected, selected_regions
    return selected

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
    if len(contours) > 5000:
        return None
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

    # Thu hep them de tang IoU: cat bo bien ria thua theo projection trong ROI moi.
    roi2_x1 = max(0, min(pw, new_x1 - sx1))
    roi2_x2 = max(0, min(pw, new_x2 - sx1))
    roi2_y1 = max(0, min(ph, new_y1 - sy1))
    roi2_y2 = max(0, min(ph, new_y2 - sy1))
    if (roi2_x2 - roi2_x1) >= 6 and (roi2_y2 - roi2_y1) >= 6:
        roi_bin = binary[roi2_y1:roi2_y2, roi2_x1:roi2_x2]
        row_ratio = np.mean(roi_bin > 0, axis=1)
        col_ratio = np.mean(roi_bin > 0, axis=0)

        row_nonzero = row_ratio[row_ratio > 0]
        col_nonzero = col_ratio[col_ratio > 0]
        row_thr = 0.10 if row_nonzero.size == 0 else max(0.08, float(np.percentile(row_nonzero, 35)) * 0.70)
        col_thr = 0.10 if col_nonzero.size == 0 else max(0.08, float(np.percentile(col_nonzero, 35)) * 0.70)

        row_idx = np.where(row_ratio >= row_thr)[0]
        col_idx = np.where(col_ratio >= col_thr)[0]
        if row_idx.size >= 2 and col_idx.size >= 2:
            trim_y1 = int(row_idx[0])
            trim_y2 = int(row_idx[-1]) + 1
            trim_x1 = int(col_idx[0])
            trim_x2 = int(col_idx[-1]) + 1

            test_x1 = roi2_x1 + trim_x1 + sx1
            test_x2 = roi2_x1 + trim_x2 + sx1
            test_y1 = roi2_y1 + trim_y1 + sy1
            test_y2 = roi2_y1 + trim_y2 + sy1
            if (test_x2 - test_x1) >= 6 and (test_y2 - test_y1) >= 6:
                new_x1, new_x2 = test_x1, test_x2
                new_y1, new_y2 = test_y1, test_y2

    # Dam bao thu tu dung
    if new_x1 >= new_x2:
        new_x1, new_x2 = x1, x2
    if new_y1 >= new_y2:
        new_y1, new_y2 = y1, y2

    return int(new_x1), int(new_y1), int(new_x2), int(new_y2)


def _extract_finder_candidates_from_patch(
    patch_gray: np.ndarray,
    relaxed: bool = False,
) -> List[Tuple[np.ndarray, float]]:
    if patch_gray is None or patch_gray.size == 0:
        return []

    h, w = patch_gray.shape[:2]
    if h < 24 or w < 24:
        return []

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
    enhanced = clahe.apply(patch_gray)
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

    binaries: List[np.ndarray] = []
    _, bin_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binaries.append(bin_otsu)

    for block, cval in ((11, 2), (15, 3), (21, 4)):
        if min(h, w) <= block:
            continue
        binaries.append(
            cv2.adaptiveThreshold(
                blurred,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                block,
                cval,
            )
        )

    area_min = 8.0 if relaxed else 10.0
    ratio_max = 2.8 if relaxed else 2.3
    fill_min = 0.18 if relaxed else 0.25
    ratio_om_min = 1.05 if relaxed else 1.15
    ratio_om_max = 7.5 if relaxed else 5.5
    ratio_mi_min = 1.05 if relaxed else 1.15
    ratio_mi_max = 9.0 if relaxed else 7.0
    center_dist_scale = 0.82 if relaxed else 0.65

    best_patterns: List[Tuple[np.ndarray, float]] = []
    for binary in binaries:
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            continue
        hierarchy = hierarchy[0]

        candidates: List[Tuple[np.ndarray, float]] = []
        for i, outer in enumerate(contours):
            child = hierarchy[i][2]
            if child == -1:
                continue
            grandchild = hierarchy[child][2]
            if grandchild == -1:
                continue

            cnts = (outer, contours[child], contours[grandchild])
            areas: List[float] = []
            centers: List[np.ndarray] = []
            ok = True
            for cnt in cnts:
                area = abs(float(cv2.contourArea(cnt)))
                if area < area_min:
                    ok = False
                    break

                x, y, cw, ch = cv2.boundingRect(cnt)
                if cw < 3 or ch < 3:
                    ok = False
                    break

                ratio = max(cw, ch) / max(1.0, min(cw, ch))
                fill = area / max(1.0, float(cw * ch))
                if ratio > ratio_max or fill < fill_min:
                    ok = False
                    break

                m = cv2.moments(cnt)
                if abs(m["m00"]) < 1e-6:
                    ok = False
                    break

                cx = float(m["m10"] / m["m00"])
                cy = float(m["m01"] / m["m00"])
                centers.append(np.array([cx, cy], dtype=np.float32))
                areas.append(area)

            if not ok:
                continue

            area_o, area_m, area_i = areas
            if not (area_o > area_m > area_i):
                continue

            ratio_om = area_o / max(1e-6, area_m)
            ratio_mi = area_m / max(1e-6, area_i)
            if not (ratio_om_min <= ratio_om <= ratio_om_max and ratio_mi_min <= ratio_mi <= ratio_mi_max):
                continue

            _, _, bw, bh = cv2.boundingRect(outer)
            if max(
                np.linalg.norm(centers[0] - centers[1]),
                np.linalg.norm(centers[0] - centers[2]),
                np.linalg.norm(centers[1] - centers[2]),
            ) > max(bw, bh) * center_dist_scale:
                continue

            candidates.append((centers[0], area_o))

        uniq: List[Tuple[np.ndarray, float]] = []
        for center, area in sorted(candidates, key=lambda x: x[1], reverse=True):
            duplicated = False
            for old_center, old_area in uniq:
                if (
                    np.linalg.norm(center - old_center) < (5.0 if relaxed else 4.0)
                    and min(area, old_area) / max(1e-6, max(area, old_area)) > 0.55
                ):
                    duplicated = True
                    break
            if not duplicated:
                uniq.append((center, area))

        if len(uniq) > len(best_patterns):
            best_patterns = uniq

    return best_patterns


def _extract_finder_centers_from_patch(
    patch_gray: np.ndarray,
    relaxed: bool = False,
) -> List[np.ndarray]:
    candidates = _extract_finder_candidates_from_patch(patch_gray, relaxed=relaxed)
    return [c for c, _ in candidates]


def _qr_transition_score(patch_gray: np.ndarray, target_size: int = 48) -> float:
    if patch_gray is None or patch_gray.size == 0:
        return 0.0

    h, w = patch_gray.shape[:2]
    interp = cv2.INTER_NEAREST if min(h, w) < target_size else cv2.INTER_AREA
    small = cv2.resize(patch_gray, (target_size, target_size), interpolation=interp)
    _, binary = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    denom = float(max(1, target_size - 1))
    row_trans = float(np.mean(np.sum(binary[:, 1:] != binary[:, :-1], axis=1) / denom))
    col_trans = float(np.mean(np.sum(binary[1:, :] != binary[:-1, :], axis=0) / denom))
    return 0.5 * (row_trans + col_trans)


def _looks_like_micro_qr_patch(patch_gray: np.ndarray) -> bool:
    """
    Heuristic nhe cho QR rat nho:
    - patch co texture den/trang dang module
    - dark ratio khong qua cao/thap
    """
    if patch_gray is None or patch_gray.size == 0:
        return False
    h, w = patch_gray.shape[:2]
    if h < 8 or w < 8:
        return False

    target = 24
    interp = cv2.INTER_NEAREST if min(h, w) < target else cv2.INTER_AREA
    small = cv2.resize(patch_gray, (target, target), interpolation=interp)
    _, binary = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    dark_ratio = float(np.mean(binary > 0))
    if dark_ratio < 0.14 or dark_ratio > 0.88:
        return False

    transition = _qr_transition_score(patch_gray, target_size=target)
    if transition < 0.125:
        return False

    # Kiem tra do da dang module theo o luoi 4x4.
    mixed_cells = 0
    step = target // 4
    for gy in range(4):
        for gx in range(4):
            y0 = gy * step
            y1 = target if gy == 3 else (gy + 1) * step
            x0 = gx * step
            x1 = target if gx == 3 else (gx + 1) * step
            ratio_dark = float(np.mean(binary[y0:y1, x0:x1] > 0))
            if 0.10 <= ratio_dark <= 0.90:
                mixed_cells += 1
    return mixed_cells >= 7


def _search_tiny_qr_in_box(
    gray_image: np.ndarray,
    box_xyxy: Tuple[int, int, int, int],
) -> Optional[Tuple[int, int, int, int]]:
    x1, y1, x2, y2 = box_xyxy
    roi = gray_image[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    h, w = roi.shape[:2]
    if h < 24 or w < 24:
        return None

    tile = 8 if min(h, w) >= 140 else 4
    enhanced = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(tile, tile)).apply(roi)
    adaptive_block = max(11, min(25, ((min(h, w) // 18) | 1)))
    variants = [
        (cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1], 1.0),
        (
            cv2.adaptiveThreshold(
                enhanced,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                adaptive_block,
                2,
            ),
            1.15,
        ),
    ]

    best_box: Optional[Tuple[int, int, int, int]] = None
    best_score = -1e9
    for binary, src_weight in variants:
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for lab in range(1, num_labels):
            area = float(stats[lab, cv2.CC_STAT_AREA])
            if area < 120.0 or area > 2200.0:
                continue

            bx = int(stats[lab, cv2.CC_STAT_LEFT])
            by = int(stats[lab, cv2.CC_STAT_TOP])
            bw = int(stats[lab, cv2.CC_STAT_WIDTH])
            bh = int(stats[lab, cv2.CC_STAT_HEIGHT])
            if bw < 12 or bh < 12 or bw > 90 or bh > 90:
                continue

            ratio = max(bw, bh) / max(1.0, min(bw, bh))
            if ratio > 2.1:
                continue

            fill = area / max(1.0, float(bw * bh))
            if fill < 0.30 or fill > 0.92:
                continue

            patch = roi[by:by + bh, bx:bx + bw]
            if patch.size == 0:
                continue

            std_val = float(np.std(patch))
            if std_val < 18.0:
                continue

            if not _looks_like_micro_qr_patch(patch):
                continue

            trans = _qr_transition_score(patch, target_size=24)
            score = src_weight + 1.8 * trans + 0.010 * std_val - 0.07 * abs(ratio - 1.0)
            if score > best_score:
                best_score = score
                best_box = (x1 + bx, y1 + by, x1 + bx + bw, y1 + by + bh)

    return best_box


def _tighten_bbox_with_finder_geometry(
    image_bgr: np.ndarray,
    box_xyxy: Tuple[int, int, int, int],
) -> Optional[np.ndarray]:
    """
    Thu hep bbox lon dua tren finder candidates.
    - Uu tien truong hop bbox qua lon do bao gom nen/chu.
    - Neu khong co finder ma bbox lon, thu tim QR tiny ben trong.
    """
    if image_bgr is None or image_bgr.size == 0:
        return None

    x1, y1, x2, y2 = box_xyxy
    bw0 = x2 - x1
    bh0 = y2 - y1
    if bw0 < 12 or bh0 < 12:
        return None

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr.copy()
    h_img, w_img = gray.shape[:2]
    img_area = float(h_img * w_img)
    box_area = float(bw0 * bh0)
    patch = gray[y1:y2, x1:x2]
    if patch.size == 0:
        return None

    finder_candidates = _extract_finder_candidates_from_patch(patch, relaxed=True)
    finder_count = len(finder_candidates)

    if finder_count == 0:
        # Bbox rat lon nhung khong thay finder -> thay bang tiny-QR neu co.
        if box_area > 0.08 * img_area:
            tiny = _search_tiny_qr_in_box(gray, box_xyxy)
            if tiny is not None:
                tx1, ty1, tx2, ty2 = tiny
                return np.array([[tx1, ty1], [tx2, ty1], [tx2, ty2], [tx1, ty2]], dtype=np.float32)
        return None

    centers = np.array([c for c, _ in finder_candidates], dtype=np.float32)
    areas = np.array([a for _, a in finder_candidates], dtype=np.float32)
    finder_side = float(np.median(np.sqrt(np.maximum(areas, 1.0))))

    dmax = 0.0
    pair = (0, 0)
    if finder_count >= 2:
        for i in range(finder_count):
            for j in range(i + 1, finder_count):
                d = float(np.linalg.norm(centers[i] - centers[j]))
                if d > dmax:
                    dmax = d
                    pair = (i, j)

    if finder_count >= 2:
        side_est = max(finder_side * 3.2, dmax + 0.9 * finder_side)
        too_large = (max(bw0, bh0) > side_est * 2.35) and (box_area > side_est * side_est * 4.8)
    else:
        side_est = max(36.0, finder_side * 4.8)
        too_large = max(bw0, bh0) > side_est * 1.55

    if not too_large:
        return None

    # Tao cac gia thuyet tam theo hinh hoc finder.
    center_hypotheses: List[np.ndarray] = []
    if finder_count >= 2:
        cmean = np.mean(centers, axis=0)
        center_hypotheses.append(cmean)
        v = centers[pair[1]] - centers[pair[0]]
        vp = np.array([-v[1], v[0]], dtype=np.float32)
        vn = float(np.linalg.norm(vp))
        if vn > 1e-6:
            vp /= vn
            center_hypotheses.append(cmean + vp * dmax * 0.45)
            center_hypotheses.append(cmean - vp * dmax * 0.45)
    else:
        fc = centers[0]
        shift = side_est * 0.36
        center_hypotheses = [
            fc,
            np.array([fc[0] - shift, fc[1] - shift], dtype=np.float32),
            np.array([fc[0] + shift, fc[1] - shift], dtype=np.float32),
            np.array([fc[0] - shift, fc[1] + shift], dtype=np.float32),
            np.array([fc[0] + shift, fc[1] + shift], dtype=np.float32),
        ]

    size_multipliers = [0.74, 0.90, 1.05, 1.22, 1.38]
    wh_multipliers = [(1.0, 1.0), (0.85, 1.25), (1.25, 0.85), (0.80, 1.45), (1.45, 0.80)]

    best_score = -1e9
    best_box_local: Optional[Tuple[int, int, int, int]] = None
    for center in center_hypotheses:
        cx = float(center[0])
        cy = float(center[1])
        for sm in size_multipliers:
            for rw, rh in wh_multipliers:
                tw = max(12, int(round(side_est * sm * rw)))
                th = max(12, int(round(side_est * sm * rh)))
                bx1 = int(round(cx - 0.5 * tw))
                by1 = int(round(cy - 0.5 * th))
                bx2 = bx1 + tw
                by2 = by1 + th

                if bx1 < 0:
                    bx2 -= bx1
                    bx1 = 0
                if by1 < 0:
                    by2 -= by1
                    by1 = 0
                if bx2 > bw0:
                    bx1 -= (bx2 - bw0)
                    bx2 = bw0
                if by2 > bh0:
                    by1 -= (by2 - bh0)
                    by2 = bh0
                bx1 = max(0, bx1)
                by1 = max(0, by1)
                bx2 = min(bw0, bx2)
                by2 = min(bh0, by2)
                if (bx2 - bx1) < 12 or (by2 - by1) < 12:
                    continue

                sub = patch[by1:by2, bx1:bx2]
                if sub.size == 0:
                    continue
                trans = _qr_transition_score(sub, target_size=48)
                if trans < 0.09:
                    continue

                inside = int(np.sum(
                    (centers[:, 0] >= bx1) & (centers[:, 0] <= bx2)
                    & (centers[:, 1] >= by1) & (centers[:, 1] <= by2)
                ))
                if inside <= 0:
                    continue

                relaxed_ok = verify_qr_finder_signature_relaxed(sub)
                strict_ok = verify_qr_finder_signature(sub)
                texture_ok = _has_qr_texture_signature(sub)

                area_sub = float((bx2 - bx1) * (by2 - by1))
                score = (
                    2.4 * trans
                    + 1.2 * float(inside)
                    + (1.8 if relaxed_ok else 0.0)
                    + (1.0 if strict_ok else 0.0)
                    + (0.8 if texture_ok else 0.0)
                    - 0.00001 * area_sub
                )
                if score > best_score:
                    best_score = score
                    best_box_local = (bx1, by1, bx2, by2)

    if best_box_local is None or best_score < 2.2:
        return None

    bx1, by1, bx2, by2 = best_box_local
    abs_x1 = x1 + bx1
    abs_y1 = y1 + by1
    abs_x2 = x1 + bx2
    abs_y2 = y1 + by2

    abs_x1, abs_y1, abs_x2, abs_y2 = refine_bbox_by_scanning(
        image_bgr, abs_x1, abs_y1, abs_x2, abs_y2, w_img, h_img, max_expand=12
    )
    if abs_x2 - abs_x1 < 10 or abs_y2 - abs_y1 < 10:
        return None

    new_area = float((abs_x2 - abs_x1) * (abs_y2 - abs_y1))
    if new_area >= box_area * 0.92:
        return None

    return np.array([[abs_x1, abs_y1], [abs_x2, abs_y1], [abs_x2, abs_y2], [abs_x1, abs_y2]], dtype=np.float32)


def _has_qr_texture_signature(patch_gray: np.ndarray) -> bool:
    if patch_gray is None or patch_gray.size == 0:
        return False
    h, w = patch_gray.shape[:2]
    if h < 20 or w < 20:
        return False

    small = cv2.resize(patch_gray, (56, 56), interpolation=cv2.INTER_AREA)
    _, binary = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    row_trans = np.mean(np.sum(binary[:, 1:] != binary[:, :-1], axis=1) / 55.0)
    col_trans = np.mean(np.sum(binary[1:, :] != binary[:-1, :], axis=0) / 55.0)
    transition = 0.5 * (row_trans + col_trans)
    if transition < 0.105:
        return False

    dark_ratio = float(np.mean(binary > 0))
    if dark_ratio < 0.12 or dark_ratio > 0.88:
        return False

    cell = 7
    dark_cells = 0
    mixed_cells = 0
    for gy in range(cell):
        for gx in range(cell):
            y0 = gy * (56 // cell)
            y1 = 56 if gy == cell - 1 else (gy + 1) * (56 // cell)
            x0 = gx * (56 // cell)
            x1 = 56 if gx == cell - 1 else (gx + 1) * (56 // cell)
            ratio_dark = float(np.mean(binary[y0:y1, x0:x1] > 0))
            if ratio_dark > 0.20:
                dark_cells += 1
            if 0.15 <= ratio_dark <= 0.85:
                mixed_cells += 1
    if dark_cells < 12 or mixed_cells < 10:
        return False
    return True


def _looks_like_datamatrix_border(patch_gray: np.ndarray) -> bool:
    if patch_gray is None or patch_gray.size == 0:
        return False
    small = cv2.resize(patch_gray, (56, 56), interpolation=cv2.INTER_AREA)
    _, binary = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    strip = 4
    top = float(np.mean(binary[:strip, :] > 0))
    bottom = float(np.mean(binary[-strip:, :] > 0))
    left = float(np.mean(binary[:, :strip] > 0))
    right = float(np.mean(binary[:, -strip:] > 0))

    sides = {"top": top, "right": right, "bottom": bottom, "left": left}
    adjacent = [("top", "left", "bottom", "right"), ("top", "right", "bottom", "left"),
                ("bottom", "left", "top", "right"), ("bottom", "right", "top", "left")]
    for a, b, oa, ob in adjacent:
        if sides[a] > 0.76 and sides[b] > 0.76 and sides[oa] < 0.66 and sides[ob] < 0.66:
            return True
    return False


def verify_qr_finder_signature(patch_gray: np.ndarray) -> bool:
    """
    Xac thuc patch co cau truc finder pattern QR:
    - Loi den 3x3
    - Vanh trang 5x5
    - Vanh den ngoai 7x7
    (thong qua contour long nhau + hinh hoc 3 finder patterns).
    """
    best_centers = _extract_finder_centers_from_patch(patch_gray, relaxed=False)
    if len(best_centers) < 3:
        return False

    n = len(best_centers)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                pts = [best_centers[i], best_centers[j], best_centers[k]]
                d01 = float(np.linalg.norm(pts[1] - pts[0]))
                d02 = float(np.linalg.norm(pts[2] - pts[0]))
                d12 = float(np.linalg.norm(pts[2] - pts[1]))
                lens = [d01, d02, d12]

                longest_idx = int(np.argmax(lens))
                long_edge = lens[longest_idx]
                pair = [(0, 1), (0, 2), (1, 2)][longest_idx]
                right_idx = ({0, 1, 2} - set(pair)).pop()
                a_idx, b_idx = [u for u in (0, 1, 2) if u != right_idx]

                va = pts[a_idx] - pts[right_idx]
                vb = pts[b_idx] - pts[right_idx]
                la = float(np.linalg.norm(va))
                lb = float(np.linalg.norm(vb))
                if la < 4.0 or lb < 4.0:
                    continue

                cos_val = abs(float(np.dot(va, vb) / (la * lb)))
                len_ratio = min(la, lb) / max(la, lb)
                hyp = float(np.sqrt(la * la + lb * lb))

                if cos_val < 0.58 and len_ratio > 0.35 and 0.70 * hyp <= long_edge <= 1.45 * hyp:
                    return True

    return False


def verify_qr_finder_signature_relaxed(patch_gray: np.ndarray) -> bool:
    """
    Ban relaxed:
    - Uu tien pass strict truoc.
    - Cho phep 2 finder + texture QR trong truong hop 1 finder bi thieu do goc chup/crop.
    """
    if verify_qr_finder_signature(patch_gray):
        return True

    if patch_gray is None or patch_gray.size == 0:
        return False
    h, w = patch_gray.shape[:2]
    if h < 24 or w < 24:
        return False

    centers = _extract_finder_centers_from_patch(patch_gray, relaxed=True)
    n = len(centers)

    # Neu co >=3 finder thi kiem tra hinh hoc voi nguong nhe hon.
    if n >= 3:
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    pts = [centers[i], centers[j], centers[k]]
                    d01 = float(np.linalg.norm(pts[1] - pts[0]))
                    d02 = float(np.linalg.norm(pts[2] - pts[0]))
                    d12 = float(np.linalg.norm(pts[2] - pts[1]))
                    lens = [d01, d02, d12]

                    longest_idx = int(np.argmax(lens))
                    long_edge = lens[longest_idx]
                    pair = [(0, 1), (0, 2), (1, 2)][longest_idx]
                    right_idx = ({0, 1, 2} - set(pair)).pop()
                    a_idx, b_idx = [u for u in (0, 1, 2) if u != right_idx]

                    va = pts[a_idx] - pts[right_idx]
                    vb = pts[b_idx] - pts[right_idx]
                    la = float(np.linalg.norm(va))
                    lb = float(np.linalg.norm(vb))
                    if la < 4.0 or lb < 4.0:
                        continue

                    cos_val = abs(float(np.dot(va, vb) / (la * lb)))
                    len_ratio = min(la, lb) / max(la, lb)
                    hyp = float(np.sqrt(la * la + lb * lb))

                    if cos_val < 0.82 and len_ratio > 0.15 and 0.56 * hyp <= long_edge <= 1.90 * hyp:
                        return True

    # Truong hop chi thay duoc 2 finder: xac thuc them texture va loai Data Matrix.
    if n >= 2 and _has_qr_texture_signature(patch_gray) and not _looks_like_datamatrix_border(patch_gray):
        return True

    # Truong hop sat bien/goc chup xau: co the chi bat duoc 1 finder.
    # Chi mo rong voi patch du lon de tranh nham text nho.
    if (
        n >= 1
        and min(h, w) >= 120
        and max(h, w) >= 170
        and _has_qr_texture_signature(patch_gray)
        and not _looks_like_datamatrix_border(patch_gray)
    ):
        return True

    return False


def detect_qr_by_outline(
    img: np.ndarray,
    max_candidates: int = 4,
    blocked_mask: Optional[np.ndarray] = None,
) -> List[np.ndarray]:
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
    low_light = float(np.mean(gray)) < 95.0

    # Bilateral denoise + LoG -> Canny de bat canh ro hon trong anh mo.
    denoised = cv2.bilateralFilter(gray, d=5, sigmaColor=35, sigmaSpace=35)
    log_img = cv2.Laplacian(denoised, cv2.CV_16S, ksize=3)
    log_abs = cv2.convertScaleAbs(log_img)
    edge = cv2.Canny(log_abs, 55, 150)

    block = max(15, min(71, min(H, W) // 18))
    if block % 2 == 0:
        block += 1
    adaptive = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block,
        2 if low_light else 3,
    )
    _, otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    variants = [adaptive, otsu, edge]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    results: List[np.ndarray] = []
    seen_boxes: List[Tuple[int, int, int, int]] = []
    use_blocked = blocked_mask is not None and cv2.countNonZero(blocked_mask) > 0

    for binary_raw in variants:
        binary = cv2.morphologyEx(binary_raw, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 5000:
            continue

        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < 160.0 or area > float(H * W) * 0.35:
                continue

            x, y, cw, ch = cv2.boundingRect(cnt)
            if cw < 10 or ch < 10:
                continue

            ratio = max(cw, ch) / max(1.0, min(cw, ch))
            if ratio > 3.3:
                continue
            fill_rect = area / max(1.0, float(cw * ch))
            if fill_rect < 0.26 or fill_rect > 0.96:
                continue

            exp_o = max(int(max(cw, ch) * 0.10), 8)
            ex1 = max(0, x - exp_o)
            ey1 = max(0, y - exp_o)
            ex2 = min(W, x + cw + exp_o)
            ey2 = min(H, y + ch + exp_o)
            patch_gray = gray[ey1:ey2, ex1:ex2]
            if patch_gray.size < 100:
                continue
            if float(np.std(patch_gray)) < 20.0:
                continue

            patch_small = cv2.resize(gray[y:y + ch, x:x + cw], (48, 48), interpolation=cv2.INTER_AREA)
            _, patch_bin = cv2.threshold(patch_small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            row_trans = np.mean(np.sum(patch_bin[:, 1:] != patch_bin[:, :-1], axis=1) / 47.0)
            col_trans = np.mean(np.sum(patch_bin[1:, :] != patch_bin[:-1, :], axis=0) / 47.0)
            transition_score = 0.5 * (row_trans + col_trans)
            if transition_score < 0.09:
                continue

            if not verify_qr_finder_signature_relaxed(patch_gray):
                continue

            box = (x, y, x + cw, y + ch)
            if use_blocked:
                block_ratio = masked_overlap_ratio(
                    np.array([[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]], dtype=np.float32),
                    blocked_mask,
                )
                if block_ratio > 0.02:
                    continue

            if any(
                bbox_iou_xyxy(box, old) > 0.35 or boxes_overlap_or_touch(box, old, margin=1)
                for old in seen_boxes
            ):
                continue

            quad = np.array([[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]], dtype=np.float32)
            results.append(quad)
            seen_boxes.append(box)

            if len(results) >= max_candidates * 2:
                break

    return results[: max_candidates * 2]


def detect_dense_small_qr_components(
    img: np.ndarray,
    blocked_mask: Optional[np.ndarray] = None,
    min_cluster_count: int = 25,
    max_candidates: int = 220,
) -> List[np.ndarray]:
    """
    Fast-path cho anh co nhieu QR nho, sat nhau.
    Y tuong:
    - Dung Otsu + connected-components de tach cac cum module QR
    - Lay cum co dien tich dong deu (mode area), ratio hop ly
    - NMS theo IoU de tranh trung lap
    """
    if img is None or img.size == 0:
        return []

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    variants = [
        (cv2.morphologyEx(
            otsu,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=1,
        ), 2.0),
        (otsu, 1.0),
    ]

    border = max(1, min(h, w) // 320)
    use_blocked_mask = blocked_mask is not None and cv2.countNonZero(blocked_mask) > 0
    area_upper = max(12000, int(h * w * 0.12))
    candidates: List[Tuple[float, np.ndarray, float, float]] = []

    for binary, src_weight in variants:
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for lab in range(1, num_labels):
            area = float(stats[lab, cv2.CC_STAT_AREA])
            if area < 650 or area > area_upper:
                continue

            x = int(stats[lab, cv2.CC_STAT_LEFT])
            y = int(stats[lab, cv2.CC_STAT_TOP])
            bw = int(stats[lab, cv2.CC_STAT_WIDTH])
            bh = int(stats[lab, cv2.CC_STAT_HEIGHT])
            x2 = x + bw
            y2 = y + bh

            if bw < 14 or bh < 14:
                continue
            if x <= border or y <= border or x2 >= w - border or y2 >= h - border:
                continue

            ratio = max(bw, bh) / max(1.0, min(bw, bh))
            if ratio > 2.6:
                continue

            # Kiem tra nhanh texture QR de tranh fast-path bat nham hoa tiet/text.
            px1 = max(0, x - 2)
            py1 = max(0, y - 2)
            px2 = min(w, x2 + 2)
            py2 = min(h, y2 + 2)
            patch = gray[py1:py2, px1:px2]
            if patch.size == 0:
                continue
            if float(np.std(patch)) < 20.0:
                continue
            if not _has_qr_texture_signature(patch):
                continue
            if _qr_transition_score(patch, target_size=40) < 0.115:
                continue

            quad = np.array([[x, y], [x2, y], [x2, y2], [x, y2]], dtype=np.float32)
            if use_blocked_mask and masked_overlap_ratio(quad, blocked_mask) > 0.03:
                continue

            score = src_weight + 0.0002 * area
            candidates.append((score, quad, area, ratio))

    if len(candidates) < min_cluster_count:
        return []

    areas = np.array([c[2] for c in candidates], dtype=np.float32)
    area_mean = float(np.mean(areas))
    area_std = float(np.std(areas))
    area_cv = area_std / max(1e-6, area_mean)
    if area_cv > 0.65:
        return []

    q1 = float(np.percentile(areas, 25))
    q3 = float(np.percentile(areas, 75))
    med = float(np.median(areas))
    area_low = max(380.0, min(q1 * 0.90, med * 0.72))
    area_high = max(area_low + 40.0, max(q3 * 1.20, med * 1.35))

    filtered: List[Tuple[float, np.ndarray]] = []
    for score, quad, area, ratio in candidates:
        if area < area_low or area > area_high:
            continue
        if ratio > 1.75:
            continue
        filtered.append((score, quad))

    if len(filtered) < max(8, min_cluster_count // 3):
        return []

    filtered.sort(key=lambda x: x[0], reverse=True)
    selected: List[np.ndarray] = []
    selected_boxes: List[Tuple[int, int, int, int]] = []
    for _, q in filtered:
        b = quad_to_box_int(q)
        if any(bbox_iou_xyxy(b, old) > 0.34 for old in selected_boxes):
            continue
        selected.append(q)
        selected_boxes.append(b)
        if len(selected) >= max_candidates:
            break
    return selected


def detect_qr_by_components(
    img: np.ndarray,
    blocked_mask: Optional[np.ndarray] = None,
    max_candidates: int = 16,
) -> List[np.ndarray]:
    """
    Detect QR theo connected-components + xac thuc finder-signature 7x7.
    Muc tieu: bo sung khi finder-pattern global bi thieu va tranh nham Data Matrix.
    """
    if img is None or img.size == 0:
        return []

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    low_light = float(np.mean(gray)) < 105.0
    # Preprocess giu bien cuc bo de khong gom ca vung toi lon thanh mot component.
    enhanced = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)

    _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    variants = [
        (
            cv2.morphologyEx(
                otsu,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                iterations=1,
            ),
            2.0,
        ),
        (otsu, 1.0),
        (
            cv2.adaptiveThreshold(
                enhanced,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                max(11, min(31, ((min(h, w) // 20) | 1))),
                3,
            ),
            1.2,
        ),
    ]

    min_area = max(280.0, float(h * w) * 0.0005)
    max_area = max(8000.0, float(h * w) * 0.28)
    border = max(1, min(h, w) // 320)

    scored: List[Tuple[float, np.ndarray]] = []
    for binary, src_weight in variants:
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for lab in range(1, num_labels):
            area = float(stats[lab, cv2.CC_STAT_AREA])
            if area < min_area or area > max_area:
                continue

            x = int(stats[lab, cv2.CC_STAT_LEFT])
            y = int(stats[lab, cv2.CC_STAT_TOP])
            bw = int(stats[lab, cv2.CC_STAT_WIDTH])
            bh = int(stats[lab, cv2.CC_STAT_HEIGHT])
            x2 = x + bw
            y2 = y + bh

            if bw < 16 or bh < 16:
                continue
            if x <= border or y <= border or x2 >= w - border or y2 >= h - border:
                continue

            ratio = max(bw, bh) / max(1.0, min(bw, bh))
            if ratio > 2.6:
                continue
            if low_light and (bw > int(0.58 * w) or bh > int(0.58 * h)):
                continue
            bbox_area = float(bw * bh)
            fill = area / max(1.0, bbox_area)
            fill_low = 0.22 if low_light else 0.15
            if fill < fill_low or fill > 0.90:
                continue

            quad = np.array([[x, y], [x2, y], [x2, y2], [x, y2]], dtype=np.float32)
            if blocked_mask is not None and masked_overlap_ratio(quad, blocked_mask) > 0.03:
                continue

            # finder pattern (vong ngoai) duoc bao gom day du trong patch
            expand_px = max(int(max(bw, bh) * 0.10), 8)
            ex1 = max(0, x - expand_px)
            ey1 = max(0, y - expand_px)
            ex2 = min(w, x2 + expand_px)
            ey2 = min(h, y2 + expand_px)
            patch_gray = gray[ey1:ey2, ex1:ex2]
            if patch_gray.size == 0:
                continue
            if float(np.std(patch_gray)) < 20.0:
                continue
            relaxed_ok = verify_qr_finder_signature_relaxed(patch_gray)
            if not relaxed_ok:
                continue

            # Chan false-positive bbox qua lon (hay nham nen/hoa tiet) bang ti le
            # giua kich thuoc bbox va kich thuoc finder tim duoc.
            finder_candidates = _extract_finder_candidates_from_patch(patch_gray, relaxed=True)
            finder_count = len(finder_candidates)
            if finder_count <= 0:
                continue
            finder_side = float(
                np.median(np.sqrt(np.maximum(np.array([a for _, a in finder_candidates], dtype=np.float32), 1.0)))
            )
            if finder_side < 1.0:
                continue
            bbox_scale_ratio = np.sqrt(max(1.0, bbox_area)) / finder_side
            if bbox_scale_ratio > 11.0:
                continue

            size_penalty = np.sqrt(bbox_area / max(1.0, float(h * w)))
            score = (
                src_weight
                + 0.00004 * area
                + 0.8 * fill
                + 0.10 * min(3.0, float(finder_count))
                - (0.35 if low_light else 0.22) * size_penalty
                - 0.08 * abs(ratio - 1.0)
            )
            scored.append((score, quad))

    if not scored:
        return []

    scored.sort(key=lambda x: x[0], reverse=True)
    selected: List[np.ndarray] = []
    selected_boxes: List[Tuple[int, int, int, int]] = []
    for _, q in scored:
        b = quad_to_box_int(q)
        if any(
            bbox_iou_xyxy(b, old) > 0.34
            or boxes_overlap_or_touch(b, old, margin=1)
            for old in selected_boxes
        ):
            continue
        selected.append(q)
        selected_boxes.append(b)
        if len(selected) >= max_candidates:
            break
    return selected


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
    low_light = float(np.mean(gray)) < 105.0

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
        if low_light and not verify_qr_finder_signature_relaxed(patch):
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


def detect_qr_by_bright_square(
    image: np.ndarray,
    blocked_mask: Optional[np.ndarray] = None,
    max_candidates: int = 2,
) -> List[np.ndarray]:
    """
    Fallback cho anh ro net co QR lon + xoay + choi sang:
    tim contour vung sang gan vuong, sau do xac thuc bang finder-signature.
    """
    if image is None or image.size == 0:
        return []

    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8, 8)).apply(gray)

    raw_binaries: List[np.ndarray] = []
    for thr in (150, 160, 170, 180, 190):
        _, b = cv2.threshold(clahe, thr, 255, cv2.THRESH_BINARY)
        raw_binaries.append(b)

    scored: List[Tuple[float, np.ndarray]] = []
    for binary in raw_binaries:
        closed = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
            iterations=2,
        )
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < 5000.0 or area > float(h * w) * 0.85:
                continue

            rect = cv2.minAreaRect(cnt)
            (_, _), (rw, rh), _ = rect
            if rw < 40.0 or rh < 40.0:
                continue
            ratio = max(rw, rh) / max(1.0, min(rw, rh))
            if ratio > 2.0:
                continue

            box = cv2.boxPoints(rect).astype(np.float32)
            x1, y1, x2, y2 = quad_to_box_int(box)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            if x2 - x1 < 40 or y2 - y1 < 40:
                continue

            quad = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
            if blocked_mask is not None and masked_overlap_ratio(quad, blocked_mask) > 0.08:
                continue

            patch = gray[y1:y2, x1:x2]
            if patch.size == 0:
                continue

            finder_candidates = _extract_finder_candidates_from_patch(patch, relaxed=True)
            if not finder_candidates:
                continue
            finder_side = float(
                np.median(np.sqrt(np.maximum(np.array([a for _, a in finder_candidates], dtype=np.float32), 1.0)))
            )
            if finder_side < 1.0:
                continue

            bbox_scale_ratio = np.sqrt(float((x2 - x1) * (y2 - y1))) / finder_side
            if bbox_scale_ratio > 11.0:
                continue

            relaxed_ok = verify_qr_finder_signature_relaxed(patch)
            trans = _qr_transition_score(patch, target_size=48)
            if (not relaxed_ok) and trans < 0.12:
                continue

            score = 0.85 * trans + 0.22 * min(3.0, float(len(finder_candidates))) - 0.02 * abs(ratio - 1.0)
            scored.append((score, quad))

    if not scored:
        return []

    scored.sort(key=lambda x: x[0], reverse=True)
    selected: List[np.ndarray] = []
    selected_boxes: List[Tuple[int, int, int, int]] = []
    for _, q in scored:
        b = quad_to_box_int(q)
        if any(bbox_iou_xyxy(b, old) > 0.45 for old in selected_boxes):
            continue
        selected.append(q)
        selected_boxes.append(b)
        if len(selected) >= max_candidates:
            break
    return selected


def _has_dense_qr_layout_evidence(
    quads: List[np.ndarray],
    image_shape: Tuple[int, int],
) -> bool:
    """
    Chi cho fast-path khi co bang chung ro anh la lo QR day dac:
    - So luong candidate lon
    - Phan bo tam box theo nhieu cell (khong tap trung 1 vung)
    - Khoang cach lan can gan nhau va khong qua loan
    """
    if len(quads) < 40:
        return False
    h, w = int(image_shape[0]), int(image_shape[1])
    if h <= 0 or w <= 0:
        return False

    boxes = [quad_to_box_int(q) for q in quads]
    centers = np.array(
        [[0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3])] for b in boxes],
        dtype=np.float32,
    )
    if centers.shape[0] < 40:
        return False

    span_x = float(np.max(centers[:, 0]) - np.min(centers[:, 0]))
    span_y = float(np.max(centers[:, 1]) - np.min(centers[:, 1]))
    if span_x < 0.30 * float(w) or span_y < 0.22 * float(h):
        return False

    # Occupancy grid de dam bao phan bo khong bi don vao mot cum.
    grid_x, grid_y = 6, 5
    occ = np.zeros((grid_y, grid_x), dtype=np.int32)
    for cx, cy in centers:
        gx = int(np.clip((cx / max(1.0, float(w))) * grid_x, 0, grid_x - 1))
        gy = int(np.clip((cy / max(1.0, float(h))) * grid_y, 0, grid_y - 1))
        occ[gy, gx] += 1
    active_cells = int(np.count_nonzero(occ > 0))
    if active_cells < 10:
        return False

    # Kiem tra nearest-neighbor spacing de loai cac cum noise ngau nhien.
    sample = centers[: min(120, centers.shape[0])]
    diff = sample[:, None, :] - sample[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    nn = np.min(dist, axis=1)
    nn = nn[np.isfinite(nn)]
    if nn.size < 20:
        return False
    nn_med = float(np.median(nn))
    nn_mean = float(np.mean(nn))
    if nn_med < 6.0 or nn_med > 55.0:
        return False
    nn_cv = float(np.std(nn)) / max(1e-6, nn_mean)
    if nn_cv > 0.90:
        return False
    return True


def detect_blurry_wide_qr_components(
    image: np.ndarray,
    blocked_mask: Optional[np.ndarray] = None,
    max_candidates: int = 8,
) -> List[np.ndarray]:
    """
    Detect QR mo/perspective co dang ngang (ratio 1.6 -> 3.4),
    thuong gap trong anh co nhieu QR nho theo cot/chuoi.
    """
    if image is None or image.size == 0:
        return []

    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    enhanced = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    binaries = [
        cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 17, 2),
        cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 3),
        cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 4),
    ]

    border = max(1, min(h, w) // 320)
    scored: List[Tuple[float, np.ndarray, int, int]] = []
    for binary in binaries:
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for lab in range(1, num_labels):
            area = float(stats[lab, cv2.CC_STAT_AREA])
            x = int(stats[lab, cv2.CC_STAT_LEFT])
            y = int(stats[lab, cv2.CC_STAT_TOP])
            bw = int(stats[lab, cv2.CC_STAT_WIDTH])
            bh = int(stats[lab, cv2.CC_STAT_HEIGHT])
            x2 = x + bw
            y2 = y + bh

            if bw < 70 or bh < 24 or bw > 220 or bh > 130:
                continue
            if x <= border or y <= border or x2 >= w - border or y2 >= h - border:
                continue

            ratio = max(bw, bh) / max(1.0, min(bw, bh))
            if ratio < 1.6 or ratio > 3.4:
                continue

            fill = area / max(1.0, float(bw * bh))
            if fill < 0.18 or fill > 0.70:
                continue

            patch = gray[y:y2, x:x2]
            if patch.size == 0:
                continue
            trans = _qr_transition_score(patch, target_size=48)
            std_val = float(np.std(patch))
            if trans < 0.22 or std_val < 30.0:
                continue
            if not _has_qr_texture_signature(patch):
                continue

            quad = np.array([[x, y], [x2, y], [x2, y2], [x, y2]], dtype=np.float32)
            if blocked_mask is not None and masked_overlap_ratio(quad, blocked_mask) > 0.03:
                continue

            score = 1.8 * trans + 0.01 * std_val + 0.12 * fill
            scored.append((score, quad, bw, bh))

    if not scored:
        return []

    scored.sort(key=lambda t: t[0], reverse=True)
    deduped: List[Tuple[float, np.ndarray, int, int]] = []
    for item in scored:
        b = quad_to_box_int(item[1])
        if any(bbox_iou_xyxy(b, quad_to_box_int(old[1])) > 0.45 for old in deduped):
            continue
        deduped.append(item)
        if len(deduped) >= 32:
            break

    # Uu tien nhom co kich thuoc tuong dong va gan cung x-center.
    clusters: List[List[int]] = []
    used = [False] * len(deduped)
    for i, cur in enumerate(deduped):
        if used[i]:
            continue
        used[i] = True
        member = [i]
        cur_box = quad_to_box_int(cur[1])
        cur_cx = 0.5 * (cur_box[0] + cur_box[2])
        cur_w = cur[2]
        cur_h = cur[3]
        for j, cand in enumerate(deduped):
            if used[j]:
                continue
            b = quad_to_box_int(cand[1])
            cx = 0.5 * (b[0] + b[2])
            if abs(cur_cx - cx) > max(28.0, 0.35 * min(float(cur_w), float(cand[2]))):
                continue
            wr = min(cur_w, cand[2]) / max(1.0, float(max(cur_w, cand[2])))
            hr = min(cur_h, cand[3]) / max(1.0, float(max(cur_h, cand[3])))
            if wr > 0.65 and hr > 0.60:
                used[j] = True
                member.append(j)
        clusters.append(member)

    if not clusters:
        return []
    clusters.sort(key=lambda ids: (len(ids), float(sum(deduped[k][0] for k in ids))), reverse=True)
    best_cluster = clusters[0]
    if len(best_cluster) < 2:
        return []

    selected: List[np.ndarray] = []
    selected_boxes: List[Tuple[int, int, int, int]] = []
    for idx in best_cluster:
        q = deduped[idx][1]
        b = quad_to_box_int(q)
        if any(bbox_iou_xyxy(b, old) > 0.35 for old in selected_boxes):
            continue
        selected.append(q)
        selected_boxes.append(b)
        if len(selected) >= max_candidates:
            break
    return selected


def detect_tiny_qr_row_clusters(
    image: np.ndarray,
    blocked_mask: Optional[np.ndarray] = None,
    max_candidates: int = 8,
) -> List[np.ndarray]:
    """
    Detect QR rat nho (seed 6-34 px) theo row ngang.
    Muc tieu: bat cac QR mo/nho ben duoi QR lon.
    """
    if image is None or image.size == 0:
        return []

    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)

    variants = [
        cv2.adaptiveThreshold(clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2),
        cv2.adaptiveThreshold(clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 2),
        cv2.morphologyEx(
            cv2.adaptiveThreshold(clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2),
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
            iterations=1,
        ),
    ]

    seeds: List[Tuple[float, Tuple[int, int, int, int]]] = []
    for binary in variants:
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for lab in range(1, num_labels):
            area = float(stats[lab, cv2.CC_STAT_AREA])
            if area < 18.0 or area > 420.0:
                continue

            x = int(stats[lab, cv2.CC_STAT_LEFT])
            y = int(stats[lab, cv2.CC_STAT_TOP])
            bw = int(stats[lab, cv2.CC_STAT_WIDTH])
            bh = int(stats[lab, cv2.CC_STAT_HEIGHT])
            if bw < 6 or bh < 6 or bw > 34 or bh > 34:
                continue

            ratio = max(bw, bh) / max(1.0, min(bw, bh))
            if ratio > 2.2:
                continue

            patch = gray[y:y + bh, x:x + bw]
            if patch.size == 0:
                continue
            std_val = float(np.std(patch))
            if std_val < 20.0:
                continue

            trans = _qr_transition_score(patch, target_size=24)
            if trans < 0.14:
                continue
            if not _looks_like_micro_qr_patch(patch):
                continue

            target = 24
            interp = cv2.INTER_NEAREST if min(patch.shape[:2]) < target else cv2.INTER_AREA
            small = cv2.resize(patch, (target, target), interpolation=interp)
            _, bb = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            dark = float(np.mean(bb > 0))
            if dark < 0.15 or dark > 0.80:
                continue

            mixed = 0
            for gy in range(4):
                for gx in range(4):
                    y0 = gy * 6
                    y1 = 24 if gy == 3 else (gy + 1) * 6
                    x0 = gx * 6
                    x1 = 24 if gx == 3 else (gx + 1) * 6
                    ratio_dark = float(np.mean(bb[y0:y1, x0:x1] > 0))
                    if 0.10 <= ratio_dark <= 0.90:
                        mixed += 1
            if mixed < 11:
                continue

            score = 1.8 * trans + 0.01 * std_val - 0.05 * abs(ratio - 1.0)
            seeds.append((score, (x, y, x + bw, y + bh)))

    if not seeds:
        return []

    seeds.sort(key=lambda t: t[0], reverse=True)
    deduped: List[Tuple[float, Tuple[int, int, int, int]]] = []
    for seed in seeds:
        if any(bbox_iou_xyxy(seed[1], old[1]) > 0.50 for old in deduped):
            continue
        deduped.append(seed)

    # Tao nhom theo hang ngang, uu tien nhom co khoang cach x deu.
    groups: List[List[int]] = []
    for i, seed in enumerate(deduped):
        cy_ref = 0.5 * (seed[1][1] + seed[1][3])
        idx = [
            j for j, s in enumerate(deduped)
            if abs(0.5 * (s[1][1] + s[1][3]) - cy_ref) <= 10.0
        ]
        if len(idx) < 2:
            continue

        idx = sorted(idx, key=lambda k: 0.5 * (deduped[k][1][0] + deduped[k][1][2]))
        filtered_idx: List[int] = []
        for k in idx:
            if filtered_idx and bbox_iou_xyxy(deduped[k][1], deduped[filtered_idx[-1]][1]) > 0.45:
                continue
            filtered_idx.append(k)
        if len(filtered_idx) < 2:
            continue

        centers_x = [0.5 * (deduped[k][1][0] + deduped[k][1][2]) for k in filtered_idx]
        centers_y = [0.5 * (deduped[k][1][1] + deduped[k][1][3]) for k in filtered_idx]
        gaps = [centers_x[t + 1] - centers_x[t] for t in range(len(centers_x) - 1)]
        if len(gaps) == 0:
            continue
        gap_med = float(np.median(np.array(gaps, dtype=np.float32)))
        if gap_med < 14.0 or gap_med > 70.0:
            continue
        if len(gaps) >= 2:
            gap_cv = float(np.std(np.array(gaps, dtype=np.float32))) / max(1e-6, float(np.mean(np.array(gaps, dtype=np.float32))))
            if gap_cv > 0.35:
                continue
        if max(centers_y) - min(centers_y) > 10.0:
            continue

        if len(filtered_idx) == 2:
            if min(deduped[filtered_idx[0]][0], deduped[filtered_idx[1]][0]) < 1.05:
                continue

        groups.append(filtered_idx)

    if not groups:
        return []

    # Chon nhom tot nhat: uu tien nhieu phan tu, sau do tong score.
    groups_unique: List[List[int]] = []
    seen = set()
    for g in groups:
        key = tuple(g)
        if key in seen:
            continue
        seen.add(key)
        groups_unique.append(g)
    groups_unique.sort(
        key=lambda g: (len(g), float(sum(deduped[k][0] for k in g))),
        reverse=True,
    )
    best = groups_unique[0]
    if len(best) < 2:
        return []

    quads: List[np.ndarray] = []
    for k in best:
        x1, y1, x2, y2 = deduped[k][1]
        # Nhe tay mo rong de phu toa do GT tiny-QR on dinh hon.
        x1 = max(0, x1 - 1)
        y1 = max(0, y1 - 1)
        x2 = min(w, x2 + 1)
        y2 = min(h, y2 + 2)
        quad = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        if blocked_mask is not None and masked_overlap_ratio(quad, blocked_mask) > 0.03:
            continue
        quads.append(quad)
        if len(quads) >= max_candidates:
            break
    return quads


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
    # FAST-PATH: Anh co rat nhieu QR nho, sat nhau (uu tien recall)
    # -----------------------------------------------------------------------
    dense_quads = detect_dense_small_qr_components(
        img,
        blocked_mask=None,
        min_cluster_count=25,
        max_candidates=220,
    )
    dense_fastpath_ok = (
        len(dense_quads) > 40
        and _has_dense_qr_layout_evidence(dense_quads, img.shape[:2])
    )
    if dense_fastpath_ok:
        dense_quads.sort(key=lambda q: (np.min(q[:, 1]), np.min(q[:, 0])))

        corners_dense: List[List[Point]] = []
        for q in dense_quads:
            r = np.asarray(q, dtype=np.float32).reshape(4, 2)
            x1 = int(np.min(r[:, 0]))
            y1 = int(np.min(r[:, 1]))
            x2 = int(np.max(r[:, 0]))
            y2 = int(np.max(r[:, 1]))
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue
            corners_dense.append([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

        if corners_dense:
            return len(corners_dense), corners_dense

    # -----------------------------------------------------------------------
    # BUOC 1: Thu tim bang finder-pattern
    # -----------------------------------------------------------------------
    prep_cache: Dict[Tuple[str, int], np.ndarray] = {}
    low_light = float(np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))) < 92.0

    def _map_patterns_to_full(
        local_patterns: List[Tuple[Contour, Point, float]],
        scale: float,
        offset_x: int,
        offset_y: int,
    ) -> List[Tuple[Contour, Point, float]]:
        if abs(scale - 1.0) < 1e-6 and offset_x == 0 and offset_y == 0:
            return local_patterns
        mapped: List[Tuple[Contour, Point, float]] = []
        inv_scale = 1.0 / max(1e-6, scale)
        for cnt, center, area in local_patterns:
            cnt_scaled = (cnt.astype(np.float32) * inv_scale).astype(np.int32)
            cnt_scaled[:, 0, 0] += int(offset_x)
            cnt_scaled[:, 0, 1] += int(offset_y)
            cx = int(round(center[0] * inv_scale + offset_x))
            cy = int(round(center[1] * inv_scale + offset_y))
            mapped.append((cnt_scaled, (cx, cy), float(area) * (inv_scale * inv_scale)))
        return mapped

    binary_tasks: List[Tuple[np.ndarray, float, int, int]] = []

    # Multi-scale nhanh: detect tho tren anh downscale 0.5.
    coarse_scale = 0.5 if min(H_img, W_img) >= 420 else 1.0
    if coarse_scale < 1.0:
        coarse_img = cv2.resize(img, None, fx=coarse_scale, fy=coarse_scale, interpolation=cv2.INTER_AREA)
        binary_tasks.append((preprocess_image(coarse_img, variant=0, cache=prep_cache), coarse_scale, 0, 0))
        if low_light:
            binary_tasks.append((preprocess_image(coarse_img, variant=1, cache=prep_cache), coarse_scale, 0, 0))

    # Full-scale pass gon nhe.
    binary_tasks.append((preprocess_image(img, variant=0, cache=prep_cache), 1.0, 0, 0))
    if low_light:
        binary_tasks.append((preprocess_image(img, variant=1, cache=prep_cache), 1.0, 0, 0))

    patterns: List[Tuple[Contour, Point, float]] = []
    max_workers = min(4, max(1, len(binary_tasks)))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_extract_patterns_bidirectional, task[0]) for task in binary_tasks]
        for fut, task in zip(futures, binary_tasks):
            local_patterns = fut.result()
            patterns.extend(_map_patterns_to_full(local_patterns, task[1], task[2], task[3]))

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

    # Neu da co rough tu 0.5x, refine lai tren ROI full-size (khong quet toan anh).
    if coarse_scale < 1.0 and len(unique_patterns) >= 3:
        coarse_rough = build_qr_quads(unique_patterns, img.shape)
        roi_tasks: List[Tuple[np.ndarray, float, int, int]] = []
        seen_roi: List[Tuple[int, int, int, int]] = []
        for rq in coarse_rough[:24]:
            q2 = np.asarray(rq, dtype=np.float32).reshape(4, 2)
            x1 = int(np.min(q2[:, 0]))
            y1 = int(np.min(q2[:, 1]))
            x2 = int(np.max(q2[:, 0]))
            y2 = int(np.max(q2[:, 1]))
            bw = x2 - x1
            bh = y2 - y1
            if bw < 10 or bh < 10:
                continue
            pad = max(8, int(0.25 * max(bw, bh)))
            rx1 = max(0, x1 - pad)
            ry1 = max(0, y1 - pad)
            rx2 = min(W_img, x2 + pad)
            ry2 = min(H_img, y2 + pad)
            roi_box = (rx1, ry1, rx2, ry2)
            if any(bbox_iou_xyxy(roi_box, old) > 0.55 for old in seen_roi):
                continue
            seen_roi.append(roi_box)
            roi_img = img[ry1:ry2, rx1:rx2]
            if roi_img.size == 0:
                continue
            roi_tasks.append((preprocess_image(roi_img, variant=0, cache=prep_cache), 1.0, rx1, ry1))

        if roi_tasks:
            with ThreadPoolExecutor(max_workers=min(4, len(roi_tasks))) as pool:
                futures = [pool.submit(_extract_patterns_bidirectional, task[0]) for task in roi_tasks]
                for fut, task in zip(futures, roi_tasks):
                    local_patterns = fut.result()
                    patterns.extend(_map_patterns_to_full(local_patterns, task[1], task[2], task[3]))

            # Dedup lai sau khi bo sung ROI patterns.
            unique_patterns = []
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
    rough_quads, blocked_regions = suppress_overlapping_quads(
        rough_quads,
        iou_threshold=0.26,
        image_shape=(H_img, W_img),
        return_regions=True,
        touch_margin=0,  
    )
    # (khong fill tu blocked_regions vi co the chua rough quad gia/qua lon)
    blocked_mask = np.zeros((H_img, W_img), dtype=np.uint8)

    # -----------------------------------------------------------------------
    # BUOC 2: Tinh chinh toa do sang axis-aligned bbox
    # -----------------------------------------------------------------------
    refined_quads: List[np.ndarray] = []
    refined_boxes: List[Tuple[int, int, int, int]] = []

    for rq in rough_quads:
        rq2 = np.asarray(rq, dtype=np.float32).reshape(4, 2)
        rq_w = float(np.max(rq2[:, 0]) - np.min(rq2[:, 0]))
        rq_h = float(np.max(rq2[:, 1]) - np.min(rq2[:, 1]))
        rq_area = rq_w * rq_h
        # Neu rough quad chiem >25% dien tich anh -> kha nang false positive
        if rq_area > H_img * W_img * 0.25:
            continue
        # Neu rough quad rong/cao qua nua anh -> skip
        if rq_w > W_img * 0.70 or rq_h > H_img * 0.70:
            continue

        rough_ratio = max(rq_w, rq_h) / max(1.0, min(rq_w, rq_h))
        # Chi refine khi thuc su can: box rough meo nhieu hoac kich thuoc bat thuong.
        need_refine = not (
            rough_ratio <= 1.18
            and 900.0 <= rq_area <= float(H_img * W_img) * 0.06
        )

        if need_refine:
            refined = refine_to_axis_aligned_bbox(img, rq)
        else:
            refined = None

        if refined is None:
            q2 = np.asarray(rq, dtype=np.float32).reshape(4, 2)
            x1, y1 = int(np.min(q2[:, 0])), int(np.min(q2[:, 1]))
            x2, y2 = int(np.max(q2[:, 0])), int(np.max(q2[:, 1]))
            refined = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)

        r = refined.reshape(4, 2)
        x1 = int(np.min(r[:, 0]))
        y1 = int(np.min(r[:, 1]))
        x2 = int(np.max(r[:, 0]))
        y2 = int(np.max(r[:, 1]))

        # Loai bo refined bbox qua lon
        ref_area = (x2 - x1) * (y2 - y1)
        if ref_area > H_img * W_img * 0.25 or (x2 - x1) > W_img * 0.70 or (y2 - y1) > H_img * 0.70:
            continue

        box = (x1, y1, x2, y2)
        if not any(
            bbox_iou_xyxy(box, old) > 0.35
            for old in refined_boxes
        ):
            refined_quads.append(refined)
            refined_boxes.append(box)
            add_quad_to_mask(blocked_mask, refined, dilate_px=0)

    # -----------------------------------------------------------------------
    # BUOC 2.5: Bo sung bang connected-components + finder signature 7x7
    # -----------------------------------------------------------------------
    comp_quads = detect_qr_by_components(img, blocked_mask=blocked_mask, max_candidates=18)
    for cq in comp_quads:
        # Component box thuong la "tight", can no them de gom du QR khi anh xoay nhe.
        rc = expand_axis_aligned_quad(cq, (H_img, W_img), pad_ratio=0.12, min_pad=4, max_pad=44)
        rr = rc.reshape(4, 2)
        cbox = (
            int(np.min(rr[:, 0])),
            int(np.min(rr[:, 1])),
            int(np.max(rr[:, 0])),
            int(np.max(rr[:, 1])),
        )
        if any(
            bbox_iou_xyxy(cbox, old) > 0.32
            for old in refined_boxes
        ):
            continue
        refined_quads.append(rc)
        refined_boxes.append(cbox)
        add_quad_to_mask(blocked_mask, rc, dilate_px=0)

    # -----------------------------------------------------------------------
    # BUOC 3: Neu chua co du QR, thu phat hien bang contour outline
    # -----------------------------------------------------------------------
    # Chi them neu chua phat hien duoc gi, hoac thu them vung co the bo qua
    outline_quads = detect_qr_by_outline(img, max_candidates=6, blocked_mask=blocked_mask)

    for oq in outline_quads:
        r = oq.reshape(4, 2)
        x1 = int(np.min(r[:, 0]))
        y1 = int(np.min(r[:, 1]))
        x2 = int(np.max(r[:, 0]))
        y2 = int(np.max(r[:, 1]))

        box = (x1, y1, x2, y2)
        if any(
            bbox_iou_xyxy(box, old) > 0.30
            for old in refined_boxes
        ):
            continue
        if masked_overlap_ratio(oq, blocked_mask) > 0.10:
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

        if not any(
            bbox_iou_xyxy(rbox, old) > 0.32
            for old in refined_boxes
        ):
            refined_quads.append(refined)
            refined_boxes.append(rbox)
            add_quad_to_mask(blocked_mask, refined, dilate_px=0)

    # -----------------------------------------------------------------------
    # BUOC 4: Fallback cho QR lon ro net (anh co choi, xoay)
    # -----------------------------------------------------------------------
    if not refined_quads:
        bright_quads = detect_qr_by_bright_square(img, blocked_mask=blocked_mask, max_candidates=2)
        for bq in bright_quads:
            r2 = bq.reshape(4, 2)
            box = (
                int(np.min(r2[:, 0])),
                int(np.min(r2[:, 1])),
                int(np.max(r2[:, 0])),
                int(np.max(r2[:, 1])),
            )
            if any(
                bbox_iou_xyxy(box, old) > 0.30 or boxes_overlap_or_touch(box, old, margin=1)
                for old in refined_boxes
            ):
                continue
            refined_quads.append(bq)
            refined_boxes.append(box)
            add_quad_to_mask(blocked_mask, bq, dilate_px=1)

    # -----------------------------------------------------------------------
    # BUOC 4.5: Fallback QR tiny (anh toi, QR rat nho)
    # -----------------------------------------------------------------------
    if not refined_quads and float(np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))) < 90.0:
        tiny_quads = detect_tiny_qr_row_clusters(img, blocked_mask=blocked_mask, max_candidates=2)
        for tq in tiny_quads:
            r2 = tq.reshape(4, 2)
            box = (
                int(np.min(r2[:, 0])),
                int(np.min(r2[:, 1])),
                int(np.max(r2[:, 0])),
                int(np.max(r2[:, 1])),
            )
            if any(
                bbox_iou_xyxy(box, old) > 0.30 or boxes_overlap_or_touch(box, old, margin=1)
                for old in refined_boxes
            ):
                continue
            refined_quads.append(tq)
            refined_boxes.append(box)
            add_quad_to_mask(blocked_mask, tq, dilate_px=1)

    # -----------------------------------------------------------------------
    # BUOC 5: Fallback neu van chua co gi
    # -----------------------------------------------------------------------
    if not refined_quads:
        fallback = fallback_corner_cluster_quads(img, max_candidates=3)
        for fq in fallback:
            if masked_overlap_ratio(fq, blocked_mask) > 0.03:
                continue
            refined = refine_to_axis_aligned_bbox(img, fq)
            if refined is None:
                q2 = np.asarray(fq, dtype=np.float32).reshape(4, 2)
                x1, y1 = int(np.min(q2[:, 0])), int(np.min(q2[:, 1]))
                x2, y2 = int(np.max(q2[:, 0])), int(np.max(q2[:, 1]))
                refined = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
            r2 = refined.reshape(4, 2)
            box = (
                int(np.min(r2[:, 0])),
                int(np.min(r2[:, 1])),
                int(np.max(r2[:, 0])),
                int(np.max(r2[:, 1])),
            )
            if any(
                bbox_iou_xyxy(box, old) > 0.30 or boxes_overlap_or_touch(box, old, margin=1)
                for old in refined_boxes
            ):
                continue
            refined_quads.append(refined)
            refined_boxes.append(box)
            add_quad_to_mask(blocked_mask, refined, dilate_px=1)

    # -----------------------------------------------------------------------
    # BUOC 6: NMS cuoi cung va sap xep ket qua
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

        # Thu hep box qua lon dua tren hinh hoc finder/tiny-qr neu co.
        tightened = _tighten_bbox_with_finder_geometry(img, (x1, y1, x2, y2))
        if tightened is not None:
            tr = tightened.reshape(4, 2)
            x1 = int(np.min(tr[:, 0]))
            y1 = int(np.min(tr[:, 1]))
            x2 = int(np.max(tr[:, 0]))
            y2 = int(np.max(tr[:, 1]))

        box = (x1, y1, x2, y2)
        if any(
            bbox_iou_xyxy(box, old) > 0.32
            for old in final_boxes
        ):
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


def build_pred_quads_from_results_rows(
    result_rows: List[Dict[str, Union[str, int]]]
) -> Dict[str, List[np.ndarray]]:
    grouped: Dict[str, List[np.ndarray]] = {}
    for row in result_rows:
        image_id = str(row.get("image_id", "")).strip()
        if image_id == "":
            continue
        grouped.setdefault(image_id, [])

        qr_index = str(row.get("qr_index", "")).strip()
        if qr_index == "":
            continue

        try:
            coords = [
                float(row["x0"]), float(row["y0"]),
                float(row["x1"]), float(row["y1"]),
                float(row["x2"]), float(row["y2"]),
                float(row["x3"]), float(row["y3"]),
            ]
        except Exception:
            continue

        quad = np.array([
            [coords[0], coords[1]], [coords[2], coords[3]],
            [coords[4], coords[5]], [coords[6], coords[7]],
        ], dtype=np.float32)
        grouped[image_id].append(order_quad_clockwise_start_top_left(quad))
    return grouped


def find_per_image_failures(
    pred_by_image: Dict[str, List[np.ndarray]],
    gt_by_image: Dict[str, List[np.ndarray]],
    iou_threshold: float = 0.5,
    restrict_ids: Optional[set] = None,
) -> Tuple[List[Tuple[str, int, int]], List[Tuple[str, int, int, int]]]:
    """
    Tra ve:
    - missing_count_images: [(image_id, pred_count, gt_count), ...]
    - iou_mismatch_images: [(image_id, matched_iou, pred_count, gt_count), ...]
    """
    all_ids = set(gt_by_image.keys()) | set(pred_by_image.keys())
    if restrict_ids is not None:
        all_ids &= set(restrict_ids)

    missing_count_images: List[Tuple[str, int, int]] = []
    iou_mismatch_images: List[Tuple[str, int, int, int]] = []

    for image_id in sorted(all_ids):
        preds = pred_by_image.get(image_id, [])
        gts = gt_by_image.get(image_id, [])
        pred_count = len(preds)
        gt_count = len(gts)

        if pred_count < gt_count:
            missing_count_images.append((image_id, pred_count, gt_count))

        matched = [False] * gt_count
        matched_iou = 0
        for p in preds:
            best_iou = 0.0
            best_j = -1
            for j, g in enumerate(gts):
                if matched[j]:
                    continue
                iou = quad_iou(p, g)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_j >= 0 and best_iou >= iou_threshold:
                matched[best_j] = True
                matched_iou += 1

        if matched_iou < gt_count:
            iou_mismatch_images.append((image_id, matched_iou, pred_count, gt_count))

    return missing_count_images, iou_mismatch_images


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
        data_name = os.path.basename(data_csv_path).lower()
        if data_name == "public_valid.csv":
            output_filename = "valid_output.csv"
        elif data_name == "public_train.csv":
            output_filename = "train_output.csv"
        else:
            output_filename = "output.csv"
        output_path = os.path.join(script_dir, output_filename)

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
                image_for_decode = read_image_any_path(img_path)
                decoded_contents = decode_qr_contents_for_quads(image_for_decode, corners_list)
                for qr_index, quad in enumerate(corners_list):
                    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = quad
                    try:
                        content = decoded_contents[qr_index] if qr_index < len(decoded_contents) else ""
                    except Exception:
                        content = ""
                    content = ensure_utf8_text(content)
                    results.append({
                        "image_id": image_id, "qr_index": qr_index,
                        "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                        "x2": x2, "y2": y2, "x3": x3, "y3": y3, "content": content,
                    })
            else:
                results.append({
                    "image_id": image_id, "qr_index": "",
                    "x0": "", "y0": "", "x1": "", "y1": "",
                    "x2": "", "y2": "", "x3": "", "y3": "", "content": "",
                })

        with open(output_path, "w", newline="", encoding="utf-8", errors="replace") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=[
                "image_id", "qr_index", "x0", "y0", "x1", "y1",
                "x2", "y2", "x3", "y3", "content"],
                quoting=csv.QUOTE_MINIMAL,
                quotechar='"',
                doublequote=True)
            writer.writeheader()
            writer.writerows(results)

        print(f"\n{'='*70}")
        print(f"DA GHI KET QUA VAO: {to_console_safe(output_path)}")
        print(f"{'='*70}\n")

        compare_gt_path = ""
        gt_path = (args.gt or "").strip()
        if gt_path:
            compare_gt_path = os.path.abspath(gt_path)
            if not os.path.isfile(compare_gt_path):
                print(f"Canh bao: khong tim thay file --gt: {to_console_safe(compare_gt_path)}")
                compare_gt_path = ""
        else:
            data_name = os.path.basename(data_csv_path).lower()
            auto_gt_candidates: List[str] = []
            if data_name.endswith("public_valid.csv"):
                auto_gt_candidates.append(os.path.join(script_dir, "output_valid.csv"))
            elif data_name.endswith("public_train.csv"):
                auto_gt_candidates.append(os.path.join(script_dir, "output_train.csv"))
            else:
                if "valid" in data_name:
                    auto_gt_candidates.append(os.path.join(script_dir, "output_valid.csv"))
                if "train" in data_name:
                    auto_gt_candidates.append(os.path.join(script_dir, "output_train.csv"))
                auto_gt_candidates.extend([
                    os.path.join(script_dir, "output_valid.csv"),
                    os.path.join(script_dir, "output_train.csv"),
                ])
            for cand in auto_gt_candidates:
                if os.path.isfile(cand):
                    compare_gt_path = cand
                    break

        if compare_gt_path:
            print(f"Su dung ground-truth: {to_console_safe(compare_gt_path)}")
            evaluate_csvs(
                output_path,
                compare_gt_path,
                iou_threshold=float(args.iou_threshold),
                filter_ids=data_image_ids if data_image_ids else None,
            )
        else:
            print("Canh bao: khong tim thay ground-truth de danh gia tu dong.")
        
        # ===== DANH GIA TOC DO =================
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
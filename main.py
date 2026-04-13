#!/usr/bin/env python3
"""
Phat hien va dinh vi QR code bang xu ly anh truyen thong.
Khong dung QRCodeDetector/deep learning.
"""

import argparse
import csv
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

Point = Tuple[int, int]
Contour = np.ndarray


def to_console_safe(text: str) -> str:
    """
    Chuyen chuoi ve dang an toan de in tren console khong ho tro Unicode day du.
    """
    return str(text).encode("ascii", errors="backslashreplace").decode("ascii")


def read_image_any_path(image_path: str) -> Optional[np.ndarray]:
    """
    Doc anh on dinh tren Windows, ke ca duong dan co Unicode.
    """
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
    """
    Sap xep 4 diem theo chieu kim dong ho, bat dau tu diem gan goc tren-trai nhat.
    """
    q = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    center = np.mean(q, axis=0)
    ang = np.arctan2(q[:, 1] - center[1], q[:, 0] - center[0])
    q = q[np.argsort(ang)]
    start = int(np.argmin(q[:, 0] + q[:, 1]))
    return np.roll(q, -start, axis=0)


def bbox_iou_xyxy(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """
    IoU cho 2 bounding-box dang (x1, y1, x2, y2).
    """
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


def fallback_corner_cluster_quads(image: np.ndarray, max_candidates: int = 1) -> List[np.ndarray]:
    """
    Fallback khi khong tao duoc quad tu finder-pattern.
    Dung mat do corner de tim vung co cau truc QR.
    """
    if image is None or image.size == 0:
        return []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=1400,
        qualityLevel=0.01,
        minDistance=3,
        blockSize=3,
        useHarrisDetector=False,
    )
    if corners is None or len(corners) < 28:
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

    for flat_idx in ranked[: grid_size * 3]:
        score = float(smooth.reshape(-1)[flat_idx])
        if score < 2.0:
            break

        gy, gx = divmod(int(flat_idx), grid_size)
        radius = 3
        x0 = max(0, int((gx - radius) * cell_w))
        y0 = max(0, int((gy - radius) * cell_h))
        x1 = min(w, int((gx + radius + 1) * cell_w))
        y1 = min(h, int((gy + radius + 1) * cell_h))
        if x1 - x0 < 18 or y1 - y0 < 18:
            continue

        mask = (pts[:, 0] >= x0) & (pts[:, 0] < x1) & (pts[:, 1] >= y0) & (pts[:, 1] < y1)
        local_pts = pts[mask]
        if len(local_pts) < 28:
            continue

        qx1 = float(np.percentile(local_pts[:, 0], 2))
        qx2 = float(np.percentile(local_pts[:, 0], 98))
        qy1 = float(np.percentile(local_pts[:, 1], 2))
        qy2 = float(np.percentile(local_pts[:, 1], 98))
        qbw = max(1.0, qx2 - qx1)
        qbh = max(1.0, qy2 - qy1)

        # Nhe x va no manh o phia duoi de bat duoc QR bi perspective.
        bx1 = max(0, int(np.floor(qx1 - 0.12 * qbw - 1)))
        bx2 = min(w, int(np.ceil(qx2 + 0.12 * qbw + 1)))
        by1 = max(0, int(np.floor(qy1 - 0.03 * qbh - 1)))
        by2 = min(h, int(np.ceil(qy2 + 0.16 * qbh + 1)))
        bw, bh = bx2 - bx1, by2 - by1
        if bw < 18 or bh < 18:
            continue

        ratio = max(bw, bh) / max(1.0, min(bw, bh))
        area = float(bw * bh)
        if ratio > 3.2 or area < 700 or area > 0.30 * h * w:
            continue

        local = local_pts - np.array([bx1, by1], dtype=np.int32)
        occ = np.zeros((3, 3), dtype=np.uint8)
        for px, py in local:
            cx = min(2, max(0, int((px * 3) / max(1, bw))))
            cy = min(2, max(0, int((py * 3) / max(1, bh))))
            occ[cy, cx] = 1
        if int(np.sum(occ)) < 5:
            continue

        patch = gray[by1:by2, bx1:bx2]
        if patch.size == 0:
            continue
        patch = cv2.resize(patch, (72, 72), interpolation=cv2.INTER_CUBIC)
        patch_bin = cv2.adaptiveThreshold(
            patch,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            21,
            3,
        )
        row_trans = np.mean(np.sum(patch_bin[:, 1:] != patch_bin[:, :-1], axis=1) / 71.0)
        col_trans = np.mean(np.sum(patch_bin[1:, :] != patch_bin[:-1, :], axis=0) / 71.0)
        transition_score = 0.5 * (row_trans + col_trans)
        if transition_score < 0.14:
            continue

        candidate_box = (bx1, by1, bx2, by2)
        if any(bbox_iou_xyxy(candidate_box, old) > 0.60 for old in selected_boxes):
            continue

        quad = np.array(
            [
                [bx1, by1],
                [bx2, by1],
                [bx2, by2],
                [bx1, by2],
            ],
            dtype=np.float32,
        )
        selected_quads.append(order_quad_clockwise_start_top_left(quad))
        selected_boxes.append(candidate_box)

        if len(selected_quads) >= max_candidates:
            break

    return selected_quads


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Tien xu ly thanh anh nhi phan de tim finder-pattern.
    """
    if image is None or image.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    h, w = gray.shape
    sigma = max(3.0, min(h, w) / 32.0)
    gray_f = gray.astype(np.float32)
    background = cv2.GaussianBlur(gray_f, (0, 0), sigmaX=sigma, sigmaY=sigma)
    normalized = cv2.divide(gray_f, background + 1.0, scale=255.0)
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=3.2, tileGridSize=(8 if min(h, w) >= 480 else 6,) * 2)
    enhanced = clahe.apply(normalized)
    denoised = cv2.medianBlur(enhanced, 3)
    sharpened = cv2.addWeighted(
        denoised,
        1.35,
        cv2.GaussianBlur(denoised, (0, 0), 1.2),
        -0.35,
        0,
    )

    block = max(15, min(51, ((min(h, w) // 24) | 1)))
    binary_g = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, 3
    )
    binary_m = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block, 5
    )
    binary_g_sharp = cv2.adaptiveThreshold(
        sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, 2
    )
    _, binary_o = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, binary_o_sharp = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary = cv2.bitwise_or(cv2.bitwise_or(binary_g, binary_m), cv2.bitwise_or(binary_o, binary_o_sharp))
    binary = cv2.bitwise_or(binary, binary_g_sharp)

    k = max(3, min(7, ((min(h, w) // 220) | 1)))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    min_area = max(35, int(h * w * 0.00004))
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    for lab in range(1, num_labels):
        if stats[lab, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == lab] = 255
    return cleaned


def find_finder_patterns(binary: np.ndarray) -> List[Tuple[Contour, Point, float]]:
    """
    Tim cac finder-pattern theo cau truc long nhau (outer -> middle -> inner).
    """
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
                    if area < 35 or peri < 1e-6:
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
                    if wh_ratio > 2.7 or fill < 0.52:
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
                    if area_o > area_m > area_i and area_i > 20:
                        ratio_om = area_o / max(1e-6, area_m)
                        ratio_mi = area_m / max(1e-6, area_i)
                        if 1.2 < ratio_om < 6.0 and 1.2 < ratio_mi < 8.0:
                            co = np.array(centers[0], dtype=np.float32)
                            cm = np.array(centers[1], dtype=np.float32)
                            ci = np.array(centers[2], dtype=np.float32)
                            _, _, bw, bh = cv2.boundingRect(outer)
                            threshold = max(bw, bh) * 0.32
                            if max(
                                np.linalg.norm(co - cm),
                                np.linalg.norm(co - ci),
                                np.linalg.norm(cm - ci),
                            ) <= threshold:
                                patterns.append((outer, centers[0], area_o))

                cur_grandchild = hierarchy[cur_grandchild][0]
            cur_child = hierarchy[cur_child][0]

    # Fallback: neu nested-pattern qua it, lay candidate contour co child
    # de bat truong hop QR ro nhung hierarchy bi vo do blur/threshold.
    if len(patterns) < 3:
        img_h, img_w = binary.shape[:2]
        img_area = float(img_h * img_w)
        for i, cnt in enumerate(contours):
            child_idx = hierarchy[i][2]
            if child_idx == -1:
                continue

            area = abs(float(cv2.contourArea(cnt)))
            if area < max(400.0, img_area * 0.002) or area > img_area * 0.28:
                continue

            child_area = abs(float(cv2.contourArea(contours[child_idx])))
            child_ratio = child_area / max(1e-6, area)
            if child_ratio < 0.02 or child_ratio > 0.90:
                continue

            peri = cv2.arcLength(cnt, True)
            if peri < 1e-6:
                continue
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) < 4 or len(approx) > 10:
                continue

            rect = cv2.minAreaRect(cnt)
            (_, _), (rw, rh), _ = rect
            if rw < 5 or rh < 5:
                continue
            wh_ratio = max(rw, rh) / max(1e-6, min(rw, rh))
            fill = area / max(1e-6, rw * rh)
            if wh_ratio > 2.3 or fill < 0.38:
                continue

            m = cv2.moments(cnt)
            if abs(m["m00"]) > 1e-6:
                cx = int(round(m["m10"] / m["m00"]))
                cy = int(round(m["m01"] / m["m00"]))
            else:
                x, y, w, h = cv2.boundingRect(cnt)
                cx, cy = x + w // 2, y + h // 2
            patterns.append((cnt, (cx, cy), area))

    # Remove near-duplicates
    unique: List[Tuple[Contour, Point, float]] = []
    for cand in patterns:
        _, c, area = cand
        dup = False
        for _, old_c, old_area in unique:
            dist = np.linalg.norm(np.array(c, dtype=np.float32) - np.array(old_c, dtype=np.float32))
            area_ratio = min(area, old_area) / max(1e-6, max(area, old_area))
            if dist < 10 and area_ratio > 0.80:
                dup = True
                break
        if not dup:
            unique.append(cand)
    return unique


def build_qr_quads(patterns: List[Tuple[Contour, Point, float]], image_shape: Tuple[int, ...]) -> List[np.ndarray]:
    """
    Ghep bo 3 finder-pattern thanh 4 goc QR (TL, TR, BR, BL).
    """
    if len(patterns) < 3:
        return []

    centers = [np.array(p[1], dtype=np.float32) for p in patterns]
    contours = [p[0] for p in patterns]
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
                if cos_val > 0.40 or len_ratio < 0.45:
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

                    # Chon goc theo huong tu tam finder-pattern
                    corner_dirs = (-ex - ey, ex - ey, -ex + ey)
                    chosen = []
                    for box, d in zip(contour_boxes, corner_dirs):
                        c = np.mean(box, axis=0)
                        dn = d / max(1e-6, np.linalg.norm(d))
                        scores = np.dot(box - c, dn)
                        chosen.append(box[int(np.argmax(scores))])

                    tl_pt, tr_pt, bl_pt = chosen
                    br_pt = tr_pt + bl_pt - tl_pt
                    quad = np.array([tl_pt, tr_pt, br_pt, bl_pt], dtype=np.float32)

                    area = abs(float(cv2.contourArea(quad)))
                    if area < 900:
                        continue

                    # Keep only quads mostly inside image
                    inside = np.sum(
                        (quad[:, 0] >= -0.05 * w)
                        & (quad[:, 0] <= 1.05 * w)
                        & (quad[:, 1] >= -0.05 * h)
                        & (quad[:, 1] <= 1.05 * h)
                    )
                    if inside < 4:
                        continue

                    # Order clockwise, starting near top-left
                    center = np.mean(quad, axis=0)
                    ang = np.arctan2(quad[:, 1] - center[1], quad[:, 0] - center[0])
                    quad = quad[np.argsort(ang)]
                    start = int(np.argmin(quad[:, 0] + quad[:, 1]))
                    quad = np.roll(quad, -start, axis=0)
                    quads.append(quad)

    return quads


def suppress_overlapping_quads(quads: List[np.ndarray], iou_threshold: float = 0.30) -> List[np.ndarray]:
    """
    Loai bo vung QR chong cheo: neu QR moi de len vung QR da tim thay thi bo qua.
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
            if iou >= iou_threshold or overlap_min >= 0.60:
                keep = False
                break
        if keep:
            selected.append(q)
            selected_boxes.append(b)
    return selected


def process_image(image_path: str) -> Tuple[int, List[List[Point]]]:
    """
    Xu ly mot anh va tra ve (so_luong_qr, danh_sach_4_goc).
    """
    img = read_image_any_path(image_path)
    if img is None:
        print(f"Khong the doc anh: {to_console_safe(image_path)}")
        return 0, []

    binaries: List[np.ndarray] = []
    binaries.append(preprocess_image(img))

    # Bien the tang net cho anh mo.
    blur = cv2.GaussianBlur(img, (0, 0), 1.3)
    sharpened_bgr = cv2.addWeighted(img, 1.45, blur, -0.45, 0)
    binaries.append(preprocess_image(sharpened_bgr))

    # Bien the to hon de gom contour tot hon khi QR nho/meo.
    h, w = img.shape[:2]
    scale = 1.35 if min(h, w) < 700 else 1.20
    up = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    binaries.append(preprocess_image(up))

    patterns: List[Tuple[Contour, Point, float]] = []
    for b_idx, binary in enumerate(binaries):
        local_patterns = find_finder_patterns(binary)
        inv_patterns = find_finder_patterns(cv2.bitwise_not(binary))
        local_patterns.extend(inv_patterns)

        # Scale nguoc pattern tu anh upsample ve anh goc.
        if b_idx == 2:
            mapped_patterns: List[Tuple[Contour, Point, float]] = []
            for cnt, center, area in local_patterns:
                cnt_scaled = (cnt.astype(np.float32) / scale).astype(np.int32)
                cx = int(round(center[0] / scale))
                cy = int(round(center[1] / scale))
                mapped_patterns.append((cnt_scaled, (cx, cy), float(area) / (scale * scale)))
            local_patterns = mapped_patterns
        patterns.extend(local_patterns)

    # Khu trung lap patterns sau khi gop tu nhieu bien the.
    unique_patterns: List[Tuple[Contour, Point, float]] = []
    for cand in patterns:
        _, c, area = cand
        duplicated = False
        for _, old_c, old_area in unique_patterns:
            dist = np.linalg.norm(np.array(c, dtype=np.float32) - np.array(old_c, dtype=np.float32))
            ar = min(area, old_area) / max(1e-6, max(area, old_area))
            if dist < 12 and ar > 0.75:
                duplicated = True
                break
        if not duplicated:
            unique_patterns.append(cand)

    quads = build_qr_quads(unique_patterns, img.shape)
    quads = suppress_overlapping_quads(quads, iou_threshold=0.30)
    if not quads:
        quads = fallback_corner_cluster_quads(img, max_candidates=1)

    corners = []
    for q in quads:
        q = order_quad_clockwise_start_top_left(q)
        corners.append([(int(round(x)), int(round(y))) for x, y in q])
    corners.sort(key=lambda quad: (min(p[1] for p in quad), min(p[0] for p in quad)))
    return len(corners), corners


def polygon_signed_area(poly: np.ndarray) -> float:
    """
    Dien tich co huong cua da giac.
    """
    if poly is None or len(poly) < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def polygon_area(poly: np.ndarray) -> float:
    """
    Dien tich tuyet doi cua da giac.
    """
    return abs(polygon_signed_area(poly))


def ensure_ccw(poly: np.ndarray) -> np.ndarray:
    """
    Dam bao da giac co thu tu nguoc chieu kim dong ho.
    """
    p = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    if len(p) >= 3 and polygon_signed_area(p) < 0:
        p = p[::-1]
    return p


def line_intersection(p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Giao diem 2 duong thang p1-p2 va q1-q2.
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
    Cat da giac loi subject boi da giac loi clipper.
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
    IoU giua 2 tu giac (dien tich that, khong dung bounding-rect).
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
    """
    Doc file CSV output/ground-truth va nhom tu giac theo image_id.
    """
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

            quad = np.array(
                [
                    [coords[0], coords[1]],
                    [coords[2], coords[3]],
                    [coords[4], coords[5]],
                    [coords[6], coords[7]],
                ],
                dtype=np.float32,
            )
            quad = order_quad_clockwise_start_top_left(quad)
            grouped[image_id].append(quad)
    return grouped


def greedy_iou_match(
    pred_by_image: Dict[str, List[np.ndarray]],
    gt_by_image: Dict[str, List[np.ndarray]],
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Greedy matching theo thu tu du doan tren tung image_id.
    """
    tp = 0
    fp = 0
    fn = 0

    all_ids = set(pred_by_image.keys()) | set(gt_by_image.keys())
    for image_id in all_ids:
        preds = pred_by_image.get(image_id, [])
        gts = gt_by_image.get(image_id, [])
        matched = [False] * len(gts)

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
                tp += 1
            else:
                fp += 1

        fn += sum(1 for used in matched if not used)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phat hien QR code trong anh (khong dung detector co san)")
    parser.add_argument("--data", required=True, help="Duong dan file CSV co cot image_path")
    parser.add_argument("--gt", default="", help="File ground-truth CSV de cham (tuy chon)")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="Nguong IoU cho matching")
    args = parser.parse_args()

    try:
        data_csv_path = os.path.abspath(args.data)
        csv_dir = os.path.dirname(data_csv_path)
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output.csv")

        with open(data_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            required_columns = {"image_id", "image_path"}
            if not required_columns.issubset(fieldnames):
                print("File CSV phai co day du 2 cot: 'image_id' va 'image_path'")
                return
            rows = list(reader)
        data_image_ids = {(row.get("image_id") or "").strip() for row in rows if (row.get("image_id") or "").strip() != ""}

        results = []
        for idx, row in enumerate(rows, start=1):
            image_id = (row.get("image_id") or "").strip()
            image_path_raw = (row.get("image_path") or "").strip()
            img_path = image_path_raw
            if not os.path.isabs(img_path):
                img_path = os.path.normpath(os.path.join(csv_dir, img_path))
            print(f"Xu ly anh [{idx}/{len(rows)}]: {to_console_safe(image_id)}")
            num_qr, corners_list = process_image(img_path)

            if num_qr > 0:
                for qr_index, quad in enumerate(corners_list):
                    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = quad
                    results.append(
                        {
                            "image_id": image_id,
                            "qr_index": qr_index,
                            "x0": x0,
                            "y0": y0,
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "x3": x3,
                            "y3": y3,
                            "content": "",
                        }
                    )
            else:
                results.append(
                    {
                        "image_id": image_id,
                        "qr_index": "",
                        "x0": "",
                        "y0": "",
                        "x1": "",
                        "y1": "",
                        "x2": "",
                        "y2": "",
                        "x3": "",
                        "y3": "",
                        "content": "",
                    }
                )

        with open(output_path, "w", newline="", encoding="utf-8") as f_out:
            writer = csv.DictWriter(
                f_out,
                fieldnames=["image_id", "qr_index", "x0", "y0", "x1", "y1", "x2", "y2", "x3", "y3", "content"],
            )
            writer.writeheader()
            writer.writerows(results)

        print("Da ghi ket qua vao: output.csv")

        gt_path = (args.gt or "").strip()
        if gt_path:
            gt_path = os.path.abspath(gt_path)
            pred_quads = load_quads_by_image_id(output_path)
            gt_quads = load_quads_by_image_id(gt_path)
            if data_image_ids:
                pred_quads = {k: v for k, v in pred_quads.items() if k in data_image_ids}
                gt_quads = {k: v for k, v in gt_quads.items() if k in data_image_ids}
            stats = greedy_iou_match(pred_quads, gt_quads, iou_threshold=float(args.iou_threshold))
            print("=== KET QUA CHAM GREEDY IoU ===")
            print(f"IoU threshold: {float(args.iou_threshold):.3f}")
            print(f"TP: {int(stats['tp'])} | FP: {int(stats['fp'])} | FN: {int(stats['fn'])}")
            print(f"Precision: {stats['precision']:.6f}")
            print(f"Recall:    {stats['recall']:.6f}")
            print(f"F1 Score:  {stats['f1']:.6f}")
    except Exception as e:
        print(f"Loi xu ly du lieu: {e}")


if __name__ == "__main__":
    main()

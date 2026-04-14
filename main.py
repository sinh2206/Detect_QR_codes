#!/usr/bin/env python3
"""
Phat hien va dinh vi QR code bang xu ly anh truyen thong.
Khong dung QRCodeDetector/deep learning.
CAI TIEN: Tang cuong phat hien QR bi meo, mo, bien dang
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


def fallback_corner_cluster_quads(image: np.ndarray, max_candidates: int = 2) -> List[np.ndarray]:
    """
    Fallback khi khong tao duoc quad tu finder-pattern.
    Dung mat do corner de tim vung co cau truc QR.
    CAI TIEN: Tang max_candidates, giam nguong de bat nhieu QR hon
    """
    if image is None or image.size == 0:
        return []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=1600,  # Tang len de bat QR nho hon
        qualityLevel=0.008,  # Giam nguong de bat nhieu corner hon
        minDistance=2,  # Giam khoang cach toi thieu
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

    for flat_idx in ranked[: grid_size * 4]:  # Tang len de xem xet nhieu vung hon
        score = float(smooth.reshape(-1)[flat_idx])
        if score < 1.5:  # Giam nguong score
            break

        gy, gx = divmod(int(flat_idx), grid_size)
        radius = 3
        x0 = max(0, int((gx - radius) * cell_w))
        y0 = max(0, int((gy - radius) * cell_h))
        x1 = min(w, int((gx + radius + 1) * cell_w))
        y1 = min(h, int((gy + radius + 1) * cell_h))
        if x1 - x0 < 16 or y1 - y0 < 16:  # Giam size toi thieu
            continue

        mask = (pts[:, 0] >= x0) & (pts[:, 0] < x1) & (pts[:, 1] >= y0) & (pts[:, 1] < y1)
        local_pts = pts[mask]
        if len(local_pts) < 24:  # Giam yeu cau so corner
            continue

        qx1 = float(np.percentile(local_pts[:, 0], 2))
        qx2 = float(np.percentile(local_pts[:, 0], 98))
        qy1 = float(np.percentile(local_pts[:, 1], 2))
        qy2 = float(np.percentile(local_pts[:, 1], 98))
        qbw = max(1.0, qx2 - qx1)
        qbh = max(1.0, qy2 - qy1)

        # No manh hon de bat QR bi bien dang
        bx1 = max(0, int(np.floor(qx1 - 0.15 * qbw - 2)))
        bx2 = min(w, int(np.ceil(qx2 + 0.15 * qbw + 2)))
        by1 = max(0, int(np.floor(qy1 - 0.05 * qbh - 2)))
        by2 = min(h, int(np.ceil(qy2 + 0.18 * qbh + 2)))
        bw, bh = bx2 - bx1, by2 - by1
        if bw < 16 or bh < 16:
            continue

        ratio = max(bw, bh) / max(1.0, min(bw, bh))
        area = float(bw * bh)
        if ratio > 3.5 or area < 600 or area > 0.32 * h * w:  # Nhe hon mot chut
            continue

        local = local_pts - np.array([bx1, by1], dtype=np.int32)
        occ = np.zeros((3, 3), dtype=np.uint8)
        for px, py in local:
            cx = min(2, max(0, int((px * 3) / max(1, bw))))
            cy = min(2, max(0, int((py * 3) / max(1, bh))))
            occ[cy, cx] = 1
        if int(np.sum(occ)) < 4:  # Giam yeu cau
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
        if transition_score < 0.12:  # Giam nguong transition
            continue

        candidate_box = (bx1, by1, bx2, by2)
        if any(bbox_iou_xyxy(candidate_box, old) > 0.55 for old in selected_boxes):  # Giam nguong IoU
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


def preprocess_image(image: np.ndarray, variant: int = 0) -> np.ndarray:
    """
    Tien xu ly thanh anh nhi phan de tim finder-pattern.
    variant: bien the tien xu ly (0: default, 1: aggressive cho anh mo, 2: soft)
    """
    if image is None or image.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    h, w = gray.shape
    
    # CAI TIEN: Xu ly dac biet cho anh mo
    if variant == 1:  # Aggressive - tot cho anh mo
        gray = cv2.equalizeHist(gray)
        gray = cv2.GaussianBlur(gray, (0, 0), 0.8)
        sharpening_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
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
        denoised,
        sharp_weight,
        cv2.GaussianBlur(denoised, (0, 0), 1.2),
        -(sharp_weight - 1.0),
        0,
    )

    block = max(15, min(51, ((min(h, w) // 24) | 1)))
    if variant == 1:
        block = max(11, block - 4)  # Block size nho hon cho anh mo
    
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
    
    # CAI TIEN: Them binary tu enhanced truc tiep (tot cho QR meo)
    if variant == 1:
        _, binary_enh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        binary = cv2.bitwise_or(binary, binary_enh)

    k = max(3, min(7, ((min(h, w) // 220) | 1)))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    min_area = max(30, int(h * w * 0.00003))  # Giam nguong area mot chut
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
                    if area < 30 or peri < 1e-6:  # Giam area toi thieu
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
                    if wh_ratio > 2.8 or fill < 0.50:  # Nhe hon mot chut
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
                    if area_o > area_m > area_i and area_i > 18:  # Giam nguong
                        ratio_om = area_o / max(1e-6, area_m)
                        ratio_mi = area_m / max(1e-6, area_i)
                        if 1.15 < ratio_om < 6.5 and 1.15 < ratio_mi < 8.5:  # Nhe hon
                            co = np.array(centers[0], dtype=np.float32)
                            cm = np.array(centers[1], dtype=np.float32)
                            ci = np.array(centers[2], dtype=np.float32)
                            _, _, bw, bh = cv2.boundingRect(outer)
                            threshold = max(bw, bh) * 0.35  # Tang muc tolerance
                            if max(
                                np.linalg.norm(co - cm),
                                np.linalg.norm(co - ci),
                                np.linalg.norm(cm - ci),
                            ) <= threshold:
                                patterns.append((outer, centers[0], area_o))

                cur_grandchild = hierarchy[cur_grandchild][0]
            cur_child = hierarchy[cur_child][0]

    # Fallback: neu nested-pattern qua it
    if len(patterns) < 3:
        img_h, img_w = binary.shape[:2]
        img_area = float(img_h * img_w)
        for i, cnt in enumerate(contours):
            child_idx = hierarchy[i][2]
            if child_idx == -1:
                continue

            area = abs(float(cv2.contourArea(cnt)))
            if area < max(350.0, img_area * 0.0018) or area > img_area * 0.30:
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

    # Remove near-duplicates
    unique: List[Tuple[Contour, Point, float]] = []
    for cand in patterns:
        _, c, area = cand
        dup = False
        for _, old_c, old_area in unique:
            dist = np.linalg.norm(np.array(c, dtype=np.float32) - np.array(old_c, dtype=np.float32))
            area_ratio = min(area, old_area) / max(1e-6, max(area, old_area))
            if dist < 10 and area_ratio > 0.78:  # Giam nguong
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
                if cos_val > 0.45 or len_ratio < 0.42:  # Nhe hon
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
                    br_pt = tr_pt + bl_pt - tl_pt
                    quad = np.array([tl_pt, tr_pt, br_pt, bl_pt], dtype=np.float32)

                    area = abs(float(cv2.contourArea(quad)))
                    if area < 800:  # Giam nguong area
                        continue

                    inside = np.sum(
                        (quad[:, 0] >= -0.05 * w)
                        & (quad[:, 0] <= 1.05 * w)
                        & (quad[:, 1] >= -0.05 * h)
                        & (quad[:, 1] <= 1.05 * h)
                    )
                    if inside < 4:
                        continue

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
            if iou >= iou_threshold or overlap_min >= 0.58:  # Giam nguong mot chut
                keep = False
                break
        if keep:
            selected.append(q)
            selected_boxes.append(b)
    return selected


def process_image(image_path: str) -> Tuple[int, List[List[Point]]]:
    """
    Xu ly mot anh va tra ve (so_luong_qr, danh_sach_4_goc).
    CAI TIEN: Them nhieu bien the preprocessing va scale de bat QR kho hon
    """
    img = read_image_any_path(image_path)
    if img is None:
        print(f"Khong the doc anh: {to_console_safe(image_path)}")
        return 0, []

    binaries: List[np.ndarray] = []
    
    # Bien the preprocessing
    binaries.append(preprocess_image(img, variant=0))  # Default
    binaries.append(preprocess_image(img, variant=1))  # Aggressive cho anh mo
    binaries.append(preprocess_image(img, variant=2))  # Soft

    # Bien the tang net
    blur = cv2.GaussianBlur(img, (0, 0), 1.3)
    sharpened_bgr = cv2.addWeighted(img, 1.45, blur, -0.45, 0)
    binaries.append(preprocess_image(sharpened_bgr, variant=0))
    binaries.append(preprocess_image(sharpened_bgr, variant=1))

    # Bien the scale - nhieu scale hon de bat QR o kich thuoc khac nhau
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

        # Scale nguoc pattern tu anh upsample
        if b_idx >= 5:  # Cac binary tu upsample
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

    # Khu trung lap
    unique_patterns: List[Tuple[Contour, Point, float]] = []
    for cand in patterns:
        _, c, area = cand
        duplicated = False
        for _, old_c, old_area in unique_patterns:
            dist = np.linalg.norm(np.array(c, dtype=np.float32) - np.array(old_c, dtype=np.float32))
            ar = min(area, old_area) / max(1e-6, max(area, old_area))
            if dist < 11 and ar > 0.72:  # Giam nguong
                duplicated = True
                break
        if not duplicated:
            unique_patterns.append(cand)

    quads = build_qr_quads(unique_patterns, img.shape)
    quads = suppress_overlapping_quads(quads, iou_threshold=0.28)  # Giam nguong
    
    if not quads:
        # Fallback voi corner clustering - thu manh hon
        quads = fallback_corner_cluster_quads(img, max_candidates=3)

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
) -> Dict[str, Dict[str, float]]:
    """
    Greedy matching theo thu tu du doan tren tung image_id.
    Tach thanh 2 phan:
    - count: So sanh so luong (qr_index)
    - iou: So sanh toa do duong bao
    """
    tp_iou = 0
    fp_iou = 0
    fn_iou = 0

    tp_count = 0
    fp_count = 0
    fn_count = 0

    all_ids = set(pred_by_image.keys()) | set(gt_by_image.keys())
    for image_id in all_ids:
        preds = pred_by_image.get(image_id, [])
        gts = gt_by_image.get(image_id, [])

        # --- Danh gia theo so luong QR (qr_index) ---
        p_len = len(preds)
        g_len = len(gts)
        tp_count += min(p_len, g_len)
        fp_count += max(0, p_len - g_len)
        fn_count += max(0, g_len - p_len)

        # --- Danh gia theo IoU ---
        matched = [False] * g_len

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
                tp_iou += 1
            else:
                fp_iou += 1

        fn_iou += sum(1 for used in matched if not used)

    # Tinh toan metrics cho Count
    precision_count = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
    recall_count = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0.0
    f1_count = (2.0 * precision_count * recall_count / (precision_count + recall_count)) if (precision_count + recall_count) > 0 else 0.0

    # Tinh toan metrics cho IoU
    precision_iou = tp_iou / (tp_iou + fp_iou) if (tp_iou + fp_iou) > 0 else 0.0
    recall_iou = tp_iou / (tp_iou + fn_iou) if (tp_iou + fn_iou) > 0 else 0.0
    f1_iou = (2.0 * precision_iou * recall_iou / (precision_iou + recall_iou)) if (precision_iou + recall_iou) > 0 else 0.0

    return {
        "count": {
            "tp": float(tp_count),
            "fp": float(fp_count),
            "fn": float(fn_count),
            "precision": precision_count,
            "recall": recall_count,
            "f1": f1_count,
        },
        "iou": {
            "tp": float(tp_iou),
            "fp": float(fp_iou),
            "fn": float(fn_iou),
            "precision": precision_iou,
            "recall": recall_iou,
            "f1": f1_iou,
        }
    }


def evaluate_csvs(
    pred_csv: str,
    gt_csv: str,
    iou_threshold: float = 0.5,
    filter_ids: Optional[set] = None,
) -> None:
    """
    So sanh pred_csv voi gt_csv va in ket qua F1, Precision, Recall ra console.
    """
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

    # Phan 1: Theo so luong QR (qr_index)
    sc = stats["count"]
    print(f"{'='*70}")
    print(f"PHAN 1: SO SANH THEO SO LUONG QR (qr_index)")
    print(f"{'='*70}")
    print(f"  True  Positives (TP): {int(sc['tp']):>6}")
    print(f"  False Positives (FP): {int(sc['fp']):>6}")
    print(f"  False Negatives (FN): {int(sc['fn']):>6}")
    print(f"{'-'*70}")
    print(f"  Precision : {sc['precision']:.6f}  ({sc['precision']*100:.2f}%)")
    print(f"  Recall    : {sc['recall']:.6f}  ({sc['recall']*100:.2f}%)")
    print(f"{'='*70}")
    print(f"  F1 SCORE  : {sc['f1']:.6f}  ({sc['f1']*100:.2f}%)")
    print(f"{'='*70}\n")

    # Phan 2: Theo IoU (x0, y0, ..., y3)
    si = stats["iou"]
    print(f"{'='*70}")
    print(f"PHAN 2: SO SANH THEO TOA DO DUONG BAO (IoU >= {iou_threshold:.2f})")
    print(f"{'='*70}")
    print(f"  True  Positives (TP): {int(si['tp']):>6}")
    print(f"  False Positives (FP): {int(si['fp']):>6}")
    print(f"  False Negatives (FN): {int(si['fn']):>6}")
    print(f"{'-'*70}")
    print(f"  Precision : {si['precision']:.6f}  ({si['precision']*100:.2f}%)")
    print(f"  Recall    : {si['recall']:.6f}  ({si['recall']*100:.2f}%)")
    print(f"{'='*70}")
    print(f"  F1 SCORE  : {si['f1']:.6f}  ({si['f1']*100:.2f}%)")
    print(f"{'='*70}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phat hien QR code trong anh (khong dung detector co san)")
    parser.add_argument("--data", default="", help="Duong dan file CSV co cot image_path")
    parser.add_argument("--gt", default="", help="File ground-truth CSV de cham (tuy chon)")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="Nguong IoU cho matching")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Chi danh gia output.csv voi output_valid.csv, khong chay detection",
    )
    parser.add_argument(
        "--pred",
        default="",
        help="File prediction CSV khi dung --eval-only (mac dinh: output.csv ben canh script)",
    )
    parser.add_argument(
        "--valid",
        default="",
        help="File ground-truth CSV khi dung --eval-only (mac dinh: output_valid.csv ben canh script)",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # ------------------------------------------------------------------ #
    # Mode: chi danh gia 2 file CSV co san                                #
    # ------------------------------------------------------------------ #
    if args.eval_only:
        pred_path = os.path.abspath(args.pred) if args.pred else os.path.join(script_dir, "output.csv")
        valid_path = os.path.abspath(args.valid) if args.valid else os.path.join(script_dir, "output_valid.csv")
        if not os.path.isfile(pred_path):
            print(f"Loi: khong tim thay file prediction: {to_console_safe(pred_path)}")
            return
        if not os.path.isfile(valid_path):
            print(f"Loi: khong tim thay file ground-truth: {to_console_safe(valid_path)}")
            return
        try:
            evaluate_csvs(pred_path, valid_path, iou_threshold=float(args.iou_threshold))
        except Exception as e:
            print(f"\nLOI KHI DANH GIA: {e}")
            import traceback
            traceback.print_exc()
        return

    if not args.data:
        parser.error("Can --data <csv_path> hoac --eval-only")

    try:  # noqa: E722
        data_csv_path = os.path.abspath(args.data)
        csv_dir = os.path.dirname(data_csv_path)
        output_path = os.path.join(script_dir, "output.csv")

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

        print(f"\n{'='*70}")
        print(f"DA GHI KET QUA VAO: {to_console_safe(output_path)}")
        print(f"{'='*70}\n")

        # --- Danh gia voi --gt neu co ---
        gt_path = (args.gt or "").strip()
        if gt_path:
            gt_abs = os.path.abspath(gt_path)
            evaluate_csvs(
                output_path,
                gt_abs,
                iou_threshold=float(args.iou_threshold),
                filter_ids=data_image_ids if data_image_ids else None,
            )

        # --- Tu dong danh gia voi output_valid.csv neu co (va chua danh gia) ---
        auto_valid = os.path.join(script_dir, "output_valid.csv")
        if not gt_path and os.path.isfile(auto_valid):
            evaluate_csvs(
                output_path,
                auto_valid,
                iou_threshold=float(args.iou_threshold),
                filter_ids=data_image_ids if data_image_ids else None,
            )
            
    except Exception as e:  # noqa: BLE001
        print(f"\nLOI XU LY DU LIEU: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
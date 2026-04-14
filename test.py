#!/usr/bin/env python3
"""
Đánh giá kết quả phát hiện QR Code dựa trên IoU (Intersection over Union).
So sánh file output.csv (dự đoán) với output_valid.csv (ground truth).
Tính Precision, Recall, F1 Score.
"""

import argparse
import csv
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# Định nghĩa bounding box: 4 điểm (x0,y0), (x1,y1), (x2,y2), (x3,y3)
BBox = List[Tuple[float, float]]  # list of 4 points

def polygon_area(pts: List[Tuple[float, float]]) -> float:
    """Tính diện tích đa giác (công thức shoelace)"""
    n = len(pts)
    area = 0.0
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i+1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0

def polygon_intersection_area(pts1: BBox, pts2: BBox) -> float:
    """Tính diện tích giao của hai đa giác lồi (dùng clipping Sutherland–Hodgman)"""
    # Đơn giản: dùng bounding rectangle để ước lượng nhanh (chấp nhận sai số nhỏ)
    # Nhưng để chính xác hơn, ta dùng thư viện shapely? Đề bài không cấm dùng shapely.
    # Tuy nhiên, để tránh phụ thuộc, ta tự implement clipping cho đa giác lồi.
    # Vì các bounding box là tứ giác lồi (hình thang, hình bình hành...), ta dùng thuật toán Sutherland–Hodgman.
    
    def inside(p, cp1, cp2):
        # Kiểm tra điểm p có nằm bên trong cạnh cp1->cp2 không (bên trái nếu đi ngược chiều kim đồng hồ)
        return (cp2[0] - cp1[0]) * (p[1] - cp1[1]) - (cp2[1] - cp1[1]) * (p[0] - cp1[0]) >= 0
    
    def intersection(cp1, cp2, s, e):
        # Giao điểm của đoạn s-e với đường thẳng cp1-cp2
        dc = (cp1[0] - cp2[0], cp1[1] - cp2[1])
        dp = (s[0] - e[0], s[1] - e[1])
        n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
        n2 = s[0] * e[1] - s[1] * e[0]
        denom = dc[0] * dp[1] - dc[1] * dp[0]
        if abs(denom) < 1e-12:
            return None
        x = (n1 * dp[0] - dc[0] * n2) / denom
        y = (n1 * dp[1] - dc[1] * n2) / denom
        return (x, y)
    
    def clip(subject_polygon, clip_polygon):
        # Clipping đa giác subject_polygon với clip_polygon (đa giác lồi)
        output_list = subject_polygon
        cp1 = clip_polygon[-1]
        for cp2 in clip_polygon:
            input_list = output_list
            output_list = []
            if not input_list:
                break
            s = input_list[-1]
            for e in input_list:
                if inside(e, cp1, cp2):
                    if not inside(s, cp1, cp2):
                        inter = intersection(cp1, cp2, s, e)
                        if inter:
                            output_list.append(inter)
                    output_list.append(e)
                elif inside(s, cp1, cp2):
                    inter = intersection(cp1, cp2, s, e)
                    if inter:
                        output_list.append(inter)
                s = e
            cp1 = cp2
        return output_list
    
    # Đảm bảo các đa giác được sắp xếp theo thứ tự (theo chiều kim đồng hồ hoặc ngược)
    # Không cần sắp xếp lại vì clipping vẫn hoạt động nếu các điểm liên tục.
    intersect_poly = clip(pts1, pts2)
    if not intersect_poly:
        return 0.0
    # Tính diện tích đa giác giao
    return polygon_area(intersect_poly)

def polygon_union_area(pts1: BBox, pts2: BBox, intersection: float) -> float:
    """Diện tích hợp = diện tích1 + diện tích2 - diện tích giao"""
    area1 = polygon_area(pts1)
    area2 = polygon_area(pts2)
    return area1 + area2 - intersection

def compute_iou(pts1: BBox, pts2: BBox) -> float:
    """Tính IoU của hai tứ giác"""
    inter = polygon_intersection_area(pts1, pts2)
    if inter == 0:
        return 0.0
    union = polygon_union_area(pts1, pts2, inter)
    return inter / union if union > 0 else 0.0

def parse_bbox(row: Dict) -> Optional[BBox]:
    """Từ dòng csv, trích xuất 4 điểm (x0,y0),...,(x3,y3)"""
    try:
        pts = []
        for i in range(4):
            x = float(row[f'x{i}'])
            y = float(row[f'y{i}'])
            pts.append((x, y))
        return pts
    except (KeyError, ValueError):
        return None

def read_gt_csv(filepath: str) -> Dict[Tuple[str, int], BBox]:
    """
    Đọc ground truth (output_valid.csv).
    Mỗi dòng: image_id, qr_index, x0,y0,x1,y1,x2,y2,x3,y3,content
    Trả về dict: key = (image_id, qr_index) -> bbox
    """
    gt = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_id = row['image_id'].strip()
            qr_idx = int(row['qr_index'])
            bbox = parse_bbox(row)
            if bbox:
                gt[(img_id, qr_idx)] = bbox
    return gt

def read_pred_csv(filepath: str) -> Dict[str, List[BBox]]:
    """
    Đọc file dự đoán (output.csv) với cột image_path, qr_count, corners.
    corners là chuỗi dạng "[(x0,y0),(x1,y1),(x2,y2),(x3,y3)];[(...)]"
    Trả về dict: image_id (lấy tên file không đuôi) -> list các bbox.
    """
    pred = defaultdict(list)
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = row['image_path'].strip()
            # Lấy image_id từ đường dẫn (giả sử file ảnh có tên như "image001.png")
            img_id = img_path.split('/')[-1].split('.')[0]
            corners_str = row['corners'].strip()
            if not corners_str:
                continue
            # Parse corners: mỗi QR cách nhau bằng dấu chấm phẩy
            for qr_str in corners_str.split(';'):
                qr_str = qr_str.strip()
                if not qr_str:
                    continue
                # qr_str có dạng "[(x0,y0),(x1,y1),(x2,y2),(x3,y3)]"
                # Loại bỏ dấu ngoặc vuông
                if qr_str.startswith('[') and qr_str.endswith(']'):
                    qr_str = qr_str[1:-1]
                points = qr_str.split('),(')
                bbox = []
                for pt in points:
                    pt = pt.replace('(', '').replace(')', '')
                    x_str, y_str = pt.split(',')
                    x = float(x_str)
                    y = float(y_str)
                    bbox.append((x, y))
                if len(bbox) == 4:
                    pred[img_id].append(bbox)
    return pred

def match_predictions_to_gt(gt_bboxes: List[BBox], pred_bboxes: List[BBox], iou_threshold=0.5):
    """
    Ghép các dự đoán với ground truth dựa trên IoU.
    Trả về:
        tp: số lượng TP
        fp: số lượng FP
        fn: số lượng FN
    """
    n_gt = len(gt_bboxes)
    n_pred = len(pred_bboxes)
    if n_gt == 0 and n_pred == 0:
        return 0, 0, 0
    if n_gt == 0:
        return 0, n_pred, 0
    if n_pred == 0:
        return 0, 0, n_gt
    
    # Ma trận IoU giữa các dự đoán và GT
    iou_matrix = np.zeros((n_pred, n_gt))
    for i, pbox in enumerate(pred_bboxes):
        for j, gbox in enumerate(gt_bboxes):
            iou_matrix[i, j] = compute_iou(pbox, gbox)
    
    # Ghép theo thuật toán tham lam: mỗi dự đoán ghép với GT có IoU max >= ngưỡng
    matched_gt = set()
    matched_pred = set()
    for i in range(n_pred):
        if iou_matrix[i, :].max() >= iou_threshold:
            j = np.argmax(iou_matrix[i, :])
            if j not in matched_gt:
                matched_gt.add(j)
                matched_pred.add(i)
    
    tp = len(matched_pred)
    fp = n_pred - tp
    fn = n_gt - len(matched_gt)
    return tp, fp, fn

def evaluate(output_csv: str, valid_csv: str):
    # Đọc dữ liệu
    gt_dict = read_gt_csv(valid_csv)          # key: (image_id, qr_index)
    pred_dict = read_pred_csv(output_csv)     # key: image_id -> list bbox
    
    # Tổng hợp các metric
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    # Nhóm GT theo image_id
    gt_by_image = defaultdict(list)
    for (img_id, _), bbox in gt_dict.items():
        gt_by_image[img_id].append(bbox)
    
    # Duyệt tất cả các ảnh có trong GT hoặc dự đoán
    all_images = set(gt_by_image.keys()).union(set(pred_dict.keys()))
    for img_id in all_images:
        gt_boxes = gt_by_image.get(img_id, [])
        pred_boxes = pred_dict.get(img_id, [])
        tp, fp, fn = match_predictions_to_gt(gt_boxes, pred_boxes, iou_threshold=0.5)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        print(f"Image {img_id}: TP={tp}, FP={fp}, FN={fn}")
    
    # Tính toán Precision, Recall, F1
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    print("\n===== KẾT QUẢ ĐÁNH GIÁ =====")
    print(f"Total True Positives (TP): {total_tp}")
    print(f"Total False Positives (FP): {total_fp}")
    print(f"Total False Negatives (FN): {total_fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    return precision, recall, f1

def main():
    parser = argparse.ArgumentParser(description="Đánh giá phát hiện QR code dựa trên IoU")
    parser.add_argument("--pred", default="output.csv", help="File dự đoán (mặc định: output.csv)")
    parser.add_argument("--gt", default="output_valid.csv", help="File ground truth (mặc định: output_valid.csv)")
    args = parser.parse_args()
    
    evaluate(args.pred, args.gt)

if __name__ == "__main__":
    main()
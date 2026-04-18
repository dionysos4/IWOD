import numpy as np

def axis_difference(a, b):
    """
    smallest angle difference between two orientations (in radians), treating direction reversals (±π) as equivalent.
    Parameters:
        a, b: angles in radians, typically in the range [-π, π)
    Returns:
        float: angle difference in [0, π/2]
    """
    diff = np.abs(a - b) % np.pi
    return min(diff, np.pi - diff)


def iou_axis_aligned(w1, l1, w2, l2):
    inter_w = min(w1, w2)
    inter_l = min(l1, l2)
    if inter_w <= 0 or inter_l <= 0:
        return 0.0
    inter_area = inter_w * inter_l
    area1 = w1 * l1
    area2 = w2 * l2
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def evaluate_detections_2d_center_distance(gt_boxes, pred_boxes, dist_thresh=2.0):
    matched_gt = set()
    tp = 0
    fp = 0
    centroid_errors = []
    scale_errors = []
    orientation_errors = []

    # Sort predictions by score
    pred_boxes = sorted(pred_boxes, key=lambda x: x.get("score", 1.0), reverse=True)

    for pred in pred_boxes:
        pred_center = np.array([pred["center_x"], pred["center_z"]])
        best_distance = float("inf")
        best_match = -1
        for i, gt in enumerate(gt_boxes):
            if i in matched_gt:
                continue
            gt_center = np.array([gt["center_x"], gt["center_z"]])
            distance = np.linalg.norm(pred_center - gt_center)
            if distance < dist_thresh and distance < best_distance:
                best_distance = distance
                best_match = i

        if best_match >= 0:
            matched_gt.add(best_match)
            tp += 1
            centroid_errors.append(best_distance)

            gt = gt_boxes[best_match]
            iou = iou_axis_aligned(pred["width"], pred["length"], gt["width"], gt["length"])
            ase = 1.0 - iou
            scale_errors.append(ase)

            aoe = axis_difference(pred["orientation"], gt["orientation"])
            orientation_errors.append(aoe)
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "MeanCentroidError": np.mean(centroid_errors) if centroid_errors else None,
        "AverageScaleError": np.mean(scale_errors) if scale_errors else None,
        "AverageOrientationError": np.mean(orientation_errors) if orientation_errors else None,
    }


def evaluate_dataset_center_distance(all_gt_boxes, all_pred_boxes, dist_thresh=2.0):
    total_tp = 0
    total_fp = 0
    total_fn = 0
    all_centroid_errors = []
    all_scale_errors = []
    all_orientation_errors = []

    for gt_boxes, pred_boxes in zip(all_gt_boxes, all_pred_boxes):
        res = evaluate_detections_2d_center_distance(gt_boxes, pred_boxes, dist_thresh)
        total_tp += res["TP"]
        total_fp += res["FP"]
        total_fn += res["FN"]

        if res["MeanCentroidError"] is not None:
            matched_gt = set()
            pred_boxes_sorted = sorted(pred_boxes, key=lambda x: x.get("score", 1.0), reverse=True)

            for pred in pred_boxes_sorted:
                pred_center = np.array([pred["center_x"], pred["center_z"]])
                best_distance = float("inf")
                best_match = -1
                for i, gt in enumerate(gt_boxes):
                    if i in matched_gt:
                        continue
                    gt_center = np.array([gt["center_x"], gt["center_z"]])
                    distance = np.linalg.norm(pred_center - gt_center)
                    if distance < dist_thresh and distance < best_distance:
                        best_distance = distance
                        best_match = i
                if best_match >= 0:
                    matched_gt.add(best_match)
                    gt = gt_boxes[best_match]
                    all_centroid_errors.append(best_distance)
                    iou = iou_axis_aligned(pred["width"], pred["length"], gt["width"], gt["length"])
                    ase = 1.0 - iou
                    all_scale_errors.append(ase)
                    aoe = axis_difference(pred["orientation"], gt["orientation"])
                    all_orientation_errors.append(aoe)

    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return {
        "Total TP": total_tp,
        "Total FP": total_fp,
        "Total FN": total_fn,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "MeanCentroidError": np.mean(all_centroid_errors) if all_centroid_errors else None,
        "AverageScaleError": np.mean(all_scale_errors) if all_scale_errors else None,
        "AverageOrientationError": np.mean(all_orientation_errors) if all_orientation_errors else None,
    }


def evaluate_dataset_center_distance_multi_thresholds(
    all_gt_boxes, all_pred_boxes, dist_thresholds=[0.5, 1.0, 2.0, 4.0, 10.0]
):
    results = {}
    for t in dist_thresholds:
        res = evaluate_dataset_center_distance(all_gt_boxes, all_pred_boxes, dist_thresh=t)
        results[f"{t:.2f}m"] = res
    return results


def compute_precision_recall_curve(all_gt_boxes, all_pred_boxes, dist_thresh=2.0):
    all_preds = []
    all_gts = []

    # Alle GTs und Predictions mit Bildindex speichern
    for image_id, (gt_boxes, pred_boxes) in enumerate(zip(all_gt_boxes, all_pred_boxes)):
        for gt in gt_boxes:
            all_gts.append((image_id, gt))
        for pred in pred_boxes:
            all_preds.append((image_id, pred))

    # Nach Score sortieren
    all_preds = sorted(all_preds, key=lambda x: x[1].get("score", 1.0), reverse=True)

    tp_list = []
    fp_list = []
    matched = set()
    total_gt = len(all_gts)

    for image_id, pred in all_preds:
        pred_center = np.array([pred["center_x"], pred["center_z"]])
        best_match = -1
        best_distance = float("inf")

        for gt_idx, (gt_image_id, gt) in enumerate(all_gts):
            if gt_image_id != image_id or gt_idx in matched:
                continue
            gt_center = np.array([gt["center_x"], gt["center_z"]])
            distance = np.linalg.norm(pred_center - gt_center)
            if distance < dist_thresh and distance < best_distance:
                best_distance = distance
                best_match = gt_idx

        if best_match >= 0:
            matched.add(best_match)
            tp_list.append(1)
            fp_list.append(0)
        else:
            tp_list.append(0)
            fp_list.append(1)

    tp_cum = np.cumsum(tp_list)
    fp_cum = np.cumsum(fp_list)
    precisions = tp_cum / (tp_cum + fp_cum + 1e-6)
    recalls = tp_cum / (total_gt + 1e-6)

    return precisions, recalls


def compute_ap(precisions, recalls):
    # Fläche unter der PR-Kurve mit Trapezregel (Integration)
    return np.trapezoid(precisions, recalls)


def compute_map(all_gt_boxes, all_pred_boxes, dist_thresholds=[0.5, 1.0, 2.0, 4.0, 10.0]):
    ap_list = []
    for t in dist_thresholds:
        precisions, recalls = compute_precision_recall_curve(all_gt_boxes, all_pred_boxes, dist_thresh=t)
        ap = compute_ap(precisions, recalls)
        ap_list.append(ap)
    mean_ap = np.mean(ap_list)
    return {"APs": ap_list, "mAP": mean_ap}
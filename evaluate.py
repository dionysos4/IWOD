from src.iwod.model.predict import Predictor
from src.iwod.eval_lcod.eval import *
import numpy as np
import tqdm
import argparse


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_path", help="Path to the config file", default="/home/dennis/git_repos/IWOD/config/train_lcod.yaml")
    parser.add_argument("--ckpt_path", help="Path to the checkpoint file", default="/mnt/deepdoubt/dennis_data/logging/kitti_stereo/1052/lcod-epoch=81-train_loss=0.00.ckpt")
    args = parser.parse_args()
    CONFIG_PATH = args.config_path
    CKPT_PATH = args.ckpt_path

    predictor = Predictor(CONFIG_PATH, CKPT_PATH, gpu_id=2, water_detection=True)

    all_detected_boxes = []
    all_gt_boxes = []

    for idx in tqdm.tqdm(range(predictor.get_dataset_length())):

        det_X, det_Z, det_scores, det_regs, det_angles = predictor.predict(idx=idx)
        
        detected_boxes_one_image = []
        for j in range(len(det_X)):
            x = det_X[j]
            z = det_Z[j]
            x = x + det_regs[j][0]
            z = z + det_regs[j][1]
            w = det_regs[j][2]
            l = det_regs[j][3]
            theta = det_angles[j]

            # l always greater than w
            if w > l:
                w, l = l, w
                theta += np.pi / 2

                # 2. Normalize theta in [−π/2, π/2]
                #    Note: (θ + π/2) % π − π/2 bringt alles in [−π/2, π/2]
                theta = (theta + np.pi / 2) % np.pi - np.pi / 2

            bbox = {
                "orientation": theta.item(),
                "length": l.item(),
                "width": w.item(),
                "center_x": x.item(),
                "center_z": z.item(),
                "score": det_scores[j].item()
            }

            detected_boxes_one_image.append(bbox)
        all_detected_boxes.append(detected_boxes_one_image)

        # annotations
        gt_annotations = predictor.get_annotations(idx)
        gt_boxes_one_image = []
        gt_annotations = gt_annotations[gt_annotations.sum(1) != 0]
        
        for anno in gt_annotations:
            x_gt = anno[1]
            z_gt = anno[3]
            w_gt = anno[4]
            l_gt = anno[6]
            theta_gt = anno[7]
            gt_bbox = {
                'orientation': theta_gt.item(),
                'length': l_gt.item(),
                'width': w_gt.item(),
                'center_x': x_gt.item(),
                'center_z': z_gt.item()
            }
            gt_boxes_one_image.append(gt_bbox)
        all_gt_boxes.append(gt_boxes_one_image)

    # Evaluation all images
    print(evaluate_dataset_center_distance_multi_thresholds(all_gt_boxes, all_detected_boxes, dist_thresholds=[0.5, 1.0, 2.0, 4.0, 10.0]))
    print(compute_map(all_gt_boxes, all_detected_boxes))


if __name__ == "__main__":
    main()
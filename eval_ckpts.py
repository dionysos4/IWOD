from src.iwod.model.predict import Predictor
from src.iwod.eval_lcod.eval import *
import numpy as np
import tqdm
import argparse
import os
import matplotlib.pyplot as plt


def extract_epoch(filename):
    return int(filename.split('epoch=')[1].split('-')[0])

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_path", help="Path to the config file", default="/home/dennis/git_repos/IWOD/config/train_lcod.yaml")
    parser.add_argument("--ckpt_path", help="Path to the checkpoints", default="/mnt/deepdoubt/dennis_data/logging/lcod/3001")
    parser.add_argument("--output_path", help="Path to save the evaluation results", default="/home/dennis/git_repos/IWOD/valid_results_3001")
    args = parser.parse_args()
    CONFIG_PATH = args.config_path
    CKPT_PATH = args.ckpt_path
    OUTPUT_FILE = args.output_path

    filtered_checkpoints = []
    checkpoint_files = os.listdir(CKPT_PATH)
    for file in checkpoint_files:
        if "train" in file:
            filtered_checkpoints.append(file)
    sorted_ckpt_files = sorted(filtered_checkpoints, key=extract_epoch)

    ap_results = []
    map_results = []

    # iterate over checkpoints
    for ckpt in tqdm.tqdm(sorted_ckpt_files):
        ckpt_path_sample = os.path.join(CKPT_PATH, ckpt)
        predictor = Predictor(CONFIG_PATH, ckpt_path_sample, gpu_id=2, water_detection=False, training="valid")

        # iterate over dataset
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
        ap = evaluate_dataset_center_distance_multi_thresholds(all_gt_boxes, all_detected_boxes, dist_thresholds=[0.5, 1.0, 2.0, 4.0, 10.0])
        map = compute_map(all_gt_boxes, all_detected_boxes)
        ap_results.append(ap)
        map_results.append(map)

        
        np.savez(OUTPUT_FILE, ap_results=ap_results, map_results=map_results)

        # Plot
        fig, ax = plt.subplots()
        x_values = []
        y_values = []

        for i in range(len(map_results)):
            y = map_results[i]["mAP"]
            x_values.append(i)
            y_values.append(y)


        ax.plot(x_values, y_values, color='blue', marker='o', linestyle='-', label='mAP')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('mAP')
        ax.legend()

        plt.savefig(OUTPUT_FILE.replace(".npy", ".png"))
        plt.close(fig)
        


if __name__ == "__main__":
    main()
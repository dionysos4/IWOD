import numpy as np
import torch
import detectron2.layers as dt2
import copy
from iwod.model.submodules import *
import yaml
import os, shutil


def get_prediction_dict(predictions, cfg, val_img_ids, score_threshold=0.6):
    """
    Takes all the predictions and converts them to the format required for the official KITTI evaluation code. Additionaly
    it applies a score threshold and non max suppression to the predictions. 
    
    params:
        predictions: (list) list of predictions for each image in the validation set.
        cfg: (dict) config dictionary containing the parameters for the evaluation.
        val_img_ids: (list) list of image ids in the validation set.
        score_threshold: (float) score threshold for the predictions. Default is 0.6
        
    returns:
        list: list of dictionaries containing the predictions for each image in the validation set.
    """
    kitti_global_detection = []
    global_counter = 0
    for idx in range(len(predictions)):
        for batch_counter in range(predictions[idx][0].shape[0]):
            score = torch.nn.functional.sigmoid(predictions[idx][1][batch_counter,0]).cpu()
            centerness = torch.nn.functional.sigmoid(predictions[idx][4][batch_counter,0]).cpu()
            score = score * centerness

            ## get world coordinates
            Z_MIN, Z_MAX = cfg["z_min"], cfg["z_max"]
            Y_MIN, Y_MAX = cfg["y_min"], cfg["y_max"]
            X_MIN, X_MAX = cfg["x_min"], cfg["x_max"]
            VOXEL_Z_SIZE = cfg["z_size"]
            VOXEL_X_SIZE = cfg["x_size"]

            shifts_z = torch.arange(Z_MAX, Z_MIN - np.sign(VOXEL_Z_SIZE) * 1e-10, step=VOXEL_Z_SIZE, 
                dtype=torch.float32) + VOXEL_Z_SIZE / 2.
            shifts_x = torch.arange(X_MIN, X_MAX - np.sign(VOXEL_X_SIZE) * 1e-10, step=VOXEL_X_SIZE,
                dtype=torch.float32) + VOXEL_X_SIZE / 2.
            shifts_z, shifts_x = torch.meshgrid(shifts_z, shifts_x)
            locations_bev = torch.stack([shifts_x, shifts_z], dim=-1)
            locations_bev = locations_bev.reshape(-1, 2)

            X = locations_bev[:,0].view(score.shape)
            Z = locations_bev[:,1].view(score.shape)
            

            score_mask = score > score_threshold

            ######
            val_idx = val_img_ids[global_counter]
            val_idx = str(val_idx)
            val_idx = val_idx.zfill(6)
            ######

            if torch.sum(score_mask) > 0:

                regression_cent = copy.deepcopy(predictions[idx][2][batch_counter].unsqueeze(0))# * stride
                regression_dim = copy.deepcopy(predictions[idx][3][batch_counter].unsqueeze(0))
                regression = torch.cat((regression_cent, regression_dim), dim=1).cpu()


                angle_regression = predictions[idx][5][batch_counter].cpu()
                ps_coder = PSCoder("le90", 4)
                angle_regression = angle_regression.permute(1,2,0)
                angle_regression = angle_regression.reshape(-1, angle_regression.shape[2])
                angle_regression = ps_coder.decode(angle_regression)
                angle_regression = angle_regression.reshape(regression.shape[2], regression.shape[3])

                zfxf = torch.nonzero(score_mask)
                zf = zfxf[:,0]
                xf = zfxf[:,1]
                detected_X = X[zf, xf]
                detected_Z = Z[zf, xf]

                detected_regressions = regression[0].permute(1,2,0)[zf, xf]
                detected_angles = angle_regression[zf,xf]

                # non max suppression
                dscores = score[score_mask]
                dx = detected_X
                dz = detected_Z
                centerx = dx + detected_regressions[:, 0]
                centery = dz + detected_regressions[:, 1]


                w = detected_regressions[:, 2]
                h = detected_regressions[:, 3]

                dtheta = detected_angles

                potential_boxes = torch.cat((centerx.unsqueeze(dim=1), centery.unsqueeze(dim=1),
                        w.unsqueeze(dim=1), h.unsqueeze(dim=1),
                        torch.rad2deg(dtheta).unsqueeze(dim=1)), dim=1)

                
                nms_idx = dt2.nms_rotated(potential_boxes, torch.tensor(dscores, dtype=potential_boxes.dtype), 0.0001)

                detected_X = detected_X[nms_idx]
                detected_Z = detected_Z[nms_idx]
                detected_scores = dscores[nms_idx]
                detected_regressions = detected_regressions[nms_idx]
                detected_angles = detected_angles[nms_idx]

                # iterate over detections per image
                kitti_dict = {"name": np.array([]), "truncated": np.array([]), "occluded": np.array([]), "alpha": np.array([]), "bbox": [], "dimensions": [], "location": [], "rotation_y": np.array([]), "score": np.array([])}
                
                # with open(os.path.join(prediction_path, val_idx + ".txt"), "w") as f:
                for i in range(detected_X.shape[0]):
                    x = detected_X[i]
                    z = detected_Z[i]
                    x = x + detected_regressions[i, 0]
                    z = z + detected_regressions[i, 1]
                    w = detected_regressions[i, 2]
                    l = detected_regressions[i, 3]
                    theta = detected_angles[i]

                    # theta to kitti format
                    if (theta < torch.pi / 2) & (theta > 0):
                        theta = theta + np.pi / 2
                    elif (theta >= torch.pi / 2) & (theta > 0):
                        theta = -np.pi + (theta - np.pi / 2)
                    elif (theta > -np.pi / 2) & (theta < 0):
                        theta = theta + np.pi / 2
                    elif (theta <= -np.pi / 2) & (theta < 0):
                        theta = theta + np.pi / 2

                    dscore = detected_scores[i]

                    y = 1.6

                    kitti_dict["name"] = np.append(kitti_dict["name"], "Car").astype(np.dtype("U3"))
                    kitti_dict["truncated"] = np.append(kitti_dict["truncated"], 0)
                    kitti_dict["occluded"] = np.append(kitti_dict["occluded"], 0)
                    kitti_dict["alpha"] = np.append(kitti_dict["alpha"], 0)
                    kitti_dict["bbox"].append([0, 0, 100, 100])
                    #kitti_dict["bbox"].append([x_min, y_min, x_max, y_max])
                    # in metrc dictionary l,h,w
                    kitti_dict["dimensions"].append([l, 1.6, w])
                    kitti_dict["location"].append([x, y, z])
                    kitti_dict["rotation_y"] = np.append(kitti_dict["rotation_y"], theta.to(dtype=torch.float32))
                    kitti_dict["score"] = np.append(kitti_dict["score"], dscore.to(dtype=torch.float32))

            else:
                kitti_dict = {"name": np.array([]), "truncated": np.array([]), "occluded": np.array([]), "alpha": np.array([]), "bbox": np.array([]).reshape(0,4), "dimensions": np.array([]).reshape(0,3), "location": np.array([]).reshape(0,3), "rotation_y": np.array([]), "score": np.array([])}

            kitti_dict["bbox"] = np.array(kitti_dict["bbox"])
            kitti_dict["dimensions"] = np.array(kitti_dict["dimensions"])
            kitti_dict["location"] = np.array(kitti_dict["location"])
            kitti_global_detection.append(kitti_dict)

            global_counter += 1

    return kitti_global_detection


def load_config(config_file):
    """
    Load the config file and return the config as a dictionary.
    
    params:
        config_file: (str) path to the config file
    
    returns:
        dict: config as a dictionary
    """
    with open(config_file) as file:
        config = yaml.safe_load(file)
    return config


def read_imageset_file(path):
    """
    Read the image set file and return a list of image ids as integers.
    
    params:
        path: (str) path to the image set file
        
    returns:        
        list: list of image ids as integers
    
    """
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]


def convert_targets(TESTFILE, LABELPATH, CALIBPATH, TARGETPATH):
    """
    Convert the center of the targets from stereo rect in stereo unrect frame. This is necessary 
    because the evaluation code expects the targets to be in the same frame as the predictions, 
    which are in stereo unrect frame.
    
    params:
        TESTFILE: (str) path to the test file containing the image ids to be evaluated
        LABELPATH: (str) path to the folder containing the label files in stereo rect frame
        CALIBPATH: (str) path to the folder containing the calibration files
        TARGETPATH: (str) path to the folder where the converted label files will be saved
    """
    # read test file
    with open(TESTFILE, 'r') as f:
        lines = f.readlines()


    # check if target_path exists and create if not or clear it
    if os.path.exists(TARGETPATH):
        shutil.rmtree(TARGETPATH)
    os.makedirs(TARGETPATH)

    # copy files to target path
    for line in lines:
        img_id = line.strip()
        tar_path = os.path.join(TARGETPATH, "{}.txt".format(img_id))
        shutil.copyfile(LABELPATH + "/{}.txt".format(img_id), tar_path)

    for file in os.listdir(TARGETPATH):
        with open(TARGETPATH + "/" + file, 'r') as f:
            lines = f.readlines()
        
        # read P2
        with open(CALIBPATH + "/" + file, 'r') as f:
            lines_calib = f.readlines()
        R2 = lines_calib[6].split(" ")[1:]
        T2 = lines_calib[10].split(" ")[1:]
        R0_rect = lines_calib[12].split(" ")[1:]
        #remove \n
        R2[-1] = R2[-1][:-1]
        T2[-1] = T2[-1][:-1]
        R0_rect[-1] = R0_rect[-1][:-1]
        R2 = np.array(R2).astype(np.float32).reshape(3,3)
        T2 = np.array(T2).astype(np.float32)
        R0_rect = np.array(R0_rect).astype(np.float32).reshape(3,3)


        with open(TARGETPATH + "/" + file, 'w') as f:
            for line in lines:
                # replace line.split[1] with 0
                line = line.split(" ")
                center = np.array([float(line[11]), float(line[12]), float(line[13])])

                # transform from cam0 rect to cam0 unrect
                center = R0_rect.T @ center
                T_cam2_cam0 = np.zeros((3,4))
                T_cam2_cam0[:3,:3] = R2
                T_cam2_cam0[:,3] = T2
                center = T_cam2_cam0 @ np.append(center, 1)

                line[11] = str(center[0])
                line[12] = str(center[1])
                line[13] = str(center[2])
                line = " ".join(line)
                f.write(line.strip() + "\n")
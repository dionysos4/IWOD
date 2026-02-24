import h5py
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import fire


def extract_data(dataset_dir, save_dir, downsample_factor, split_file_path):
    """
    Extracts data from hdf5 files and save them as png and npz files. It 
    already splits the training data into a training and validation dataset. This split
    is provided in train.txt and val.txt
    Args:
        dataset_dir: directory of the hdf5-dataset
        save_dir: directory to save extracted data
        downsample_factor: downsample factor
        split_file_path: path to the directory which contains the split files for training and validation
    """

    print(dataset_dir, save_dir, downsample_factor)

    DATASET_DIR = dataset_dir
    DOWNSAMPLE_FACTOR = downsample_factor
    SAVE_DIR = save_dir


    TRAIN_DIR = os.path.join(DATASET_DIR, "train")
    TEST_DIR = os.path.join(DATASET_DIR, "test")
    CLASSES = {"motorboat": 0, "sailboat": 1, "sailboat under bare poles": 2, "stand-up-paddle": 3, "catamaran": 4,
                        "ferry": 5, "pedal boat": 6, "motor vessel": 7, "pile": 8}
    np.random.seed(1)

    # create save dir
    EXTRACTED_DIR = os.path.join(SAVE_DIR, "extracted_dataset_factor_" + str(DOWNSAMPLE_FACTOR))
    os.makedirs(EXTRACTED_DIR, exist_ok=True)

    # train dir
    train_dir = os.path.join(EXTRACTED_DIR, "train")
    os.makedirs(train_dir, exist_ok=True)
    train_left_img_dir = os.path.join(train_dir, "left_img")
    os.makedirs(train_left_img_dir, exist_ok=True)
    train_right_img_dir = os.path.join(train_dir, "right_img")
    os.makedirs(train_right_img_dir, exist_ok=True)
    train_annotation_dir = os.path.join(train_dir, "annotations")
    os.makedirs(train_annotation_dir, exist_ok=True)
    train_calib_dir = os.path.join(train_dir, "calibration")
    os.makedirs(train_calib_dir, exist_ok=True)
    train_lidar_dir = os.path.join(train_dir, "lidar")
    os.makedirs(train_lidar_dir, exist_ok=True)

    # valid dir
    valid_dir = os.path.join(EXTRACTED_DIR, "valid")
    os.makedirs(valid_dir, exist_ok=True)
    valid_left_img_dir = os.path.join(valid_dir, "left_img")
    os.makedirs(valid_left_img_dir, exist_ok=True)
    valid_right_img_dir = os.path.join(valid_dir, "right_img")
    os.makedirs(valid_right_img_dir, exist_ok=True)
    valid_annotation_dir = os.path.join(valid_dir, "annotations")
    os.makedirs(valid_annotation_dir, exist_ok=True)
    valid_calib_dir = os.path.join(valid_dir, "calibration")
    os.makedirs(valid_calib_dir, exist_ok=True)
    valid_lidar_dir = os.path.join(valid_dir, "lidar")
    os.makedirs(valid_lidar_dir, exist_ok=True)

    # test dir
    test_dir = os.path.join(EXTRACTED_DIR, "test")
    os.makedirs(test_dir, exist_ok=True)
    test_left_img_dir = os.path.join(test_dir, "left_img")
    os.makedirs(test_left_img_dir, exist_ok=True)
    test_right_img_dir = os.path.join(test_dir, "right_img")
    os.makedirs(test_right_img_dir, exist_ok=True)
    test_annotation_dir = os.path.join(test_dir, "annotations")
    os.makedirs(test_annotation_dir, exist_ok=True)
    test_calib_dir = os.path.join(test_dir, "calibration")
    os.makedirs(test_calib_dir, exist_ok=True)
    test_lidar_dir = os.path.join(test_dir, "lidar")
    os.makedirs(test_lidar_dir, exist_ok=True)

    # load split file
    with open(os.path.join(split_file_path, "train.txt"), "r") as f:
        train_split = f.readlines()
        train_split = [x.strip() for x in train_split]
    with open(os.path.join(split_file_path, "val.txt"), "r") as f:
        valid_split = f.readlines()
        valid_split = [x.strip() for x in valid_split]


    for i, dir in enumerate([TRAIN_DIR, TEST_DIR]):
        file_list = sorted(os.listdir(dir))
        for file in file_list:
            if file in train_split:
                # train dir
                left_img_dir = train_left_img_dir
                right_img_dir = train_right_img_dir
                annotation_dir = train_annotation_dir
                calib_dir = train_calib_dir
                lidar_dir = train_lidar_dir
            elif file in valid_split:
                # valid dir
                left_img_dir = valid_left_img_dir
                right_img_dir = valid_right_img_dir
                annotation_dir = valid_annotation_dir
                calib_dir = valid_calib_dir
                lidar_dir = valid_lidar_dir
            else:
                # test dir
                left_img_dir = test_left_img_dir
                right_img_dir = test_right_img_dir
                annotation_dir = test_annotation_dir
                calib_dir = test_calib_dir
                lidar_dir = test_lidar_dir


            hdf5_file = file
            # open hdf5 file and read data
            with h5py.File(os.path.join(dir, hdf5_file),'r') as f:
                # Annotations
                bbox_label_int = []
                bbox_label_str = []
                bbox_location = []
                bbox_dimensions = []
                bbox_rotation_y = []
                bbox_visibility = []
                bbox_occlusion = []
                k = 0
                while True:
                    try:
                        label_str = f["bounding_boxes/bounding_box_" + str(k) + "/category/name"][()].decode("utf-8")
                        label_int = CLASSES[label_str]
                        bbox_label_int.append(label_int)
                        bbox_label_str.append(label_str)
                        bbox_location.append(f["bounding_boxes/bounding_box_" + str(k) + "/location"][:])
                        bbox_dimensions.append(f["bounding_boxes/bounding_box_" + str(k) + "/dimensions"][:])
                        bbox_rotation_y.append(f["bounding_boxes/bounding_box_" + str(k) + "/rotation_y"][()])
                        bbox_visibility.append(f["bounding_boxes/bounding_box_" + str(k) + "/visibility"][()])
                        bbox_occlusion.append(f["bounding_boxes/bounding_box_" + str(k) + "/occlusion"][()])
                        k += 1
                    except:
                        break
        
                # Calibrations
                K_l = f['left_image/K'][:].reshape(3,3)
                K_r = f['right_image/K'][:].reshape(3,3)
                D_l = f['left_image/D'][:]
                D_r = f['right_image/D'][:]
                R_l = f['left_image/R'][:].reshape(3,3)
                R_r = f['right_image/R'][:].reshape(3,3)
                P_l = f['left_image/P'][:].reshape(3,4)
                P_r = f['right_image/P'][:].reshape(3,4)

                # IMAGES
                left_img = f['left_image/image'][:1200,:,:] / np.iinfo("uint16").max
                right_img = f['right_image/image'][:1200,:,:] / np.iinfo("uint16").max
                height, width = left_img.shape[:2]
                left_img = cv2.undistort(left_img, K_l, D_l)
                right_img = cv2.undistort(right_img, K_r, D_r)
                
                # resize images and intrinsics
                left_img = cv2.resize(left_img, (width//DOWNSAMPLE_FACTOR, height//DOWNSAMPLE_FACTOR),
                                    interpolation=cv2.INTER_CUBIC)
                right_img = cv2.resize(right_img, (width//DOWNSAMPLE_FACTOR, height//DOWNSAMPLE_FACTOR),
                                    interpolation=cv2.INTER_CUBIC)
                # clip images
                left_img = np.clip(left_img, 0, 1)
                right_img = np.clip(right_img, 0, 1)
                
                K_l[:2] /= DOWNSAMPLE_FACTOR
                K_r[:2] /= DOWNSAMPLE_FACTOR
                P_l[:2] /= DOWNSAMPLE_FACTOR
                P_r[:2] /= DOWNSAMPLE_FACTOR

                # map from lidar to cam
                Tr_cam_velo = np.zeros((4,4))
                Tr_cam_velo[3,3] = 1
                Tr_cam_velo[:3,:3] = f["calib_lidar_to_cam/R"][:]
                Tr_cam_velo[:3,3] = f["calib_lidar_to_cam/t"][:]
                
                # map from right cam to left cam
                Tr_lcam_rcam = np.zeros((4,4))
                Tr_lcam_rcam[3,3] = 1
                Tr_lcam_rcam[:3,:3] = f["calib_cam_r_to_cam_l/R"][:]
                Tr_lcam_rcam[:3,3] = f["calib_cam_r_to_cam_l/t"][:]
                
                # map from cam to plane
                Tr_plane_cam = np.zeros((4,4))
                Tr_plane_cam[3,3] = 1
                Tr_plane_cam[:3,:3] = f["calib_cam_to_plane/R"][:]
                Tr_plane_cam[:3,3] = f["calib_cam_to_plane/t"][:]

                # lidar
                lidar_pc = f["pointcloud"][:]

                plt.imsave(os.path.join(left_img_dir, hdf5_file[:-5] + ".png"), left_img)
                plt.imsave(os.path.join(right_img_dir, hdf5_file[:-5] + ".png"), right_img)

                np.savez(os.path.join(annotation_dir, hdf5_file[:-5] + ".npz"), 
                                        label_str=label_str, 
                                        label_int=label_int,
                                        bbox_label_int=bbox_label_int, 
                                        bbox_label_str=bbox_label_str,
                                        bbox_location=bbox_location, 
                                        bbox_dimensions=bbox_dimensions,
                                        bbox_rotation_y=bbox_rotation_y, 
                                        bbox_visibility=bbox_visibility,
                                        bbox_occlusion=bbox_occlusion)

                np.savez(os.path.join(calib_dir, hdf5_file[:-5] + ".npz"), 
                                    K_l=K_l, K_r=K_r, 
                                    D_l=D_l, D_r=D_r, 
                                    R_l=R_l, R_r=R_r, 
                                    P_l=P_l, P_r=P_r,
                                    Tr_cam_velo=Tr_cam_velo,
                                    Tr_lcam_rcam=Tr_lcam_rcam,
                                    Tr_plane_cam=Tr_plane_cam)
                
                np.savez(os.path.join(lidar_dir, hdf5_file[:-5] + ".npz"),
                                    lidar_pc=lidar_pc)



def main():
    fire.Fire(extract_data)


if __name__ == "__main__":
    main()
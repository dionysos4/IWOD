import os
from sklearn import pipeline
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import cv2
from torchvision import models, transforms
import torch
import sys
import copy
from tqdm import tqdm
from iwod.utils.waterplane import TPlaneCamEstimator


class LCDDataset(Dataset):
    """
    Pytorch lake constance detection dataset

    Parameters
    ----------
    dataset_dir : str
        directory where the dataset is stored
    training : string
        choose train, valid or test
    transform : torch transform objects
        transformation which are applied to the input
    depth : str
        type of depth image to be generated, either "lidar" or "stereo"
    
    return: dict
        {'left_img'; 'right_img', 'calibration', 'annotations', 'filename', 'idx'}
    """

    def __init__(self, dataset_dir, training, transform, depth="lidar", water_detection=False, custom_T_plane_cam=None):
        
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.custom_T_plane_cam = custom_T_plane_cam
        self.data_dir = os.path.join(dataset_dir, training)
        if not os.path.exists(self.data_dir):
            raise Exception("Incorrect training parameter (train, valid, test), your input: " + training)
        self.left_img_dir = os.path.join(self.data_dir, "left_img")
        self.right_img_dir = os.path.join(self.data_dir, "right_img")
        self.calib_dir = os.path.join(self.data_dir, "calibration")
        self.annotations_dir = os.path.join(self.data_dir, "annotations")
        self.depth = depth
        if self.depth == "lidar":
            self.depth_dir = os.path.join(self.data_dir, "lidar")
        else:
            self.depth_dir = os.path.join(self.data_dir, "stereo")
        
        self.left_img_list = sorted(os.listdir(self.left_img_dir))
        self.right_img_list = sorted(os.listdir(self.right_img_dir))
        self.depth_list = sorted(os.listdir(self.depth_dir))


        self.K_l = {}
        self.K_r = {}
        self.D_l = {}
        self.D_r = {}
        self.P_l = {}
        self.P_r = {}
        self.R_l = {}
        self.R_r = {}
        self.Tr_plane_cam = {}
        self.Tr_cam_velo = {}
        self.Tr_lcam_rcam = {}
        
        self.bbox_label_int = {}
        self.bbox_label_str = {}
        self.bbox_location = {}
        self.bbox_dimensions = {}
        self.bbox_rotation_y = {}
        self.bbox_visibility = {}
        self.bbox_occlusion = {}
        
        self.__metadata_to_ram()
        self.water_detection = water_detection
        # if water_detection:
        #     self.detected_Tr_plane_cam = self.get_estimated_transformations()
        if water_detection:
            self.T_plane_cam_estimator = TPlaneCamEstimator()


    def __len__(self):
        return len(self.left_img_list)


    def __getitem__(self, idx):
        # load already undistorted images
        left_img_file = self.left_img_list[idx]
        right_img_file = self.right_img_list[idx]
        if left_img_file != right_img_file:
            raise Exception("Left and right image do not match")
        depth_file = self.depth_list[idx]
        file_key = left_img_file.split(".")[0]
        left_img_path = os.path.join(self.left_img_dir, left_img_file)
        right_img_path = os.path.join(self.right_img_dir, right_img_file)
        depth_path = os.path.join(self.depth_dir, depth_file)

        img_l = plt.imread(left_img_path)[:,:,:3]
        img_r = plt.imread(right_img_path)[:,:,:3]

        K_l = self.K_l[file_key]
        K_r = self.K_r[file_key]
        D_l = self.D_l[file_key]
        D_r = self.D_r[file_key]
        R_l = self.R_l[file_key]
        R_r = self.R_r[file_key]
        P_l = self.P_l[file_key]
        P_r = self.P_r[file_key]

        # load annotations
        annotations = np.zeros((7, 10))

        for i in range(len(self.bbox_location[file_key])):
            bbox_label_int = self.bbox_label_int[file_key][i]
            bbox_location = self.bbox_location[file_key][i]
            bbox_dimensions = self.bbox_dimensions[file_key][i]
            bbox_rotation_y = self.bbox_rotation_y[file_key][i]
            bbox_occlusion = self.bbox_occlusion[file_key][i]
            bbox_visibility = self.bbox_visibility[file_key][i]
            annotations[i][0] = bbox_label_int
            annotations[i][1:4] = bbox_location
            annotations[i][4:7] = bbox_dimensions
            annotations[i][7] = bbox_rotation_y
            annotations[i][8] = bbox_occlusion
            annotations[i][9] = bbox_visibility
            
            # reorder the classes and use only th class ship for ferry, catamaran, motor vessel
            # new order motorboat, sailboat, sailboat under bare poles, stand-up-paddle, ship, pedal boat, pile
            if annotations[i][0] == 5:
                annotations[i][0] = 4
            elif annotations[i][0] == 6:
                annotations[i][0] = 5
            elif annotations[i][0] == 7:
                annotations[i][0] = 4
            elif annotations[i][0] == 8:
                annotations[i][0] = 6
            

        # Tr_plane_cam
        if self.water_detection:
            #Tr_plane_cam_hom = self.detected_Tr_plane_cam[idx]
            Tr_plane_cam_hom = self.T_plane_cam_estimator.estimate_T_plane_cam(K_l, K_r, np.array([0,0,0,0,0]), np.array([0,0,0,0,0]), 
                                                         R_l, R_r, P_l, P_r, img_l, img_r)
        else:
            Tr_plane_cam_hom = self.Tr_plane_cam[file_key]

        if self.custom_T_plane_cam is not None:
            Tr_plane_cam_hom = self.custom_T_plane_cam


        # Since this is the transformation from stereo rectified camera to the plane, we need to adapt it
        R_l_hom = np.eye(4)
        R_l_hom[:3, :3] = R_l
        Tr_plane_cam_hom = Tr_plane_cam_hom @ R_l_hom
        Tr_cam_plane_hom = np.linalg.inv(Tr_plane_cam_hom).astype("float32")

        # Tr_rcam_lcam
        Tr_rcam_lcam_hom = np.linalg.inv(self.Tr_lcam_rcam[file_key])

        # Tr_cam_velo
        Tr_cam_velo_hom = self.Tr_cam_velo[file_key]

        # lidar
        if self.depth == "lidar":
            lidar_pc = np.load(depth_path)["lidar_pc"]
        else:
            # load point cloud pcd file
            pcd = o3d.io.read_point_cloud(depth_path)
            # convert to numpy array
            lidar_pc = np.asarray(pcd.points)
            # add intensity to point cloud
            if lidar_pc.shape[1] == 3:
                lidar_pc = np.hstack((lidar_pc, np.ones((lidar_pc.shape[0], 1))))

        # get depth image
        T_lcam_lcam = np.eye(4)
        depth_img_left = self.__get_depth_img(lidar_pc[:,:4], K_l, R_l, T_lcam_lcam[:3], Tr_cam_velo_hom[:3], img_l.shape)
        depth_img_right = self.__get_depth_img(lidar_pc[:,:4], K_r, R_l, Tr_rcam_lcam_hom[:3], Tr_cam_velo_hom[:3], img_r.shape)

        img_targets = [img_r]
        transform_mats = [Tr_rcam_lcam_hom[:3].astype("float32")]
        K_list = [K_r.astype("float32")]
        K_ref_inv = np.linalg.inv(K_l).astype("float32")
        depth_img_right = [depth_img_right]

        sample = (img_l, img_targets, transform_mats, K_list, K_ref_inv, Tr_cam_plane_hom), (depth_img_left, depth_img_right), annotations

        if self.transform:
            sample = self.transform(sample)
        return sample


    def __metadata_to_ram(self):
        """
        stores annotations and calibration data into ram
        """
        calib_files = sorted(os.listdir(self.calib_dir))
        annotations_files = sorted(os.listdir(self.annotations_dir))
        for calib_file, annotations_file in zip(calib_files, annotations_files):
            if calib_file != annotations_file:
                raise Exception("Calibration file and annotations file do not match")
            file_key = calib_file.split(".")[0]
            calib_file_path = os.path.join(self.calib_dir, calib_file)
            annotation_file_path = os.path.join(self.annotations_dir, annotations_file)
            calib_data = np.load(calib_file_path)
            annotation_data = np.load(annotation_file_path)
            self.K_l[file_key] = calib_data["K_l"]
            self.K_r[file_key] = calib_data["K_r"]
            self.D_l[file_key] = calib_data["D_l"]
            self.D_r[file_key] = calib_data["D_r"]
            self.P_l[file_key] = calib_data["P_l"]
            self.P_r[file_key] = calib_data["P_r"]
            self.R_l[file_key] = calib_data["R_l"]
            self.R_r[file_key] = calib_data["R_r"]
            self.Tr_plane_cam[file_key] = calib_data["Tr_plane_cam"]
            self.Tr_cam_velo[file_key] = calib_data["Tr_cam_velo"]
            self.Tr_lcam_rcam[file_key] = calib_data["Tr_lcam_rcam"]

            self.bbox_label_int[file_key] = annotation_data["bbox_label_int"]
            self.bbox_label_str[file_key] = annotation_data["bbox_label_str"]
            self.bbox_location[file_key] = annotation_data["bbox_location"]
            self.bbox_dimensions[file_key] = annotation_data["bbox_dimensions"]
            self.bbox_rotation_y[file_key] = annotation_data["bbox_rotation_y"]
            self.bbox_visibility[file_key] = annotation_data["bbox_visibility"]
            self.bbox_occlusion[file_key] = annotation_data["bbox_occlusion"]

    
    def __get_depth_img(self, pc, K, R_rect, T_target_cam, T_cam_velo, img_shape):
        """
        Get the depth image from the point cloud
        parameters:
        -----------
        pc : np.array
            point cloud
        K : np.array
            projection matrix of the target camera
        R_rect : np.array
            stereo rectification matrix
        T_target_cam : np.array
            transformation matrix from cam0 to target camera
        T_cam_velo : np.array
            transformation matrix from velodyne to camera
        img_shape : tuple
            shape of the image
        return:
        -------
        depth_img : np.array
            depth image
        """
        depth_img = np.zeros(img_shape[:2])
        pc = pc[pc[:,0] > 0.1]
        pc[:,3] = 1

        # einsum 3x4 * n*4
        pc_in_cam0 = np.einsum("ij,nj->ni", T_cam_velo, pc)
        # transform in stereo unrectified coordinate system
        pc_in_cam0 = np.einsum("ij,nj->ni", R_rect.T, pc_in_cam0)
        # make it homogeneous by adding 1 in the last column        
        pc_in_cam0 = np.hstack((pc_in_cam0, np.ones((pc_in_cam0.shape[0], 1))))
        # transform to target camera
        pc_in_target_cam = np.einsum("ij,nj->ni", T_target_cam, pc_in_cam0)
        # project the points to the image plane
        pc_in_img = np.einsum("ij,nj->ni", K, pc_in_target_cam)
        # normalize the points
        z = pc_in_img[:,2]
        pc_in_img = pc_in_img / pc_in_img[:,2][:,None]
        # round the points
        pc_in_img = np.round(pc_in_img).astype("int")
        # remove points that are outside the image
        mask = (pc_in_img[:,0] >= 0) & (pc_in_img[:,0] < img_shape[1]) & \
                (pc_in_img[:,1] >= 0) & (pc_in_img[:,1] < img_shape[0])
        pc_in_img = pc_in_img[mask]
        z = z[mask]
        # get the depth
        depth_img[pc_in_img[:,1], pc_in_img[:,0]] = z

        return depth_img


# a = LCDDataset("/mnt/deepdoubt/dennis_data/extracted_dataset_factor_2", "train", None, water_detection=True)
# c = a[0]
# b = 1
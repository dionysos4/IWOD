import os
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import copy

class KittiMultiviewDataset(Dataset):
    """
    Kitti stereo dataset

    Parameters
    ----------
    dataset_dir : str
        directory where the dataset is stored
    training : string
        choose train, valid, test or valid_valid, valid_test. The last two are the splits of the validation set into validation and test
    transform : torch transform objects
        transformation which are applied to the input
    cfg : dict
        configuration dictionary
    cameras : dict
        which cameras to use, e.g. {"cam0" : True, "cam1" : True, "cam2" : False, "cam3" : False}
    
    return: dict
        {'left_img'; 'right_img', 'calibration', 'annotations', 'filename', 'idx'}
    """

    def __init__(self, dataset_dir, training, transform, cfg, cameras={"cam0" : True, "cam1" : True, "cam2" : False, "cam3" : False}):
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.data_dir = dataset_dir
        self.training = training

        self.calib_dir = os.path.join(self.data_dir, "data_object_calib/training/calib")
        self.annotations_dir = os.path.join(self.data_dir, "data_object_label_2/training/label_2")
        self.lidar_dir = os.path.join(self.data_dir, "data_object_velodyne/training/velodyne")

        self.cameras = cameras

        if cameras["cam0"]:
            self.cam0_img_dir = os.path.join(self.data_dir, "data_object_image_0/training/image_0")
        if cameras["cam1"]:
            self.cam1_img_dir = os.path.join(self.data_dir, "data_object_image_1/training/image_1")
        if cameras["cam2"]:
            self.cam2_img_dir = os.path.join(self.data_dir, "data_object_image_2/training/image_2")
        if cameras["cam3"]:
            self.cam3_img_dir = os.path.join(self.data_dir, "data_object_image_3/training/image_3")


        self.__classes =  {'Car': 0, 'Van': 1, 'Truck': 2, 'Pedestrian': 3, 'Person_sitting': 4, 'Cyclist': 5, 'Tram': 6, 'Misc': 7, 'DontCare': 8}

        ## read train split from txt file
        with open(os.path.join("/home/dennis/git_repos/multiview_detection_v3/eval_detection/eval_files/split/train.txt"), "r") as f:
            self.train_list = f.read().splitlines()
        
        ## read valid split from txt file
        if self.training == "valid":
            with open(os.path.join("/home/dennis/git_repos/multiview_detection_v3/eval_detection/eval_files/split/val.txt"), "r") as f:
                self.valid_list = f.read().splitlines()
        elif self.training == "valid_valid":
            with open(os.path.join("/home/dennis/git_repos/multiview_detection_v3/eval_detection/eval_files/split/val_val_split.txt"), "r") as f:
                self.valid_list = f.read().splitlines()

        if self.training == "test":
            with open(os.path.join("/home/dennis/git_repos/multiview_detection_v3/eval_detection/eval_files/split/val.txt"), "r") as f:
                self.test_list = f.read().splitlines()
        elif self.training == "valid_test":
            with open(os.path.join("/home/dennis/git_repos/multiview_detection_v3/eval_detection/eval_files/split/val_test_split.txt"), "r") as f:
                self.test_list = f.read().splitlines()

        
        # filter train and valid list ######
        remove_list = []
        for f in self.train_list:
            labels = self.__get_labels(os.path.join(self.annotations_dir, f + ".txt"))
            if len(labels) == 0:
                remove_list.append(f)
                continue
        # remove all elements in remove_list from train_list
        for i in remove_list:
            self.train_list.remove(i)

        if self.training == "valid":
            remove_list = []
            for f in self.valid_list:
                labels = self.__get_labels(os.path.join(self.annotations_dir, f + ".txt"))
                if len(labels) == 0:
                    remove_list.append(f)
                    continue
            for i in remove_list:
                self.valid_list.remove(i)
            #####################################

    
    def __len__(self):
        if self.training == "train":
            return len(self.train_list)
        elif self.training == "valid" or self.training == "valid_valid":
            return len(self.valid_list)
        elif self.training == "test" or self.training == "valid_test":
            return len(self.test_list)
        else:
            raise ValueError("training must be train or valid or test")
        

    def __getitem__(self, idx):
        if self.training == "train":
            img_id = self.train_list[idx]
        elif self.training == "valid" or self.training == "valid_valid":
            img_id = self.valid_list[idx]
        elif self.training == "test" or self.training == "valid_test":
            img_id = self.test_list[idx]
        else:
            raise ValueError("training must be train or valid")
        
        if self.cameras["cam0"]:
            cam0_img = plt.imread(os.path.join(self.cam0_img_dir, img_id + ".png"))[:,:,0]
            cam0_img = np.expand_dims(cam0_img, axis=2)
        if self.cameras["cam1"]:
            cam1_img = plt.imread(os.path.join(self.cam1_img_dir, img_id + ".png"))[:,:,0]
            cam1_img = np.expand_dims(cam1_img, axis=2)
        if self.cameras["cam2"]:
            cam2_img = plt.imread(os.path.join(self.cam2_img_dir, img_id + ".png"))[:,:,:3]
        if self.cameras["cam3"]:
            cam3_img = plt.imread(os.path.join(self.cam3_img_dir, img_id + ".png"))[:,:,:3]

        pc = np.fromfile(os.path.join(self.lidar_dir, img_id + ".bin"), dtype=np.float32).reshape(-1, 4)


        K_dict, R0_rect, R_dict, T_dict, R, T = self.__get_calibration(os.path.join(self.calib_dir, img_id + ".txt"), self.cameras)
        T_cam_velo = self.__get_transformation_matrix(R, T)[:3]
        
        # if cam2 is used cam2 is the most left camera
        if self.cameras["cam2"]:
            depth_img_left = self.__get_depth_img(pc, K_dict["K2"], R_dict["R2"], T_dict["T2"], T_cam_velo, cam2_img.shape)
            depth_img_right = []
            if self.cameras["cam0"]:
                depth_img_right.append(self.__get_depth_img(pc, K_dict["K0"], R_dict["R0"], T_dict["T0"], T_cam_velo, cam0_img.shape))
            if self.cameras["cam3"]:
                depth_img_right.append(self.__get_depth_img(pc, K_dict["K3"], R_dict["R3"], T_dict["T3"], T_cam_velo, cam3_img.shape))
            if self.cameras["cam1"]:
                depth_img_right.append(self.__get_depth_img(pc, K_dict["K1"], R_dict["R1"], T_dict["T1"], T_cam_velo, cam1_img.shape))
        elif self.cameras["cam0"]:
            depth_img_left = self.__get_depth_img(pc, K_dict["K0"], R_dict["R0"], T_dict["T0"], T_cam_velo, cam0_img.shape)
            depth_img_right = []
            if self.cameras["cam3"]:
                depth_img_right.append(self.__get_depth_img(pc, K_dict["K3"], R_dict["R3"], T_dict["T3"], T_cam_velo, cam3_img.shape))
            if self.cameras["cam1"]:
                depth_img_right.append(self.__get_depth_img(pc, K_dict["K1"], R_dict["R1"], T_dict["T1"], T_cam_velo, cam1_img.shape))

        # get transformation matrices
        T_cam0_world = self.__get_transformation_matrix(R_dict["R0"], T_dict["T0"])
        T_cam1_cam0 = self.__get_transformation_matrix(R_dict["R1"], T_dict["T1"])
        T_cam2_cam0 = self.__get_transformation_matrix(R_dict["R2"], T_dict["T2"])
        T_cam3_cam0 = self.__get_transformation_matrix(R_dict["R3"], T_dict["T3"])
        
        # get the annotations
        annotation = self.__get_labels(os.path.join(self.annotations_dir, img_id + ".txt"))
        
        # transform annotations from rect cam0 to unrect cam2. So that cam2 is world frame
        if len(annotation) > 0:
            center = annotation[:,11:14]
            center_unrect = np.einsum("ij,nj->ni", R0_rect.T, center)
            # check which is the most left camera
            if self.cameras["cam2"]:
                center_in_cam_coords = np.einsum("ij,nj->ni", T_cam2_cam0[:3], np.hstack((center_unrect, np.ones((center_unrect.shape[0], 1)))))
            else:
                center_in_cam_coords = np.einsum("ij,nj->ni", T_cam0_world[:3], np.hstack((center_unrect, np.ones((center_unrect.shape[0], 1)))))
            annotation[:,11:14] = center_in_cam_coords
        annotations = np.zeros((24, 8))
        
        if len(annotation) > 0:
            for j in range(len(annotation)):
                h, w, l, x, y, z, rot = (annotation[j][8], 
                                        annotation[j][9], 
                                        annotation[j][10], 
                                        annotation[j][11], 
                                        annotation[j][12], 
                                        annotation[j][13],
                                        annotation[j][14])
                # to mslp angle
                rot = self.kitti_to_lcod_angle(rot)

                annotations[j][0] = annotation[j][0]
                annotations[j][1] = x
                annotations[j][2] = y
                annotations[j][3] = z
                annotations[j][4] = w
                annotations[j][5] = h
                annotations[j][6] = l
                annotations[j][7] = rot


        # compute Transformations to the reference matrix if cam2 is used it is the reference matrix else cam0
        T_cam0_cam2 = np.linalg.inv(T_cam2_cam0)
        T_cam1_cam2 = T_cam1_cam0 @ np.linalg.inv(T_cam2_cam0)
        T_cam3_cam2 = T_cam3_cam0 @ np.linalg.inv(T_cam2_cam0)
 

        # we assume either cam2 or cam0 is the reference camera
        if self.cameras["cam2"]:
            K_ref_inv = np.linalg.inv(K_dict["K2"]).astype("float32")
            img_ref = cam2_img.astype("float32")
            # add all other cameras wihich in dict are true
            img_targets = []
            transform_mats = []
            K_list = []
            if self.cameras["cam0"]:
                img_targets.append(cam0_img.astype("float32"))
                transform_mats.append(T_cam0_cam2.astype("float32"))
                K_list.append(K_dict["K0"].astype("float32"))
            if self.cameras["cam3"]:
                img_targets.append(cam3_img.astype("float32"))
                transform_mats.append(T_cam3_cam2.astype("float32"))
                K_list.append(K_dict["K3"].astype("float32"))
            if self.cameras["cam1"]:
                img_targets.append(cam1_img.astype("float32"))
                transform_mats.append(T_cam1_cam2.astype("float32"))
                K_list.append(K_dict["K1"].astype("float32"))
        else:
            K_ref_inv = np.linalg.inv(K_dict["K0"]).astype("float32")
            img_ref = cam0_img.astype("float32")
            img_targets = []
            transform_mats = []
            K_list = []
            if self.cameras["cam3"]:
                img_targets.append(cam3_img.astype("float32"))
                transform_mats.append(T_cam3_cam2.astype("float32"))
                K_list.append(K_dict["K3"].astype("float32"))
            if self.cameras["cam1"]:
                img_targets.append(cam1_img.astype("float32"))
                transform_mats.append(T_cam1_cam2.astype("float32"))
                K_list.append(K_dict["K1"].astype("float32"))


        depth_img_left = depth_img_left.astype("float32")
        [depth_img_right[i].astype("float32") for i in range(len(depth_img_right))]
        annotations = annotations.astype("float32")
        
        sample = (img_ref, img_targets, transform_mats, K_list, K_ref_inv), (depth_img_left, depth_img_right), annotations
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    

    def __get_calibration(self, calibration_path, cameras):
        """
        Read calibration file and return the calibration matrices
        parameters:
        -----------
        calibration_path : str
            path to the calibration file
        cameras : dict
            which cameras are used
        return:
        -------
        K_dict : dict
            dict if intrinsics matrices
        R_cam0 : np.array
            Rectification matrix for cam0
        R_dict : dict
            dict of rotation matrices
        T_dict : dict
            dict of translation vectors
        R : np.array
            rotation matrix for velo to cam0
        T : np.array
            translation vector for velo to cam0
        """
        with open(calibration_path, "r") as file:
            K_dict = {}
            R_dict = {}
            T_dict = {}
            for line in file:
                line = line.rstrip().split(" ")
                if line[0] == "K0:":
                    K_dict["K0"] = np.array(line[1:]).astype("float").reshape(3, 3)
                if line[0] == "K1:":
                    K_dict["K1"] = np.array(line[1:]).astype("float").reshape(3, 3)
                if line[0] == "K2:":
                    K_dict["K2"] = np.array(line[1:]).astype("float").reshape(3, 3)
                if line[0] == "K3:":
                    K_dict["K3"] = np.array(line[1:]).astype("float").reshape(3, 3)
                
                if line[0] == "R0_rect:":
                    R0_rect = np.array(line[1:]).astype("float").reshape(3, 3)
                
                if line[0] == "R0:":
                    R_dict["R0"] = np.array(line[1:]).astype("float").reshape(3, 3)
                if line[0] == "R1:":
                    R_dict["R1"] = np.array(line[1:]).astype("float").reshape(3, 3)
                if line[0] == "R2:":
                    R_dict["R2"] = np.array(line[1:]).astype("float").reshape(3, 3)
                if line[0] == "R3:":
                    R_dict["R3"] = np.array(line[1:]).astype("float").reshape(3, 3)

                if line[0] == "T0:":
                    T_dict["T0"] = np.array(line[1:]).astype("float")
                if line[0] == "T1:":
                    T_dict["T1"] = np.array(line[1:]).astype("float")
                if line[0] == "T2:":
                    T_dict["T2"] = np.array(line[1:]).astype("float")
                if line[0] == "T3:":
                    T_dict["T3"] = np.array(line[1:]).astype("float")
                
                if line[0] == "R:":
                    R = np.array(line[1:]).astype("float").reshape(3, 3)
                if line[0] == "T:":
                    T = np.array(line[1:]).astype("float")
        return K_dict, R0_rect, R_dict, T_dict, R, T
    

    def __get_depth_img(self, pc, K, R_camt, T_camt, T_cam_velo, img_shape):
        """
        Get the depth image from the point cloud
        parameters:
        -----------
        pc : np.array
            point cloud
        K : np.array
            projection matrix of the target camera
        R_camt : np.array
            rotation matrix to project from cam0 to target camera
        T_camt : np.array
            translation vector to project from cam0 to target camera
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
        # make pc_in_rect homogeneous by adding 1 in the last column
        pc_in_cam0 = np.hstack((pc_in_cam0, np.ones((pc_in_cam0.shape[0], 1))))
        # transform to target camera
        T_camt_cam0 = np.zeros((3,4))
        T_camt_cam0[:3,:3] = R_camt
        T_camt_cam0[:,3] = T_camt
        pc_in_target_cam = np.einsum("ij,nj->ni", T_camt_cam0, pc_in_cam0)
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
    


    def __get_labels(self, label_path):
        """
        Read label file and return the labels
        parameters:
        -----------
        label_path : str
            path to the label file
        return:
        -------
        label_list : np.array
            list of labels
        """
        label_list = []
        with open(label_path, "r") as file:
            for i, line in enumerate(file):
                line = line.rstrip().split(" ")
                line[0] = self.__classes[line[0]]
                # remove small objects
                # if float(line[7]) - float(line[5]) < 10:
                #     continue
                # remove truncations
                if float(line[1]) >= 0.98:
                    continue
                # remove unknown occlusions
                # if int(line[2]) == 3:
                #     continue
                # only class 0 is considered
                if line[0] == 0:
                    label_list.append(np.array(line).astype("float"))
                # add Van to Car class because if the detector predicts a Van it penelizes the loss (very high)
                if line[0] == 1:
                    line[0] = 0
                    label_list.append(np.array(line).astype("float"))
        return np.array(label_list)
    
    
    def __check_imgs(self, img_l, img_r):
        """
        Check if the images have standard size
        parameters:
        -----------
        img_l : np.array
            left image
        img_r : np.array
            right image
        return:
        -------
        img_l : np.array
            left image
        img_r : np.array
            right image
        """

        if img_l.shape[0] != 376:
            img_l = np.vstack((img_l, np.zeros((376 - img_l.shape[0], img_l.shape[1], img_l.shape[2]))))
        if img_r.shape[0] != 376:
            img_r = np.vstack((img_r, np.zeros((376 - img_r.shape[0], img_r.shape[1], img_r.shape[2]))))
        if img_l.shape[1] != 1242:
            img_l = np.hstack((img_l, np.zeros((img_l.shape[0], 1242 - img_l.shape[1], img_l.shape[2]))))
        if img_r.shape[1] != 1242:
            img_r = np.hstack((img_r, np.zeros((img_r.shape[0], 1242 - img_r.shape[1], img_r.shape[2]))))

        return img_l, img_r
    

    def __get_rotation_matrix(self, y_rot):
        """
        Get the rotation matrix for the y rotation
        parameters:
        -----------
        y_rot : float
            rotation angle
        return:
        -------
        np.array
            rotation matrix
        """
        return np.array([[np.cos(y_rot), 0, np.sin(y_rot)], [0, 1, 0], [-np.sin(y_rot), 0, np.cos(y_rot)]])


    ### converts the kitti angle to mslp angle
    def kitti_to_lcod_angle(self, y_rot):
        """
        Convert the kitti angle to mslp angle
        parameters:
        -----------
        y_rot : float
            rotation angle
        return:
            rotation angle
        -------
        """
        rot_mat = self.__get_rotation_matrix(y_rot)
        cam_mat = np.array([[1,0,0], [0,-1,0], [0,0,1]])
        x_axis = (rot_mat @ cam_mat)[:,0]
        x_axis = x_axis / np.linalg.norm(x_axis)
        angle = np.arccos(x_axis @ np.array([0,0,1]))
        if x_axis[0] < 0:
            return -angle
        if y_rot > np.pi/2:
            return angle
        return angle
    

    def __get_transformation_matrix(self, R, T):
        """
        Returns the 4x4 transformation matrix
        parameters:
        -----------
        R : np.array
            rotation matrix
        T : np.array
            translation vector
        return:
        -------
        np.array
            4x4 transformation matrix
        """
        Tr = np.zeros((4,4))
        Tr[:3,:3] = R
        Tr[:3,3] = T
        Tr[3,3] = 1
        return Tr
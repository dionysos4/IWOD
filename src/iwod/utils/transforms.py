import torch
import numpy as np
import os
import matplotlib.pyplot as plt


class Disp2Depth(torch.nn.Module):
    """
    Convert disparity to depth
    """
    def __init__(self):
        pass

    def __call__(self, sample):
        img_l, img_r, T_camr_caml, P_r, P_l = sample[0]
        disp = sample[1]
        depth = np.zeros_like(disp)
        mask = disp > 0
        depth[mask] = (np.linalg.inv(P_l[:3,:3])[0,0] * -T_camr_caml[0][0,3]) / disp[mask]
        return sample[0], depth


class Normalize(torch.nn.Module):
    """
    Normalize the image with the mean and standard deviation of the images
    * params:
        - mean_ref: mean of the reference image
        - std_ref: standard deviation of the reference image
        - means: list of means of the other images
        - stds: list of standard deviations of the other images
    * return:
        - normalized images
    """
    def __init__(self, mean_ref, std_ref, means, stds):
        self.mean_ref = np.array(mean_ref).astype("float32")
        self.std_ref = np.array(std_ref).astype("float32")
        self.means = []
        self.stds = []
        for i in range(len(means)):
            self.means.append(np.array(means[i]).astype("float32"))
            self.stds.append(np.array(stds[i]).astype("float32"))

    def __call__(self, sample):
        if len(sample[0]) == 5:
            img_l, img_r, T_camr_caml, P_r, P_l = sample[0]
        else:
            img_l, img_r, T_camr_caml, P_r, P_l, T_cam_plane = sample[0]
        img_l = (img_l - self.mean_ref) / self.std_ref
        for i in range(len(img_r)):
            img_r[i] = (img_r[i] - self.means[i]) / self.stds[i]
        
        if len(sample[0]) == 5:
            return (img_l, img_r, T_camr_caml, P_r, P_l), sample[1], sample[2]
        else:
            return (img_l, img_r, T_camr_caml, P_r, P_l, T_cam_plane), sample[1], sample[2]


class ToTensor(torch.nn.Module):
    """
    Convert the numpy arrays to torch tensors
    """
    def __init__(self):
        pass

    def __call__(self, sample):
        if len(sample[0]) == 5:
            img_l, img_r, T_camr_caml, P_r, P_l = sample[0]
        else:
            img_l, img_r, T_camr_caml, P_r, P_l, T_cam_plane = sample[0]
            T_cam_plane = torch.from_numpy(T_cam_plane[:3])
        img_l = torch.from_numpy(img_l).permute(2,0,1)
        for i in range(len(img_r)):
            img_r[i] = torch.from_numpy(img_r[i]).permute(2,0,1)
            T_camr_caml[i] = torch.from_numpy(T_camr_caml[i][:3])
            P_r[i] = torch.from_numpy(P_r[i])
        P_l = torch.from_numpy(P_l)

        for i in range(len(sample[1][1])):
            sample[1][1][i] = torch.from_numpy(sample[1][1][i].copy())

        if len(sample[0]) == 5:
            s1 =  (img_l, img_r, T_camr_caml, P_r, P_l)
        else:
            s1 =  (img_l, img_r, T_camr_caml, P_r, P_l, T_cam_plane)
        
        return s1, \
                (torch.from_numpy(sample[1][0].copy()), sample[1][1]), \
                torch.from_numpy(sample[2])
    

class PadImages(torch.nn.Module):
    """
    Pad the images to the desired size
    * params:
        - size: desired size
    * return:
        - padded images
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        if len(sample[0]) == 5:
            img_l, img_r, T_camr_caml, P_r, P_l = sample[0]
        else:
            img_l, img_r, T_camr_caml, P_r, P_l, T_cam_plane = sample[0]
        padded_img_l = np.zeros((self.size[0], self.size[1], img_l.shape[2]), dtype=np.float32)
        hl, wl = img_l.shape[0], img_l.shape[1]
        padded_img_l[:hl,:wl] = img_l

        padded_img_r = []
        for i in range(len(img_r)):
            padded_img_r.append(np.zeros((self.size[0], self.size[1], img_r[i].shape[2]), dtype=np.float32))
            h, w = img_r[i].shape[0], img_r[i].shape[1]
            padded_img_r[i][:h,:w] = img_r[i]

        depth_img_l = sample[1][0]
        padded_depth_img_l = np.zeros((self.size[0], self.size[1]), dtype=np.float32)
        h, w = depth_img_l.shape[0], depth_img_l.shape[1]
        padded_depth_img_l[:h,:w] = depth_img_l

        depth_img_r = sample[1][1]
        padded_depth_img_r = []
        for i in range(len(depth_img_r)):
            padded_depth_img = np.zeros((self.size[0], self.size[1]), dtype=np.float32)
            h, w = depth_img_r[i].shape[0], depth_img_r[i].shape[1]
            padded_depth_img[:h,:w] = depth_img_r[i]
            padded_depth_img_r.append(padded_depth_img)
        
        if len(sample[0]) == 5:
            return (padded_img_l, padded_img_r, T_camr_caml, P_r, P_l), (padded_depth_img_l, padded_depth_img_r), sample[2]
        else:
            return (padded_img_l, padded_img_r, T_camr_caml, P_r, P_l, T_cam_plane), (padded_depth_img_l, padded_depth_img_r), sample[2]


class HorizontalFlip(torch.nn.Module):
    """
    Flip the images horizontally
    * return:
        - flipped images
        - flipped depth images
        - flipped annotations
    """

    def __init__(self):
        pass

    def __call__(self, sample):
        img_l, img_r, T_camr_caml, P_r, P_l_inv = sample[0]
        # Works only for two views
        img_l_flip = img_r[0][:,::-1]
        img_r_flip = img_l[:,::-1]

        P_l = np.linalg.inv(P_l_inv[:3,:3])
        P_l[0,2] = img_l.shape[1] - P_l[0,2]
        new_P_r = np.zeros_like(P_l_inv)
        new_P_r[:3,:3] = P_l
        new_P_r[:3,3] = P_l_inv[:3,3]
        P_r[0,2] = img_r[0].shape[1] - P_r[0,2]
        new_P_l = np.zeros_like(P_r)
        new_P_l[:3,:3] = np.linalg.inv(P_r[:3,:3])
        new_P_l[:3,3] = P_r[:3,3]

        depth_img_l = sample[1][0]
        depth_img_r = sample[1][1]
        depth_img_l_flip = depth_img_r[:,::-1]
        depth_img_r_flip = depth_img_l[:,::-1]

        annotations = sample[2]

        mask = annotations[:,1:4].sum(axis=1) > 0
        annotations[mask,1:4] += T_camr_caml[0][:,3]

        annotations[:,1] *= -1
        annotations[:,7] *= -1

        return (img_l_flip, [img_r_flip], T_camr_caml, new_P_r, new_P_l), (depth_img_l_flip, depth_img_r_flip), annotations
    

class HorizontalFlipUnrect(torch.nn.Module):
    """
    Flip the images horizontally
    * return:
        - flipped images
        - flipped depth images
        - flipped annotations
        - first flip the right camera with T @ (-1, 0, 0) and R' = [[-1,0,0], [0,1,0], [0,0,1]] --> R'@R@R'
        - then flip X and theta from annotations
        - then transform annotations to new left Camera with R@P+T
        - then new R and T is R.T and -R.T@T
    """

    def __init__(self):
        pass

    def __call__(self, sample):       
        # TODO: IMPORTANT Works only for two views
        img_l, img_r, T_cam3_cam2, K3, K2_inv = sample[0]
        # Works only for two views

        img_l_flip = img_r[0][:,::-1]
        img_r_flip = img_l[:,::-1]

        K2 = np.linalg.inv(K2_inv)
        K2[0,2] = img_l.shape[1] - K2[0,2]
        new_K3 = K2
        
        K3[0][0,2] = img_r[0].shape[1] - K3[0][0,2]
        new_K2_inv = np.linalg.inv(K3[0])

        T_cam3_cam2_new = T_cam3_cam2[0]
        T_cam3_cam2_new[0,3] *= -1
        Rinv = np.eye(3)
        Rinv[0,0] = -1
        T_cam3_cam2_new[:3,:3] = Rinv @ T_cam3_cam2_new[:3,:3] @ Rinv

        annotations = sample[2]
        annotations[:,1] *= -1
        annotations[:,7] *= -1
        mask = annotations[:,1:4].sum(axis=1) > 0
        annotations[mask,1:4] = (T_cam3_cam2_new[:3,:3] @ annotations[mask,1:4].T + T_cam3_cam2_new[:3,3][:,None]).T

        T_cam3_cam2_new[:3,:3] = T_cam3_cam2_new[:3,:3].T
        T_cam3_cam2_new[:3,3] = -T_cam3_cam2_new[:3,:3] @ T_cam3_cam2_new[:3,3]

        depth_img_l = sample[1][0]
        depth_img_r = sample[1][1][0]
        depth_img_l_flip = depth_img_r[:,::-1]
        depth_img_r_flip = depth_img_l[:,::-1]

        return (img_l_flip, [img_r_flip], [T_cam3_cam2_new], [new_K3], new_K2_inv), (depth_img_l_flip, [depth_img_r_flip]), annotations


class HorizontalFlipUnrectWithoutCamFlip(torch.nn.Module):
    """
    Flip the images horizontally
    * return:
        - flipped images
        - flipped depth images
        - flipped annotations
        - first flip the right camera with T @ (-1, 0, 0) and R' = [[-1,0,0], [0,1,0], [0,0,1]] --> R'@R@R'
        - then flip X and theta from annotations
        - then transform annotations to new left Camera with R@P+T
        - then new R and T is R.T and -R.T@T
    """

    def __init__(self):
        pass

    def __call__(self, sample):
        if len(sample[0]) == 5:
            img_l, img_r, T_cam3_cam2, K3, K2_inv = sample[0]
        else:
            img_l, img_r, T_cam3_cam2, K3, K2_inv, T_cam_plane = sample[0]
            # mirror matrix
            mirror_matrix = np.eye(3)
            mirror_matrix[0,0] = -1
            T_cam_plane[:3,:3] = mirror_matrix @ T_cam_plane[:3,:3]
            T_cam_plane[:3,3] = mirror_matrix @ T_cam_plane[:3,3]


        img_l_flip = img_l[:,::-1]
        img_r_flip = []
        for i in range(len(img_r)):
            img_r_flip.append(img_r[i][:,::-1])

        K2 = np.linalg.inv(K2_inv)
        K2[0,2] = img_l.shape[1] - K2[0,2]
        new_K2_inv = np.linalg.inv(K2)
        
        new_K3 = []
        for i in range(len(K3)):
            K3[i][0,2] = img_r[i].shape[1] - K3[i][0,2]
            new_K3.append(K3[i])

        annotations = sample[2]
        annotations[:,1] *= -1
        annotations[:,7] *= -1

        A = np.eye(3)
        A[0,0] = -1
        T_cam3_cam2_new = []
        for i in range(len(T_cam3_cam2)):
            local_transform = T_cam3_cam2[i]
            local_transform[:3,:3] = A @ local_transform[:3,:3] @ A
            local_transform[0,3] *= -1
            T_cam3_cam2_new.append(local_transform)

        depth_img_l = sample[1][0]
        depth_img_r = sample[1][1]
        depth_img_l_flip = depth_img_l[:,::-1]
        depth_img_r_flip = []
        for i in range(len(depth_img_r)):
            depth_img_r_flip.append(depth_img_r[i][:,::-1])

        if len(sample[0]) == 5:
            return (img_l_flip, img_r_flip, T_cam3_cam2_new, new_K3, new_K2_inv), (depth_img_l_flip, depth_img_r_flip), annotations
        else:
            return (img_l_flip, img_r_flip, T_cam3_cam2_new, new_K3, new_K2_inv, T_cam_plane), (depth_img_l_flip, depth_img_r_flip), annotations


class CropImages(torch.nn.Module):
    """
    Crop the images to the desired size
    * params:
        - size: desired size
    * return:
        - cropped images
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img_l = sample[0][0]
        img_r = sample[0][1]
        
        img_l = img_l[:self.size[0], :self.size[1]]
        for i in range(len(img_r)):
            img_r[i] = img_r[i][:self.size[0], :self.size[1]]

        depth_img_l = sample[1][0]
        depth_img_r = sample[1][1]
        depth_img_l = depth_img_l[:self.size[0], :self.size[1]]

        [depth_img_r[i][:self.size[0], :self.size[1]] for i in range(len(depth_img_r))]

        if len(sample[0]) == 5:
            return (img_l, img_r, sample[0][2], sample[0][3], sample[0][4]), (depth_img_l, depth_img_r), sample[2]
        else:
            return (img_l, img_r, sample[0][2], sample[0][3], sample[0][4], sample[0][5]), (depth_img_l, depth_img_r), sample[2]
    

class ZeroImage(torch.nn.Module):
    """
    randomly set one camera image to zero
    """
    def __init__(self, camera_idx=None):
        self.camera_idx = camera_idx

    def __call__(self, sample):
        # number of cameras (ref + target cams)
        num_cams = len(sample[0][1]) + 1
        # randomly choose one camera
        if self.camera_idx is None:
            selected_cam = np.random.randint(0, num_cams)
        else:
            selected_cam = self.camera_idx
        # set the selected camera image to zero
        if selected_cam == 0:
            img_l = np.zeros_like(sample[0][0])
            img_r_list = sample[0][1]
        else:
            img_l = sample[0][0]
            img_r_list = sample[0][1]
            zero_img = np.zeros_like(sample[0][1][selected_cam-1])
            img_r_list[selected_cam-1] = zero_img
        return (img_l, img_r_list, sample[0][2], sample[0][3], sample[0][4]), sample[1], sample[2]
    

class AddThermalNoise(torch.nn.Module):
    """
    add thermal noise to the image
    """
    def __init__(self, cfg, camera_idx=None):
        self.sigma = cfg["noise_sigma"]
        self.camera_idx = camera_idx

    def __call__(self, sample):
        num_cams = len(sample[0][1]) + 1
        # randomly choose one camera
        if self.camera_idx is None:
            selected_cam = np.random.randint(0, num_cams)
        else:
            selected_cam = self.camera_idx
        selected_cam = np.random.randint(0, num_cams)
        # set the selected camera image to zero
        if selected_cam == 0:
            img_l = sample[0][0]
            noise = np.random.normal(0, self.sigma, img_l.shape)
            img_l = img_l + noise
            #clip to [0, 1]
            img_l = np.clip(img_l, 0, 1)
            img_r_list = sample[0][1]
        else:
            img_l = sample[0][0]
            img_r_list = sample[0][1]
            img_r = sample[0][1][selected_cam-1]
            noise = np.random.normal(0, self.sigma, img_r.shape)
            img_r = img_r + noise
            img_r = np.clip(img_r, 0, 1)
            img_r_list[selected_cam-1] = img_r

        return (img_l, img_r_list, sample[0][2], sample[0][3], sample[0][4]), sample[1], sample[2]
    

class ImageFreeze(torch.nn.Module):
    """
    randomly choose one camera image to simulate image freeze
    """
    def __init__(self, cfg, camera_idx=None, train="train"):
        self.dataset_path = cfg["data_directory"]
        self.img2_path = os.path.join(self.dataset_path, "data_object_image_2/training/image_2")
        self.img0_path = os.path.join(self.dataset_path, "data_object_image_0/training/image_0")
        self.img3_path = os.path.join(self.dataset_path, "data_object_image_3/training/image_3")
        self.img1_path = os.path.join(self.dataset_path, "data_object_image_1/training/image_1")
        self.target_path = [self.img0_path, self.img3_path, self.img1_path]
        if train == "train":
            training_idx_file_path = cfg["train_file"]
        else:
            training_idx_file_path = cfg["val_test_file"]
        with open(training_idx_file_path, "r") as f:
            self.training_idx = f.readlines()
        self.training_idx = [int(idx.strip()) for idx in self.training_idx]
        self.camera_idx = camera_idx

    
    def __insert_new_img(self, img_old, new_img):
        # check if new img is larger or smaller than old img it should have the same size. if to big crop else pad
        h_old, w_old = img_old.shape[0], img_old.shape[1]
        h_new, w_new = new_img.shape[0], new_img.shape[1]
        if h_old > h_new:
            new_img = np.pad(new_img, ((0, h_old - h_new), (0, 0), (0, 0)), mode='constant', constant_values=0)
        elif h_old < h_new:
            new_img = new_img[:h_old, :, :]
        if w_old > w_new:
            new_img = np.pad(new_img, ((0, 0), (0, w_old - w_new), (0, 0)), mode='constant', constant_values=0)
        elif w_old < w_new:
            new_img = new_img[:, :w_old, :]
        return new_img


    def __call__(self, sample):
        # number of cameras (ref + target cams)
        num_cams = len(sample[0][1]) + 1
        # randomly choose one camera
        if self.camera_idx is None:
            selected_cam = np.random.randint(0, num_cams)
        else:
            selected_cam = self.camera_idx
        random_idx = np.random.randint(0, len(self.training_idx))
        random_idx = self.training_idx[random_idx]
        # fill with 0 that string has at least 6 digits
        file_name = str(random_idx).zfill(6) + ".png"
        if selected_cam == 0:
            img_l = sample[0][0]
            img_new = plt.imread(os.path.join(self.img2_path, file_name))[:,:,:3]
            img_l = self.__insert_new_img(img_l, img_new)
            img_r_list = sample[0][1]
        else:
            img_l = sample[0][0]
            img_r_list = sample[0][1]
            if selected_cam == 1 or selected_cam == 3:
                img_new = plt.imread(os.path.join(self.target_path[selected_cam-1], file_name))[:,:,0]
                # add last dimension to img_new
                img_new = np.expand_dims(img_new, axis=2)
                img_r_list[selected_cam-1] = self.__insert_new_img(img_r_list[selected_cam-1], img_new)
            else:
                img_new = plt.imread(os.path.join(self.target_path[selected_cam-1], file_name))[:,:,:3]
                img_r_list[selected_cam-1] = self.__insert_new_img(img_r_list[selected_cam-1], img_new)

        return (img_l, img_r_list, sample[0][2], sample[0][3], sample[0][4]), sample[1], sample[2]
    
    
class TCamPlaneNoise(torch.nn.Module):
    """
    random noise to the T_cam_plane for variations in sampling
    """
    def __call__(self, sample):
        
        #  sample = (img_l, img_targets, transform_mats, K_list, K_ref_inv, Tr_cam_plane_hom), (depth_img_left, depth_img_right), annotations
        Tr_cam_plane_hom = sample[0][5]
        Rot = Tr_cam_plane_hom[:3,:3]
        R_plane_cam = Rot.T
        ## noise to rotation
        annotations = sample[2]
        # find annotations with largest depth
        max_depth_idx = np.argmax(annotations[:,3])
        max_depth = annotations[max_depth_idx,3]
        # max x displacemen == 1.5m, max y displacement == 0.7m
        max_angle_yaw = np.arctan(1.5/max_depth)
        max_angle_pitch = np.arctan(0.7/max_depth)
        noise_yaw = np.random.uniform(-max_angle_yaw, max_angle_yaw)
        noise_pitch = np.random.uniform(-max_angle_pitch, max_angle_pitch)
        R_noise_yaw = np.array([[np.cos(noise_yaw), 0, np.sin(noise_yaw)],
                                [0, 1, 0],
                                [-np.sin(noise_yaw), 0, np.cos(noise_yaw)]])
        R_noise_pitch = np.array([[1, 0, 0],
                                [0, np.cos(noise_pitch), -np.sin(noise_pitch)],
                                [0, np.sin(noise_pitch), np.cos(noise_pitch)]])
        # apply noise to rotation
        # decide with 0.5 probability if yaw noise or pitch noise should be applied
        if np.random.rand() <= 0.5:
            R_plane_cam = R_noise_yaw @ R_plane_cam
        else:
            R_plane_cam = R_noise_pitch @ R_plane_cam
        Tr_cam_plane_hom[:3,:3] = R_plane_cam.T

        return (sample[0][0], sample[0][1], sample[0][2], sample[0][3], sample[0][4], Tr_cam_plane_hom), sample[1], sample[2]
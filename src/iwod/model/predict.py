import torch
import torchvision
import numpy as np
import detectron2.layers as dt2
from iwod.dataset import lake_constance_detection as lcod
from iwod.utils.transforms import Normalize, PadImages, ToTensor
from iwod.model.lightning_module import LitPSDepth
from iwod.model.submodules import PSCoder
from iwod.utils.helper import load_config


class Predictor:
    """This class encapsulates the entire prediction pipeline, including data loading, model inference, and post-processing."""
    def __init__(self, config_path, checkpoint_path, gpu_id=0, water_detection=False, custom_T_plane_cam=None, training="test"):
        """
        Args:
            config_path (str): Path to the model configuration file.
            checkpoint_path (str): Path to the trained model checkpoint.
            gpu_id (int): ID of the GPU to use for inference.
            water_detection (bool): Whether to enable water detection mode.
            custom_T_plane_cam (torch.Tensor, optional): Custom transformation from plane to camera coordinates. If None, it will be loaded from the dataset or estimated if water_detection is True.
            training (str): Whether to run the datset in training, valid or test mode.
        """
        # 1. Setup Device
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        
        # 2. Load Config
        self.cfg = load_config(config_path)
        dataset_path = self.cfg["data_directory"]
        
        # 3. Setup Transform & Dataset
        transform = torchvision.transforms.Compose([
            Normalize(mean_ref=self.cfg["mean_ref"],
                      std_ref=self.cfg["std_ref"],
                      means=self.cfg["mean_target"],
                      stds=self.cfg["std_target"]),
            PadImages((self.cfg["pad_image_h"], self.cfg["pad_image_w"])),
            ToTensor()
        ])
        self.dataset = lcod.LCDDataset(dataset_path, training, transform=transform, water_detection=water_detection)
        
        # 4. Load Model
        self.model = LitPSDepth.load_from_checkpoint(checkpoint_path)
        self.model.to(self.device)
        self.model.eval()
        
        # 5. Init PSCoder
        self.ps_coder = PSCoder("le90", 4, device=self.device)
        
        # 6. Pre-compute BEV Grid
        self._init_grid()

    def _init_grid(self):
        Z_MIN, Z_MAX = self.cfg["z_min"], self.cfg["z_max"]
        X_MIN, X_MAX = self.cfg["x_min"], self.cfg["x_max"]
        VOXEL_Z_SIZE = self.cfg["z_size"]
        VOXEL_X_SIZE = self.cfg["x_size"]

        shifts_z = torch.arange(Z_MAX, Z_MIN - np.sign(VOXEL_Z_SIZE) * 1e-10, step=VOXEL_Z_SIZE, 
            dtype=torch.float32, device=self.device) + VOXEL_Z_SIZE / 2.
        shifts_x = torch.arange(X_MIN, X_MAX - np.sign(VOXEL_X_SIZE) * 1e-10, step=VOXEL_X_SIZE,
            dtype=torch.float32, device=self.device) + VOXEL_X_SIZE / 2.
            
        # indexing='ij' verhindert Warnungen bei neueren PyTorch-Versionen
        shifts_z, shifts_x = torch.meshgrid(shifts_z, shifts_x, indexing='ij') 
        self.locations_bev = torch.stack([shifts_x, shifts_z], dim=-1).reshape(-1, 2)


    def get_annotations(self, idx):
        """
        args:
            idx (int): Index of the sample in the dataset.
        returns:     
            Ground truth annotations for the sample with the given index.
        """
        _, _, annotations = self.dataset[idx]
        return annotations
    

    def get_dataset_length(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.dataset)


    def predict(self, idx, score_threshold=0.3, nms_threshold=0.0001, apply_nms=True):
        """
        Runs the prediction pipeline for a single sample in the dataset.

        args:
            idx (int): Index of the sample in the dataset.
            score_threshold (float): Threshold for filtering predictions based on confidence score.
            nms_threshold (float): IoU threshold for Non-Maximum Suppression (NMS).
            apply_nms (bool): Whether to apply NMS to the detected boxes.

        returns:
            Tuple of detected bounding boxes and their properties.
        """
        #prepare input data
        data, depth_target, annotations = self.dataset[idx]
        
        ref_img = data[0].unsqueeze(0).to(self.device)
        src_imgs = [img.unsqueeze(0).to(self.device) for img in data[1]]
        T_rcam_lcam = [t.unsqueeze(0).to(self.device) for t in data[2]]
        intrinsics = [i.unsqueeze(0).to(self.device) for i in data[3]]
        ref_cam_inv_intrinsic = data[4].unsqueeze(0).to(self.device)
        Tr_cam_plane_hom = data[5].unsqueeze(0).to(self.device)

        network_input = (ref_img, src_imgs, T_rcam_lcam, intrinsics, ref_cam_inv_intrinsic, Tr_cam_plane_hom)

        # inference
        with torch.no_grad():
            depth, score, bbox_cent, bbox_dim, centerness, bbox_reg_angle = self.model(network_input)
            
            # Squeeze Batch Dimension
            score = score.squeeze(0)
            bbox_cent = bbox_cent.squeeze(0)
            bbox_dim = bbox_dim.squeeze(0)
            centerness = centerness.squeeze(0)
            bbox_reg_angle = bbox_reg_angle.squeeze(0)

        # --- POST-PROCESSING ---
        score_sigmoid = torch.sigmoid(score[0])
        centerness_sigmoid = torch.sigmoid(centerness[0])
        weighted_score = score_sigmoid * centerness_sigmoid
        score_mask = weighted_score > score_threshold

        X = self.locations_bev[:,0].view(weighted_score.shape)
        Z = self.locations_bev[:,1].view(weighted_score.shape)

        regression = torch.cat((bbox_cent, bbox_dim), dim=0)

        # decode angle
        angle_regression = bbox_reg_angle.permute(1, 2, 0)
        angle_regression = angle_regression.reshape(-1, angle_regression.shape[2])
        angle_regression = self.ps_coder.decode(angle_regression)
        angle_regression = angle_regression.reshape(regression.shape[1], regression.shape[2])

        # Filter detections based on score threshold
        zfxf = torch.nonzero(score_mask)
        zf, xf = zfxf[:,0], zfxf[:,1]
        
        detected_X = X[zf, xf]
        detected_Z = Z[zf, xf]
        detected_regressions = regression.permute(1,2,0)[zf, xf]
        detected_angles = angle_regression[zf, xf]
        detected_scores = weighted_score[score_mask]

        # --- NMS ---
        if apply_nms and len(detected_scores) > 0:
            x_center = detected_X + detected_regressions[:, 0]
            z_center = detected_Z + detected_regressions[:, 1]
            w = detected_regressions[:, 2]
            l = detected_regressions[:, 3]
            
            potential_boxes = torch.cat((
                x_center.unsqueeze(dim=1), 
                z_center.unsqueeze(dim=1),
                w.unsqueeze(dim=1), 
                l.unsqueeze(dim=1),
                torch.rad2deg(detected_angles).unsqueeze(dim=1)
            ), dim=1)

            nms_idx = dt2.nms_rotated(potential_boxes, detected_scores, nms_threshold)

            detected_X = detected_X[nms_idx]
            detected_Z = detected_Z[nms_idx]
            detected_scores = detected_scores[nms_idx]
            detected_regressions = detected_regressions[nms_idx]
            detected_angles = detected_angles[nms_idx]

        return detected_X, detected_Z, detected_scores, detected_regressions, detected_angles


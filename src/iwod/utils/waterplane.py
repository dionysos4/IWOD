from waterplane import StereoFrame, OpenCVRectifier, SGBMStereo, FCNWaterSegmentation, RANSACPlaneFit, WaterPlanePipeline, ResNet, FCN8s
from torchvision.models import ResNet34_Weights
from torchvision import models
import torch

class TPlaneCamEstimator:

    def __init__(self):
        # define segmentation model
        model_path = "/home/dennis/git_repos/water_surface_detector/experiments/models/ResNet34_Enc_Dec_Misc_Constance"
        resnet34 = models.resnet34(weights=ResNet34_Weights.DEFAULT)
        res_model = ResNet(resnet34)
        seg_model = FCN8s(res_model, 1)
        seg_model.load_state_dict(torch.load(model_path))
        self.seg_model = seg_model.to("cuda")


    def estimate_T_plane_cam(self, K_l, K_r, D_l, D_r, R_l, R_r, P_l, P_r, left_img, right_img):
        calib = {
            "K_l": K_l,
            "K_r": K_r,
            "D_l": D_l,
            "D_r": D_r,
            "R_l": R_l,
            "R_r": R_r,
            "P_l": P_l,
            "P_r": P_r
        }
        
        frame = StereoFrame(left_img, right_img, calib)
        pipeline = WaterPlanePipeline(
            rectifier=OpenCVRectifier(),
            stereo=SGBMStereo(z_clip=100),
            segmenter=FCNWaterSegmentation(self.seg_model),
            plane_fit=RANSACPlaneFit()
        )

        # Run the pipeline
        result = pipeline.run(frame)
        return result.T_plane_cam
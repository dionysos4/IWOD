import lightning as L
import torch
from iwod.model.submodules import *
from iwod.model.loss import SimpleScoreLoss


class LitPSDepth(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.lr = cfg["lr"]
        self.weight_decay = cfg["weight_decay"]
        self.mindepth = cfg["mindepth"]
        self.maxdepth = cfg["maxdepth"]
        self.depth_sampling = cfg["depth_sampling"]
        self.beta1 = cfg["beta1"]
        self.beta2 = cfg["beta2"]
        self.beta3 = cfg["beta3"]
        self.beta4 = cfg["beta4"]
        self.beta5 = cfg["beta5"]

        self.depth_regression_model = DepthRegression(cfg)

        self.score_loss = SimpleScoreLoss(cfg=cfg)

        self.loss_function = torch.nn.SmoothL1Loss(reduction='mean')
        self.save_hyperparameters()
        self.cfg = cfg


    def forward(self, input):
        if len(input) == 5:
            ref, targets, pose, P_r, P_l_inv = input[0], input[1], input[2], input[3], input[4]
            depth, score, bbox_cent, bbox_dim, centerness, bbox_reg_angle = self.depth_regression_model(ref, targets, pose, P_r, P_l_inv[:,:3,:3])
            return depth, score, bbox_cent, bbox_dim, centerness, bbox_reg_angle
        else:
            ref, targets, pose, P_r, P_l_inv, T_cam_plane = input[0], input[1], input[2], input[3], input[4], input[5]
            depth, score, bbox_cent, bbox_dim, centerness, bbox_reg_angle = self.depth_regression_model(ref, targets, pose, P_r, P_l_inv[:,:3,:3], T_cam_plane)
            return depth, score, bbox_cent, bbox_dim, centerness, bbox_reg_angle


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer


    def training_step(self, batch, batch_idx):
        data, depth_target, annotations = batch
        depth_target = depth_target[0]
        output = self(data)
        ### depth ##
        depth_final = output[0].squeeze(1)
        mask = ((depth_target > self.mindepth) & (depth_target <= self.maxdepth))
        depth_loss = self.loss_function(depth_final[mask], depth_target[mask])
        self.log("training/disp_loss", depth_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        ############

        ### score and regesssion ##
        score = output[1]
        bbox_cent = output[2]
        bbox_dim = output[3]
        centerness = output[4]
        angle = output[5]

        x_feature_scale = data[0].shape[-1] / score.shape[-1]
        K_l_inv = torch.clone(data[4].to(dtype=score.dtype))
        K_l = torch.linalg.inv(K_l_inv[:,:3,:3])

        score_loss, regression_cent_loss, regression_dim_loss, angle_regression_loss, centerness_loss, diou_loss = self.score_loss(score, bbox_cent, bbox_dim, centerness, angle, annotations, K_l, x_feature_scale)

        # angle, centerness and dimension regression losses are monitored for logging purposes, but not included in the final 
        # loss calculation as they are fully optimized with diou loss.
        self.log("training/cls_loss", score_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("training/reg_cent_loss", regression_cent_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("training/reg_dim_loss", regression_dim_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("training/angle_loss", angle_regression_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("training/centerness", centerness_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("training/iou_loss", diou_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        loss = depth_loss + score_loss + centerness_loss + diou_loss
        self.log("training/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # scale losses
        depth_loss = self.beta1 * depth_loss
        score_loss = self.beta2 * score_loss
        diou_loss = self.beta3 * diou_loss
        centerness_loss = self.beta4 * centerness_loss

        final_loss = depth_loss + score_loss + centerness_loss + diou_loss
        return final_loss


    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        data, depth_target, annotations = batch
        depth_target = depth_target[0]
        output = self(data)
        ### depth ##
        depth_final = output[0].squeeze(1)
        mask = ((depth_target > 0) & (depth_target <= self.maxdepth))
        depth_loss = self.loss_function(depth_final[mask], depth_target[mask])
        self.log("validation/disp_loss", depth_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        ############

        ### score ##
        score = output[1]
        bbox_cent = output[2]
        bbox_dim = output[3]
        centerness = output[4]
        angle = output[5]

        x_feature_scale = data[0].shape[-1] / score.shape[-1]
        K_l_inv = torch.clone(data[4].to(dtype=score.dtype))
        K_l = torch.linalg.inv(K_l_inv[:,:3,:3])

        score_loss, regression_cent_loss, regression_dim_loss, angle_regression_loss, centerness_loss, diou_loss = self.score_loss(score, bbox_cent, bbox_dim, centerness, angle, annotations, K_l, x_feature_scale)

        # angle, centerness and dimension regression losses are monitored for logging purposes, but not included in the final 
        # loss calculation as they are fully optimized with diou loss.
        self.log("validation/cls_loss", score_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("validation/reg_cent_loss", regression_cent_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("validation/reg_dim_loss", regression_dim_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("validation/angle_loss", angle_regression_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("validation/centerness", centerness_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("validation/iou_loss", diou_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        loss = depth_loss + score_loss + centerness_loss + diou_loss
        self.log("validation/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)


    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        data, depth_target, annotations = batch
        res = self(data)
        return res
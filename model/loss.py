from operator import gt
import torch
from scipy.spatial.transform import Rotation as R
import numpy as np
from typing import Optional
from torch import Tensor
from torchvision.ops import sigmoid_focal_loss
from typing import List
from model.submodules import PSCoder
from iou_utils.oriented_iou_loss import cal_diou


def compute_locations_bev(Z_MIN, Z_MAX, VOXEL_Z_SIZE, X_MIN, X_MAX, VOXEL_X_SIZE, device):
    shifts_z = torch.arange(Z_MIN, Z_MAX - np.sign(VOXEL_Z_SIZE) * 1e-10, step=VOXEL_Z_SIZE, 
        dtype=torch.float32, device=device) + VOXEL_Z_SIZE / 2.
    shifts_x = torch.arange(X_MIN, X_MAX - np.sign(VOXEL_X_SIZE) * 1e-10, step=VOXEL_X_SIZE,
        dtype=torch.float32, device=device) + VOXEL_X_SIZE / 2.
    shifts_z, shifts_x = torch.meshgrid(shifts_z, shifts_x)
    locations_bev = torch.stack([shifts_x, shifts_z], dim=-1)
    locations_bev = locations_bev.reshape(-1, 2)
    return locations_bev


def get_X_Z_grid(output, K_l, min_depth, depth_sampling):

    n_depth = output.shape[1]
    max_depth = n_depth  * depth_sampling

    z_range = torch.arange(min_depth, max_depth+min_depth, depth_sampling, device=K_l.device, dtype=output.dtype)
            
    x = torch.arange(output.shape[2], device=output.device, dtype=output.dtype)
    grid_x = x.repeat(output.shape[1]).view(output.shape[1:])
    
    # compute x value (x- cx) * (z/f)
    X = grid_x - K_l[0,2]
    X = X * z_range.repeat_interleave(x.shape[0]).view(X.shape)
    X = X / K_l[0,0]
    Z = z_range.repeat_interleave(x.shape[0]).view(X.shape)

    return X, Z


def _sum(x: List[Tensor]) -> Tensor:
    res = x[0]
    for i in x[1:]:
        res = res + i
    return res


def rot(theta, device):
    theta = -theta
    return torch.tensor([[torch.cos(theta), -torch.sin(theta)], 
                         [torch.sin(theta), torch.cos(theta)]], device=device)


def get_bbox_coordinates(x, z, w, l, theta, device):
    c1 = torch.tensor([-w/2, l/2], device=device)
    c2 = torch.tensor([w/2, l/2], device=device)
    c3 = torch.tensor([w/2, -l/2], device=device)
    c4 = torch.tensor([-w/2, -l/2], device=device)

    # rotate corners around up axis
    c1 = rot(theta, device) @ c1
    c2 = rot(theta, device) @ c2
    c3 = rot(theta, device) @ c3
    c4 = rot(theta, device) @ c4

    corners = torch.concat((c1, c2, c3, c4)).view(-1,2)
    corners[:,0] = corners[:,0] + x
    corners[:,1] = corners[:,1] + z
    return corners


def get_n_bbox_coordinates(x, z, w, l, theta, device):
    c1 = torch.cat(((-w/2).unsqueeze(1), (l/2).unsqueeze(1)), dim=1)
    c2 = torch.cat(((w/2).unsqueeze(1), (l/2).unsqueeze(1)), dim=1)
    c3 = torch.cat(((w/2).unsqueeze(1), (-l/2).unsqueeze(1)), dim=1)
    c4 = torch.cat(((-w/2).unsqueeze(1), (-l/2).unsqueeze(1)), dim=1)

    r0 = torch.cos(-theta).unsqueeze(1)
    r1 = -torch.sin(-theta).unsqueeze(1)
    r2 = torch.sin(-theta).unsqueeze(1)
    r3 = torch.cos(-theta).unsqueeze(1)
    rot_mats = torch.cat((r0, r1, r2, r3), dim=1).view(-1,2,2)

    c1 = torch.matmul(rot_mats, c1.unsqueeze(2)).squeeze(2).unsqueeze(1)
    c2 = torch.matmul(rot_mats, c2.unsqueeze(2)).squeeze(2).unsqueeze(1)
    c3 = torch.matmul(rot_mats, c3.unsqueeze(2)).squeeze(2).unsqueeze(1)
    c4 = torch.matmul(rot_mats, c4.unsqueeze(2)).squeeze(2).unsqueeze(1)

    corners = torch.cat((c1, c2, c3, c4), dim=1)
    corners[:,:,0] = corners[:,:,0] + x.unsqueeze(1).repeat(1,4)
    corners[:,:,1] = corners[:,:,1] + z.unsqueeze(1).repeat(1,4)
    return corners



def points_inside_obb_corners(X, Z, corners):
    """
    Check if a point is inside an oriented bounding box (OBB) defined by its corner points.

    Parameters:
    - X tensor of x coordinates of the points to check (shape: (Z,X))
    - Z tensor of z coordinates of the points to check (shape: (Z,X))
    - corners: 4x2 Tensor of the corner points of the OBB (shape: (4,2))

    Returns:
    - Mask Tensor with true values if the point is inside the OBB, False otherwise.
    """
    z, x = X.shape
    X = X.flatten()
    Z = Z.flatten()

    # Convert the point and corners to NumPy arrays for easier manipulation
    points = torch.stack((X,Z)).permute(1,0)

    # Create vectors from the corners
    vectors = corners - torch.roll(corners, 1, dims=0)

    # Create vectors from the corners to the point
    to_point_vectors = torch.tile(points,(4,)).view(-1,4,2) - corners

    # Calculate the dot products and cross products
    dot_products = torch.multiply(vectors, to_point_vectors).sum(dim=2)

    # pad the inputs to 3d vectors because pytorch can only handle cross product for 3d vectors
    pad_vectors = torch.nn.functional.pad(vectors, (0,1)).repeat(to_point_vectors.shape[0], 1, 1)
    pad_to_point_vectors = torch.nn.functional.pad(to_point_vectors, (0,1))
    cross_products = torch.linalg.cross(pad_vectors, pad_to_point_vectors)[:,:,-1]

    # Check if the point is on the same side of each edge
    dot_mask = (torch.all(dot_products > 0, dim=1) | torch.all(dot_products < 0, dim=1))
    cross_mask = torch.all(cross_products > 0, dim=1) | torch.all(cross_products < 0, dim=1)
    mask = cross_mask & dot_mask
    return mask.view(z,x)


def get_regression_targets(points_inbbox, x_pos, z_pos, theta, corners, device):
    """
    Transforms the points inside the bounding box back to origin and aligns it
    with world coords to compute distances between the points and the bounding box borders

    Parameters:
    - points_inbbox: tensor of points inside the bounding box (shape: (N,2))
    - x_pos: x coordinate of the bounding box center
    - z_pos: z coordinate of the bounding box center
    - theta: rotation angle of the bounding box

    Returns:
    - lt: distance point to left boarder, 
    - rt: distance point to right boarder,
    - tt: distance point to top boarder,
    - bt: distance point to bottom boarder
    """
    # transform points back to origin
    points_inbbox[:,0] = points_inbbox[:,0] - x_pos
    points_inbbox[:,1] = points_inbbox[:,1] - z_pos

    corners[:,0] = corners[:,0] - x_pos
    corners[:,1] = corners[:,1] - z_pos

    # rotate points around up axis, use -theta because we want to rotate back
    theta = -theta
    points_inbbox = torch.einsum("ij,Nj->Ni", rot(theta, device), points_inbbox)
    corners = torch.einsum("ij,Nj->Ni", rot(theta, device), corners)
    lt = points_inbbox[:,0] - corners[:,0].min()
    rt = corners[:,0].max() - points_inbbox[:,0]
    tt = points_inbbox[:,1] - corners[:,1].min()
    bt = corners[:,1].max() - points_inbbox[:,1]

    # clamp due to nurmerical issues
    lt = torch.clamp(lt, 1e-7, None)
    rt = torch.clamp(rt, 1e-7, None)
    tt = torch.clamp(tt, 1e-7, None)
    bt = torch.clamp(bt, 1e-7, None)
    
    return lt, rt, tt, bt


class SimpleScoreLoss(torch.nn.modules.loss._WeightedLoss):
    
    __constants__ = ['weight', 'ignore_index', 'reduction']

    def __init__(self, weight: Optional[Tensor] = None, size_average=None,
                 reduce=None, reduction: str = 'mean', cfg={"mindepth": 2.0, "depth_sampling": 0.2}):
        super(SimpleScoreLoss, self).__init__(weight, size_average, reduce, reduction)
        self.min_depth = cfg["mindepth"]
        self.depth_sampling = cfg["depth_sampling"]

        self.Z_MIN, self.Z_MAX = cfg["z_min"], cfg["z_max"]
        self.Y_MIN, self.Y_MAX = cfg["y_min"], cfg["y_max"]
        self.X_MIN, self.X_MAX = cfg["x_min"], cfg["x_max"]
        self.VOXEL_Z_SIZE, self.VOXEL_Y_SIZE, self.VOXEL_X_SIZE = cfg["z_size"], cfg["y_size"], cfg["x_size"]

    
    def forward(self, score, bbox_cent, bbox_dim, centerness, angle, target, K_l, x_feature_scale):
        gt_boxes = target.to(dtype=score.dtype)

        locations_bev = compute_locations_bev(self.Z_MAX, self.Z_MIN, self.VOXEL_Z_SIZE, 
            self.X_MIN, self.X_MAX, self.VOXEL_X_SIZE, score.device)
        
        X = locations_bev[:,0].view(score[0,0].shape)
        Z = locations_bev[:,1].view(score[0,0].shape)

        K_l[:,0] /= x_feature_scale

        score = score
        num_classes = score.shape[1]
     
        losses = []
        regression_cent_losses = []
        regression_dim_losses = []
        angle_regression_losses = []
        centerness_losses = []
        diou_losses = []

        batch_size = score.shape[0]

        ps_coder = PSCoder("le90", num_step=4, device=score.device)

        for batch in range(batch_size):
            N_pos = 0
            N_neg = 0
            gt_boxes_batch = gt_boxes[batch]

            gt_boxes_batch = gt_boxes_batch[gt_boxes_batch.sum(dim=1) != 0]
            
            # define negative mask
            negative_mask_per_img = torch.ones_like(score[0,0], dtype=torch.int, device=score.device)
            samples_per_img = torch.empty(0,num_classes, device=score.device)
            target_sample_per_img = torch.empty(0, device=score.device)
            regression_cent_per_img = torch.empty(0, device=score.device)
            target_regression_cent_per_img = torch.empty(0, device=score.device)
            regression_dim_per_img = torch.empty(0, device=score.device)
            target_regression_dim_per_img = torch.empty(0, device=score.device)
            regression_angle_per_img = torch.empty(0, device=score.device)
            target_regression_angle_per_img = torch.empty(0, device=score.device)
            centerness_per_img = torch.empty(0, device=score.device)
            target_centerness_per_img = torch.empty(0, device=score.device)
            diou_per_img = torch.empty(0, device=score.device)


            ##### compute for each ground truth the iou to each prediction ####
            for gt_box in gt_boxes_batch:
                
                x_pos = gt_box[1]
                z_pos = gt_box[3]
                w = gt_box[4]
                l = gt_box[6]
                theta = gt_box[7]

                if theta < -np.pi/2:
                    theta = theta + (np.pi)
                if theta > np.pi/2:
                    theta = theta - (np.pi)

                # get bounding box corners in 3d bev view
                corners = get_bbox_coordinates(x_pos, z_pos, w, l, theta, device=score.device)
                # which sampling points lie inside the bounding box
                positive_mask = points_inside_obb_corners(X, Z, corners)

                if positive_mask.sum() > 0:

                    positive_samples = score[batch].permute(1,2,0)[positive_mask]

                    N_pos += len(positive_samples)
                    
                    # add eps for numerical reasons
                    eps=1e-7
                    positive_sample_mask = positive_samples == 0
                    positive_samples[positive_sample_mask] = eps
                    samples_per_img = torch.cat((samples_per_img, positive_samples))

                    # create target
                    target = torch.ones_like(positive_samples, device=score.device)
                    target_sample_per_img = torch.cat((target_sample_per_img, target))

                    negative_mask_per_img[positive_mask] = 0

                    # compute the regression targets (l, r, t, b)
                    points_inbbox = torch.stack((X[positive_mask], Z[positive_mask])).permute(1,0)

                    x_diff = x_pos - points_inbbox[:,0]
                    z_diff = z_pos - points_inbbox[:,1]

                    # compute centerness
                    centerness_one_bbox = x_diff**2 + z_diff**2
                    centerness_one_bbox = centerness_one_bbox / centerness_one_bbox.max()
                    centerness_one_bbox = torch.exp(-0.25 * centerness_one_bbox)
                    
                    centerness_per_img = torch.cat((centerness_per_img, centerness[batch].squeeze()[positive_mask]))
                    target_centerness_per_img = torch.cat((target_centerness_per_img, centerness_one_bbox))

                    wt = w
                    lt = l

                    target_regression_cent_per_img = torch.cat((target_regression_cent_per_img, torch.stack((x_diff, z_diff)).permute(1,0)))
                    target_regression_dim_per_img = torch.cat((target_regression_dim_per_img, torch.stack((wt.repeat(x_diff.shape[0]), lt.repeat(x_diff.shape[0]))).permute(1,0)))
                    regression_cent_prediction_one_ex = bbox_cent[batch].permute(1,2,0)[positive_mask]
                    regression_dim_prediction_one_ex = bbox_dim[batch].permute(1,2,0)[positive_mask]        
                    regression_cent_per_img = torch.cat((regression_cent_per_img, regression_cent_prediction_one_ex))
                    regression_dim_per_img = torch.cat((regression_dim_per_img, regression_dim_prediction_one_ex))

                    # loss for theta
                    theta_pred_one_ex = angle[batch].permute(1,2,0)[positive_mask]
                    regression_angle_per_img = torch.cat((regression_angle_per_img, theta_pred_one_ex))
                    theta_target_one_ex = ps_coder.encode(theta).repeat(theta_pred_one_ex.shape[0]).view(theta_pred_one_ex.shape[0], 4)
                    target_regression_angle_per_img = torch.cat((target_regression_angle_per_img, theta_target_one_ex))

                    # use -theta
                    gt_one_ex = torch.tensor([[[x_pos, z_pos, w, l, -theta]]], device=score.device)
                    predicted_xpos = points_inbbox[:,0] + regression_cent_prediction_one_ex[:,0]
                    predicted_zpos = points_inbbox[:,1] + regression_cent_prediction_one_ex[:,1]
                    predicted_w = regression_dim_prediction_one_ex[:,0]
                    predicted_l = regression_dim_prediction_one_ex[:,1]
                    predicted_theta = ps_coder.decode(theta_pred_one_ex)
                    predicted_one_ex = torch.stack((predicted_xpos, predicted_zpos, predicted_w, predicted_l, -predicted_theta)).permute(1,0).unsqueeze(0)
                    gt_one_ex = gt_one_ex.repeat(1, predicted_one_ex.shape[1], 1)
                    diou_loss, iou = cal_diou(gt_one_ex, predicted_one_ex)
                    diou_per_img = torch.cat((diou_per_img, diou_loss.squeeze(0)))

            # compute negative loss and loss
            negative_samples = score[batch].permute(1,2,0)[negative_mask_per_img.bool()]
            N_neg += len(negative_samples)
            target = torch.zeros_like(negative_samples, device=score.device)

            samples_per_img = torch.cat((samples_per_img, negative_samples))
            target_sample_per_img = torch.cat((target_sample_per_img, target))
            
            losses.append(sigmoid_focal_loss(samples_per_img, target_sample_per_img, alpha=0.25, gamma=2, reduction="sum")
                          / max(1, N_pos))
            
            regression_cent_losses.append((torch.nn.functional.smooth_l1_loss(regression_cent_per_img, target_regression_cent_per_img, reduction="none") * target_centerness_per_img[:,None]).sum()
                          / max(1, N_pos))
            
            regression_dim_losses.append((torch.nn.functional.smooth_l1_loss(regression_dim_per_img, target_regression_dim_per_img, reduction="none") * target_centerness_per_img[:,None]).sum()
                          / max(1, N_pos))
            
            angle_regression_losses.append(torch.nn.functional.smooth_l1_loss(regression_angle_per_img, target_regression_angle_per_img, reduction="sum")
                          / max(1, N_pos))
            
            centerness_losses.append(torch.nn.functional.binary_cross_entropy_with_logits(centerness_per_img, 
                                                                              target_centerness_per_img, reduction="sum")
                        / max(1, N_pos))
            
            diou_losses.append((diou_per_img * target_centerness_per_img).sum() / max(1, N_pos))

        # check if there are no positive samples
        if len(losses) == 0:
            return torch.nn.functional.mse_loss(score, score), torch.nn.functional.mse_loss(bbox_cent, bbox_cent), torch.nn.functional.mse_loss(bbox_dim, bbox_dim), torch.nn.functional.mse_loss(angle, angle), \
                    torch.nn.functional.mse_loss(centerness, centerness), (torch.nn.functional.mse_loss(bbox_cent, bbox_cent) + torch.nn.functional.mse_loss(bbox_dim, bbox_dim) + torch.nn.functional.mse_loss(angle, angle))
        
        if _sum(losses) == 0 or _sum(regression_cent_losses) == 0 or _sum(angle_regression_losses) == 0:
            return _sum(losses) / batch_size, torch.nn.functional.mse_loss(bbox_cent, bbox_cent), torch.nn.functional.mse_loss(bbox_dim, bbox_dim), torch.nn.functional.mse_loss(angle, angle), \
                    torch.nn.functional.mse_loss(centerness, centerness), (torch.nn.functional.mse_loss(bbox_cent, bbox_cent) + torch.nn.functional.mse_loss(bbox_dim, bbox_dim) + torch.nn.functional.mse_loss(angle, angle))

        return _sum(losses) / batch_size, _sum(regression_cent_losses) / batch_size, _sum(regression_dim_losses) / batch_size, _sum(angle_regression_losses) / batch_size, \
                    _sum(centerness_losses) / batch_size, _sum(diou_losses) / batch_size
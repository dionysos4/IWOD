from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np
from model.inverse_warp import inverse_warp


class PSCoder(torch.nn.Module):
    """Simple Phase-Shifting Coder (PSC) without dual frequency.

    Args:
        angle_version (str): Angle definition.
            Only 'le90' is supported at present.
        num_step (int, optional): Number of phase steps. Default: 3.
        thr_mod (float): Threshold of modulation. Default: 0.47.
    """

    def __init__(self,
                 angle_version: str,
                 num_step: int = 3,
                 thr_mod: float = 0.47):
        super().__init__()
        self.angle_version = angle_version
        assert angle_version in ['le90']  # Ensure valid angle version
        self.num_step = num_step
        self.thr_mod = thr_mod
        self.encode_size = self.num_step  # No dual frequency, just num_step

        # Calculate sin and cos coefficients for phase-shifting
        coef_sin_cpu = torch.tensor(
            [torch.sin(2 * k * torch.pi / torch.tensor([self.num_step])) for k in range(self.num_step)]
        )
        coef_cos_cpu = torch.tensor(
            [torch.cos(2 * k * torch.pi / torch.tensor([self.num_step])) for k in range(self.num_step)]
        )

        self.register_buffer("coef_sin", coef_sin_cpu)
        self.register_buffer("coef_cos", coef_cos_cpu)


    def encode(self, angle_targets: torch.Tensor) -> torch.Tensor:
        """Phase-Shifting Encoder.

        Args:
            angle_targets (Tensor): Angle offset for each scale level.
                Shape (num_anchors * H * W, 1)

        Returns:
            Tensor: The PSC coded data (phase-shifting patterns)
                for each scale level.
                Shape (num_anchors * H * W, encode_size)
        """
        phase_targets = angle_targets * 2  # Double the angles
        # Compute phase shift targets for encoding
        phase_shift_targets = tuple(
            torch.cos(phase_targets + 2 * torch.pi * x / self.num_step)
            for x in range(self.num_step)
        )

        # Concatenate results along the last dimension
        return torch.stack(phase_shift_targets, dim=-1)


    def decode(self, angle_preds: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
        """Phase-Shifting Decoder.

        Args:
            angle_preds (Tensor): The PSC coded data (phase-shifting patterns).
                Shape (num_anchors * H * W, encode_size)
            keepdim (bool): Whether the output tensor has dim retained or not.

        Returns:
            Tensor: Angle offset for each scale level.
                Shape (num_anchors * H * W, 1) when keepdim is True,
                (num_anchors * H * W) otherwise.
        """
        # Move coefficients to same device as input
        self.coef_sin = self.coef_sin
        self.coef_cos = self.coef_cos

        # Calculate phase using sin and cos components
        phase_sin = torch.sum(angle_preds * self.coef_sin, dim=-1, keepdim=keepdim)
        phase_cos = torch.sum(angle_preds * self.coef_cos, dim=-1, keepdim=keepdim)

        # Compute modulation
        phase_mod = phase_cos**2 + phase_sin**2

        # Calculate phase angle in range [-pi, pi)
        phase = -torch.atan2(phase_sin, phase_cos)

        # Set angle of isotropic objects to zero based on modulation threshold
        phase[phase_mod < self.thr_mod] *= 0

        # Final decoded angle
        angle_pred = phase / 2
        return angle_pred


# The following function is based on code by Sunghoon Im, 
# licensed under the MIT License.
# See LICENSE file for details.
class disparityregression(nn.Module):
    def __init__(self, maxdisp):
        super(disparityregression, self).__init__()
        # self.disp = Variable(torch.arange(maxdisp, device="cuda:0"), requires_grad=False)#Variable(torch.Tensor(np.array(range(maxdisp))).cuda(), requires_grad=False)

    def forward(self, x, depth):
        out = torch.sum(x * depth[None, :, None, None],1)
        return out


# The following class is based on code by Sunghoon Im, 
# licensed under the MIT License.
# See LICENSE file for details.
class hourglass(nn.Module):
    def __init__(self, inplanes, gn=False):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(inplanes, inplanes * 2, kernel_size=3, stride=2, pad=1, gn=gn),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1, gn=gn)

        self.conv3 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=2, pad=1, gn=gn),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1, gn=gn),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(inplanes * 2, inplanes * 2, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(inplanes * 2) if not gn else nn.GroupNorm(32, inplanes * 2))  # +conv2

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(inplanes * 2, inplanes, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(inplanes) if not gn else nn.GroupNorm(32, inplanes))  # +x

    def forward(self, x, presqu, postsqu):

        out = self.conv1(x)  # in:1/4 out:1/8
        pre = self.conv2(out)  # in:1/8 out:1/8
        if postsqu is not None:
            pre = F.relu(pre + postsqu, inplace=True)
        else:
            pre = F.relu(pre, inplace=True)

        out = self.conv3(pre)  # in:1/8 out:1/16
        out = self.conv4(out)  # in:1/16 out:1/16

        if presqu is not None:
            post = F.relu(self.conv5(out) + presqu, inplace=True)  # in:1/16 out:1/8
        else:
            post = F.relu(self.conv5(out) + pre, inplace=True)

        out = self.conv6(post)  # in:1/8 out:1/4

        return out, pre, post


# The following class is based on code by Sunghoon Im, 
# licensed under the MIT License.
# See LICENSE file for details.
class hourglass2d(nn.Module):
    def __init__(self, inplanes, gn=False, groups=32):
        super(hourglass2d, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, inplanes * 2, kernel_size=3, stride=2, pad=1, dilation=1, gn=gn, groups=groups),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1, dilation=1, gn=gn, groups=groups)

        self.conv3 = nn.Sequential(convbn(inplanes * 2, inplanes * 2, kernel_size=3, stride=2, pad=1, dilation=1, gn=gn, groups=groups),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1, dilation=1, gn=gn, groups=groups),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(inplanes * 2, inplanes * 2, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm2d(inplanes * 2) if not gn else nn.GroupNorm(groups, inplanes * 2))  # +conv2

        self.conv6 = nn.Sequential(
            nn.ConvTranspose2d(inplanes * 2, inplanes, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm2d(inplanes) if not gn else nn.GroupNorm(groups, inplanes))  # +x

    def forward(self, x, presqu, postsqu):

        out = self.conv1(x)  # in:1/4 out:1/8
        pre = self.conv2(out)  # in:1/8 out:1/8
        if postsqu is not None:
            pre = F.relu(pre + postsqu, inplace=True)
        else:
            pre = F.relu(pre, inplace=True)

        out = self.conv3(pre)  # in:1/8 out:1/16
        out = self.conv4(out)  # in:1/16 out:1/16

        if presqu is not None:
            post = F.relu(self.conv5(out) + presqu, inplace=True)  # in:1/16 out:1/8
        else:
            post = F.relu(self.conv5(out) + pre, inplace=True)

        out = self.conv6(post)  # in:1/8 out:1/4

        return out, pre, post


# The following function is based on code by Sunghoon Im, 
# licensed under the MIT License.
# See LICENSE file for details.
def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation, gn=False, groups=32):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation = dilation, bias=False),
                         nn.BatchNorm2d(out_planes) if not gn else nn.GroupNorm(groups, out_planes))


# The following function is based on code by Sunghoon Im, 
# licensed under the MIT License.
# See LICENSE file for details.
def convbn_3d(in_planes, out_planes, kernel_size, stride, pad, gn=False, groups=32):
    return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride,bias=False),
                         nn.BatchNorm3d(out_planes) if not gn else nn.GroupNorm(groups, out_planes))


# The following class is based on code by Sunghoon Im, 
# licensed under the MIT License.
# See LICENSE file for details.
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride, downsample, pad, dilation, gn=False):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation, gn=gn),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation, gn=gn)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out


# The following class is based on code by Sunghoon Im, 
# licensed under the MIT License.
# See LICENSE file for details.
class FeatureExtractor(nn.Module):
    def __init__(self, cfg, rgb=True, ref_cam=False):
        super(FeatureExtractor, self).__init__()
        gn = cfg["group_norm"]
        self.inplanes = 32
        if rgb:
            in_channels = 3
        else:
            in_channels = 1
        self.firstconv = nn.Sequential(convbn(in_channels, 32, 3, 2, 1, 1, gn=gn),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1, gn=gn),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1, gn=gn),
                                       nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1,1,1, gn=gn)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2,1,1, gn=gn) 
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1,1,1, gn=gn)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1,1,2, gn=gn)


        self.branch1 = nn.Sequential(nn.AvgPool2d((32, 32), stride=(32,32)),
                                     convbn(128, 32, 1, 1, 0, 1, gn=gn),
                                     nn.ReLU(inplace=True))

        self.branch2 = nn.Sequential(nn.AvgPool2d((16, 16), stride=(16,16)),
                                     convbn(128, 32, 1, 1, 0, 1, gn=gn),
                                     nn.ReLU(inplace=True))

        self.branch3 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8,8)),
                                     convbn(128, 32, 1, 1, 0, 1, gn=gn),
                                     nn.ReLU(inplace=True))

        self.branch4 = nn.Sequential(nn.AvgPool2d((4, 4), stride=(4,4)),
                                     convbn(128, 32, 1, 1, 0, 1, gn=gn),
                                     nn.ReLU(inplace=True))

        self.lastconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1, gn),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(128, 32, kernel_size=1, padding=0, stride = 1, bias=False))
        
        if ref_cam:
            self.rpnconv = nn.Sequential(convbn(320, 64, 3, 1, 1, 1, gn),
                                        nn.ReLU(inplace=True),
                                        convbn(64, 32, 3, 1, 1, 1, gn),
                                        nn.ReLU(inplace=True))


    def _make_layer(self, block, planes, blocks, stride, pad, dilation, gn=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
           downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion) if not gn else nn.GroupNorm(32, planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation, gn=gn))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,1,None,pad,dilation, gn=gn))

        return nn.Sequential(*layers)


    def forward(self, x, is_left=True):
        output      = self.firstconv(x)
        output      = self.layer1(output)
        output_raw  = self.layer2(output)
        output      = self.layer3(output_raw)
        output_skip = self.layer4(output)


        output_branch1 = self.branch1(output_skip)
        output_branch1 = F.interpolate(output_branch1, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear')

        output_branch2 = self.branch2(output_skip)
        output_branch2 = F.interpolate(output_branch2, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear')

        output_branch3 = self.branch3(output_skip)
        output_branch3 = F.interpolate(output_branch3, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear')

        output_branch4 = self.branch4(output_skip)
        output_branch4 = F.interpolate(output_branch4, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear')

        output_feature = torch.cat((output_raw, output_skip, output_branch4, output_branch3, output_branch2, output_branch1), 1)
        output_feature1 = self.lastconv(output_feature)

        if is_left:
            rpns_feature = self.rpnconv(output_feature)
        else:
            rpns_feature = None
        return output_feature1, rpns_feature



class DepthRegression(nn.Module):
    def __init__(self, cfg):
        super(DepthRegression, self).__init__()

        self.cfg = cfg
        gn = cfg["group_norm"]
        self.mindepth = cfg["mindepth"]
        self.maxdepth = cfg["maxdepth"]
        self.depth_sampling = cfg["depth_sampling"]

        # check if rgb or grey cameras and initialize the feature extractor
        if cfg["cameras"]["cam0"] or cfg["cameras"]["cam1"]:
            if cfg["ref_cam"] == "cam0" or cfg["ref_cam"] == "cam1" or cfg["fusion"]:
                self.feature_extraction_grey = FeatureExtractor(cfg, rgb=False, ref_cam=True)
            else:
                self.feature_extraction_grey = FeatureExtractor(cfg, rgb=False, ref_cam=False)
        if cfg["cameras"]["cam2"] or cfg["cameras"]["cam3"]:
            if cfg["ref_cam"] == "cam2" or cfg["ref_cam"] == "cam3" or cfg["fusion"]:
                self.feature_extraction_rgb = FeatureExtractor(cfg, rgb=True, ref_cam=True)
            else:
                self.feature_extraction_rgb = FeatureExtractor(cfg, rgb=True, ref_cam=False)
            
        cam_num = 0
        for key in cfg["cameras"]:
            if cfg["cameras"][key]:
                cam_num += 1

        self.depth_scale = round((self.maxdepth - self.mindepth) / self.depth_sampling) / 4
        self.compressed_depths = torch.arange(self.mindepth, self.maxdepth, self.depth_sampling)[2::4]
        self.depth_range_cpu = torch.arange(self.mindepth, self.maxdepth, self.depth_sampling)
        self.register_buffer("depth_range", self.depth_range_cpu)
        

        self.Z_MIN, self.Z_MAX = cfg["z_min"], cfg["z_max"]
        self.Y_MIN, self.Y_MAX = cfg["y_min"], cfg["y_max"]
        self.X_MIN, self.X_MAX = cfg["x_min"], cfg["x_max"]
        self.VOXEL_Z_SIZE, self.VOXEL_Y_SIZE, self.VOXEL_X_SIZE = cfg["z_size"], cfg["y_size"], cfg["x_size"]

        # /2 to get the center of the voxel
        zs = torch.arange(self.Z_MAX, self.Z_MIN, self.VOXEL_Z_SIZE) + self.VOXEL_Z_SIZE / 2.
        ys = torch.arange(self.Y_MIN, self.Y_MAX, self.VOXEL_Y_SIZE).flip(0) + self.VOXEL_Y_SIZE / 2.
        xs = torch.arange(self.X_MIN, self.X_MAX, self.VOXEL_X_SIZE) + self.VOXEL_X_SIZE / 2.
        zs, ys, xs = torch.meshgrid(zs, ys, xs)
        coord_rect = torch.stack([xs, ys, zs], dim=-1)
        self.register_buffer("coord_rect", coord_rect)


        res_dim = 64
        input_res_dim = 32 * cam_num

        self.dres0 = nn.Sequential(convbn_3d(input_res_dim, res_dim, 3, 1, 1, gn=gn),
                                    nn.ReLU(inplace=True),
                                    convbn_3d(res_dim, res_dim, 3, 1, 1, gn=gn),
                                    nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(res_dim, res_dim, 3, 1, 1, gn=gn),
                                    nn.ReLU(inplace=True),
                                    convbn_3d(res_dim, res_dim, 3, 1, 1, gn=gn))
        
        self.dres2 = hourglass(res_dim, gn=gn)

        self.classif1 = nn.Sequential(convbn_3d(res_dim, res_dim, 3, 1, 1, gn=gn),
                                              nn.ReLU(inplace=True),
                                              nn.Conv3d(res_dim, 1, kernel_size=3, padding=1, stride=1, bias=False))
        
        self.dispregression = disparityregression(len(self.depth_range))
        
        self.xy_hourglass = hourglass2d(99, gn=gn, groups=9)
        self.xz_hourglass = hourglass2d(99, gn=gn, groups=9)
        self.yz_hourglass = hourglass2d(99, gn=gn, groups=9)

        self.view_aggregation = nn.Sequential(convbn(3*99, 3*99, 3, 1, 1, 1, gn=gn, groups=11),
                nn.ReLU(inplace=True),
                convbn(3*99, 3*99, 3, 1, 1, 1, gn=gn, groups=11))

        if cfg["is_lcod"]:
            in_feature_pool = 297
        else:
            in_feature_pool = 297

        self.rpn3d_conv2 = nn.Sequential(convbn(in_feature_pool, 128, 3, 1, 1, 1, gn=gn, groups=32),
                    nn.ReLU(inplace=True))
        
        self.rpn3d_conv3 = hourglass2d(128, gn=gn)

        self.rpn3d_cls_convs = nn.Sequential(convbn(128, 128, 3, 1, 1, 1, gn=gn),
                nn.ReLU(inplace=True))
        self.rpn3d_bbox_convs = nn.Sequential(convbn(128, 128, 3, 1, 1, 1, gn=gn),
                nn.ReLU(inplace=True))
        
        self.rpn3d_cls_convs2 = nn.Sequential(convbn(128, 128, 3, 1, 1, 1, gn=gn),
                nn.ReLU(inplace=True))
        self.rpn3d_bbox_convs2 = nn.Sequential(convbn(128, 128, 3, 1, 1, 1, gn=gn),
                nn.ReLU(inplace=True))
        
        self.bbox_cls = nn.Conv2d(128, 1, kernel_size=3, padding=1, stride=1)
        self.bbox_reg_cent = nn.Conv2d(128, 2, kernel_size=3, padding=1, stride=1)
        self.bbox_reg_dim = nn.Conv2d(128, 2, kernel_size=3, padding=1, stride=1)
        self.bbox_reg_angle = nn.Conv2d(128, 4, kernel_size=3, padding=1, stride=1)
        self.bbox_centerness = nn.Conv2d(128, 1, kernel_size=3, padding=1, stride=1)


        #torch.manual_seed(1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()


        torch.nn.init.normal_(self.bbox_cls.weight, std=0.1)
        torch.nn.init.constant_(self.bbox_cls.bias, 0)
        torch.nn.init.normal_(self.bbox_centerness.weight, std=0.1)
        torch.nn.init.constant_(self.bbox_centerness.bias, 0)
        torch.nn.init.normal_(self.bbox_reg_cent.weight, std=0.02)
        torch.nn.init.constant_(self.bbox_reg_cent.bias, 0)
        torch.nn.init.normal_(self.bbox_reg_dim.weight, std=0.02)
        torch.nn.init.constant_(self.bbox_reg_dim.bias, 0)
        torch.nn.init.normal_(self.bbox_reg_angle.weight, std=0.02)
        torch.nn.init.constant_(self.bbox_reg_angle.bias, 0)

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.bbox_cls.bias, bias_value)

        if cfg["load_weights"]:
            checkpoint = torch.load(cfg["checkpoint_path_fe"], weights_only=True, map_location='cpu')
            state_dict = checkpoint["state_dict"]
            # Remove "depth_regression_model." prefix from all keys
            new_state_dict = {k.replace("depth_regression_model.", ""): v for k, v in state_dict.items()}
            self.load_state_dict(new_state_dict, strict=False)
            # frezze layers in state dict
            for name, param in self.named_parameters():
                if name in new_state_dict:  # Freeze only existing layers
                    param.requires_grad = False
                    

        if cfg["fusion"]:
            self.fusion_2d_rpn_features = torch.nn.Sequential(
                torch.nn.Conv3d(int(32*cam_num), 32, kernel_size=3, padding=1, stride=1, bias=False),
                torch.nn.GroupNorm(32, 32),
                torch.nn.ReLU(inplace=True)
            )


    def initialize_weights(self, modules):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
                
                
    def get_normalized_img_points(self, coords, K, T_cam_world, x_max, y_max):
        ones = torch.ones_like(coords[..., 0:1])
        coord_hom = torch.cat([coords, ones], dim=-1)
        coord_cam = torch.einsum('bij, dhwj -> bdhwi', T_cam_world, coord_hom)
        
        img_points = torch.einsum('bij, bdhwj -> bdhwi', K, coord_cam)
        img_points[:,:,:,:,0] /= img_points[:,:,:,:,2]
        img_points[:,:,:,:,1] /= img_points[:,:,:,:,2]
        img_points = img_points[:,:,:,:,:2]
        # add z value to img points
        img_points = torch.cat([img_points, self.coord_rect[..., 2:].unsqueeze(0).repeat(K.size(0),1,1,1,1)], dim=-1)

        # normalize for grid sampling 0 to -1 and max img size to +1
        x_min, x_max = 0, x_max
        y_min, y_max = 0, y_max
        z_min, z_max = self.Z_MIN, self.Z_MAX
        min_norm_const = torch.tensor([x_min, y_min, z_max], device=img_points.device, dtype=img_points.dtype)
        max_norm_const = torch.tensor([x_max, y_max, z_min], device=img_points.device, dtype=img_points.dtype)
        norm_img_points = (img_points - min_norm_const) / (max_norm_const - min_norm_const)
        norm_img_points = norm_img_points * 2 - 1.
        return norm_img_points


    def forward(self, ref, targets, pose, intrinsics, intrinsics_inv, T_cam_plane=None):
        #intrinsics4 = intrinsics.clone()
        intrinsics4 = intrinsics
        intrinsics_inv4 = intrinsics_inv.clone()
        intrinsics_inv4[:,:2,:2] = intrinsics_inv4[:,:2,:2] * 4

        if ref.shape[1] == 1:
            refimg_fea, rpn_feature_left     = self.feature_extraction_grey(ref, is_left=True)
        else:
            refimg_fea, rpn_feature_left     = self.feature_extraction_rgb(ref, is_left=True)

        rpn_feature_list = []

        disp2depth = torch.ones(refimg_fea.size(0), refimg_fea.size(2), refimg_fea.size(3), device=refimg_fea.device)
        cost = torch.zeros(refimg_fea.size(0), refimg_fea.size(1) * (len(targets) + 1), int(self.depth_scale), refimg_fea.size(2), refimg_fea.size(3), device=refimg_fea.device)
        for j, target in enumerate(targets):
            intrinsics4[j][:,:2,:] = intrinsics4[j][:,:2,:] / 4
            intrinsics_tmp = intrinsics4[j]
            if target.shape[1] == 1:
                if self.cfg["fusion"]:
                    targetimg_fea, rpn_feature = self.feature_extraction_grey(target, is_left=True)
                    rpn_feature_list.append(rpn_feature)
                else:
                    targetimg_fea, rpn_feature = self.feature_extraction_grey(target, is_left=False)
            else:
                if self.cfg["fusion"]:
                    targetimg_fea, rpn_feature = self.feature_extraction_rgb(target, is_left=True)
                    rpn_feature_list.append(rpn_feature)
                else:
                    targetimg_fea, rpn_feature = self.feature_extraction_rgb(target, is_left=False)
            for i in range(int(self.depth_scale)):
                depth = self.compressed_depths.flip(0)[i] * disp2depth

                targetimg_fea_t = inverse_warp(targetimg_fea, depth, pose[j], intrinsics_tmp, intrinsics_inv4)
                if j == 0:
                    cost[:, :refimg_fea.size()[1], i, :,:] = refimg_fea
                    cost[:, refimg_fea.size()[1]:refimg_fea.size()[1]*2, i, :,:] = targetimg_fea_t
                else:
                    cost[:, refimg_fea.size()[1]*(j+1):refimg_fea.size()[1]*(j+2), i, :,:] = targetimg_fea_t

        cost = cost.contiguous()

        cost0 = self.dres0(cost)   # 2 x conv
        cost0 = self.dres1(cost0) + cost0 # 2 x conv

        out1, _, _ = self.dres2(cost0, None, None)
        out1 = out1 + cost0

        cost1 = self.classif1(out1) # 2+ 3d conv

        out, cost = out1, cost1

        cost1 = F.interpolate(cost1, [len(self.depth_range), ref.size()[2], ref.size()[3]], mode='trilinear', align_corners=True)
        cost1 = torch.squeeze(cost1, 1)
        pred1_softmax = F.softmax(cost1, dim=1)
        pred1 = self.dispregression(pred1_softmax, depth=self.depth_range.flip(0))


        # define pointcloud [b,Z,Y,X,3]
        coord_rect = self.coord_rect
        # project to image space
        # Use K_l not sclaed because we normalize image coords and therefore no sclae is needed
        K_l = torch.linalg.inv(intrinsics_inv)
        
        # for left camera standard transform
        Rt_identity = torch.cat((torch.eye(3, device=K_l.device), torch.zeros((3, 1), device=K_l.device)), dim=1).unsqueeze(0).repeat(K_l.shape[0], 1, 1)
        
        # transform from plane to camera
        if self.cfg["is_lcod"]:
            ones = torch.ones_like(coord_rect[..., 0:1])
            coord_rect_hom = torch.cat([coord_rect, ones], dim=-1)
            coord_rect = torch.einsum('bij, dhwj -> dhwi', T_cam_plane, coord_rect_hom)

        norm_img_points = self.get_normalized_img_points(coord_rect, K_l, Rt_identity, ref.size(3), ref.size(2))

        valids = (norm_img_points[..., 0] >= -1.) & (norm_img_points[..., 0] <= 1.) & \
            (norm_img_points[..., 1] >= -1.) & (norm_img_points[..., 1] <= 1.) & \
            (norm_img_points[..., 2] >= -1.) & (norm_img_points[..., 2] <= 1.)
        valids = valids.float()

        CV_feature = torch.cat([out, cost.detach()], dim= 1)

        Voxel = F.grid_sample(CV_feature, norm_img_points, align_corners=True)
        Voxel = Voxel * valids[:, None, :, :, :]

        # get voxel feature from pred dist
        pred_disp = F.grid_sample(pred1_softmax.detach()[:, None], norm_img_points, align_corners=True)
        pred_disp = pred_disp * valids[:, None, :, :, :]

        # get voxel feature from left image features
        valids = (norm_img_points[..., 0] >= -1.) & (norm_img_points[..., 0] <= 1.) & \
                (norm_img_points[..., 1] >= -1.) & (norm_img_points[..., 1] <= 1.)          
        valids = valids.float()

        batch_size = ref.size(0)
        X = Voxel.size(4)
        Y = Voxel.size(3)
        Z = Voxel.size(2)
        Voxel_2D = []
        for l in range(batch_size):
            RPN_feature_per_im = rpn_feature_left[l:l+1]
            # # LOOP is SLOW but need less memory
            # for k in range(len(norm_img_points[l])):
            #     Voxel_2D_feature = F.grid_sample(RPN_feature_per_im, norm_img_points[l, k:k+1, :, :, :2], align_corners=True)
            #     Voxel_2D.append(Voxel_2D_feature)

            # # FASTER but need more memory, alternative could be mini batches
            RPN_feature_per_im = RPN_feature_per_im.expand(len(norm_img_points[l]), -1, -1, -1)
            Voxel_2D_feature = F.grid_sample(RPN_feature_per_im, norm_img_points[l,:,:,:,:2], align_corners=True)
            Voxel_2D.append(Voxel_2D_feature)

        Voxel_2D = torch.cat(Voxel_2D, dim=0)
        Voxel_2D = Voxel_2D.reshape(batch_size, Z, -1, Y, X).transpose(1,2)
        Voxel_2D = Voxel_2D * valids[:, None, :, :, :]

        # multiply with depth prediction weights
        Voxel_2D = Voxel_2D * pred_disp

        Voxel_2D_targets = []
        if self.cfg["fusion"]:
            for j in range(len(targets)):
                K = intrinsics[j]
                target_img = targets[j]
                single_pose = pose[j]
                norm_img_points = self.get_normalized_img_points(coord_rect, K, single_pose, target_img.size(3), target_img.size(2))
                valids = (norm_img_points[..., 0] >= -1.) & (norm_img_points[..., 0] <= 1.) & \
                        (norm_img_points[..., 1] >= -1.) & (norm_img_points[..., 1] <= 1.)          
                valids = valids.float()
                
                Voxel_2D_one_target = []
                for l in range(batch_size):
                    RPN_feature_per_im = rpn_feature_list[j][l:l+1]
                    # # FASTER but need more memory, alternative could be mini batches
                    RPN_feature_per_im = RPN_feature_per_im.expand(len(norm_img_points[l]), -1, -1, -1)
                    Voxel_2D_feature = F.grid_sample(RPN_feature_per_im, norm_img_points[l,:,:,:,:2], align_corners=True)
                    Voxel_2D_one_target.append(Voxel_2D_feature)
                Voxel_2D_one_target = torch.cat(Voxel_2D_one_target, dim=0)
                Voxel_2D_one_target = Voxel_2D_one_target.reshape(batch_size, Z, -1, Y, X).transpose(1,2)
                Voxel_2D_one_target = Voxel_2D_one_target * valids[:, None, :, :, :]
                Voxel_2D_one_target = Voxel_2D_one_target * pred_disp
                Voxel_2D_targets.append(Voxel_2D_one_target)
            Voxel_2D_targets = torch.cat(Voxel_2D_targets, dim=1)
            
            Voxel_2D = torch.cat([Voxel_2D, Voxel_2D_targets], dim=1)
            # fusion layers
            Voxel_2D = self.fusion_2d_rpn_features(Voxel_2D)

        # concat 2d and 3d features
        Voxel = torch.cat([Voxel, Voxel_2D], dim=1)

        xy_view = Voxel.mean(2)
        xz_view = Voxel.mean(3)
        yz_view = Voxel.mean(4)

        xy_coord_feat = self.coord_rect[0,:,:,:2].permute(2,0,1).unsqueeze(0).expand(batch_size,-1,-1,-1) * 0
        xz_coord_feat = self.coord_rect[:,0,:,[0,2]].permute(2,0,1).unsqueeze(0).expand(batch_size,-1,-1,-1) * 0
        yz_coord_feat = self.coord_rect[:,:,0,1:].permute(2,0,1).unsqueeze(0).expand(batch_size,-1,-1,-1) * 0

        xy_view = torch.cat([xy_view, xy_coord_feat], dim=1)
        xz_view = torch.cat([xz_view, xz_coord_feat], dim=1)
        yz_view = torch.cat([yz_view, yz_coord_feat], dim=1)

        xy_feat_resized = F.interpolate(xy_view, size=(Voxel.size(2), Voxel.size(4)), mode='bilinear', align_corners=False)
        yz_feat_resized = F.interpolate(yz_view, size=(Voxel.size(2), Voxel.size(4)), mode='bilinear', align_corners=False)

        del Voxel, Voxel_2D

        xy_feat,_,_ = self.xy_hourglass(xy_feat_resized, None, None)
        yz_feat,_,_ = self.yz_hourglass(yz_feat_resized, None, None)
        xz_feat,_,_ = self.xz_hourglass(xz_view, None, None)

        tri_perpsective_feat = torch.cat([xy_feat, yz_feat, xz_feat], dim=1)
        tri_perpsective_feat_agg = self.view_aggregation(tri_perpsective_feat)
        tri_perpsective_feat_agg = F.relu(tri_perpsective_feat + tri_perpsective_feat_agg, inplace=True)

        Voxel_BEV = self.rpn3d_conv2(tri_perpsective_feat_agg)
        
        # 2d hourglass
        Voxel_BEV, pre_BEV, post_BEV = self.rpn3d_conv3(Voxel_BEV, None, None)

        Voxel_BEV_cls = self.rpn3d_cls_convs(Voxel_BEV)
        Voxel_BEV_bbox = self.rpn3d_bbox_convs(Voxel_BEV)

        Voxel_BEV_cls = self.rpn3d_cls_convs2(Voxel_BEV_cls)
        Voxel_BEV_bbox = self.rpn3d_bbox_convs2(Voxel_BEV_bbox)
        
        bbox_cls = self.bbox_cls(Voxel_BEV_cls)
        bbox_reg_cent = self.bbox_reg_cent(Voxel_BEV_bbox)
        bbox_reg_dim = self.bbox_reg_dim(Voxel_BEV_bbox)
        bbox_centerness = self.bbox_centerness(Voxel_BEV_bbox)
        bbox_reg_angle = self.bbox_reg_angle(Voxel_BEV_bbox)

        bbox_reg_angle = 2 * torch.nn.functional.sigmoid(bbox_reg_angle) -1
        bbox_reg_dim = torch.nn.functional.softplus(bbox_reg_dim)

        return pred1, bbox_cls, bbox_reg_cent, bbox_reg_dim, bbox_centerness, bbox_reg_angle
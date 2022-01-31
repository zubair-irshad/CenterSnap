import torch
import torch.nn as nn
import torch.nn.functional as F
from simnet.lib.net.models.layers.residual_blocks import PreactBasicResidualBlock
from simnet.lib.net.models import panoptic_backbone

#RGB-D
class RGBDStem(nn.Module):

  def __init__(self, hparams, in_channels=3):
    super().__init__()
    self.rgb_stem = panoptic_backbone.BasicStem(in_channels=in_channels, out_channels=32)
    self.depth_stem = panoptic_backbone.BasicStem(in_channels=1, out_channels=1)
    self.disp_features = PreactBasicResidualBlock(
        1, 32, stride=1, dilation_rate=5, add_preact=True, add_last_norm=False
    )

  def forward(self, stacked_img, robot_joint_angles=None):
    small_disp = self.depth_stem.forward(stacked_img[:, 3:], None)
    left_rgb_features = self.rgb_stem.forward(stacked_img[:, 0:3], robot_joint_angles)
    disp_features = self.disp_features(small_disp)
    return torch.cat((disp_features, left_rgb_features), axis=1), small_disp

  @property
  def out_channels(self):
    return self.rgb_stem.out_channels

  @property
  def stride(self):
    return 4  # = stride 2 conv -> stride 2 max pool
import numpy as np
import torch.nn as nn

from simnet.lib.net.models.panoptic_backbone import SemSegFPNHead, PoseFPNHead, ShapeSpec, output_shape, build_resnet_fpn_backbone
from simnet.lib.net.models.basic_stem import RGBDStem
from simnet.lib.net.post_processing import segmentation_outputs
from simnet.lib.net.post_processing import depth_outputs
from simnet.lib.net.post_processing import pose_outputs
from simnet.lib.net.post_processing import abs_pose_outputs


def res_fpn(hparams):
  return PanopticNet(hparams)

class DepthHead(nn.Module):
  def __init__(self, backbone_output_shape, hparams):
    super().__init__()
    self.head = SemSegFPNHead(
        backbone_output_shape,
        num_classes=1,
        model_norm=hparams.model_norm,
        num_filters_scale=hparams.num_filters_scale
    )
    self.hparams = hparams
  def forward(self, features):
    depth_pred = self.head.forward(features).squeeze(dim=1)
    return depth_outputs.DepthOutput(depth_pred, self.hparams)


class SegmentationHead(nn.Module):
  def __init__(self, backbone_output_shape, num_classes, hparams):
    super().__init__()
    self.head = SemSegFPNHead(
        backbone_output_shape,
        num_classes=num_classes,
        model_norm=hparams.model_norm,
        num_filters_scale=hparams.num_filters_scale
    )
    self.hparams = hparams
  def forward(self, features):
    pred = self.head.forward(features)
    return segmentation_outputs.SegmentationOutput(pred, self.hparams)


class PoseHead(nn.Module):

  def __init__(self, backbone_output_shape, hparams):
    super().__init__()
    self.hparams = hparams
    self.heatmap_head = SemSegFPNHead(
        backbone_output_shape,
        num_classes=1,
        model_norm=hparams.model_norm,
        num_filters_scale=hparams.num_filters_scale
    )
    self.vertex_head = PoseFPNHead(
        backbone_output_shape,
        num_classes=16,
        model_norm=hparams.model_norm,
        num_filters_scale=hparams.num_filters_scale
    )
    self.z_centroid_head = PoseFPNHead(
        backbone_output_shape,
        num_classes=1,
        model_norm=hparams.model_norm,
        num_filters_scale=hparams.num_filters_scale
    )

  def forward(self, features):
    z_centroid_output = self.z_centroid_head.forward(features).squeeze(dim=1)
    heatmap_output = self.heatmap_head.forward(features).squeeze(dim=1)
    vertex_output = self.vertex_head.forward(features)
    return pose_outputs.PoseOutput(heatmap_output, vertex_output, z_centroid_output, self.hparams)


class OBBHead(nn.Module):
  def __init__(self, backbone_output_shape, hparams):
    super().__init__()
    self.hparams = hparams
    self.heatmap_head = SemSegFPNHead(
        backbone_output_shape,
        num_classes=1,
        model_norm=hparams.model_norm,
        num_filters_scale=hparams.num_filters_scale
    )
    self.latent_embedding_head = PoseFPNHead(
        backbone_output_shape,
        num_classes=128,
        model_norm=hparams.model_norm,
        num_filters_scale=hparams.num_filters_scale
    )
    self.abs_pose_head = PoseFPNHead(
        backbone_output_shape,
        num_classes=13,
        model_norm=hparams.model_norm,
        num_filters_scale=hparams.num_filters_scale
    )
  def forward(self, features):
    heatmap_output = self.heatmap_head.forward(features).squeeze(dim=1)
    latent_emb_output = self.latent_embedding_head.forward(features)
    abs_pose_output = self.abs_pose_head.forward(features)
    return abs_pose_outputs.OBBOutput(
        heatmap_output, latent_emb_output, abs_pose_output, self.hparams
    )

class PanopticNet(nn.Module):

  def __init__(self, hparams):
    super().__init__()
    self.hparams = hparams
    input_shape = ShapeSpec(channels=3, height=512, width=640)
    stereo_stem = RGBDStem(hparams)
    self.backbone = build_resnet_fpn_backbone(
        input_shape,
        stereo_stem,
        model_norm=hparams.model_norm,
        num_filters_scale=hparams.num_filters_scale
    )
    shape = output_shape(self.backbone)
    # Add depth head.
    self.depth_head = DepthHead(shape, hparams)
    # Add segmentation head. 6+1 categories for NOCS
    self.seg_head = SegmentationHead(shape, 7, hparams)
    self.pose_head = OBBHead(shape, hparams)

  def forward(self, image):
    #fine tune only
    # self.backbone.eval()
    # with torch.no_grad():
    #   features, small_disp_output = self.backbone.forward(image)
    features, small_disp_output = self.backbone.forward(image)
    small_disp_output = small_disp_output.squeeze(dim=1)
    
    if self.hparams.frozen_stereo_checkpoint is not None:
      small_disp_output = small_disp_output.detach()
    small_depth_output = depth_outputs.DepthOutput(small_disp_output, self.hparams)
    seg_output = self.seg_head.forward(features)
    depth_output = self.depth_head.forward(features)
    pose_output = self.pose_head.forward(features)
    return seg_output, depth_output, small_depth_output, pose_output

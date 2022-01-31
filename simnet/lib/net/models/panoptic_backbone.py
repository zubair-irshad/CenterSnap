import abc
import math
import collections

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import fvcore.nn.weight_init as weight_init

import logging
loger = logging.getLogger('lightning')
TORCH_VERSION = (1, 4)

#Model cfg
#Hard code parameters for now here.
MODEL_RESNETS_DEPTH = 50
MODEL_RESNETS_OUT_FEATURES = ['res2', 'res3', 'res4', 'res5']
MODEL_META_ARCHITECTURE = "PanopticFPN"
# https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md
# COCO panoptic weights:
# https://dl.fbaipublicfiles.com/detectron2/COCO-PanopticSegmentation/panoptic_fpn_R_50_3x/139514569/model_final_c10459.pkl
#MODEL_WEIGHTS = '/data/good_ckpts/pretrained_cocopanoptic_c10459_pkl'

USE_STEREO_STEM = False

NUM_BLOCKS_PER_STAGE = [3, 4, 6, 3]
# NUM_BLOCKS_PER_STAGE = [2, 2, 2, 2]
USE_TOP_BLOCK_P6 = False

MODEL_FPN_IN_FEATURES = ['res2', 'res3', 'res4', 'res5']
# Options: FrozenBN, GN, "SyncBN", "BN"
MODEL_RESNETS_NORM = "GN"  # originally FrozenBN
# from: detectron2/modeling/backbone/resnet.py
MODEL_BACKBONE_FREEZE_AT = -1  # originally 2
# Parameters for seg fpn head.
if USE_TOP_BLOCK_P6:
  MODEL_SEM_SEG_HEAD_IN_FEATURES = ['p2', 'p3', 'p4', 'p5', 'p6']
else:
  MODEL_SEM_SEG_HEAD_IN_FEATURES = ['p2', 'p3', 'p4', 'p5']

MODEL_POSE_HEAD_IN_FEATURES = ['p3', 'p4', 'p5']
#MODEL_SEM_SEG_HEAD_CONVS_DIM = 128 // SCALE
MODEL_SEM_SEG_HEAD_IGNORE_VALUE = 255
MODEL_SEM_SEG_HEAD_COMMON_STRIDE = 4
MODEL_POSE_HEAD_COMMON_STRIDE = 8
#MODEL_SEM_SEG_HEAD_NORM = "GN"
MODEL_SEM_SEG_HEAD_LOSS_WEIGHT = 1.0

MODEL_RESNETS_NUM_GROUPS = 1
# Output width of res2. Scaling this parameters will scale the width of all 1x1 convs in ResNet
# For R18 and R34, this needs to be set to 64
MODEL_RESNETS_STEM_OUT_CHANNELS = 64  # // SCALE

# Baseline width of each group.
# Scaling this parameters will scale the width of all bottleneck layers.
#MODEL_RESNETS_RES2_OUT_CHANNELS = 256 // SCALE
MODEL_RESNETS_RES2_OUT_CHANNELS_MAX = 19999

#MODEL_RESNETS_WIDTH_PER_GROUP = max(64 // SCALE, 32)
MODEL_RESNETS_WIDTH_PER_GROUP_MAX = 19999

# Place the stride 2 conv on the 1x1 filter
# Use True only for the original MSRA ResNet; use False for C2 and Torch models
MODEL_RESNETS_STRIDE_IN_1X1 = True

# Apply dilation in stage "res5"
MODEL_RESNETS_RES5_DILATION = 1

#MODEL_FPN_NORM = 'GN'
# Types for fusing the FPN top-down and lateral features. Can be either "sum" or "avg"
MODEL_FPN_FUSE_TYPE = "sum"


def main():
  module = SimnetResFPNGrasp()

def c2_normal_fill(module: nn.Module) -> None:
    """
    Initialize `module.weight` using the "MSRAFill" implemented in Caffe2.
    Also initializes `module.bias` to 0.

    Args:
        module (torch.nn.Module): module to initialize.
    """
    # nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

    latent_size = 128
    nn.init.normal_(
        module.weight,
        0.0,
        1.0 / math.sqrt(latent_size),
    )

    if module.bias is not None:
        # pyre-fixme[6]: Expected `Tensor` for 1st param but got `Union[nn.Module,
        #  torch.Tensor]`.
        nn.init.constant_(module.bias, 0)



def build_resnet_fpn_backbone(input_shape, stereo_stem, model_norm='BN', num_filters_scale=4):
  # For now, we are just hardcode the global parameters. We plan to depracte dectron2 dependence this quarter.
  MODEL_FPN_NORM = model_norm
  MODEL_FPN_OUT_CHANNELS = 256 // num_filters_scale

  if USE_TOP_BLOCK_P6:
    top_block = LastLevelMaxPool()
  elif False:
    in_channels_p6p7 = bottom_up.output_shape()["res5"].channels
    top_block = LastLevelP6P7(in_channels_p6p7, out_channels)
  else:
    top_block = None

  bottom_up = build_resnet_backbone(
      input_shape, stereo_stem, model_norm=model_norm, num_filters_scale=num_filters_scale
  )
  in_features = MODEL_FPN_IN_FEATURES
  out_channels = MODEL_FPN_OUT_CHANNELS
  backbone = FPN(
      bottom_up=bottom_up,
      in_features=in_features,
      out_channels=out_channels,
      norm=MODEL_FPN_NORM,
      top_block=top_block,
      fuse_type=MODEL_FPN_FUSE_TYPE,
  )
  return backbone


def get_group_gn(dim, dim_per_gp, num_groups):
  """get number of groups used by GroupNorm, based on number of channels."""
  assert dim_per_gp == -1 or num_groups == -1, \
      "GroupNorm: can only specify G or C/G."

  if dim_per_gp > 0:
    assert dim % dim_per_gp == 0, \
        "dim: {}, dim_per_gp: {}".format(dim, dim_per_gp)
    group_gn = dim // dim_per_gp
  else:
    assert dim % num_groups == 0, \
        "dim: {}, num_groups: {}".format(dim, num_groups)
    group_gn = num_groups

  return group_gn


def group_norm(out_channels, affine=True, divisor=1):
  out_channels = out_channels // divisor
  dim_per_gp = -1 // divisor
  num_groups = 16 // divisor
  eps = 1e-5  # default: 1e-5
  return torch.nn.GroupNorm(
      get_group_gn(out_channels, dim_per_gp, num_groups), out_channels, eps, affine
  )


class FPNXconvFeatureExtractor(nn.Module):
  """
    Heads for FPN for classification
    """

  def __init__(self, in_channels, num_stacked_convs=4, conv_head_dim=16, use_gn=False):
    super(FPNXconvFeatureExtractor, self).__init__()
    xconvs = []
    for ix in range(num_stacked_convs):
      xconvs.append(
          nn.Conv2d(
              in_channels,
              conv_head_dim,
              kernel_size=3,
              stride=1,
              padding=1,
              bias=False if use_gn else True
          )
      )
      in_channels = conv_head_dim
      if use_gn:
        xconvs.append(group_norm(in_channels))
      xconvs.append(nn.ReLU(inplace=True))

    self.add_module("xconvs", nn.Sequential(*xconvs))
    for modules in [
        self.xconvs,
    ]:
      for l in modules.modules():
        if isinstance(l, nn.Conv2d):
          torch.nn.init.normal_(l.weight, std=0.01)
          if not use_gn:
            torch.nn.init.constant_(l.bias, 0)

  def forward(self, x):
    x = self.xconvs(x)
    return x


class SemSegFPNHead(nn.Module):
  """
  A semantic segmentation head described in detail in the Panoptic Feature Pyramid Networks paper
  (https://arxiv.org/abs/1901.02446). It takes FPN features as input and merges information from
  all levels of the FPN into single output.
  """

  def __init__(self, input_shape, num_classes, model_norm='BN', num_filters_scale=4):
    super().__init__()
    MODEL_SEM_SEG_HEAD_NORM = model_norm
    MODEL_SEM_SEG_HEAD_CONVS_DIM = 128 // num_filters_scale

    self.in_features = MODEL_SEM_SEG_HEAD_IN_FEATURES
    feature_strides = {k: v.stride for k, v in input_shape.items()}
    feature_channels = {k: v.channels for k, v in input_shape.items()}
    self_ignore_value = MODEL_SEM_SEG_HEAD_IGNORE_VALUE
    conv_dims = MODEL_SEM_SEG_HEAD_CONVS_DIM
    self.common_stride = MODEL_SEM_SEG_HEAD_COMMON_STRIDE
    norm = MODEL_SEM_SEG_HEAD_NORM
    self.bilinear_upsample = nn.Upsample(
        scale_factor=self.common_stride, mode="bilinear", align_corners=False
    )

    self.scale_heads = []
    for in_feature in self.in_features:
      head_ops = []
      head_length = max(1, int(np.log2(feature_strides[in_feature]) - np.log2(self.common_stride)))
      for k in range(head_length):
        norm_module = get_norm(norm, conv_dims)
        conv = Conv2d(
            feature_channels[in_feature] if k == 0 else conv_dims,
            conv_dims,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not norm,
            norm=norm_module,
            activation=F.relu,
        )
        weight_init.c2_msra_fill(conv)
        head_ops.append(conv)
        if feature_strides[in_feature] != self.common_stride:
          head_ops.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
      self.scale_heads.append(nn.Sequential(*head_ops))
      self.add_module(in_feature, self.scale_heads[-1])
    self.predictor = Conv2d(conv_dims, num_classes, kernel_size=1, stride=1, padding=0)
    weight_init.c2_msra_fill(self.predictor)

  def forward(self, features, targets=None):
    """
      Returns:
        In training, returns (None, dict of losses)
        In inference, returns (predictions, {})
      """
    x = self.layers(features)
    x = self.bilinear_upsample(x)
    return x

  def layers(self, features):
    for i, f in enumerate(self.in_features):
      if i == 0:
        x = self.scale_heads[i](F.relu(features[f]))
      else:
        x = x - -self.scale_heads[i](F.relu(features[f]))
    x = self.predictor(x)
    return x

  def losses(self, predictions, targets):
    predictions = F.interpolate(
        predictions, scale_factor=self.common_stride, mode="bilinear", align_corners=False
    )
    loss = F.cross_entropy(predictions, targets, reduction="mean", ignore_index=self.ignore_value)
    return loss


class PoseFPNHead(nn.Module):
  """
  A semantic segmentation head described in detail in the Panoptic Feature Pyramid Networks paper
  (https://arxiv.org/abs/1901.02446). It takes FPN features as input and merges information from
  all levels of the FPN into single output.
  """

  def __init__(self, input_shape, num_classes, model_norm='BN', num_filters_scale=4):
    super().__init__()
    MODEL_SEM_SEG_HEAD_NORM = model_norm
    MODEL_SEM_SEG_HEAD_CONVS_DIM = 128 // num_filters_scale
    self.in_features = MODEL_POSE_HEAD_IN_FEATURES
    feature_strides = {k: v.stride for k, v in input_shape.items()}
    feature_channels = {k: v.channels for k, v in input_shape.items()}
    self_ignore_value = MODEL_SEM_SEG_HEAD_IGNORE_VALUE
    conv_dims = MODEL_SEM_SEG_HEAD_CONVS_DIM
    self.common_stride = MODEL_POSE_HEAD_COMMON_STRIDE
    norm = MODEL_SEM_SEG_HEAD_NORM

    self.scale_heads = []
    for in_feature in self.in_features:
      head_ops = []
      head_length = max(1, int(np.log2(feature_strides[in_feature]) - np.log2(self.common_stride)))

      # loger.info("head length", head_length)
      # loger.info("substraction value",int(np.log2(feature_strides[in_feature]) - np.log2(self.common_stride)))
      # print("head length", head_length)
      for k in range(head_length):
        norm_module = get_norm(norm, conv_dims)
        conv = Conv2d(
            feature_channels[in_feature] if k == 0 else conv_dims,
            conv_dims,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not norm,
            norm=norm_module,
            activation=F.relu,
        )
        # c2_normal_fill(conv)
        weight_init.c2_msra_fill(conv)
        head_ops.append(conv)
        if feature_strides[in_feature] != self.common_stride:
          head_ops.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
      self.scale_heads.append(nn.Sequential(*head_ops))
      self.add_module(in_feature, self.scale_heads[-1])
    self.predictor = Conv2d(conv_dims, num_classes, kernel_size=1, stride=1, padding=0)
    # c2_normal_fill(self.predictor)
    weight_init.c2_msra_fill(self.predictor)

  def forward(self, features, targets=None):
    """
      Returns:
        In training, returns (None, dict of losses)
        In inference, returns (predictions, {})
      """
    x = self.layers(features)
    return x

  def layers(self, features):
    for i, f in enumerate(self.in_features):
      if i == 0:
        x = self.scale_heads[i](F.relu(features[f]))
      else:
        x = x - -self.scale_heads[i](F.relu(features[f]))
    x = self.predictor(x)
    return x


class PoseFPNHead_Latent(nn.Module):
  """
  A semantic segmentation head described in detail in the Panoptic Feature Pyramid Networks paper
  (https://arxiv.org/abs/1901.02446). It takes FPN features as input and merges information from
  all levels of the FPN into single output.
  """

  def __init__(self, input_shape, num_classes, model_norm='BN', num_filters_scale=4):
    super().__init__()
    MODEL_SEM_SEG_HEAD_NORM = model_norm
    MODEL_SEM_SEG_HEAD_CONVS_DIM = 128 // num_filters_scale
    self.in_features = MODEL_POSE_HEAD_IN_FEATURES
    feature_strides = {k: v.stride for k, v in input_shape.items()}
    feature_channels = {k: v.channels for k, v in input_shape.items()}
    self_ignore_value = MODEL_SEM_SEG_HEAD_IGNORE_VALUE
    conv_dims = MODEL_SEM_SEG_HEAD_CONVS_DIM
    self.common_stride = MODEL_POSE_HEAD_COMMON_STRIDE
    norm = MODEL_SEM_SEG_HEAD_NORM

    self.scale_heads = []
    for in_feature in self.in_features:
      head_ops = []
      head_length = max(1, int(np.log2(feature_strides[in_feature]) - np.log2(self.common_stride)))

      # loger.info("head length", head_length)
      # loger.info("substraction value",int(np.log2(feature_strides[in_feature]) - np.log2(self.common_stride)))
      # print("head length", head_length)
      # head_length = 4
      for k in range(head_length):

        # print("k", k)
        # print("feature_channels[in_feature]", feature_channels[in_feature])
        norm_module = get_norm(norm, conv_dims)
        conv = Conv2d(
            feature_channels[in_feature] if k == 0 else conv_dims,
            conv_dims,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not norm,
            norm=norm_module,
            activation=F.relu,
        )
        # c2_normal_fill(conv)
        weight_init.c2_msra_fill(conv)
        head_ops.append(conv)

        for _ in range(32):
            conv_head = Conv2d(conv_dims, conv_dims, kernel_size=3, stride=1, padding=1, bias=not norm, norm=norm_module, activation=F.relu,)
            weight_init.c2_msra_fill(conv_head)
            head_ops.append(conv_head)

        if feature_strides[in_feature] != self.common_stride:
          head_ops.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
      self.scale_heads.append(nn.Sequential(*head_ops))
      self.add_module(in_feature, self.scale_heads[-1])
    self.predictor = Conv2d(conv_dims, num_classes, kernel_size=1, stride=1, padding=0)
    # c2_normal_fill(self.predictor)
    weight_init.c2_msra_fill(self.predictor)

  def forward(self, features, targets=None):
    """
      Returns:
        In training, returns (None, dict of losses)
        In inference, returns (predictions, {})
      """
    x = self.layers(features)
    return x

  def layers(self, features):
    for i, f in enumerate(self.in_features):
      # print("i, f", i, f)

      # print("features[f])", features[f].shape)
      if i == 0:
        x = self.scale_heads[i](F.relu(features[f]))
      else:
        x = x - -self.scale_heads[i](F.relu(features[f]))
      # print("x", x.shape)
    x = self.predictor(x)
    # print("x after predictor", x.shape)
    # print("======================\n\n")
    return x


class ShapeSpec(collections.namedtuple("_ShapeSpec", ["channels", "height", "width", "stride"])):
  """
  A simple structure that contains basic shape specification about a tensor.
  It is often used as the auxiliary inputs/outputs of models,
  to obtain the shape inference ability among pytorch modules.

  Attributes:
    channels:
    height:
    width:
    stride:
  """

  def __new__(cls, *, channels=None, height=None, width=None, stride=None):
    return super().__new__(cls, channels, height, width, stride)


class Backbone(nn.Module, metaclass=abc.ABCMeta):

  def __init__(self):
    """
    The `__init__` method of any subclass can specify its own set of arguments.
    """
    super().__init__()

  @abc.abstractmethod
  def forward(self):
    """
    Subclasses must override this method, but adhere to the same return type.

    Returns:
      dict[str->Tensor]: mapping from feature name (e.g., "res2") to tensor
    """
    pass

  @property
  def size_divisibility(self):
    """
    Some backbones require the input height and width to be divisible by a
    specific integer. This is typically true for encoder / decoder type networks
    with lateral connection (e.g., FPN) for which feature maps need to match
    dimension in the "bottom up" and "top down" paths. Set to 0 if no specific
    input size divisibility is required.
    """
    return 0

  def output_shape(self):
    """
    Returns:
      dict[str->ShapeSpec]
    """
    # this is a backward-compatible default
    return {
        name: ShapeSpec(
            channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
        ) for name in self._out_features
    }


def get_norm(norm, out_channels):
  """
  Args:
    norm (str or callable): either one of BN, SyncBN, FrozenBN, GN;
      or a callable that takes a channel number and returns
      the normalization layer as a nn.Module.

  Returns:
    nn.Module or None: the normalization layer
  """
  if out_channels == 32:
    N = 16
  else:
    N = 32
  if isinstance(norm, str):
    if len(norm) == 0:
      return None
    norm = {
        "BN": BatchNorm2d,
        #"SyncBN": NaiveSyncBatchNorm,
        #"FrozenBN": FrozenBatchNorm2d,
        "GN": lambda channels: nn.GroupNorm(N, channels),
        #"nnSyncBN": nn.SyncBatchNorm,  # keep for debugging
    }[norm]
  return norm(out_channels)


def cat(tensors, dim=0):
  """
  Efficient version of torch.cat that avoids a copy if there is only a single element in a list
  """
  assert isinstance(tensors, (list, tuple))
  if len(tensors) == 1:
    return tensors[0]
  return torch.cat(tensors, dim)


class _NewEmptyTensorOp(torch.autograd.Function):

  @staticmethod
  def forward(ctx, x, new_shape):
    ctx.shape = x.shape
    return x.new_empty(new_shape)

  @staticmethod
  def backward(ctx, grad):
    shape = ctx.shape
    return _NewEmptyTensorOp.apply(grad, shape), None


class Conv2d(torch.nn.Conv2d):
  """
  A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
  """

  def __init__(self, *args, **kwargs):
    """
    Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

    Args:
      norm (nn.Module, optional): a normalization layer
      activation (callable(Tensor) -> Tensor): a callable activation function

    It assumes that norm layer is used before activation.
    """
    norm = kwargs.pop("norm", None)
    activation = kwargs.pop("activation", None)
    super().__init__(*args, **kwargs)

    self.norm = norm
    self.activation = activation

  def forward(self, x):
    if x.numel() == 0 and self.training:
      # https://github.com/pytorch/pytorch/issues/12013
      assert not isinstance(
          self.norm, torch.nn.SyncBatchNorm
      ), "SyncBatchNorm does not support empty inputs!"

    if x.numel() == 0 and TORCH_VERSION <= (1, 4):
      assert not isinstance(
          self.norm, torch.nn.GroupNorm
      ), "GroupNorm does not support empty inputs in PyTorch <=1.4!"
      # When input is empty, we want to return a empty tensor with "correct" shape,
      # So that the following operations will not panic
      # if they check for the shape of the tensor.
      # This computes the height and width of the output tensor
      output_shape = [(i + 2 * p - (di * (k - 1) + 1)) // s + 1 for i, p, di, k, s in
                      zip(x.shape[-2:], self.padding, self.dilation, self.kernel_size, self.stride)]
      output_shape = [x.shape[0], self.weight.shape[0]] + output_shape
      empty = _NewEmptyTensorOp.apply(x, output_shape)
      if self.training:
        # This is to make DDP happy.
        # DDP expects all workers to have gradient w.r.t the same set of parameters.
        _dummy = sum(x.view(-1)[0] for x in self.parameters()) * 0.0
        return empty + _dummy
      else:
        return empty

    x = super().forward(x)
    if self.norm is not None:
      x = self.norm(x)
    if self.activation is not None:
      x = self.activation(x)
    return x


if TORCH_VERSION > (1, 4):
  ConvTranspose2d = torch.nn.ConvTranspose2d
else:

  class ConvTranspose2d(torch.nn.ConvTranspose2d):
    """
    A wrapper around :class:`torch.nn.ConvTranspose2d` to support zero-size tensor.
    """

    def forward(self, x):
      if x.numel() > 0:
        return super(ConvTranspose2d, self).forward(x)
      # get output shape

      # When input is empty, we want to return a empty tensor with "correct" shape,
      # So that the following operations will not panic
      # if they check for the shape of the tensor.
      # This computes the height and width of the output tensor
      output_shape = [(i - 1) * d - 2 * p + (di * (k - 1) + 1) + op for i, p, di, k, d, op in zip(
          x.shape[-2:],
          self.padding,
          self.dilation,
          self.kernel_size,
          self.stride,
          self.output_padding,
      )]
      output_shape = [x.shape[0], self.out_channels] + output_shape
      # This is to make DDP happy.
      # DDP expects all workers to have gradient w.r.t the same set of parameters.
      _dummy = sum(x.view(-1)[0] for x in self.parameters()) * 0.0
      return _NewEmptyTensorOp.apply(x, output_shape) + _dummy


if TORCH_VERSION > (1, 4):
  BatchNorm2d = torch.nn.BatchNorm2d
else:

  class BatchNorm2d(torch.nn.BatchNorm2d):
    """
    A wrapper around :class:`torch.nn.BatchNorm2d` to support zero-size tensor.
    """

    def forward(self, x):
      if x.numel() > 0:
        return super(BatchNorm2d, self).forward(x)
      # get output shape
      output_shape = x.shape
      return _NewEmptyTensorOp.apply(x, output_shape)


if False:  # not yet fixed in pytorch
  Linear = torch.nn.Linear
else:

  class Linear(torch.nn.Linear):
    """
    A wrapper around :class:`torch.nn.Linear` to support empty inputs and more features.
    Because of https://github.com/pytorch/pytorch/issues/34202
    """

    def forward(self, x):
      if x.numel() == 0:
        output_shape = [x.shape[0], self.weight.shape[0]]

        empty = _NewEmptyTensorOp.apply(x, output_shape)
        if self.training:
          # This is to make DDP happy.
          # DDP expects all workers to have gradient w.r.t the same set of parameters.
          _dummy = sum(x.view(-1)[0] for x in self.parameters()) * 0.0
          return empty + _dummy
        else:
          return empty

      x = super().forward(x)
      return x


def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
  """
  A wrapper around :func:`torch.nn.functional.interpolate` to support zero-size tensor.
  """
  if input.numel() > 0:
    return torch.nn.functional.interpolate(
        input, size, scale_factor, mode, align_corners=align_corners
    )

  def _check_size_scale_factor(dim):
    if size is None and scale_factor is None:
      raise ValueError("either size or scale_factor should be defined")
    if size is not None and scale_factor is not None:
      raise ValueError("only one of size or scale_factor should be defined")
    if (scale_factor is not None and isinstance(scale_factor, tuple) and len(scale_factor) != dim):
      raise ValueError(
          "scale_factor shape must match input shape. "
          "Input is {}D, scale_factor size is {}".format(dim, len(scale_factor))
      )

  def _output_size(dim):
    _check_size_scale_factor(dim)
    if size is not None:
      return size
    scale_factors = _ntuple(dim)(scale_factor)
    # math.floor might return float in py2.7
    return [int(math.floor(input.size(i + 2) * scale_factors[i])) for i in range(dim)]

  output_shape = tuple(_output_size(2))
  output_shape = input.shape[:-2] + output_shape
  return _NewEmptyTensorOp.apply(input, output_shape)


class FPN(Backbone):
  """
  This module implements Feature Pyramid Network.
  It creates pyramid features built on top of some input feature maps.
  """

  def __init__(
      self, bottom_up, in_features, out_channels, norm="", top_block=None, fuse_type="sum"
  ):
    """
    Args:
      bottom_up (Backbone): module representing the bottom up subnetwork.
        Must be a subclass of :class:`Backbone`. The multi-scale feature
        maps generated by the bottom up network, and listed in `in_features`,
        are used to generate FPN levels.
      in_features (list[str]): names of the input feature maps coming
        from the backbone to which FPN is attached. For example, if the
        backbone produces ["res2", "res3", "res4"], any *contiguous* sublist
        of these may be used; order must be from high to low resolution.
      out_channels (int): number of channels in the output feature maps.
      norm (str): the normalization to use.
      top_block (nn.Module or None): if provided, an extra operation will
        be performed on the output of the last (smallest resolution)
        FPN output, and the result will extend the result list. The top_block
        further downsamples the feature map. It must have an attribute
        "num_levels", meaning the number of extra FPN levels added by
        this block, and "in_feature", which is a string representing
        its input feature (e.g., p5).
      fuse_type (str): types for fusing the top down features and the lateral
        ones. It can be "sum" (default), which sums up element-wise; or "avg",
        which takes the element-wise mean of the two.
    """
    super(FPN, self).__init__()
    assert isinstance(bottom_up, Backbone)

    # Feature map strides and channels from the bottom up network (e.g. ResNet)
    input_shapes = bottom_up.output_shape()
    in_strides = [input_shapes[f].stride for f in in_features]
    in_channels = [input_shapes[f].channels for f in in_features]

    _assert_strides_are_log2_contiguous(in_strides)
    lateral_convs = []
    output_convs = []

    use_bias = norm == ""
    for idx, in_channels in enumerate(in_channels):
      lateral_norm = get_norm(norm, out_channels)
      output_norm = get_norm(norm, out_channels)

      lateral_conv = Conv2d(
          in_channels, out_channels, kernel_size=1, bias=use_bias, norm=lateral_norm
      )
      output_conv = Conv2d(
          out_channels,
          out_channels,
          kernel_size=3,
          stride=1,
          padding=1,
          bias=use_bias,
          norm=output_norm,
      )
      weight_init.c2_xavier_fill(lateral_conv)
      weight_init.c2_xavier_fill(output_conv)
      stage = int(math.log2(in_strides[idx]))
      self.add_module("fpn_lateral{}".format(stage), lateral_conv)
      self.add_module("fpn_output{}".format(stage), output_conv)

      lateral_convs.append(lateral_conv)
      output_convs.append(output_conv)
    # Place convs into top-down order (from low to high resolution)
    # to make the top-down computation in forward clearer.
    self.lateral_convs = lateral_convs[::-1]
    self.output_convs = output_convs[::-1]
    self.top_block = top_block
    self.in_features = in_features
    self.bottom_up = bottom_up
    # Return feature names are "p<stage>", like ["p2", "p3", ..., "p6"]
    self._out_feature_strides = {"p{}".format(int(math.log2(s))): s for s in in_strides}
    # top block output feature maps.
    if self.top_block is not None:
      for s in range(stage, stage + self.top_block.num_levels):
        self._out_feature_strides["p{}".format(s + 1)] = 2**(s + 1)

    self._out_features = list(self._out_feature_strides.keys())
    self._out_feature_channels = {k: out_channels for k in self._out_features}
    self._size_divisibility = in_strides[-1]
    assert fuse_type in {"avg", "sum"}
    self._fuse_type = fuse_type
    self.nearest_two_upsample = nn.Upsample(scale_factor=2, mode="nearest")

  @property
  def size_divisibility(self):
    return self._size_divisibility

  def forward(self, x):
    """
    Args:
      input (dict[str->Tensor]): mapping feature map name (e.g., "res5") to
        feature map tensor for each feature level in high to low resolution order.

    Returns:
      dict[str->Tensor]:
        mapping from feature map name to FPN feature map tensor
        in high to low resolution order. Returned feature names follow the FPN
        paper convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
        ["p2", "p3", ..., "p6"].
    """
    # Reverse feature maps into top-down order (from low to high resolution)
    bottom_up_features = self.bottom_up(x)
    x = [bottom_up_features[f] for f in self.in_features[::-1]]
    results = []
    prev_features = self.lateral_convs[0](x[0])
    results.append(self.output_convs[0](prev_features))
    for features, lateral_conv, output_conv in zip(
        x[1:], self.lateral_convs[1:], self.output_convs[1:]
    ):
      top_down_features = self.nearest_two_upsample(prev_features)
      lateral_features = lateral_conv(features)
      prev_features = lateral_features + top_down_features
      if self._fuse_type == "avg":
        prev_features /= 2
      results.insert(0, output_conv(prev_features))

    if self.top_block is not None:
      top_block_in_feature = bottom_up_features.get(self.top_block.in_feature, None)
      if top_block_in_feature is None:
        top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
      results.extend(self.top_block(top_block_in_feature))
    assert len(self._out_features) == len(results)
    return dict(zip(self._out_features, results)), bottom_up_features['disp_small']

  def output_shape(self):
    return {
        name: ShapeSpec(
            channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
        ) for name in self._out_features
    }


def _assert_strides_are_log2_contiguous(strides):
  """
  Assert that each stride is 2x times its preceding stride, i.e. "contiguous in log2".
  """
  for i, stride in enumerate(strides[1:], 1):
    assert stride == 2 * strides[i - 1], "Strides {} {} are not log2 contiguous".format(
        stride, strides[i - 1]
    )


class LastLevelMaxPool(nn.Module):
  """
  This module is used in the original FPN to generate a downsampled
  P6 feature from P5.
  """

  def __init__(self):
    super().__init__()
    self.num_levels = 1
    self.in_feature = "p5"

  def forward(self, x):
    return [F.max_pool2d(x, kernel_size=1, stride=2, padding=0)]


class LastLevelP6P7(nn.Module):
  """
  This module is used in RetinaNet to generate extra layers, P6 and P7 from
  C5 feature.
  """

  def __init__(self, in_channels, out_channels, in_feature="res5"):
    super().__init__()
    self.num_levels = 2
    self.in_feature = in_feature
    self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
    self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
    for module in [self.p6, self.p7]:
      weight_init.c2_xavier_fill(module)

  def forward(self, c5):
    p6 = self.p6(c5)
    p7 = self.p7(F.relu(p6))
    return [p6, p7]


class ResNetBlockBase(nn.Module):

  def __init__(self, in_channels, out_channels, stride):
    """
    The `__init__` method of any subclass should also contain these arguments.

    Args:
      in_channels (int):
      out_channels (int):
      stride (int):
    """
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.stride = stride

  def freeze(self):
    for p in self.parameters():
      p.requires_grad = False
    FrozenBatchNorm2d.convert_frozen_batchnorm(self)
    return self


class BasicBlock(ResNetBlockBase):

  def __init__(self, in_channels, out_channels, *, stride=1, norm="BN"):
    """
    The standard block type for ResNet18 and ResNet34.

    Args:
      in_channels (int): Number of input channels.
      out_channels (int): Number of output channels.
      stride (int): Stride for the first conv.
      norm (str or callable): A callable that takes the number of
        channels and returns a `nn.Module`, or a pre-defined string
        (one of {"FrozenBN", "BN", "GN"}).
    """
    super().__init__(in_channels, out_channels, stride)

    if in_channels != out_channels:
      self.shortcut = Conv2d(
          in_channels,
          out_channels,
          kernel_size=1,
          stride=stride,
          bias=False,
          norm=get_norm(norm, out_channels),
      )
    else:
      self.shortcut = None

    self.conv1 = Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
        norm=get_norm(norm, out_channels),
    )

    self.conv2 = Conv2d(
        out_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        norm=get_norm(norm, out_channels),
    )

    for layer in [self.conv1, self.conv2, self.shortcut]:
      if layer is not None:  # shortcut can be None
        weight_init.c2_msra_fill(layer)

  def forward(self, x):
    out = self.conv1(x)
    out = F.relu_(out)
    out = self.conv2(out)

    if self.shortcut is not None:
      shortcut = self.shortcut(x)
    else:
      shortcut = x

    out += shortcut
    out = F.relu_(out)
    return out


class BottleneckBlock(ResNetBlockBase):

  def __init__(
      self,
      in_channels,
      out_channels,
      *,
      bottleneck_channels,
      stride=1,
      num_groups=1,
      norm="BN",
      stride_in_1x1=False,
      dilation=1,
  ):
    """
    Args:
      norm (str or callable): a callable that takes the number of
        channels and return a `nn.Module`, or a pre-defined string
        (one of {"FrozenBN", "BN", "GN"}).
      stride_in_1x1 (bool): when stride==2, whether to put stride in the
        first 1x1 convolution or the bottleneck 3x3 convolution.
    """
    super().__init__(in_channels, out_channels, stride)

    if in_channels != out_channels:
      self.shortcut = Conv2d(
          in_channels,
          out_channels,
          kernel_size=1,
          stride=stride,
          bias=False,
          norm=get_norm(norm, out_channels),
      )
    else:
      self.shortcut = None

    # The original MSRA ResNet models have stride in the first 1x1 conv
    # The subsequent fb.torch.resnet and Caffe2 ResNe[X]t implementations have
    # stride in the 3x3 conv
    stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

    self.conv1 = Conv2d(
        in_channels,
        bottleneck_channels,
        kernel_size=1,
        stride=stride_1x1,
        bias=False,
        norm=get_norm(norm, bottleneck_channels),
    )

    self.conv2 = Conv2d(
        bottleneck_channels,
        bottleneck_channels,
        kernel_size=3,
        stride=stride_3x3,
        padding=1 * dilation,
        bias=False,
        groups=num_groups,
        dilation=dilation,
        norm=get_norm(norm, bottleneck_channels),
    )

    self.conv3 = Conv2d(
        bottleneck_channels,
        out_channels,
        kernel_size=1,
        bias=False,
        norm=get_norm(norm, out_channels),
    )

    for layer in [self.conv1, self.conv2, self.conv3, self.shortcut]:
      if layer is not None:  # shortcut can be None
        weight_init.c2_msra_fill(layer)

    # Zero-initialize the last normalization in each residual branch,
    # so that at the beginning, the residual branch starts with zeros,
    # and each residual block behaves like an identity.
    # See Sec 5.1 in "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour":
    # "For BN layers, the learnable scaling coefficient γ is initialized
    # to be 1, except for each residual block's last BN
    # where γ is initialized to be 0."

    # nn.init.constant_(self.conv3.norm.weight, 0)
    # TODO this somehow hurts performance when training GN models from scratch.
    # Add it as an option when we need to use this code to train a backbone.

  def forward(self, x):
    out = self.conv1(x)
    out = F.relu_(out)

    out = self.conv2(out)
    out = F.relu_(out)

    out = self.conv3(out)

    if self.shortcut is not None:
      shortcut = self.shortcut(x)
    else:
      shortcut = x

    out += shortcut
    out = F.relu_(out)
    return out


def make_stage(block_class, num_blocks, first_stride, **kwargs):
  """
  Create a resnet stage by creating many blocks.

  Args:
    block_class (class): a subclass of ResNetBlockBase
    num_blocks (int):
    first_stride (int): the stride of the first block. The other blocks will have stride=1.
      A `stride` argument will be passed to the block constructor.
    kwargs: other arguments passed to the block constructor.

  Returns:
    list[nn.Module]: a list of block module.
  """
  blocks = []
  for i in range(num_blocks):
    blocks.append(block_class(stride=first_stride if i == 0 else 1, **kwargs))
    kwargs["in_channels"] = kwargs["out_channels"]
  return blocks


class JointAngleStem(nn.Module):

  def __init__(self, in_channels=3, out_channels=64, norm="BN"):
    super().__init__()
    self.conv1 = Conv2d(
        in_channels,
        out_channels,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False,
        norm=get_norm(norm, out_channels),
    )

    D_in = 50
    hidden = 256
    self.D_out = out_channels
    self.l1 = torch.nn.Linear(D_in, hidden)
    self.b1 = torch.nn.BatchNorm1d(hidden, momentum=0.1)
    self.l2 = torch.nn.Linear(hidden, hidden)
    self.b2 = torch.nn.BatchNorm1d(hidden, momentum=0.1)
    self.l3 = torch.nn.Linear(hidden, self.D_out)

    weight_init.c2_msra_fill(self.conv1)

  def forward(self, x, robot_joint_angles):
    x = self.conv1(x)
    x = F.relu_(x)
    x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

    b, c, h, w = x.size()

    ja = robot_joint_angles
    ja = F.relu(self.b1(self.l1(ja)))
    ja = F.relu(self.b2(self.l2(ja)))
    ja = self.l3(ja)

    ja = ja.view(-1, self.D_out, 1, 1).expand(-1, -1, h, w)
    return torch.cat([x, ja], dim=-3)

  @property
  def out_channels(self):
    return self.conv1.out_channels

  @property
  def stride(self):
    return 4  # = stride 2 conv -> stride 2 max pool


class StereoStem(nn.Module):

  def __init__(self, in_channels=3, out_channels=64, norm="BN"):
    """
    Args:
      norm (str or callable): a callable that takes the number of
        channels and return a `nn.Module`, or a pre-defined string
        (one of {"FrozenBN", "BN", "GN"}).
    """
    super().__init__()
    assert in_channels == 3
    self.conv1 = Conv2d(
        1, 32, kernel_size=7, stride=2, padding=3, bias=False, norm=get_norm(norm, 32)
    )
    self.conv2 = Conv2d(
        1, 32, kernel_size=7, stride=2, padding=3, bias=False, norm=get_norm(norm, 32)
    )
    self.conv3 = Conv2d(
        1, 64, kernel_size=7, stride=2, padding=3, bias=False, norm=get_norm(norm, 64)
    )
    weight_init.c2_msra_fill(self.conv1)
    weight_init.c2_msra_fill(self.conv2)
    weight_init.c2_msra_fill(self.conv3)

  def forward(self, x):
    assert x.shape[1] == 3
    c1 = x[:, 0:1, :, :]
    c2 = x[:, 1:2, :, :]
    c3 = x[:, 2:3, :, :]
    c1 = F.max_pool2d(F.relu_(self.conv1(c1)), kernel_size=3, stride=2, padding=1)
    c2 = F.max_pool2d(F.relu_(self.conv2(c2)), kernel_size=3, stride=2, padding=1)
    c3 = F.max_pool2d(F.relu_(self.conv3(c3)), kernel_size=3, stride=2, padding=1)
    diff = (c1 - c2).abs()
    both = c1 + c2
    lr = torch.cat((diff, both), 1)
    return lr + c3

  @property
  def out_channels(self):
    return self.conv1.out_channels

  @property
  def stride(self):
    return 4  # = stride 2 conv -> stride 2 max pool


class BasicStem(nn.Module):

  def __init__(self, in_channels=3, out_channels=64, norm="BN"):
    """
    Args:
      norm (str or callable): a callable that takes the number of
        channels and return a `nn.Module`, or a pre-defined string
        (one of {"FrozenBN", "BN", "GN"}).
    """
    super().__init__()
    self.conv1 = Conv2d(
        in_channels,
        out_channels,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False,
        norm=get_norm(norm, out_channels),
    )
    weight_init.c2_msra_fill(self.conv1)

  def forward(self, x, robot_joint_angles):
    x = self.conv1(x)
    x = F.relu_(x)
    x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
    return x

  @property
  def out_channels(self):
    return self.conv1.out_channels

  @property
  def stride(self):
    return 4  # = stride 2 conv -> stride 2 max pool


class ResNet(Backbone):

  def __init__(self, stem, stages, num_classes=None, out_features=None):
    """
    Args:
      stem (nn.Module): a stem module
      stages (list[list[ResNetBlock]]): several (typically 4) stages,
        each contains multiple :class:`ResNetBlockBase`.
      num_classes (None or int): if None, will not perform classification.
      out_features (list[str]): name of the layers whose outputs should
        be returned in forward. Can be anything in "stem", "linear", or "res2" ...
        If None, will return the output of the last layer.
    """
    super(ResNet, self).__init__()
    self.stem_new = stem
    self.num_classes = num_classes

    current_stride = self.stem_new.stride
    self._out_feature_strides = {"stem": current_stride}
    self._out_feature_channels = {"stem": self.stem_new.out_channels}

    self.stages_and_names = []
    for i, blocks in enumerate(stages):
      for block in blocks:
        assert isinstance(block, ResNetBlockBase), block
        curr_channels = block.out_channels
      stage = nn.Sequential(*blocks)
      name = "res" + str(i + 2)
      self.add_module(name, stage)
      self.stages_and_names.append((stage, name))
      self._out_feature_strides[name] = current_stride = int(
          current_stride * np.prod([k.stride for k in blocks])
      )
      self._out_feature_channels[name] = blocks[-1].out_channels

    if num_classes is not None:
      self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
      self.linear = nn.Linear(curr_channels, num_classes)

      # Sec 5.1 in "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour":
      # "The 1000-way fully-connected layer is initialized by
      # drawing weights from a zero-mean Gaussian with standard deviation of 0.01."
      nn.init.normal_(self.linear.weight, std=0.01)
      name = "linear"

    if out_features is None:
      out_features = [name]
    self._out_features = out_features
    assert len(self._out_features)
    children = [x[0] for x in self.named_children()]
    for out_feature in self._out_features:
      assert out_feature in children, "Available children: {}".format(", ".join(children))

  def forward(self, x, robot_joint_angles=None):
    outputs = {}
    x, disp_small = self.stem_new(x, robot_joint_angles)
    outputs['disp_small'] = disp_small
    if "stem" in self._out_features:
      outputs["stem"] = x
    for stage, name in self.stages_and_names:
      x = stage(x)
      if name in self._out_features:
        outputs[name] = x
    if self.num_classes is not None:
      x = self.avgpool(x)
      x = torch.flatten(x, 1)
      x = self.linear(x)
      if "linear" in self._out_features:
        outputs["linear"] = x
    return outputs

  def output_shape(self):
    return {
        name: ShapeSpec(
            channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
        ) for name in self._out_features
    }


def build_resnet_backbone(input_shape, stereo_stem, model_norm='BN', num_filters_scale=4):
  """
  Create a ResNet instance from config.

  Returns:
    ResNet: a :class:`ResNet` instance.
  """
  # need registration of new blocks/stems?
  MODEL_RESNETS_RES2_OUT_CHANNELS = 256 // num_filters_scale
  MODEL_RESNETS_WIDTH_PER_GROUP = max(64 // num_filters_scale, 32)
  MODEL_RESNETS_NORM = model_norm

  norm = MODEL_RESNETS_NORM
  #stem = BasicStem(
  #    in_channels=input_shape.channels,
  #    out_channels=MODEL_RESNETS_STEM_OUT_CHANNELS,
  #    norm=norm,
  #)
  stem = stereo_stem
  stem_out_channels = MODEL_RESNETS_STEM_OUT_CHANNELS

  # fmt: off
  out_features = MODEL_RESNETS_OUT_FEATURES
  depth = MODEL_RESNETS_DEPTH
  num_groups = MODEL_RESNETS_NUM_GROUPS
  width_per_group = MODEL_RESNETS_WIDTH_PER_GROUP
  bottleneck_channels = num_groups * width_per_group
  in_channels = stem_out_channels
  out_channels = MODEL_RESNETS_RES2_OUT_CHANNELS
  stride_in_1x1 = MODEL_RESNETS_STRIDE_IN_1X1
  res5_dilation = MODEL_RESNETS_RES5_DILATION
  # fmt: on
  assert res5_dilation in {1, 2}, "res5_dilation cannot be {}.".format(res5_dilation)

  num_blocks_per_stage = {
      18: [2, 2, 2, 2],
      34: [3, 4, 6, 3],
      50: NUM_BLOCKS_PER_STAGE,
      101: [3, 4, 23, 3],
      152: [3, 8, 36, 3],
  }[depth]

  if depth in [18, 34]:
    assert out_channels == 64, "Must set MODEL.RESNETS.RES2_OUT_CHANNELS = 64 for R18/R34"
    assert res5_dilation == 1, "Must set MODEL.RESNETS.RES5_DILATION = 1 for R18/R34"
    assert num_groups == 1, "Must set MODEL.RESNETS.NUM_GROUPS = 1 for R18/R34"

  stages = []

  # Avoid creating variables without gradients
  # It consumes extra memory and may cause allreduce to fail
  out_stage_idx = [{"res2": 2, "res3": 3, "res4": 4, "res5": 5}[f] for f in out_features]
  max_stage_idx = max(out_stage_idx)
  for idx, stage_idx in enumerate(range(2, max_stage_idx + 1)):
    dilation = res5_dilation if stage_idx == 5 else 1
    first_stride = 1 if idx == 0 or (stage_idx == 5 and dilation == 2) else 2
    stage_kargs = {
        "num_blocks": num_blocks_per_stage[idx],
        "first_stride": first_stride,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "norm": norm,
    }
    # Use BasicBlock for R18 and R34.
    if depth in [18, 34]:
      stage_kargs["block_class"] = BasicBlock
    else:
      stage_kargs["bottleneck_channels"] = bottleneck_channels
      stage_kargs["stride_in_1x1"] = stride_in_1x1
      stage_kargs["dilation"] = dilation
      stage_kargs["num_groups"] = num_groups
      stage_kargs["block_class"] = BottleneckBlock
    blocks = make_stage(**stage_kargs)
    # print('stage', idx)
    # print(stage_kargs)
    # print()
    in_channels = out_channels
    out_channels *= 2
    bottleneck_channels *= 2

    out_channels = min(MODEL_RESNETS_RES2_OUT_CHANNELS_MAX, out_channels)
    bottleneck_channels = min(
        MODEL_RESNETS_WIDTH_PER_GROUP_MAX * MODEL_RESNETS_NUM_GROUPS, bottleneck_channels
    )

    stages.append(blocks)
  return ResNet(stem, stages, out_features=out_features)


def output_shape(backbone):
  """
    Returns:
      dict[str->ShapeSpec]
    """
  # this is a backward-compatible default
  return {
      name: ShapeSpec(
          channels=backbone._out_feature_channels[name], stride=backbone._out_feature_strides[name]
      ) for name in backbone._out_features
  }


# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# from .config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

# _C = CN()
#
# # The version number, to upgrade from old configs to new ones if any
# # changes happen. It's recommended to keep a VERSION in your config file.
# _C.VERSION = 2
#
# _C.MODEL = CN()
# _C.MODEL.LOAD_PROPOSALS = False
# _C.MODEL.MASK_ON = False
# _C.MODEL.KEYPOINT_ON = False
# _C.MODEL.DEVICE = "cuda"
# _C.MODEL.META_ARCHITECTURE = "GeneralizedRCNN"
#
# # Path (possibly with schema like catalog:// or detectron2://) to a checkpoint file
# # to be loaded to the model. You can find available models in the model zoo.
# _C.MODEL.WEIGHTS = ""
#
# # Values to be used for image normalization (BGR order, since INPUT.FORMAT defaults to BGR).
# # To train on images of different number of channels, just set different mean & std.
# # Default values are the mean pixel value from ImageNet: [103.53, 116.28, 123.675]
# _C.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675]
# # When using pre-trained models in Detectron1 or any MSRA models,
# # std has been absorbed into its conv1 weights, so the std needs to be set 1.
# # Otherwise, you can use [57.375, 57.120, 58.395] (ImageNet std)
# _C.MODEL.PIXEL_STD = [1.0, 1.0, 1.0]
#
#
# # -----------------------------------------------------------------------------
# # INPUT
# # -----------------------------------------------------------------------------
# _C.INPUT = CN()
# # Size of the smallest side of the image during training
# _C.INPUT.MIN_SIZE_TRAIN = (800,)
# # Sample size of smallest side by choice or random selection from range give by
# # INPUT.MIN_SIZE_TRAIN
# _C.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
# # Maximum size of the side of the image during training
# _C.INPUT.MAX_SIZE_TRAIN = 1333
# # Size of the smallest side of the image during testing. Set to zero to disable resize in testing.
# _C.INPUT.MIN_SIZE_TEST = 800
# # Maximum size of the side of the image during testing
# _C.INPUT.MAX_SIZE_TEST = 1333
#
# # `True` if cropping is used for data augmentation during training
# _C.INPUT.CROP = CN({"ENABLED": False})
# # Cropping type:
# # - "relative" crop (H * CROP.SIZE[0], W * CROP.SIZE[1]) part of an input of size (H, W)
# # - "relative_range" uniformly sample relative crop size from between [CROP.SIZE[0], [CROP.SIZE[1]].
# #   and  [1, 1] and use it as in "relative" scenario.
# # - "absolute" crop part of an input with absolute size: (CROP.SIZE[0], CROP.SIZE[1]).
# _C.INPUT.CROP.TYPE = "relative_range"
# # Size of crop in range (0, 1] if CROP.TYPE is "relative" or "relative_range" and in number of
# # pixels if CROP.TYPE is "absolute"
# _C.INPUT.CROP.SIZE = [0.9, 0.9]
#
#
# # Whether the model needs RGB, YUV, HSV etc.
# # Should be one of the modes defined here, as we use PIL to read the image:
# # https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes
# # with BGR being the one exception. One can set image format to BGR, we will
# # internally use RGB for conversion and flip the channels over
# _C.INPUT.FORMAT = "BGR"
# # The ground truth mask format that the model will use.
# # Mask R-CNN supports either "polygon" or "bitmask" as ground truth.
# _C.INPUT.MASK_FORMAT = "polygon"  # alternative: "bitmask"
#
#
# # -----------------------------------------------------------------------------
# # Dataset
# # -----------------------------------------------------------------------------
# _C.DATASETS = CN()
# # List of the dataset names for training. Must be registered in DatasetCatalog
# _C.DATASETS.TRAIN = ()
# # List of the pre-computed proposal files for training, which must be consistent
# # with datasets listed in DATASETS.TRAIN.
# _C.DATASETS.PROPOSAL_FILES_TRAIN = ()
# # Number of top scoring precomputed proposals to keep for training
# _C.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN = 2000
# # List of the dataset names for testing. Must be registered in DatasetCatalog
# _C.DATASETS.TEST = ()
# # List of the pre-computed proposal files for test, which must be consistent
# # with datasets listed in DATASETS.TEST.
# _C.DATASETS.PROPOSAL_FILES_TEST = ()
# # Number of top scoring precomputed proposals to keep for test
# _C.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST = 1000
#
# # -----------------------------------------------------------------------------
# # DataLoader
# # -----------------------------------------------------------------------------
# _C.DATALOADER = CN()
# # Number of data loading threads
# _C.DATALOADER.NUM_WORKERS = 4
# # If True, each batch should contain only images for which the aspect ratio
# # is compatible. This groups portrait images together, and landscape images
# # are not batched with portrait images.
# _C.DATALOADER.ASPECT_RATIO_GROUPING = True
# # Options: TrainingSampler, RepeatFactorTrainingSampler
# _C.DATALOADER.SAMPLER_TRAIN = "TrainingSampler"
# # Repeat threshold for RepeatFactorTrainingSampler
# _C.DATALOADER.REPEAT_THRESHOLD = 0.0
# # if True, the dataloader will filter out images that have no associated
# # annotations at train time.
# _C.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True
#
# # ---------------------------------------------------------------------------- #
# # Backbone options
# # ---------------------------------------------------------------------------- #
# _C.MODEL.BACKBONE = CN()
#
# _C.MODEL.BACKBONE.NAME = "build_resnet_backbone"
# # Freeze the first several stages so they are not trained.
# # There are 5 stages in ResNet. The first is a convolution, and the following
# # stages are each group of residual blocks.
# _C.MODEL.BACKBONE.FREEZE_AT = 2
#
#
# # ---------------------------------------------------------------------------- #
# # FPN options
# # ---------------------------------------------------------------------------- #
# _C.MODEL.FPN = CN()
# # Names of the input feature maps to be used by FPN
# # They must have contiguous power of 2 strides
# # e.g., ["res2", "res3", "res4", "res5"]
# _C.MODEL.FPN.IN_FEATURES = []
# _C.MODEL.FPN.OUT_CHANNELS = 256
#
# # Options: "" (no norm), "GN"
# _C.MODEL.FPN.NORM = ""
#
# # Types for fusing the FPN top-down and lateral features. Can be either "sum" or "avg"
# _C.MODEL.FPN.FUSE_TYPE = "sum"
#
#
# # ---------------------------------------------------------------------------- #
# # Proposal generator options
# # ---------------------------------------------------------------------------- #
# _C.MODEL.PROPOSAL_GENERATOR = CN()
# # Current proposal generators include "RPN", "RRPN" and "PrecomputedProposals"
# _C.MODEL.PROPOSAL_GENERATOR.NAME = "RPN"
# # Proposal height and width both need to be greater than MIN_SIZE
# # (a the scale used during training or inference)
# _C.MODEL.PROPOSAL_GENERATOR.MIN_SIZE = 0
#
#
# # ---------------------------------------------------------------------------- #
# # Anchor generator options
# # ---------------------------------------------------------------------------- #
# _C.MODEL.ANCHOR_GENERATOR = CN()
# # The generator can be any name in the ANCHOR_GENERATOR registry
# _C.MODEL.ANCHOR_GENERATOR.NAME = "DefaultAnchorGenerator"
# # Anchor sizes (i.e. sqrt of area) in absolute pixels w.r.t. the network input.
# # Format: list[list[int]]. SIZES[i] specifies the list of sizes
# # to use for IN_FEATURES[i]; len(SIZES) == len(IN_FEATURES) must be true,
# # or len(SIZES) == 1 is true and size list SIZES[0] is used for all
# # IN_FEATURES.
# _C.MODEL.ANCHOR_GENERATOR.SIZES = [[32, 64, 128, 256, 512]]
# # Anchor aspect ratios. For each area given in `SIZES`, anchors with different aspect
# # ratios are generated by an anchor generator.
# # Format: list[list[int]]. ASPECT_RATIOS[i] specifies the list of aspect ratios
# # to use for IN_FEATURES[i]; len(ASPECT_RATIOS) == len(IN_FEATURES) must be true,
# # or len(ASPECT_RATIOS) == 1 is true and aspect ratio list ASPECT_RATIOS[0] is used
# # for all IN_FEATURES.
# _C.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]
# # Anchor angles.
# # list[float], the angle in degrees, for each input feature map.
# # ANGLES[i] specifies the list of angles for IN_FEATURES[i].
# _C.MODEL.ANCHOR_GENERATOR.ANGLES = [[-90, 0, 90]]
# # Relative offset between the center of the first anchor and the top-left corner of the image
# # Units: fraction of feature map stride (e.g., 0.5 means half stride)
# # Allowed values are floats in [0, 1) range inclusive.
# # Recommended value is 0.5, although it is not expected to affect model accuracy.
# _C.MODEL.ANCHOR_GENERATOR.OFFSET = 0.0
#
# # ---------------------------------------------------------------------------- #
# # RPN options
# # ---------------------------------------------------------------------------- #
# _C.MODEL.RPN = CN()
# _C.MODEL.RPN.HEAD_NAME = "StandardRPNHead"  # used by RPN_HEAD_REGISTRY
#
# # Names of the input feature maps to be used by RPN
# # e.g., ["p2", "p3", "p4", "p5", "p6"] for FPN
# _C.MODEL.RPN.IN_FEATURES = ["res4"]
# # Remove RPN anchors that go outside the image by BOUNDARY_THRESH pixels
# # Set to -1 or a large value, e.g. 100000, to disable pruning anchors
# _C.MODEL.RPN.BOUNDARY_THRESH = -1
# # IOU overlap ratios [BG_IOU_THRESHOLD, FG_IOU_THRESHOLD]
# # Minimum overlap required between an anchor and ground-truth box for the
# # (anchor, gt box) pair to be a positive example (IoU >= FG_IOU_THRESHOLD
# # ==> positive RPN example: 1)
# # Maximum overlap allowed between an anchor and ground-truth box for the
# # (anchor, gt box) pair to be a negative examples (IoU < BG_IOU_THRESHOLD
# # ==> negative RPN example: 0)
# # Anchors with overlap in between (BG_IOU_THRESHOLD <= IoU < FG_IOU_THRESHOLD)
# # are ignored (-1)
# _C.MODEL.RPN.IOU_THRESHOLDS = [0.3, 0.7]
# _C.MODEL.RPN.IOU_LABELS = [0, -1, 1]
# # Total number of RPN examples per image
# _C.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 256
# # Target fraction of foreground (positive) examples per RPN minibatch
# _C.MODEL.RPN.POSITIVE_FRACTION = 0.5
# # Weights on (dx, dy, dw, dh) for normalizing RPN anchor regression targets
# _C.MODEL.RPN.BBOX_REG_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
# # The transition point from L1 to L2 loss. Set to 0.0 to make the loss simply L1.
# _C.MODEL.RPN.SMOOTH_L1_BETA = 0.0
# _C.MODEL.RPN.LOSS_WEIGHT = 1.0
# # Number of top scoring RPN proposals to keep before applying NMS
# # When FPN is used, this is *per FPN level* (not total)
# _C.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 12000
# _C.MODEL.RPN.PRE_NMS_TOPK_TEST = 6000
# # Number of top scoring RPN proposals to keep after applying NMS
# # When FPN is used, this limit is applied per level and then again to the union
# # of proposals from all levels
# # NOTE: When FPN is used, the meaning of this config is different from Detectron1.
# # It means per-batch topk in Detectron1, but per-image topk here.
# # See "modeling/rpn/rpn_outputs.py" for details.
# _C.MODEL.RPN.POST_NMS_TOPK_TRAIN = 2000
# _C.MODEL.RPN.POST_NMS_TOPK_TEST = 1000
# # NMS threshold used on RPN proposals
# _C.MODEL.RPN.NMS_THRESH = 0.7
#
# # ---------------------------------------------------------------------------- #
# # ROI HEADS options
# # ---------------------------------------------------------------------------- #
# _C.MODEL.ROI_HEADS = CN()
# _C.MODEL.ROI_HEADS.NAME = "Res5ROIHeads"
# # Number of foreground classes
# _C.MODEL.ROI_HEADS.NUM_CLASSES = 80
# # Names of the input feature maps to be used by ROI heads
# # Currently all heads (box, mask, ...) use the same input feature map list
# # e.g., ["p2", "p3", "p4", "p5"] is commonly used for FPN
# _C.MODEL.ROI_HEADS.IN_FEATURES = ["res4"]
# # IOU overlap ratios [IOU_THRESHOLD]
# # Overlap threshold for an RoI to be considered background (if < IOU_THRESHOLD)
# # Overlap threshold for an RoI to be considered foreground (if >= IOU_THRESHOLD)
# _C.MODEL.ROI_HEADS.IOU_THRESHOLDS = [0.5]
# _C.MODEL.ROI_HEADS.IOU_LABELS = [0, 1]
# # RoI minibatch size *per image* (number of regions of interest [ROIs])
# # Total number of RoIs per training minibatch =
# #   ROI_HEADS.BATCH_SIZE_PER_IMAGE * SOLVER.IMS_PER_BATCH
# # E.g., a common configuration is: 512 * 16 = 8192
# _C.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
# # Target fraction of RoI minibatch that is labeled foreground (i.e. class > 0)
# _C.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.25
#
# # Only used on test mode
#
# # Minimum score threshold (assuming scores in a [0, 1] range); a value chosen to
# # balance obtaining high recall with not having too many low precision
# # detections that will slow down inference post processing steps (like NMS)
# # A default threshold of 0.0 increases AP by ~0.2-0.3 but significantly slows down
# # inference.
# _C.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
# # Overlap threshold used for non-maximum suppression (suppress boxes with
# # IoU >= this threshold)
# _C.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
# # If True, augment proposals with ground-truth boxes before sampling proposals to
# # train ROI heads.
# _C.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT = True
#
# # ---------------------------------------------------------------------------- #
# # Box Head
# # ---------------------------------------------------------------------------- #
# _C.MODEL.ROI_BOX_HEAD = CN()
# # C4 don't use head name option
# # Options for non-C4 models: FastRCNNConvFCHead,
# _C.MODEL.ROI_BOX_HEAD.NAME = ""
# # Default weights on (dx, dy, dw, dh) for normalizing bbox regression targets
# # These are empirically chosen to approximately lead to unit variance targets
# _C.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS = (10.0, 10.0, 5.0, 5.0)
# # The transition point from L1 to L2 loss. Set to 0.0 to make the loss simply L1.
# _C.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA = 0.0
# _C.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 14
# _C.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO = 0
# # Type of pooling operation applied to the incoming feature map for each RoI
# _C.MODEL.ROI_BOX_HEAD.POOLER_TYPE = "ROIAlignV2"
#
# _C.MODEL.ROI_BOX_HEAD.NUM_FC = 0
# # Hidden layer dimension for FC layers in the RoI box head
# _C.MODEL.ROI_BOX_HEAD.FC_DIM = 1024
# _C.MODEL.ROI_BOX_HEAD.NUM_CONV = 0
# # Channel dimension for Conv layers in the RoI box head
# _C.MODEL.ROI_BOX_HEAD.CONV_DIM = 256
# # Normalization method for the convolution layers.
# # Options: "" (no norm), "GN", "SyncBN".
# _C.MODEL.ROI_BOX_HEAD.NORM = ""
# # Whether to use class agnostic for bbox regression
# _C.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG = False
# # If true, RoI heads use bounding boxes predicted by the box head rather than proposal boxes.
# _C.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES = False
#
# # ---------------------------------------------------------------------------- #
# # Cascaded Box Head
# # ---------------------------------------------------------------------------- #
# _C.MODEL.ROI_BOX_CASCADE_HEAD = CN()
# # The number of cascade stages is implicitly defined by the length of the following two configs.
# _C.MODEL.ROI_BOX_CASCADE_HEAD.BBOX_REG_WEIGHTS = (
#   (10.0, 10.0, 5.0, 5.0),
#   (20.0, 20.0, 10.0, 10.0),
#   (30.0, 30.0, 15.0, 15.0),
# )
# _C.MODEL.ROI_BOX_CASCADE_HEAD.IOUS = (0.5, 0.6, 0.7)
#
#
# # ---------------------------------------------------------------------------- #
# # Mask Head
# # ---------------------------------------------------------------------------- #
# _C.MODEL.ROI_MASK_HEAD = CN()
# _C.MODEL.ROI_MASK_HEAD.NAME = "MaskRCNNConvUpsampleHead"
# _C.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 14
# _C.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO = 0
# _C.MODEL.ROI_MASK_HEAD.NUM_CONV = 0  # The number of convs in the mask head
# _C.MODEL.ROI_MASK_HEAD.CONV_DIM = 256
# # Normalization method for the convolution layers.
# # Options: "" (no norm), "GN", "SyncBN".
# _C.MODEL.ROI_MASK_HEAD.NORM = ""
# # Whether to use class agnostic for mask prediction
# _C.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK = False
# # Type of pooling operation applied to the incoming feature map for each RoI
# _C.MODEL.ROI_MASK_HEAD.POOLER_TYPE = "ROIAlignV2"
#
#
# # ---------------------------------------------------------------------------- #
# # Keypoint Head
# # ---------------------------------------------------------------------------- #
# _C.MODEL.ROI_KEYPOINT_HEAD = CN()
# _C.MODEL.ROI_KEYPOINT_HEAD.NAME = "KRCNNConvDeconvUpsampleHead"
# _C.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION = 14
# _C.MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO = 0
# _C.MODEL.ROI_KEYPOINT_HEAD.CONV_DIMS = tuple(512 for _ in range(8))
# _C.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 17  # 17 is the number of keypoints in COCO.
#
# # Images with too few (or no) keypoints are excluded from training.
# _C.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE = 1
# # Normalize by the total number of visible keypoints in the minibatch if True.
# # Otherwise, normalize by the total number of keypoints that could ever exist
# # in the minibatch.
# # The keypoint softmax loss is only calculated on visible keypoints.
# # Since the number of visible keypoints can vary significantly between
# # minibatches, this has the effect of up-weighting the importance of
# # minibatches with few visible keypoints. (Imagine the extreme case of
# # only one visible keypoint versus N: in the case of N, each one
# # contributes 1/N to the gradient compared to the single keypoint
# # determining the gradient direction). Instead, we can normalize the
# # loss by the total number of keypoints, if it were the case that all
# # keypoints were visible in a full minibatch. (Returning to the example,
# # this means that the one visible keypoint contributes as much as each
# # of the N keypoints.)
# _C.MODEL.ROI_KEYPOINT_HEAD.NORMALIZE_LOSS_BY_VISIBLE_KEYPOINTS = True
# # Multi-task loss weight to use for keypoints
# # Recommended values:
# #   - use 1.0 if NORMALIZE_LOSS_BY_VISIBLE_KEYPOINTS is True
# #   - use 4.0 if NORMALIZE_LOSS_BY_VISIBLE_KEYPOINTS is False
# _C.MODEL.ROI_KEYPOINT_HEAD.LOSS_WEIGHT = 1.0
# # Type of pooling operation applied to the incoming feature map for each RoI
# _C.MODEL.ROI_KEYPOINT_HEAD.POOLER_TYPE = "ROIAlignV2"
#
# # ---------------------------------------------------------------------------- #
# # Semantic Segmentation Head
# # ---------------------------------------------------------------------------- #
# _C.MODEL.SEM_SEG_HEAD = CN()
# _C.MODEL.SEM_SEG_HEAD.NAME = "SemSegFPNHead"
# _C.MODEL.SEM_SEG_HEAD.IN_FEATURES = ["p2", "p3", "p4", "p5"]
# # Label in the semantic segmentation ground truth that is ignored, i.e., no loss is calculated for
# # the correposnding pixel.
# _C.MODEL.SEM_SEG_HEAD.IGNORE_VALUE = 255
# # Number of classes in the semantic segmentation head
# _C.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 54
# # Number of channels in the 3x3 convs inside semantic-FPN heads.
# _C.MODEL.SEM_SEG_HEAD.CONVS_DIM = 128
# # Outputs from semantic-FPN heads are up-scaled to the COMMON_STRIDE stride.
# _C.MODEL.SEM_SEG_HEAD.COMMON_STRIDE = 4
# # Normalization method for the convolution layers. Options: "" (no norm), "GN".
# _C.MODEL.SEM_SEG_HEAD.NORM = "GN"
# _C.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT = 1.0
#
# _C.MODEL.PANOPTIC_FPN = CN()
# # Scaling of all losses from instance detection / segmentation head.
# _C.MODEL.PANOPTIC_FPN.INSTANCE_LOSS_WEIGHT = 1.0
#
# # options when combining instance & semantic segmentation outputs
# _C.MODEL.PANOPTIC_FPN.COMBINE = CN({"ENABLED": True})
# _C.MODEL.PANOPTIC_FPN.COMBINE.OVERLAP_THRESH = 0.5
# _C.MODEL.PANOPTIC_FPN.COMBINE.STUFF_AREA_LIMIT = 4096
# _C.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.5
#
#
# # ---------------------------------------------------------------------------- #
# # RetinaNet Head
# # ---------------------------------------------------------------------------- #
# _C.MODEL.RETINANET = CN()
#
# # This is the number of foreground classes.
# _C.MODEL.RETINANET.NUM_CLASSES = 80
#
# _C.MODEL.RETINANET.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]
#
# # Convolutions to use in the cls and bbox tower
# # NOTE: this doesn't include the last conv for logits
# _C.MODEL.RETINANET.NUM_CONVS = 4
#
# # IoU overlap ratio [bg, fg] for labeling anchors.
# # Anchors with < bg are labeled negative (0)
# # Anchors  with >= bg and < fg are ignored (-1)
# # Anchors with >= fg are labeled positive (1)
# _C.MODEL.RETINANET.IOU_THRESHOLDS = [0.4, 0.5]
# _C.MODEL.RETINANET.IOU_LABELS = [0, -1, 1]
#
# # Prior prob for rare case (i.e. foreground) at the beginning of training.
# # This is used to set the bias for the logits layer of the classifier subnet.
# # This improves training stability in the case of heavy class imbalance.
# _C.MODEL.RETINANET.PRIOR_PROB = 0.01
#
# # Inference cls score threshold, only anchors with score > INFERENCE_TH are
# # considered for inference (to improve speed)
# _C.MODEL.RETINANET.SCORE_THRESH_TEST = 0.05
# _C.MODEL.RETINANET.TOPK_CANDIDATES_TEST = 1000
# _C.MODEL.RETINANET.NMS_THRESH_TEST = 0.5
#
# # Weights on (dx, dy, dw, dh) for normalizing Retinanet anchor regression targets
# _C.MODEL.RETINANET.BBOX_REG_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
#
# # Loss parameters
# _C.MODEL.RETINANET.FOCAL_LOSS_GAMMA = 2.0
# _C.MODEL.RETINANET.FOCAL_LOSS_ALPHA = 0.25
# _C.MODEL.RETINANET.SMOOTH_L1_LOSS_BETA = 0.1
#
#
# # ---------------------------------------------------------------------------- #
# # ResNe[X]t options (ResNets = {ResNet, ResNeXt}
# # Note that parts of a resnet may be used for both the backbone and the head
# # These options apply to both
# # ---------------------------------------------------------------------------- #
# _C.MODEL.RESNETS = CN()
#
# _C.MODEL.RESNETS.DEPTH = 50
# _C.MODEL.RESNETS.OUT_FEATURES = ["res4"]  # res4 for C4 backbone, res2..5 for FPN backbone
#
# # Number of groups to use; 1 ==> ResNet; > 1 ==> ResNeXt
# _C.MODEL.RESNETS.NUM_GROUPS = 1
#
# # Options: FrozenBN, GN, "SyncBN", "BN"
# _C.MODEL.RESNETS.NORM = "FrozenBN"
#
# # Baseline width of each group.
# # Scaling this parameters will scale the width of all bottleneck layers.
# _C.MODEL.RESNETS.WIDTH_PER_GROUP = 64
#
# # Place the stride 2 conv on the 1x1 filter
# # Use True only for the original MSRA ResNet; use False for C2 and Torch models
# _C.MODEL.RESNETS.STRIDE_IN_1X1 = True
#
# # Apply dilation in stage "res5"
# _C.MODEL.RESNETS.RES5_DILATION = 1
#
# # Output width of res2. Scaling this parameters will scale the width of all 1x1 convs in ResNet
# # For R18 and R34, this needs to be set to 64
# _C.MODEL.RESNETS.RES2_OUT_CHANNELS = 256
# _C.MODEL.RESNETS.STEM_OUT_CHANNELS = 64
#
#
#
# # ---------------------------------------------------------------------------- #
# # Solver
# # ---------------------------------------------------------------------------- #
# _C.SOLVER = CN()
#
# # See detectron2/solver/build.py for LR scheduler options
# _C.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"
#
# _C.SOLVER.MAX_ITER = 40000
#
# _C.SOLVER.BASE_LR = 0.001
#
# _C.SOLVER.MOMENTUM = 0.9
#
# _C.SOLVER.WEIGHT_DECAY = 0.0001
# # The weight decay that's applied to parameters of normalization layers
# # (typically the affine transformation)
# _C.SOLVER.WEIGHT_DECAY_NORM = 0.0
#
# _C.SOLVER.GAMMA = 0.1
# # The iteration number to decrease learning rate by GAMMA.
# _C.SOLVER.STEPS = (30000,)
#
# _C.SOLVER.WARMUP_FACTOR = 1.0 / 1000
# _C.SOLVER.WARMUP_ITERS = 1000
# _C.SOLVER.WARMUP_METHOD = "linear"
#
# # Save a checkpoint after every this number of iterations
# _C.SOLVER.CHECKPOINT_PERIOD = 5000
#
# # Number of images per batch across all machines.
# # If we have 16 GPUs and IMS_PER_BATCH = 32,
# # each GPU will see 2 images per batch.
# _C.SOLVER.IMS_PER_BATCH = 16
#
# # Detectron v1 (and previous detection code) used a 2x higher LR and 0 WD for
# # biases. This is not useful (at least for recent models). You should avoid
# # changing these and they exist only to reproduce Detectron v1 training if
# # desired.
# _C.SOLVER.BIAS_LR_FACTOR = 1.0
# _C.SOLVER.WEIGHT_DECAY_BIAS = _C.SOLVER.WEIGHT_DECAY
#
# # Gradient clipping
# _C.SOLVER.CLIP_GRADIENTS = CN({"ENABLED": False})
# # Type of gradient clipping, currently 2 values are supported:
# # - "value": the absolute values of elements of each gradients are clipped
# # - "norm": the norm of the gradient for each parameter is clipped thus
# #   affecting all elements in the parameter
# _C.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "value"
# # Maximum absolute value used for clipping gradients
# _C.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
# # Floating point number p for L-p norm to be used with the "norm"
# # gradient clipping type; for L-inf, please specify .inf
# _C.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0
#
# # ---------------------------------------------------------------------------- #
# # Specific test options
# # ---------------------------------------------------------------------------- #
# _C.TEST = CN()
# # For end-to-end tests to verify the expected accuracy.
# # Each item is [task, metric, value, tolerance]
# # e.g.: [['bbox', 'AP', 38.5, 0.2]]
# _C.TEST.EXPECTED_RESULTS = []
# # The period (in terms of steps) to evaluate the model during training.
# # Set to 0 to disable.
# _C.TEST.EVAL_PERIOD = 0
# # The sigmas used to calculate keypoint OKS. See http://cocodataset.org/#keypoints-eval
# # When empty it will use the defaults in COCO.
# # Otherwise it should have the same length as ROI_KEYPOINT_HEAD.NUM_KEYPOINTS.
# _C.TEST.KEYPOINT_OKS_SIGMAS = []
# # Maximum number of detections to return per image during inference (100 is
# # based on the limit established for the COCO dataset).
# _C.TEST.DETECTIONS_PER_IMAGE = 100
#
# _C.TEST.AUG = CN({"ENABLED": False})
# _C.TEST.AUG.MIN_SIZES = (400, 500, 600, 700, 800, 900, 1000, 1100, 1200)
# _C.TEST.AUG.MAX_SIZE = 4000
# _C.TEST.AUG.FLIP = True
#
# _C.TEST.PRECISE_BN = CN({"ENABLED": False})
# _C.TEST.PRECISE_BN.NUM_ITER = 200
#
# # ---------------------------------------------------------------------------- #
# # Misc options
# # ---------------------------------------------------------------------------- #
# # Directory where output files are written
# _C.OUTPUT_DIR = "./output"
# # Set seed to negative to fully randomize everything.
# # Set seed to positive to use a fixed seed. Note that a fixed seed does not
# # guarantee fully deterministic behavior.
# _C.SEED = -1
# # Benchmark different cudnn algorithms.
# # If input images have very different sizes, this option will have large overhead
# # for about 10k iterations. It usually hurts total time, but can benefit for certain models.
# # If input images have the same or similar sizes, benchmark is often helpful.
# _C.CUDNN_BENCHMARK = False
# # The period (in terms of steps) for minibatch visualization at train time.
# # Set to 0 to disable.
# _C.VIS_PERIOD = 0
#
# # global config is for quick hack purposes.
# # You can set them in command line or config files,
# # and access it with:
# #
# # from detectron2.config import global_cfg
# # print(global_cfg.HACK)
# #
# # Do not commit any configs into it.
# _C.GLOBAL = CN()
# _C.GLOBAL.HACK = 1.0

if __name__ == '__main__':
  main()

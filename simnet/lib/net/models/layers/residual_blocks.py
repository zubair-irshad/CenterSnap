# Copyright 2018-2020 Toyota Research Institute.  All rights reserved.
#
# Originally from Koichiro Yamaguchi's pixwislab repo mirrored at:
# https://github.awsinternal.tri.global/driving/pixwislab

import torch.nn as nn


def resnet_shortcut(in_channels, out_channels, stride, preact=False):
  """Shortcut layer for residual block.

    When the numbers of input and output channels are the same and stride is
    equal to 1, no layer is made.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        stride (int): Stride of the residual block.
        preact (bool, optional): If True, make a shortcut for pre-activation
                                 residual block.

    Returns:
        Module of shortcut layers.
    """
  if stride == 1 and in_channels == out_channels:
    return None

  if preact:
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
  else:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(out_channels)
    )


class ResidualBlock(nn.Module):
  """Base class for residual block."""

  @classmethod
  def expansion(cls):
    """Expansion rate."""
    raise NotImplementedError

  @classmethod
  def preact(cls):
    """Pre-activation flag."""
    raise NotImplementedError


class PreactBasicResidualBlock(ResidualBlock):
  """Pre-activation basic residual block."""

  def __init__(
      self,
      in_channels,
      base_channels,
      stride=1,
      dilation_rate=1,
      add_preact=True,
      add_last_norm=False
  ):
    """
        Args:
            in_channels (int): The number of input channels.
            base_channels (int): The number of output channels.
            stride (int, optional): Stride of the residual block.
            dilation_rate (int, optional): Dilation rate of the residual block.
            add_preact (bool, optional): If True, add pre-activation.
            add_last_norm (bool, optional): If True, add batch normalization
                                            after the last convolution.
        """
    super().__init__()
    if add_preact:
      self.preact_bn = nn.BatchNorm2d(in_channels)
    else:
      self.preact_bn = None
    self.conv_shortcut = resnet_shortcut(in_channels, base_channels, stride, preact=True)
    self.conv1 = nn.Conv2d(
        in_channels,
        base_channels,
        kernel_size=3,
        stride=stride,
        padding=dilation_rate,
        dilation=dilation_rate,
        bias=False
    )
    self.bn1 = nn.BatchNorm2d(base_channels)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = nn.Conv2d(
        base_channels,
        base_channels,
        kernel_size=3,
        padding=dilation_rate,
        dilation=dilation_rate,
        bias=False
    )
    self.bn_last = nn.BatchNorm2d(base_channels) if add_last_norm else None

  @classmethod
  def expansion(cls):
    """Expansion rate, which is a ratio of the number of the output
        channels to the number of the base channels in the residual block.

        Returns:
            Expansion rate (= 1).
        """
    return 1

  @classmethod
  def preact(cls):
    """Pre-activation flag.

        Returns:
            Flag (= True).
        """
    return True

  def forward(self, inputs):
    """Forward computation.

        Args:
            inputs (Tensor): Input tensor.

        Returns:
            Output tensor.
        """
    if self.conv_shortcut is None:
      shortcut_inputs = inputs
    else:
      shortcut_inputs = self.conv_shortcut(inputs)

    if self.preact_bn is not None:
      inputs = self.relu(self.preact_bn(inputs))
    outputs = self.relu(self.bn1(self.conv1(inputs)))
    outputs = self.conv2(outputs)

    outputs += shortcut_inputs

    if self.bn_last is not None:
      outputs = self.relu(self.bn_last(outputs))
    return outputs


def preact_resnet_group(
    block_func, in_channels, base_channels, num_blocks, stride=1, dilation_rate=1
):
  """Make a group of pre-activation residual blocks.

    Args:
        block_func (ResidualBlock): Function of a residual block.
        in_channels (int): The number of input channels.
        base_channels (int): The number of base channels of the residual block.
        num_blocks (int): The number of residual blocks.
        stride (int, optional): Stride of the first residual block.
        dilation_rate (int, optional): Dilation rate of residual blocks.

    Returns:
        Module of a group of residual blocks.
    """
  assert block_func.preact()

  residual_blocks = [
      block_func(
          in_channels, base_channels, stride=stride, dilation_rate=dilation_rate, add_preact=False
      )
  ]
  in_channels = block_func.expansion() * base_channels
  for idx in range(1, num_blocks):
    residual_blocks.append(
        block_func(
            in_channels,
            base_channels,
            dilation_rate=dilation_rate,
            add_preact=True,
            add_last_norm=idx == num_blocks - 1
        )
    )
  return nn.Sequential(*residual_blocks)

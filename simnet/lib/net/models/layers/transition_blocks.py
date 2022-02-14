# Copyright 2018-2020 Toyota Research Institute.  All rights reserved.
#
# Originally from Koichiro Yamaguchi's pixwislab repo mirrored at:
# https://github.awsinternal.tri.global/driving/pixwislab

import torch.nn as nn


class TransitionBlock(nn.Module):
  """Transition block for changing resolution or the number of channels."""

  def __init__(self, in_channels, out_channels, stride):
    """
        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            stride (int): Stride (1 or 2).
        """
    assert stride in (1, 2)
    assert not (in_channels == out_channels and stride == 1)
    super().__init__()

    if stride == 1:
      self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
    else:
      self.conv = nn.Conv2d(
          in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False
      )
    self.bn = nn.BatchNorm2d(out_channels)
    self.relu = nn.ReLU(inplace=True)

  def forward(self, inputs):
    """Forward computation.

        Args:
            inputs (Tensor): Input tensor.

        Returns:
            Output tensor.
        """
    return self.relu(self.bn(self.conv(inputs)))

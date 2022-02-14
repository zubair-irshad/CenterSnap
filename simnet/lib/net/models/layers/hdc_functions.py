# Copyright 2018-2020 Toyota Research Institute.  All rights reserved.
#
# Originally from Koichiro Yamaguchi's pixwislab repo mirrored at:
# https://github.awsinternal.tri.global/driving/pixwislab

import torch.nn as nn


def hdc_resnet_group(block_func, in_channels, base_channels, num_blocks, dilation_rates):
  """Make a group of pre-activation residual blocks with Hybrid Dilated
    Convolution (HDC).

    "Understanding Convolution for Semantic Segmentation",
    https://arxiv.org/abs/1702.08502.

    Args:
        block_func (ResidualBlock): Function of a residual block.
        in_channels (int): The number of input channels.
        base_channels (int): The number of base channels of the residual block.
        num_blocks (int): The number of residual blocks.
        dilation_rates (list): List of dilation rates.

    Returns:
        Module of a group of residual blocks.
    """
  assert block_func.preact()

  num_rates = len(dilation_rates)
  residual_blocks = [
      block_func(in_channels, base_channels, dilation_rate=dilation_rates[0], add_preact=False)
  ]
  in_channels = block_func.expansion() * base_channels
  for idx in range(1, num_blocks):
    residual_blocks.append(
        block_func(
            in_channels,
            base_channels,
            dilation_rate=dilation_rates[idx % num_rates],
            add_preact=True,
            add_last_norm=idx == num_blocks - 1
        )
    )
  return nn.Sequential(*residual_blocks)

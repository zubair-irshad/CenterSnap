# Copyright 2020 Toyota Research Institute.  All rights reserved.

import torch
import torch.nn as nn


@torch.jit.script
def cost_volume(left, right, num_disparities: int, is_right: bool):
  batch_size, channels, height, width = left.shape

  output = torch.zeros((batch_size, channels, num_disparities, height, width),
                       dtype=left.dtype,
                       device=left.device)

  for i in range(num_disparities):
    if not is_right:
      output[:, :, i, :, i:] = left[:, :, :, i:] * right[:, :, :, :width - i]
    else:
      output[:, :, i, :, :width - i] = left[:, :, :, i:] * right[:, :, :, :width - i]

  return output


class CostVolume(nn.Module):
  """Compute cost volume using cross correlation of left and right feature maps"""

  def __init__(self, num_disparities, is_right=False):
    super().__init__()
    self.num_disparities = num_disparities
    self.is_right = is_right

  def forward(self, left, right):
    if torch.jit.is_scripting():
      return cost_volume(left, right, self.num_disparities, self.is_right)
    else:
      return self.forward_with_amp(left, right)

  @torch.jit.unused
  def forward_with_amp(self, left, right):
    """This operation is unstable at float16, so compute at float32 even when using mixed precision"""
    with torch.cuda.amp.autocast(enabled=False):
      left = left.to(torch.float32)
      right = right.to(torch.float32)
      output = cost_volume(left, right, self.num_disparities, self.is_right)
      output = torch.clamp(output, -1e3, 1e3)
      return output


@torch.jit.script
def dot_product_cost_volume(left, right, num_disparities: int, is_right: bool):
  batch_size, channels, height, width = left.shape

  output = torch.zeros((batch_size, num_disparities, height, width),
                       dtype=left.dtype,
                       device=left.device)

  for i in range(num_disparities):
    if not is_right:
      output[:, i, :, i:] = (left[:, :, :, i:] * right[:, :, :, :width - i]).mean(dim=1)
    else:
      output[:, i, :, width - i] = (left[:, :, :, i:] * right[:, :, :, :width - i]).mean(dim=1)

  return output


class DotProductCostVolume(nn.Module):
  """Compute cost volume using dot product of left and right feature maps"""

  def __init__(self, num_disparities, is_right=False):
    super().__init__()
    self.num_disparities = num_disparities
    self.is_right = is_right

  def forward(self, left, right):
    return dot_product_cost_volume(left, right, self.num_disparities, self.is_right)

  @torch.jit.unused
  def forward_with_amp(self, left, right):
    """This operation is unstable at float16, so compute at float32 even when using mixed precision"""
    with torch.cuda.amp.autocast(enabled=False):
      left = left.to(torch.float32)
      right = right.to(torch.float32)
      output = dot_product_cost_volume(left, right, self.num_disparities, self.is_right)
      output = torch.clamp(output, -1e3, 1e3)
      return output

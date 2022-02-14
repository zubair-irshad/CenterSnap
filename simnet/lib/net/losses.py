# Copyright 2019 Toyota Research Institute.  All rights reserved.

import torch
import torch.nn as nn
import numpy as np
import IPython


class MaskedL1Loss(nn.Module):

  def __init__(self, centroid_threshold=0.3, downscale_factor=8):
    super().__init__()
    self.loss = nn.L1Loss(reduction='none')
    self.centroid_threshold = centroid_threshold
    self.downscale_factor = downscale_factor

  def forward(self, output, target, valid_mask):
    '''
        output: [N,16,H,W]
        target: [N,16,H,W]
        valid_mask: [N,H,W]
        '''
    valid_count = torch.sum(
        valid_mask[:, ::self.downscale_factor, ::self.downscale_factor] > self.centroid_threshold
    )
    loss = self.loss(output, target)
    if len(output.shape) == 4:
      loss = torch.sum(loss, dim=1)
    loss[valid_mask[:, ::self.downscale_factor, ::self.downscale_factor] < self.centroid_threshold
        ] = 0.0
    if valid_count == 0:
      return torch.sum(loss)
    return torch.sum(loss) / valid_count


class MSELoss(nn.Module):

  def __init__(self):
    super().__init__()
    self.loss = nn.MSELoss(reduction='none')

  def forward(self, output, target):
    '''
        output: [N,H,W]
        target: [N,H,W]
        ignore_mask: [N,H,W]
        '''
    loss = self.loss(output, target)
    return torch.mean(loss)


class MaskedMSELoss(nn.Module):

  def __init__(self):
    super().__init__()
    self.loss = nn.MSELoss(reduction='none')

  def forward(self, output, target, ignore_mask):
    '''
        output: [N,H,W]
        target: [N,H,W]
        ignore_mask: [N,H,W]
        '''
    valid_sum = torch.sum(torch.logical_not(ignore_mask))
    loss = self.loss(output, target)
    loss[ignore_mask > 0] = 0.0
    return torch.sum(loss) / valid_sum

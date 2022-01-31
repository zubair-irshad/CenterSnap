# Copyright 2020 Toyota Research Institute.  All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.jit.script
def soft_argmin(input):
  _, channels, _, _ = input.shape

  softmin = F.softmin(input, dim=1)
  index_tensor = torch.arange(0, channels, dtype=softmin.dtype,
                              device=softmin.device).view(1, channels, 1, 1)
  output = torch.sum(softmin * index_tensor, dim=1, keepdim=True)
  return output


class SoftArgmin(nn.Module):
  """Compute soft argmin operation for given cost volume"""

  def forward(self, input):
    return soft_argmin(input)

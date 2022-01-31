# Copyright 2020 Toyota Research Institute.  All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

#


@torch.jit.script
def matchability(input):
  softmin = F.softmin(input, dim=1)
  log_softmin = F.log_softmax(-input, dim=1)
  output = torch.sum(softmin * log_softmin, dim=1, keepdim=True)
  return output


class Matchability(nn.Module):
  """Compute disparity matchability value from https://arxiv.org/abs/2008.04800"""

  def forward(self, input):
    if torch.jit.is_scripting():
      # Torchscript generation can't handle mixed precision, so always compute at float32.
      return matchability(input)
    else:
      return self.forward_with_amp(input)

  @torch.jit.unused
  def forward_with_amp(self, input):
    """This operation is unstable at float16, so compute at float32 even when using mixed precision"""
    with torch.cuda.amp.autocast(enabled=False):
      input = input.to(torch.float32)
      return matchability(input)

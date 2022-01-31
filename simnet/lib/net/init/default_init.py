import torch.nn as nn


def default_init(module):
  """Initialize parameters of the module.

    For convolution, weights are initialized by Kaiming method and
    biases are initialized to zero.
    For batch normalization, scales and biases are set to 1 and 0,
    respectively.
    """
  if isinstance(module, nn.Conv2d):
    nn.init.kaiming_normal_(module.weight.data)
    if module.bias is not None:
      module.bias.data.zero_()
  elif isinstance(module, nn.Conv3d):
    nn.init.kaiming_normal_(module.weight.data)
    if module.bias is not None:
      module.bias.data.zero_()
  elif isinstance(module, nn.BatchNorm2d):
    module.weight.data.fill_(1)
    module.bias.data.zero_()
  elif isinstance(module, nn.BatchNorm3d):
    module.weight.data.fill_(1)
    module.bias.data.zero_()

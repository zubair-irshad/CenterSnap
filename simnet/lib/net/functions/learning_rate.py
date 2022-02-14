# Copyright 2018 Toyota Research Institute.  All rights reserved.
#
# Originally from Koichiro Yamaguchi's pixwislab repo mirrored at:
# https://github.awsinternal.tri.global/driving/pixwislab


def lambda_learning_rate_poly(max_epochs, exponent):
  """Make a function for computing learning rate by "poly" policy.

    This policy does a polynomial decay of the learning rate over the epochs
    of training.

    Args:
        max_epochs (int): max numbers of epochs
        exponent (float): exponent value
    """
  return lambda epoch: pow((1.0 - epoch / max_epochs), exponent)


def lambda_warmup(warmup_period, warmup_factor, wrapped_lambda):

  def warmup(epoch, warmup_period, warmup_factor):
    if epoch > warmup_period:
      return 1.0
    else:
      return warmup_factor + (1.0 - warmup_factor) * (epoch / warmup_period)

  return lambda epoch: warmup(epoch, warmup_period, warmup_factor) * wrapped_lambda(epoch)

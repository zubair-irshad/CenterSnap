import pathlib
from importlib.machinery import SourceFileLoader
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader
from simnet.lib.net.init.default_init import default_init
from simnet.lib.net.dataset import Dataset


def add_dataset_args(parser, prefix):
  group = parser.add_argument_group("{}_dataset".format(prefix))
  group.add_argument("--{}_path".format(prefix), type=str, required=True)
  #group.add_argument("--{}_fraction".format(prefix), type=str, default=None)
  group.add_argument("--{}_batch_size".format(prefix), default=16, type=int)
  group.add_argument("--{}_num_workers".format(prefix), default=7, type=int)
  #group.add_argument("--{}_random_crop".format(prefix), default=None, type=int, nargs=2)


def add_train_args(parser):
  parser.add_argument("--max_steps", type=int, required=True)
  parser.add_argument("--output", type=str, required=True)

  add_dataset_args(parser, "train")
  add_dataset_args(parser, "val")

  optim_group = parser.add_argument_group("optim")
  optim_group.add_argument("--optim_type", default='sgd', type=str)
  optim_group.add_argument("--optim_learning_rate", default=0.02, type=float)
  optim_group.add_argument("--optim_momentum", default=0.9, type=float)
  optim_group.add_argument("--optim_weight_decay", default=1e-4, type=float)
  optim_group.add_argument("--optim_poly_exp", default=0.9, type=float)
  optim_group.add_argument("--optim_warmup_epochs", default=None, type=int)
  parser.add_argument("--model_file", type=str, required=True)
  parser.add_argument("--model_name", type=str, required=True)
  parser.add_argument("--checkpoint", default=None, type=str)
  parser.add_argument("--wandb_name", type=str, required=True)
  # Ignore Mask Search.
  parser.add_argument("--min_height", default=0.0, type=float)
  parser.add_argument("--min_occlusion", default=0.0, type=float)
  parser.add_argument("--min_truncation", default=0.0, type=float)
  # Backbone configs
  parser.add_argument("--model_norm", default='BN', type=str)
  parser.add_argument("--num_filters_scale", default=4, type=int)

  # Loss weights
  parser.add_argument("--frozen_stereo_checkpoint", default=None, type=str)
  parser.add_argument("--loss_seg_mult", default=1.0, type=float)
  parser.add_argument("--loss_depth_mult", default=1.0, type=float)
  parser.add_argument("--loss_heatmap_mult", default=100.0, type=float)
  parser.add_argument("--loss_vertex_mult", default=0.1, type=float)
  parser.add_argument("--loss_z_centroid_mult", default=0.1, type=float)
  parser.add_argument("--loss_rotation_mult", default=0.1, type=float)
  parser.add_argument("--loss_keypoint_mult", default=0.1, type=float)
  parser.add_argument("--loss_latent_emb_mult", default=0.1, type=float)
  parser.add_argument("--loss_abs_pose_mult", default=0.1, type=float)
  # Stereo Stem Args
  parser.add_argument(
      "--loss_disparity_stdmean_scaled",
      action="store_true",
      help="If true, the loss will be scaled based on the standard deviation and mean of the "
      "ground truth disparities"
  )
  parser.add_argument("--cost_volume_downsample_factor", default=4, type=int)
  parser.add_argument("--max_disparity", default=90, type=int)
  parser.add_argument(
      "--fe_features",
      default=16,
      type=int,
      help="Number of output features in feature extraction stage"
  )
  parser.add_argument(
      "--fe_internal_features",
      default=32,
      type=int,
      help="Number of features in the first block of the feature extraction"
  )
  # keypoint head args
  parser.add_argument("--num_keypoints", default=1, type=int)



def get_config_value(hparams, prefix, key):
  full_key = "{}_{}".format(prefix, key)
  if hasattr(hparams, full_key):
    return getattr(hparams, full_key)
  else:
    return None


def get_loader(hparams, prefix, preprocess_func=None, datapoint_dataset=None):
  datasets = []
  path = get_config_value(hparams, prefix, 'path')
  datasets.append(
      Dataset(
          path, hparams, preprocess_image_func=preprocess_func, datapoint_dataset=datapoint_dataset
      )
  )
  batch_size = get_config_value(hparams, prefix, "batch_size")

  collate_fn = simnet_collate
  if prefix == 'train':
    return DataLoader(
        ConcatDataset(datasets),
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=get_config_value(hparams, prefix, "num_workers"),
        pin_memory=False,
        drop_last=True, 
        shuffle=True
    )
  else:
    return DataLoader(
      ConcatDataset(datasets),
      batch_size=batch_size,
      collate_fn=collate_fn,
      num_workers=get_config_value(hparams, prefix, "num_workers"),
      pin_memory=False,
      drop_last=True, 
      shuffle=False
  )


def simnet_collate(batch):
  # list of elements per patch
  # Each element is a tuple of (stereo,imgs)
  targets = []
  for ii in range(len(batch[0])):
    targets.append([batch_element[ii] for batch_element in batch])
  stacked_images = torch.stack(targets[0])
  return stacked_images, targets[1], targets[2], targets[3], targets[4], targets[5]


def prune_state_dict(state_dict):
  for key in list(state_dict.keys()):
    state_dict[key[6:]] = state_dict.pop(key)
  return state_dict


def keep_only_stereo_weights(state_dict):
  pruned_state_dict = {}
  for key in list(state_dict.keys()):
    if 'stereo' in key:
      pruned_state_dict[key] = state_dict[key]
  return pruned_state_dict


def get_model(hparams):
  model_path = (pathlib.Path(__file__).parent / hparams.model_file).resolve()
  print('Using model class from:', model_path)
  net_module = SourceFileLoader(hparams.model_name, str(model_path)).load_module()
  net_attr = getattr(net_module, hparams.model_name)
  model = net_attr(hparams)
  model.apply(default_init)
  # For large models use imagenet weights.
  # This speeds up training and can give a +2 mAP score on car detections
  if hparams.num_filters_scale == 1:
    model.load_imagenet_weights()

  if hparams.frozen_stereo_checkpoint is not None:
    print('Restoring stereo weights from checkpoint:', hparams.frozen_stereo_checkpoint)
    state_dict = torch.load(hparams.frozen_stereo_checkpoint, map_location='cpu')['state_dict']
    state_dict = prune_state_dict(state_dict)
    state_dict = keep_only_stereo_weights(state_dict)
    model.load_state_dict(state_dict, strict=False)

  if hparams.checkpoint is not None:
    print('Restoring from checkpoint:', hparams.checkpoint)
    state_dict = torch.load(hparams.checkpoint, map_location='cpu')['state_dict']
    state_dict = prune_state_dict(state_dict)
    model.load_state_dict(state_dict, strict=False)

  return model

import os

os.environ['PYTHONHASHSEED'] = str(1)
import argparse
from importlib.machinery import SourceFileLoader
import sys

import random

random.seed(12345)
import numpy as np

np.random.seed(12345)
import torch

torch.manual_seed(12345)

import wandb

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers

from simnet.lib.net import common
from simnet.lib import datapoint
from simnet.lib import camera
from simnet.lib.net.panoptic_trainer import PanopticModel

_GPU_TO_USE = 0


class EvalMethod():

  def __init__(self):

    self.eval_3d = None
    self.camera_model = camera.NOCS_Camera()

  def process_sample(self, pose_outputs, box_outputs, seg_outputs, detections_gt, scene_name):
    return True

  def process_all_dataset(self, log):
    return True
    # log['all 3Dmap'] = self.eval_3d.process_all_3D_dataset()

  def draw_detections(
      self,seg_outputs,left_image_np, llog, prefix
  ):
    seg_vis = seg_outputs.get_visualization_img(np.copy(left_image_np))
    llog[f'{prefix}/seg'] = wandb.Image(seg_vis, caption=prefix)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
  common.add_train_args(parser)
  hparams = parser.parse_args()
  train_ds = datapoint.make_dataset(hparams.train_path)
  samples_per_epoch = len(train_ds.list())
  samples_per_step = hparams.train_batch_size
  steps = hparams.max_steps
  steps_per_epoch = samples_per_epoch // samples_per_step
  epochs = int(np.ceil(steps / steps_per_epoch))
  actual_steps = epochs * steps_per_epoch
  print('Samples per epoch', samples_per_epoch)
  print('Steps per epoch', steps_per_epoch)
  print('Target steps:', steps)
  print('Actual steps:', actual_steps)
  print('Epochs:', epochs)

  model = PanopticModel(hparams, epochs, train_ds, EvalMethod())
  model_checkpoint = ModelCheckpoint(filepath=hparams.output, save_top_k=-1, period=1, mode='max')
  wandb_logger = loggers.WandbLogger(name=hparams.wandb_name, project='CenterSnap')

  if hparams.finetune_real:
    trainer = pl.Trainer(
        max_nb_epochs=epochs,
        early_stop_callback=None,
        gpus=[_GPU_TO_USE],
        checkpoint_callback=model_checkpoint,
        val_check_interval=1.0,
        logger=wandb_logger,
        default_save_path=hparams.output,
        use_amp=False,
        print_nan_grads=True,
        resume_from_checkpoint=hparams.checkpoint
    )
  else:
    trainer = pl.Trainer(
        max_nb_epochs=epochs,
        early_stop_callback=None,
        gpus=[_GPU_TO_USE],
        checkpoint_callback=model_checkpoint,
        val_check_interval=1.0,
        logger=wandb_logger,
        default_save_path=hparams.output,
        use_amp=False,
        print_nan_grads=True,
    )

  trainer.fit(model)

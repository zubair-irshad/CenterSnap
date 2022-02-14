import os
import copy

os.environ['PYTHONHASHSEED'] = str(1)
from importlib.machinery import SourceFileLoader
import random
random.seed(123456)
import numpy as np
np.random.seed(123456)
import torch
import wandb
import pytorch_lightning as pl

from simnet.lib.net import common
from simnet.lib.net.dataset import extract_left_numpy_img
from simnet.lib.net.functions.learning_rate import lambda_learning_rate_poly, lambda_warmup

_GPU_TO_USE = 0
class PanopticModel(pl.LightningModule):
  def __init__(
      self, hparams, epochs=None, train_dataset=None, eval_metric=None, preprocess_func=None
  ):
    super().__init__()

    self.hparams = hparams
    self.epochs = epochs
    self.train_dataset = train_dataset

    self.model = common.get_model(hparams)
    self.eval_metrics = eval_metric
    self.preprocess_func = preprocess_func

  def forward(self, image):
    seg_output, depth_output, small_depth_output, pose_output = self.model(
        image
    )
    return seg_output, depth_output, small_depth_output, pose_output
  
  def optimizer_step(self, epoch_nb, batch_nb, optimizer, optimizer_i, second_order_closure=None):
    super().optimizer_step(epoch_nb, batch_nb, optimizer, optimizer_i, second_order_closure)
    if batch_nb == 0:
      for param_group in optimizer.param_groups:
        learning_rate = param_group['lr']

  def training_step(self, batch, batch_idx):
    image, seg_target, depth_target, pose_targets, _, _ = batch
    seg_output, depth_output, small_depth_output, pose_outputs = self.forward(
        image
    )
    log = {}
    prefix = 'train'
    
    loss = depth_output.compute_loss(copy.deepcopy(depth_target), log, f'{prefix}_detailed/loss/refined_disp')
    if self.hparams.frozen_stereo_checkpoint is None:
      loss = loss + small_depth_output.compute_loss(depth_target, log, f'{prefix}_detailed/loss/train_cost_volume_disp')
    loss = loss + seg_output.compute_loss(seg_target, log, f'{prefix}_detailed/loss/seg')
    if pose_targets[0] is not None:
      loss = loss + pose_outputs.compute_loss(pose_targets, log, f'{prefix}_detailed/pose')
    log['train/loss/total'] = loss

    if (batch_idx % 200) == 0:
      with torch.no_grad():
        llog = {}
        prefix = 'train'
        left_image_np = extract_left_numpy_img(image[0])
        logger = self.logger.experiment
        seg_pred_vis = seg_output.get_visualization_img(np.copy(left_image_np))
        llog[f'{prefix}/seg'] = wandb.Image(seg_pred_vis, caption=prefix)
        depth_vis = depth_output.get_visualization_img(np.copy(left_image_np))
        llog[f'{prefix}/disparity'] = wandb.Image(depth_vis, caption=prefix)
        small_depth_vis = small_depth_output.get_visualization_img(np.copy(left_image_np))
        llog[f'{prefix}/small_disparity'] = wandb.Image(small_depth_vis, caption=prefix)
        logger.log(llog)
    return {'loss': loss, 'log': log}

  def validation_step(self, batch, batch_idx):
    image, seg_target, depth_target, pose_targets, detections_gt, scene_name = batch
    dt = [di.depth_pred.cuda().unsqueeze(0) for di in depth_target]
    dt = torch.stack(dt)
    real_image = torch.cat([image[:,:3,:,:], dt], dim=1)
    seg_output, depth_output, small_depth_output, pose_outputs = self.forward(
        real_image
    )
    log = {}
    logger = self.logger.experiment
    with torch.no_grad():
      prefix_loss = 'validation'
      loss = depth_output.compute_loss(copy.deepcopy(depth_target), log, f'{prefix_loss}_detailed/loss/refined_disp')
      if self.hparams.frozen_stereo_checkpoint is None:
        loss = loss + small_depth_output.compute_loss(depth_target, log, f'{prefix_loss}_detailed_loss/train_cost_volume_disp')
      loss = loss + seg_output.compute_loss(seg_target, log, f'{prefix_loss}_detailed/loss/seg')      
      if pose_targets[0] is not None:
        loss = loss + pose_outputs.compute_loss(pose_targets, log, f'{prefix_loss}_detailed/pose')
      log['validation/loss/total'] = loss.item()
      if batch_idx < 5 or scene_name[0] == 'fmk':
        llog = {}
        left_image_np = extract_left_numpy_img(image[0])
        prefix = f'val/{batch_idx}'
        depth_vis = depth_output.get_visualization_img(np.copy(left_image_np))
        llog[f'{prefix}/disparity'] = wandb.Image(depth_vis, caption=prefix)
        small_depth_vis = small_depth_output.get_visualization_img(np.copy(left_image_np))
        llog[f'{prefix}/small_disparity'] = wandb.Image(small_depth_vis, caption=prefix)
        self.eval_metrics.draw_detections(
           seg_output,left_image_np, llog, prefix
        )
        logger.log(llog)
    return log

  def validation_epoch_end(self, outputs):
    self.trainer.checkpoint_callback.save_best_only = False
    mean_dict = {}
    for key in outputs[0].keys():
        mean_dict[key] = np.mean([d[key] for d in outputs], axis=0)
    logger = self.logger.experiment
    logger.log(mean_dict)
    log = {}
    return {'log': log}

  @pl.data_loader
  def train_dataloader(self):
    return common.get_loader(
        self.hparams,
        "train",
        preprocess_func=self.preprocess_func,
        datapoint_dataset=self.train_dataset
    )
  @pl.data_loader
  def val_dataloader(self):
    return common.get_loader(self.hparams, "val", preprocess_func=self.preprocess_func)

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.optim_learning_rate)
    lr_lambda = lambda_learning_rate_poly(self.epochs, self.hparams.optim_poly_exp)
    if self.hparams.optim_warmup_epochs is not None and self.hparams.optim_warmup_epochs > 0:
      lr_lambda = lambda_warmup(self.hparams.optim_warmup_epochs, 0.2, lr_lambda)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return [optimizer], [scheduler]

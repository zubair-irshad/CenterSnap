import numpy as np
import cv2
import IPython
import torch
import torch.nn as nn
from skimage.feature import peak_local_max
LOSS = nn.BCELoss()
# LOSS = nn.MSELoss()


class KeypointOutput:

  def __init__(self, heatmap, hparams, ignore_mask=None):
    self.heatmap = heatmap
    self.first_heatmap = None
    self.ignore_mask = ignore_mask
    self.is_numpy = False
    self.hparams = hparams
    self.loss = LOSS
    self.num_keypoints = hparams.num_keypoints
    self.all_keypoints = None

  # Converters for torch to numpy
  def convert_to_numpy_from_torch(self):
    self.heatmap = np.ascontiguousarray(self.heatmap.cpu().numpy())
    self.is_numpy = True

  def convert_to_torch_from_numpy(self):
    self.heatmap = torch.from_numpy(np.ascontiguousarray(self.heatmap)).float()
    self.is_numpy = False

  def evaluation_metrics(self, targ_kp_output):
    targ_keypoints = targ_kp_output.get_keypoints()
    results = {}
    for i in range(len(targ_keypoints)):
      results[i] = {}
    for confidence in np.linspace(0, 1.01, 10):
      all_keypoints = self.get_keypoints(min_confidence=confidence)
      for i, (pred_class, targ_class) in enumerate(zip(all_keypoints, targ_keypoints)):
        tp, fp, fn = evaluate_keypoints(pred_class, targ_class)
        precision = tp / (tp + fp) if tp + fp > 0 else 1
        recall = tp / (tp + fn) if tp + fn > 0 else 1
        results[i][confidence] = (precision, recall)
    return results

  def get_visualization_img(self, left_img):
    if not self.is_numpy:
      self.convert_to_numpy_from_torch()
    if self.num_keypoints == 1:
      self.first_heatmap = self.heatmap[:, None, :, :][0]
    else:
      self.first_heatmap = self.heatmap[0]
    return vis_network_outputs(self.first_heatmap, left_img)

  def compute_loss(self, keypoint_targets, log):
    if self.is_numpy:
      raise ValueError("Output is not in torch mode")
    heatmap_target = torch.stack([
        torch.squeeze(keypoint_target.heatmap) for keypoint_target in keypoint_targets
    ])

    # Move to GPU
    heatmap_target = heatmap_target.to(torch.device('cuda:0'))

    heatmap_loss = self.loss(self.heatmap, heatmap_target)
    log['keypoint'] = heatmap_loss
    return self.hparams.loss_keypoint_mult * heatmap_loss


def vis_network_outputs(heatmaps, left_img, idx=0):
  heatmap_vis = []
  gray_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
  for heatmap in heatmaps:
    img = np.copy(gray_img)
    heatmap /= np.max(heatmap)
    img = cv2.addWeighted(heatmap.astype(float), 0.999, img.astype(float), 0.001, 0)
    img /= img.max() / 255
    heatmap_vis.append(img[:, :, np.newaxis].astype(np.float32))
  return heatmap_vis

def extract_peaks_from_heatmap(heatmap, min_distance=40, min_confidence=0.3):
  peaks = peak_local_max(
      heatmap,
      min_distance=min_distance,
      threshold_abs=min_confidence,
      exclude_border=False,
      num_peaks=2
  )
  return peaks

def evaluate_keypoints(pred_kp, targ_kp, distance_threshold=20):
  true_positives = 0
  false_negatives = 0
  counted = []
  all_distances = []
  for px in pred_kp.pixels:
    distances = np.linalg.norm(targ_kp.pixels - px, axis=1)
    all_distances.append(distances)
  all_distances = np.array(all_distances)
  for i in range(len(targ_kp.pixels)):
    if len(all_distances) == 0:
      break
    closest = all_distances[:, i].argmin()  # prediction that is closest
    if all_distances[closest, i] < distance_threshold:
      true_positives += 1  # correct prediction
      all_distances[closest] = 1e10  # don't let this prediction be a positive for anything else
    else:
      false_negatives += 1  # no prediction was sufficiently close
  false_positives = len(pred_kp.pixels) - true_positives
  false_negatives = len(targ_kp.pixels) - true_positives
  return true_positives, false_positives, false_negatives

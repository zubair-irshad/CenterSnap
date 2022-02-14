import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib import rc,rcParams
# activate latex text rendering
rc('text', usetex=True)
rc('axes', linewidth=2)
rc('font', weight='bold')
rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']

def plot_mAP_combined(iou_aps, pose_aps, iou_aps_2, pose_aps_2, out_dir, iou_thres_list, degree_thres_list, shift_thres_list):
    """ Draw iou 3d AP vs. iou thresholds.
    """
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Ubuntu'
    plt.rcParams['font.sans-serif'] = []
    plt.rcParams['font.monospace'] = 'Ubuntu Mono'
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['font.size'] = 12
    plt.rcParams['legend.fontsize'] = 8
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.style.use('seaborn-notebook')
    
    plt.set_cmap('plasma')
#     ax1.set_prop_cycle( cycler('color', ['c', 'm', 'y', 'k']) )

    labels = ['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug', 'mean', 'nocs']
#     plasma = cm.get_cmap('plasma', 7)
#     viridis = cm.get_cmap('viridis', 7)
#     print("plasma", plasma)
#     colors = plasma.colors
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:pink', 'tab:olive', 'tab:purple', 'tab:red', 'tab:gray']
    styles = ['-', '-', '-', '-', '-', '-', '--', ':']

    fig, (axes) = plt.subplots(2, 3, figsize=(8, 7), dpi=200)
    ax_iou, ax_degree, ax_shift, ax_iou_2, ax_degree_2, ax_shift_2 = axes[0,0], axes[0,1], axes[0,2], axes[1,0], axes[1,1], axes[1,2]
    # IoU subplot
    ax_iou.set_title('3D IoU AP', fontsize=10)
    ax_iou.set_ylabel('AP %')
    ax_iou.set_ylim(0, 100)
#     ax_iou.set_xlabel('3D IOU %')
    ax_iou.set_xlim(0, 100)
    ax_iou.xaxis.set_ticks([0, 25, 50, 75, 100])
    ax_iou.grid()
    for i in range(1, iou_aps.shape[0]):
        ax_iou.plot(100*np.array(iou_thres_list), 100*iou_aps[i, :],
                    color=colors[i-1], linestyle=styles[i-1], label=labels[i-1])
    # rotation subplot
    ax_degree.set_title('Rotation', fontsize=10)
    ax_degree.set_ylim(0, 100)
    ax_degree.yaxis.set_ticklabels([])
#     ax_degree.set_xlabel('Rotation Error/Degree')
    ax_degree.set_xlim(0, 60)
    ax_degree.xaxis.set_ticks([0, 20, 40, 60])
    ax_degree.grid()
    for i in range(1, pose_aps.shape[0]):
        ax_degree.plot(np.array(degree_thres_list), 100*pose_aps[i, :len(degree_thres_list), -1],
                       color=colors[i-1], linestyle=styles[i-1], label=labels[i-1])
    # translation subplot
    ax_shift.set_title('Translation', fontsize=10)
    ax_shift.set_ylim(0, 100)
    ax_shift.yaxis.set_ticklabels([])
#     ax_shift.set_xlabel('Translation Error/cm')
    ax_shift.set_xlim(0, 10)
    ax_shift.xaxis.set_ticks([0, 5, 10])
    ax_shift.grid()
    for i in range(1, pose_aps.shape[0]):
        ax_shift.plot(np.array(shift_thres_list), 100*pose_aps[i, -1, :len(shift_thres_list)],
                      color=colors[i-1], linestyle=styles[i-1], label=labels[i-1])
        
    # IoU subplot
#     ax_iou_2.set_title('3D IoU AP', fontsize=10)
    ax_iou_2.set_ylabel('\large{\textbf{CAMERA25} \n AP %')
    ax_iou_2.set_ylim(0, 100)
    ax_iou_2.set_xlabel('3D IOU %')
    ax_iou_2.set_xlim(0, 100)
    ax_iou_2.xaxis.set_ticks([0, 25, 50, 75, 100])
    ax_iou_2.grid()
    for i in range(1, iou_aps.shape[0]):
        ax_iou_2.plot(100*np.array(iou_thres_list), 100*iou_aps_2[i, :],
                    color=colors[i-1], linestyle=styles[i-1], label=labels[i-1])
    # rotation subplot
#     ax_degree_2.set_title('Rotation', fontsize=10)
    ax_degree_2.set_ylim(0, 100)
    ax_degree_2.yaxis.set_ticklabels([])
    ax_degree_2.set_xlabel('Rotation Error/Degree')
    ax_degree_2.set_xlim(0, 60)
    ax_degree_2.xaxis.set_ticks([0, 20, 40, 60])
    ax_degree_2.grid()
    for i in range(1, pose_aps.shape[0]):
        ax_degree_2.plot(np.array(degree_thres_list), 100*pose_aps_2[i, :len(degree_thres_list), -1],
                       color=colors[i-1], linestyle=styles[i-1], label=labels[i-1])
    # translation subplot
#     ax_shift_2.set_title('Translation', fontsize=10)
    ax_shift_2.set_ylim(0, 100)
    ax_shift_2.yaxis.set_ticklabels([])
    ax_shift_2.set_xlabel('Translation Error/cm')
    ax_shift_2.set_xlim(0, 10)
    ax_shift_2.xaxis.set_ticks([0, 5, 10])
    ax_shift_2.grid()
    for i in range(1, pose_aps.shape[0]):
        ax_shift_2.plot(np.array(shift_thres_list), 100*pose_aps_2[i, -1, :len(shift_thres_list)],
                      color=colors[i-1], linestyle=styles[i-1], label=labels[i-1])
        
        
        
    ax_shift_2.legend(loc='lower right', fontsize='small')
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(out_dir, 'mAP_combined.png'))
    plt.close(fig)
    return

import os
import _pickle as cPickle
real_file='/home/zubairirshad/fleet/simnet/data/nocs_results/rgbd_nocs_Real_finetune_with_ICP_refine_inv_mcrnn_ids_selected_norm_24eval_results/validation_inference'
cam_file = '/home/zubairirshad/fleet/simnet/data/nocs_results/rgbd_nocs_CAM_conventional_with_ICP_selected_norm_23eval_results/validation_inference'
pkl_path = os.path.join(cam_file, 'mAP_Acc.pkl')

with open(pkl_path, 'rb') as f:
  nocs_results = cPickle.load(f)

iou_aps = nocs_results['iou_aps']
pose_aps = nocs_results['pose_aps']
iou_thres_list = nocs_results['iou_thres_list']
degree_thres_list = nocs_results['degree_thres_list']
shift_thres_list = nocs_results['shift_thres_list']

pkl_path = os.path.join(real_file, 'mAP_Acc.pkl')
with open(pkl_path, 'rb') as f:
  nocs_results = cPickle.load(f)

iou_aps_real = nocs_results['iou_aps']
pose_aps_real = nocs_results['pose_aps']
iou_thres_list_real = nocs_results['iou_thres_list']
degree_thres_list_real = nocs_results['degree_thres_list']
shift_thres_list_real = nocs_results['shift_thres_list']

result_dir = '/home/zubairirshad'

plot_mAP_combined(iou_aps, pose_aps, iou_aps_real, pose_aps_real, result_dir, iou_thres_list, degree_thres_list, shift_thres_list)
import os
import time
import argparse
import torch
import wandb
from external.shape_pretraining.model.auto_encoder import PointCloudAE
from pytorch3d.loss import chamfer_distance
from external.shape_pretraining.dataset.shape_dataset import ShapeDataset
from utils.shape_utils import setup_logger
import wandb
wandb.init(project="train_ae")

parser = argparse.ArgumentParser()
parser.add_argument('--num_point', type=int, default=2048, help='number of points, needed if use points')
parser.add_argument('--emb_dim', type=int, default=128, help='dimension of latent embedding [default: 512]')
parser.add_argument('--h5_file', type=str, default='obj_models_dir/ShapeNetCore_2048.h5', help='h5 file')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--num_workers', type=int, default=10, help='number of data loading workers')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
parser.add_argument('--start_epoch', type=int, default=1, help='which epoch to start')
parser.add_argument('--max_epoch', type=int, default=50, help='max number of epochs to train')
parser.add_argument('--resume_model', type=str, default='', help='resume from saved model')
parser.add_argument('--result_dir', type=str, default='results/ae_2048', help='directory to save train results')
opt = parser.parse_args()

opt.repeat_epoch = 30
opt.decay_step = 5000
opt.decay_rate = [1.0, 0.6, 0.3, 0.1]


def train_net():
    if not os.path.exists(opt.result_dir):
        os.makedirs(opt.result_dir)
    logger = setup_logger('train_log', os.path.join(opt.result_dir, 'log.txt'))
    for key, value in vars(opt).items():
        logger.info(key + ': ' + str(value))
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    # model & loss
    estimator = PointCloudAE(opt.emb_dim, opt.num_point)
    estimator.cuda()
    if opt.resume_model != '':
        estimator.load_state_dict(torch.load(opt.resume_model))
    # dataset
    train_dataset = ShapeDataset(opt.h5_file, mode='train', augment=True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size,
                                                   shuffle=True, num_workers=opt.num_workers)
    val_dataset = ShapeDataset(opt.h5_file, mode='val', augment=False)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size,
                                                 shuffle=False, num_workers=opt.num_workers)
    # train
    st_time = time.time()
    global_step = ((train_dataset.length + opt.batch_size - 1) // opt.batch_size) * opt.repeat_epoch * (opt.start_epoch - 1)
    print("train_dataset",train_dataset.length)
    print("global step", global_step)
    decay_count = -1
    for epoch in range(opt.start_epoch, opt.max_epoch+1):
        # train one epoch
        logger.info('Time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + \
                    ', ' + 'Epoch %02d' % epoch + ', ' + 'Training started'))
        # create optimizer and adjust learning rate if needed
        if global_step // opt.decay_step > decay_count:
            decay_count += 1
            if decay_count < len(opt.decay_rate):
                current_lr = opt.lr * opt.decay_rate[decay_count]
                optimizer = torch.optim.Adam(estimator.parameters(), lr=current_lr)
        batch_idx = 0
        estimator.train()
        for rep in range(opt.repeat_epoch):
            for i, data in enumerate(train_dataloader):
                # label must be zero_indexed
                batch_xyz, batch_label = data
                batch_xyz = batch_xyz[:, :, :3].cuda()
                optimizer.zero_grad()
                embedding, point_cloud = estimator(batch_xyz)
                loss, _ = chamfer_distance(point_cloud, batch_xyz)
                wandb.log({'train_loss': loss})
                loss.backward()
                optimizer.step()
                global_step += 1
                batch_idx += 1
                # write results to tensorboard
                # tb_writer.add_summary(summary, global_step)
                if batch_idx % 100 == 0:
                    logger.info('Batch {0} Loss:{1:f}'.format(batch_idx, loss))
        logger.info('>>>>>>>>----------Epoch {:02d} train finish---------<<<<<<<<'.format(epoch))
        # evaluate one epoch
        logger.info('Time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + \
                    ', ' + 'Epoch %02d' % epoch + ', ' + 'Testing started'))
        estimator.eval()
        val_loss = 0.0
        for i, data in enumerate(val_dataloader, 1):
            batch_xyz, batch_label = data
            batch_xyz = batch_xyz[:, :, :3].cuda()
            embedding, point_cloud = estimator(batch_xyz)
            loss, _ = chamfer_distance(point_cloud, batch_xyz)
            # loss, _, _ = criterion(point_cloud, batch_xyz)
            val_loss += loss.item()
            logger.info('Batch {0} Loss:{1:f}'.format(i, loss))
        val_loss = val_loss / i
        wandb.log({'val_loss': val_loss})
        # summary = tf.Summary(value=[tf.Summary.Value(tag='val_loss', simple_value=val_loss)])
        # tb_writer.add_summary(summary, global_step)
        logger.info('Epoch {0:02d} test average loss: {1:06f}'.format(epoch, val_loss))
        logger.info('>>>>>>>>----------Epoch {:02d} test finish---------<<<<<<<<'.format(epoch))
        # save model after each epoch
        torch.save(estimator.state_dict(), '{0}/model_{1:02d}.pth'.format(opt.result_dir, epoch))

if __name__ == '__main__':
    train_net()
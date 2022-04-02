import os
import torch
import torch.nn.functional as F
import sys

sys.path.append('./models')
import numpy as np
from datetime import datetime
from models.TriTransNet import TriTransNet
from data import get_loader, test_dataset
from utils import clip_gradient, adjust_lr
import logging
import torch.backends.cudnn as cudnn
from options import config


if config.gpu_id == '0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('USE GPU 0')
elif config.gpu_id == '1':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('USE GPU 1')
cudnn.benchmark = True


model = TriTransNet()

if (config.load is not None):
    model.load_state_dict(torch.load(config.load))
    print('load model from ', config.load)

model.cuda()
params = model.parameters()
optimizer = torch.optim.Adam(params, config.lr)

image_root = config.rgb_root
gt_root = config.gt_root
depth_root = config.depth_root
test_image_root = config.test_rgb_root
test_gt_root = config.test_gt_root
test_depth_root = config.test_depth_root
save_path = config.save_path

if not os.path.exists(save_path):
    os.makedirs(save_path)


print('load data...')
train_loader = get_loader(image_root, gt_root, depth_root, batchsize=config.batchsize, trainsize=config.trainsize)
test_loader = test_dataset(test_image_root, test_gt_root, test_depth_root, config.trainsize)
total_step = len(train_loader)

logging.basicConfig(filename=save_path + 'log.log', format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
logging.info("TransformerNet-Train")
logging.info("Config")
logging.info(
    'epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};load:{};save_path:{};decay_epoch:{}'.format(
        config.epoch, config.lr, config.batchsize, config.trainsize, config.clip, config.decay_rate, config.load,
        save_path, config.decay_epoch))


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

step = 0
best_mae = 1
best_epoch = 0


def train(train_loader, model, optimizer, epoch, save_path):
    global step
    model.train()
    loss_all = 0
    epoch_step = 0
    try:
        for i, (images, gts, depths) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            images = images.cuda()
            gts = gts.cuda()
            depths = depths.cuda()

            s1, s2, s3, s4 = model(images, depths)
            loss1 = structure_loss(s1, gts)
            loss2 = structure_loss(s2, gts)
            loss3 = structure_loss(s3, gts)
            loss4 = structure_loss(s4, gts)
            loss = loss1 + loss2 + loss3 + loss4

            loss.backward()

            clip_gradient(optimizer, config.clip)
            optimizer.step()
            step += 1
            epoch_step += 1
            loss_all += loss.data
            if i % 100 == 0 or i == total_step or i == 1:
                print(
                    '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss1: {:.4f} Loss2: {:0.4f} Loss3: {:.4f} Loss4: {:.4f}'.
                    format(datetime.now(), epoch, config.epoch, i, total_step, loss1.data, loss2.data, loss3.data,
                           loss4.data))
                logging.info(
                    '#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss1: {:.4f} Loss2: {:.4f} Loss3: {:.4f} Loss4: {:.4f}'.
                    format(epoch, config.epoch, i, total_step, loss1.data, loss2.data, loss3.data, loss4.data))

        loss_all /= epoch_step
        logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, config.epoch, loss_all))
        if (epoch) % 5 == 0:
            torch.save(model.state_dict(), save_path + 'TriTransNet_epoch_{}.pth'.format(epoch))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + 'TriTransNet_epoch_{}.pth'.format(epoch + 1))
        print('save checkpoints successfully!')
        raise


def test(test_loader, model, epoch, save_path):
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        for i in range(test_loader.size):
            image, gt, depth, name, img_for_post = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            depth = depth.cuda()
            res, _, _, _ = model(image, depth)
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])
        mae = mae_sum / test_loader.size
        print('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch, mae, best_mae, best_epoch))
        if epoch == 1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'TriTransNet_epoch_best.pth')
                print('best epoch:{}'.format(epoch))
        logging.info('#TEST#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))


if __name__ == '__main__':
    print("Start train...")
    for epoch in range(1, config.epoch):
        cur_lr = adjust_lr(optimizer, config.lr, epoch, config.decay_rate, config.decay_epoch)
        train(train_loader, model, optimizer, epoch, save_path)
        test(test_loader, model, epoch, save_path)

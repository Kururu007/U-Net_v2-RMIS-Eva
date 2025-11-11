import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from lib.pvt import PolypPVT
from tabulate import tabulate
from utils.dataloader import get_loader, get_loader_QaTa, test_dataset
from utils.utils import clip_gradient, adjust_lr, AvgMeter
import torch.nn.functional as F
import numpy as np
import logging
import wandb
from unet_v2.UNet_v2 import UNetV2

import matplotlib.pyplot as plt

import torchmetrics
from torchmetrics import Dice, Accuracy
from torchmetrics.classification import BinaryJaccardIndex

class SampleMeanBinaryJaccard(torchmetrics.Metric):
    """先对 batch 内每个样本独立计算 Binary Jaccard Index，
       然后对这些样本分数求平均。
    """
    higher_is_better: bool = True  # IoU 越大越好

    def __init__(self, **kwargs):
        super().__init__(dist_sync_on_step=False)
        # 内部仍然用官方 BinaryJaccardIndex
        self._jac = BinaryJaccardIndex(**kwargs)

        # 累加样本级 IoU 之和与样本计数
        self.add_state("sum_iou", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_items", default=torch.tensor(0),   dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Args:
            preds  : [B, ...]  二值预测
            target : [B, ...]  二值标签
        """
        B = preds.shape[0]

        # 将每个样本展平到一维后单独喂入 BinaryJaccardIndex
        preds_flat  = preds.reshape(B, -1)
        target_flat = target.reshape(B, -1)

        for i in range(B):
            score = self._jac(preds_flat[i], target_flat[i])  # 单样本 IoU
            self._jac.reset()  # 清掉内部状态，下一样本重新计算
            self.sum_iou += score
            self.n_items += 1

    def compute(self):
        # 返回所有样本 IoU 的平均值
        return self.sum_iou / self.n_items

def structure_loss(pred, mask):
    if mask.shape[-1] != pred.shape[-1]:
        pred = F.interpolate(pred, size=mask.shape[-2:],  mode='bilinear', align_corners=False)

    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


def test(model, path):
    image_root = '{}/images/'.format(path)
    gt_root = '{}/masks/'.format(path)
    model.eval()
    num1 = len(os.listdir(gt_root))
    test_loader = test_dataset(image_root, gt_root, 224)
    
    macro_dice_meter = Dice(average='samples').cuda()
    macro_miou_meter = SampleMeanBinaryJaccard().cuda()
    micro_dice_meter = Dice().cuda()
    micro_miou_meter = BinaryJaccardIndex().cuda()
    micro_acc_meter = Accuracy(task='binary').cuda()
    
    for i in range(num1):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        gt = torch.from_numpy(gt).cuda()

        res, res1 = model(image)
        res1 = F.interpolate(res1, res.shape[-2:],  mode='bilinear', align_corners=False)
        res = F.upsample(res + res1, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().squeeze(0)
        gt = gt.unsqueeze(0)
        # print(res.size(), gt.size())
        macro_dice_meter.update(res, gt.long())
        macro_miou_meter.update(res, gt.long())
        micro_dice_meter.update(res, gt.long())
        micro_miou_meter.update(res, gt.long())
        micro_acc_meter.update(res, gt.long())

    # return DSC / num1
    test_macro_dice = macro_dice_meter.compute().item()
    test_macro_miou = macro_miou_meter.compute().item()
    test_micro_dice = micro_dice_meter.compute().item()
    test_micro_miou = micro_miou_meter.compute().item()
    test_micro_acc = micro_acc_meter.compute().item()
    return test_macro_dice, test_macro_miou, test_micro_dice, test_micro_miou, test_micro_acc

def train(train_loader, model, optimizer, epoch, test_path, state):
    model.train()
    global best
    size_rates = [0.75, 1, 1.25] 
    loss_P2_record = AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            # ---- rescale ----
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            # ---- forward ----
            P1, P2 = model(images)
            # ---- loss function ----
            loss_P1 = structure_loss(P1, gts)
            loss_P2 = structure_loss(P2, gts)
            loss = loss_P1 + loss_P2 
            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            # ---- recording loss ----
            if rate == 1:
                loss_P2_record.update(loss_P2.data, opt.batchsize)
        # ---- train visualization ----
        if i % 10 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  ' lateral-5: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_P2_record.show()))
    # save model 
    save_path = opt.train_save

    latest_path = os.path.join(save_path, "latest")
    best_path = os.path.join(save_path, "best")
    if not os.path.exists(latest_path):
        os.makedirs(latest_path)
    if not os.path.exists(best_path):
        os.makedirs(best_path)

    torch.save(model.state_dict(), os.path.join(latest_path, f"last.pth"))
    # choose the best model
   
    test1path = opt.test_path

    
    if (epoch + 1) % 1 == 0:
        test_macro_dice, test_macro_miou, test_micro_dice, test_micro_miou, test_micro_acc = test(model, test1path)

        wandb.log({f"QaTa/mDice": test_macro_dice}, step=epoch)
        wandb.log({f"QaTa/mIoU": test_macro_miou}, step=epoch)
        wandb.log({f"QaTa/gDice": test_micro_dice}, step=epoch)
        wandb.log({f"QaTa/gIoU": test_micro_miou}, step=epoch)
        wandb.log({f"QaTa/Acc": test_micro_acc}, step=epoch)
        
        print(f'Testing performance in val model: mDice : %f, mIoU : %f, gDice : %f, gIoU : %f, Acc. : %f' % (test_macro_dice, test_macro_miou, test_micro_dice, test_micro_miou, test_micro_acc))
        
        if test_macro_dice > state["best_mdice"]:
            state["best_mdice"] = float(test_macro_dice)
            state["best_epoch"] = epoch
            torch.save(model.state_dict(), os.path.join(best_path, f"best_epoch_{epoch}.pth"))
            print(f'got best mdice {state["best_mdice"]} at epoch {epoch}'.center(70, '='))
    
    
if __name__ == '__main__':
    dict_plot = {'CVC-300':[], 'CVC-ClinicDB':[], 'Kvasir':[], 'CVC-ColonDB':[], 'ETIS-LaribPolypDB':[], 'test':[]}
    name = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'test']
    ##################model_name#############################
    # model_name = 'PolypPVT'
    ###############################################
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', type=int,
                        default=100, help='epoch number')

    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')

    parser.add_argument('--optimizer', type=str,
                        default='AdamW', help='choosing optimizer AdamW or SGD')

    parser.add_argument('--augmentation',
                        default=False, help='choose to do random flip rotation')

    parser.add_argument('--batchsize', type=int,
                        default=24, help='training batch size')

    parser.add_argument('--trainsize', type=int,
                        default=352, help='training dataset size')

    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')

    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')

    parser.add_argument('--decay_epoch', type=int,
                        default=50, help='every n epochs decay learning rate')

    parser.add_argument('--train_path', type=str,
                        default=None, required=True,
                        help='path to train dataset')

    parser.add_argument('--test_path', type=str,
                        default=None, required=True,
                        help='path to testing Kvasir dataset')

    parser.add_argument('--train_save', type=str,
                        default=None, required=True)

    opt = parser.parse_args()
    logging.basicConfig(filename='train_log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

    # ---- build models ----

    # from lib.pvt_3 import PVTNetwork
    # model = PVTNetwork().cuda()

    model = UNetV2(pretrained_path=f"{os.getcwd()}/PolypSeg/pvt_pth/pvt_v2_b2.pth").cuda()

    model_name = model.__class__.__name__
    print(model.__class__.__name__.center(50, "="))

    opt.train_save = os.path.join(opt.train_save, model_name)

    # wandb.login(key="66b58ac7004a123a43487d7a6cf34ebb4571a7ea")
    wandb.login(key="cd32109501327dc0c5d7a1e1600e2f122ff5a0ed")
    wandb.init(project="UNet_v2_QaTa",
            #    dir="./wandb",
               name=model.__class__.__name__,
               resume="allow",  # must resume, otherwise crash
               # id=id,
               config={"class_name": str(model.__class__.__name__)})

    best = 0

    params = model.parameters()

    if opt.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(params, opt.lr, weight_decay=1e-4, momentum=0.9)

    print(optimizer)
    image_root = '{}/images/'.format(opt.train_path)
    gt_root = '{}/masks/'.format(opt.train_path)

    train_loader = get_loader_QaTa(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize, augmentation=opt.augmentation)
    total_step = len(train_loader)

    print("#" * 20, "Start Training", "#" * 20)
    
    state = {"best_mdice": -1.0, "best_epoch": -1}

    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, 0.1, 200)
        train(train_loader, model, optimizer, epoch, opt.test_path, state)
    

import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
from lib.pvt import PolypPVT
from unet_v2.UNet_v2 import UNetV2
from utils.dataloader import test_dataset
import cv2

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=352, help='testing size')
    parser.add_argument('--pth_path', type=str, default='./model_pth/PolypPVT.pth')
    opt = parser.parse_args()
    model = UNetV2(pretrained_path=f"{os.getcwd()}/PolypSeg/pvt_pth/pvt_v2_b2.pth")
    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    model.eval()

    ##### put data_path here #####
    data_path = './PolypSeg/data/MosMedTestDataset'
    ##### save_path #####
    save_path = './result_map/UNetV2_MosMed/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    
    macro_dice_meter = Dice(average='samples').cuda()
    macro_miou_meter = SampleMeanBinaryJaccard().cuda()
    micro_dice_meter = Dice().cuda()
    micro_miou_meter = BinaryJaccardIndex().cuda()
    micro_acc_meter = Accuracy(task='binary').cuda()
    
    num1 = len(os.listdir(gt_root))
    test_loader = test_dataset(image_root, gt_root, opt.testsize)
    for i in range(num1):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        gt = torch.from_numpy(gt).cuda()
        P1, P2 = model(image)
        P2 = F.interpolate(P2, P1.shape[-2:],  mode='bilinear', align_corners=False)
        res = F.upsample(P1+P2, size=gt.shape, mode='bilinear', align_corners=False)
        out_meitrc = res.sigmoid().squeeze(0)
        gt = gt.unsqueeze(0)
        macro_dice_meter.update(out_meitrc, gt.long())
        macro_miou_meter.update(out_meitrc, gt.long())
        micro_dice_meter.update(out_meitrc, gt.long())
        micro_miou_meter.update(out_meitrc, gt.long())
        micro_acc_meter.update(out_meitrc, gt.long())
        
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        cv2.imwrite(save_path+name, res*255)
    
    test_macro_dice = macro_dice_meter.compute().item()
    test_macro_miou = macro_miou_meter.compute().item()
    test_micro_dice = micro_dice_meter.compute().item()
    test_micro_miou = micro_miou_meter.compute().item()
    test_micro_acc = micro_acc_meter.compute().item()
    print('Finish!')
    print(f'Testing performance: mDice : %f, mIoU : %f, gDice : %f, gIoU : %f, Acc. : %f' % (test_macro_dice, test_macro_miou, test_micro_dice, test_micro_miou, test_micro_acc))

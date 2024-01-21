import math
from functools import partial

import numpy as np
import torch


# 三元组损失函数
def triplet_loss(alpha=0.2):
    def _triplet_loss(y_pred, Batch_size):
        anchor, positive, negative = y_pred[:int(Batch_size)], y_pred[int(Batch_size):int(2 * Batch_size)], y_pred[
                                                                                                            int(2 * Batch_size):]

        pos_dist = torch.sqrt(torch.sum(torch.pow(anchor - positive, 2), axis=-1))
        neg_dist = torch.sqrt(torch.sum(torch.pow(anchor - negative, 2), axis=-1))

        keep_all = (neg_dist - pos_dist < alpha).cpu().numpy().flatten()
        hard_triplets = np.where(keep_all == 1)

        pos_dist = pos_dist[hard_triplets]
        neg_dist = neg_dist[hard_triplets]

        basic_loss = pos_dist - neg_dist + alpha
        loss = torch.sum(basic_loss) / torch.max(torch.tensor(1), torch.tensor(len(hard_triplets[0])))
        return loss

    return _triplet_loss


# 获取学习率下降函数
def get_Lr_Fun(lr_decay_type, maxLR, minLR, totalEpoch, config):
    # yolox里的warmcos学习率方法
    def yolox_warmcos(lr, minLr, total_Epoch, upToMax_Epoch, startToMax_LR, keepWithMin_Epoch, currentEpoch):
        # 前面周期从startToMax_LR上升到maxLR
        if currentEpoch <= upToMax_Epoch:
            lr = (lr - startToMax_LR) * pow(currentEpoch / float(upToMax_Epoch), 2) + startToMax_LR
        # 最后几个周期保持minLR
        elif currentEpoch >= total_Epoch - keepWithMin_Epoch:
            lr = minLr
        # 中间周期从maxLR下降到minLR
        else:
            lr = minLr + 0.5 * (lr - minLr) * (1.0 + math.cos(math.pi * (currentEpoch - upToMax_Epoch)
                                                              / (total_Epoch - upToMax_Epoch - keepWithMin_Epoch)))
        return lr

    # 步降学习率
    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n = iters // step_size
        out_lr = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        upToMaxEpoch = min(max(config.cos.start * totalEpoch, 1), config.cos.startLimit)
        startToMaxLR = max(config.cos.lrRate * maxLR, config.cos.lrLimit)
        keepWithMinEpoch = min(max(config.cos.end * totalEpoch, 1), config.cos.endLimit)
        func = partial(yolox_warmcos, maxLR, minLR, totalEpoch, upToMaxEpoch, startToMaxLR, keepWithMinEpoch)
    else:
        decay_rate = (minLR / maxLR) ** (1 / (config.step.stepNum - 1))
        step_size = totalEpoch / config.step.stepNum
        func = partial(step_lr, maxLR, decay_rate, step_size)
    return func


# 设置优化器学习率
def set_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

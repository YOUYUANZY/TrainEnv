import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim

from nets.arcface import Arcface
from nets.facenet import Facenet
from torch.utils.data import DataLoader

from utils.lossRecord import LossHistory
from utils.dataloader import FacenetDataset, LFWDataset, face_dataset_collate, arcFaceDataset, arc_dataset_collate
from utils.training import get_Lr_Fun, set_lr, triplet_loss
from utils.utils import get_num_classes, seed_everything
from utils.epochTrain import epochTrain


def train(config, lfw):
    seed_everything(11)
    # 获取训练设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 一个标记(既负责提示信息又代表设备序号)
    flag = 0
    # 获取标签数量
    num_classes = get_num_classes(config.dataPath)
    # 加载模型
    if config.model == 'facenet':
        model = Facenet(backbone=config.backbone, attention=config.attention, num_classes=num_classes,
                        pretrained=config.preTrained)
    elif config.model == 'arcface':
        model = Arcface(num_classes=num_classes, backbone=config.backbone, pretrained=config.preTrained)
    else:
        raise ValueError('model unsupported')
    # 加载权重
    if config.weightPath != '':
        if flag == 0:
            print('Load weights {}.'.format(config.weightPath))
        model_dict = model.state_dict()
        pretrained_dict = torch.load(config.weightPath, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        # 没加载上的权重
        if flag == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")

    # 锁定权重
    if config.onlyAttention:
        for name, value in model.named_parameters():
            if not name.startswith("attention"):
                value.requires_grad = False

    # 获取损失函数
    loss = triplet_loss()
    # 记录Loss
    if flag == 0:
        loss_history = LossHistory(config.saveDir, model, input_shape=config.inputSize)
    else:
        loss_history = None
    # torch 1.2不支持amp，建议使用torch 1.7.1及以上正确使用fp16
    if config.fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None
    # 启用模型训练
    model_train = model.train()
    # 是否gpu加速
    if config.cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()
    # LFW评估加载
    LFW_loader = torch.utils.data.DataLoader(
        LFWDataset(dir=lfw.dirPath, pairs_path=lfw.PairPath, image_size=lfw.inputSize), batch_size=32,
        shuffle=False) if config.lfwEval else None
    # 划分训练集和验证集
    with open(config.dataPath, "r") as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * config.valRate)
    num_train = len(lines) - num_val
    print(
        "num_classes: " + str(num_classes) + "\n" + "num_train: " + str(num_train) + "\n" + "num_val: " + str(num_val))
    # 自适应调整学习率
    if config.batchSize % 3 != 0:
        raise ValueError("Batch_size must be the multiple of 3.")
    maxLR = min(max(config.batchSize / config.nbs * config.maxLR, config.minLimitLR), config.maxLimitLR)
    minLR = min(max(config.batchSize / config.nbs * config.minLR, config.minLimitLR * 1e-2),
                config.maxLimitLR * 1e-2)
    # 获得优化器
    optimizer = {
        'adam': optim.Adam(model.parameters(), maxLR, betas=(config.momentum, 0.999),
                           weight_decay=config.weightDecay),
        'sgd': optim.SGD(model.parameters(), maxLR, momentum=config.momentum, nesterov=True,
                         weight_decay=config.weightDecay)
    }[config.optimizer]
    # 获得学习率下降的公式
    lr_func = get_Lr_Fun(config.LrDecayType, maxLR, minLR, config.endEpoch, config.LRscheduler)
    # 判断每个轮次的批次数
    epoch_step = num_train // config.batchSize
    epoch_step_val = num_val // config.batchSize
    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")
    # 构建数据集加载器
    if config.model == 'facenet':
        train_dataset = FacenetDataset(config.inputSize, lines[:num_train], num_classes, random=True)
        val_dataset = FacenetDataset(config.inputSize, lines[num_train:], num_classes, random=False)
    elif config.model == 'arcface':
        train_dataset = arcFaceDataset(config.inputSize, lines[:num_train], random=True)
        val_dataset = arcFaceDataset(config.inputSize, lines[num_train:], random=False)
    else:
        raise ValueError('dataset unsupported')
    # 获得训练和验证数据集
    train_sampler = None
    val_sampler = None
    shuffle = True
    batchSize = config.batchSize // 3 if config.model == 'facenet' else config.batchSize
    collate_fn = face_dataset_collate if config.model == 'facenet' else arc_dataset_collate
    gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batchSize,
                     num_workers=config.numWorkers,
                     pin_memory=True,
                     drop_last=True, collate_fn=collate_fn, sampler=train_sampler)
    gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batchSize,
                         num_workers=config.numWorkers,
                         pin_memory=True,
                         drop_last=True, collate_fn=collate_fn, sampler=val_sampler)
    # 开始训练

    for epoch in range(config.startEpoch, config.endEpoch):
        set_lr(optimizer, lr_func, epoch)
        epochTrain(config.model, model_train, model, loss_history, loss, optimizer, epoch, epoch_step, epoch_step_val,
                   gen,
                   gen_val, config.endEpoch, config.cuda, LFW_loader, config.batchSize // 3, config.lfwEval,
                   config.fp16, scaler, config.savePeriod, config.saveDir, flag)
    # 训练结束
    if flag == 0:
        loss_history.writer.close()

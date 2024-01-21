import numpy as np
import torch
from scipy import interpolate
from sklearn.model_selection import KFold
from tqdm import tqdm
"""
tp:真阳，即判断为真，且实际为真
fp:假阳，即判断为真，但实际为假
tn:真阴，即判断为假，且实际为假
tf:假阴，即判断为假，但实际为真
tpr:真阳率/召回率,tp/(tp+fn),判断正确的正样本占全体正样本的比率
fpr:假阳率/误识率,fp/(fp+tn),判断错误的负样本占全体负样本的比率
"""


# 距离，标签，交叉验证数量
def evaluate(distances, labels, foldsNum=10):
    thresholds = np.arange(0, 4, 0.01)
    tpr, fpr, accuracy, best_thresholds = calculate_roc(thresholds, distances,
                                                        labels, foldsNum=foldsNum)
    thresholds = np.arange(0, 4, 0.01)
    val, val_std, far = calculate_val(thresholds, distances,
                                      labels, 1e-3, foldsNum=foldsNum)
    # 真阳率、假阳率、正确率、区分度、区分度标准差、误识率、最佳阈值
    return tpr, fpr, accuracy, val, val_std, far, best_thresholds


# 阈值，距离，标签，交叉验证数量
def calculate_roc(thresholds, distances, labels, foldsNum=10):
    # 有效数据对数
    pairsNum = min(len(labels), len(distances))
    # 测试阈值数
    thresholdsNum = len(thresholds)
    # 每次交叉验证下各阈值下的真阳率
    tprs = np.zeros((foldsNum, thresholdsNum))
    # 每次交叉验证下各阈值下的假阳率
    fprs = np.zeros((foldsNum, thresholdsNum))
    # 每次交叉验证下的正确率
    accuracy = np.zeros((foldsNum))
    # 每次交叉验证下的最佳阈值
    best_thresholds = np.zeros((foldsNum))
    # 数据对标号
    indices = np.arange(pairsNum)
    # 交叉验证
    k_fold = KFold(n_splits=foldsNum, shuffle=False)
    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # 各阈值下训练集的正确率
        acc_train = np.zeros((thresholdsNum))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, distances[train_set], labels[train_set])
        # 最佳阈值
        best_thresholds[fold_idx] = acc_train[np.argmax(acc_train)]
        # 验证集测试真阳率、假阳率
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(threshold,
                                                                                                 distances[test_set],
                                                                                                 labels[test_set])
        # 获取最佳阈值在验证集的正确率
        _, _, accuracy[fold_idx] = calculate_accuracy(best_thresholds[fold_idx], distances[test_set],
                                                      labels[test_set])
    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy, np.mean(best_thresholds)


# 计算真阳率、假阳率、正确率
def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


def calculate_val(thresholds, distances, labels, far_target=1e-3, foldsNum=10):
    pairsNum = min(len(labels), len(distances))
    thresholdsNum = len(thresholds)
    val = np.zeros(foldsNum)
    far = np.zeros(foldsNum)
    indices = np.arange(pairsNum)
    k_fold = KFold(n_splits=foldsNum, shuffle=False)
    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        far_train = np.zeros(thresholdsNum)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_vf(threshold, distances[train_set], labels[train_set])
        if np.max(far_train) >= far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0
        val[fold_idx], far[fold_idx] = calculate_vf(threshold, distances[test_set], labels[test_set])

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean


# 和上面是一样的，计算真阳率、假阳率，变量名和计算方法不一样而已
def calculate_vf(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    if n_diff == 0:
        n_diff = 1
    if n_same == 0:
        return 0, 0
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far


def startEval(test_loader, model, png_save_path, log_interval, batch_size, cuda):
    labels, distances = [], []
    pbar = tqdm(enumerate(test_loader))
    for batch_idx, (data_a, data_p, label) in pbar:
        with torch.no_grad():
            # 加载数据，设置成cuda
            data_a, data_p = data_a.type(torch.FloatTensor), data_p.type(torch.FloatTensor)
            if cuda:
                data_a, data_p = data_a.cuda(), data_p.cuda()
            # 传入模型预测，获得预测结果
            out_a, out_p = model(data_a), model(data_p)
            # 获得预测结果的距离
            dists = torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))
        # 将结果添加进列表中
        distances.append(dists.data.cpu().numpy())
        labels.append(label.data.cpu().numpy())
        # 打印进度条
        if batch_idx % log_interval == 0:
            pbar.set_description('Test Epoch: [{}/{} ({:.0f}%)]'.format(
                batch_idx * batch_size, len(test_loader.dataset),
                100. * batch_idx / len(test_loader)))
    # 转换成numpy
    labels = np.array([sublabel for label in labels for sublabel in label])
    distances = np.array([subdist for dist in distances for subdist in dist])

    tpr, fpr, accuracy, val, val_std, far, best_thresholds = evaluate(distances, labels)
    print('Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
    print('Best_thresholds: %2.5f' % best_thresholds)
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
    # 画roc图
    plot_roc(fpr, tpr, figure_name=png_save_path)


# 画roc图
def plot_roc(fpr, tpr, figure_name="roc.png"):
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc
    roc_auc = auc(fpr, tpr)
    fig = plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    fig.savefig(figure_name, dpi=fig.dpi)

import argparse

from matplotlib import pyplot as plt

import torch
import torchvision.transforms as transforms
from sklearn import metrics
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

import numpy as np


import torchvision.transforms as transforms

from torch.utils.data import DataLoader


# 需要改为Vgg
from models.resnet50_2 import CustomResNet50 as model
# from models.ConvNetT import CustomConvNeXtT as model
from models.vgg import *
from conf import settings
from utils import get_network, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights
from dataset import CESM

# 指定保存数据的路径和文件名
file_path = r'C:\Users\Administrator\Desktop\临时\testdata.csv'

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument('-net', type=str, required=True, help='net type')
    # parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    # parser.add_argument('-weights2', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('-gpu', action='store_true', default=True, help='use gpu or not')
    parser.add_argument('-b', type=int, default=1, help='batch size for dataloader')
    args = parser.parse_args()

    # 检查是否有可用的 CUDA 设备，如果没有，则使用 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        print("Device GPUid:{} is available".format(torch.cuda.current_device()))
    else:
        print("CUDA is not available, using CPU for training.")



    CESMdata2 = CESM(base_dir=r"C:\Users\Administrator\Desktop\临时\valid",
                     transform=transforms.Compose([
                         transforms.ToTensor(),
                     ]))

    CESM_10_test_l = DataLoader(CESMdata2, batch_size=1, shuffle=False, drop_last=False,
                                pin_memory=torch.cuda.is_available())
    for i, x in enumerate(CESM_10_test_l):

        csv = x['csv'].to(device).squeeze(2)
        csv_shape = csv.shape[1]

        break  # 终止循环，只获取第一个批次的数据

    net = model( in_channels=3,num_classes=2,chunk=2,csv_shape = csv_shape,CSV=False)

    net.to(device)

    path1 = r'E:\pycharmproject\Breast\checkpoint\Resnet50\Monday_24_June_2024_00h_09m_37s\Resnet50-24-best.pth'

    net.load_state_dict(torch.load(path1))

    # print(net)
    net.eval()

    test_loss = 0.0  # cost function error
    correct = 0.0
    correct2 = 0.0
    class_correct = list(0. for i in range(2))
    class_total = list(0. for i in range(2))
    correct_1 = 0.0
    correct_5 = 0.0
    total = 0
    tp_l = 0.0
    tn_l = 0.0
    fp_l = 0.0
    fn_l = 0.0
    tp_r = 0.0
    tn_r = 0.0
    fp_r = 0.0
    fn_r = 0.0
    pro = []
    label = []
    with torch.no_grad():
        for i, x in enumerate(CESM_10_test_l):

            data_DCE = x['data_DCE'].to(device)
            data_T2 = x['data_T2'].to(device)
            csv = x['csv'].to(device).squeeze(2)

            labels = x['label']
            labels = torch.LongTensor(labels.numpy())
            label.append(labels.cpu().item())

            labels = labels.to(device)

            outputs= net(data_DCE, data_T2,csv)



            sot1 = torch.softmax(outputs, dim=1)

            pro1 = torch.index_select(sot1.cpu(), dim=1, index=torch.tensor(1))
            pro.append(pro1.cpu().item())

            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum()

            tp_l += torch.sum(np.logical_and(preds.cpu() == 1, labels.cpu() == 1))
            tn_l += torch.sum(np.logical_and(preds.cpu() == 0, labels.cpu() == 0))
            fp_l += torch.sum(np.logical_and(preds.cpu() == 1, labels.cpu() == 0))
            fn_l += torch.sum(np.logical_and(preds.cpu() == 0, labels.cpu() == 1))

        recall_l = tp_l / (tp_l + fn_l)

        # 计算precision
        precision_l = tp_l / (tp_l + fp_l)
        npv_l = tn_l / (tn_l + fn_l)
        ppv_l = tp_l / (tp_l + fp_l)

        # 计算f1-score
        f1_score_l = 2 * recall_l * precision_l / (recall_l + precision_l)
        # 计算falseAlarmRate
        spe_l = tn_l / (tn_l + fp_l)
        false_alarm_rate_l = fp_l / (tn_l + fp_l)
        auc_score = roc_auc_score(label, pro)

        print('auc_score', auc_score)

        print(
            ' Recall_L: {:.4f}\tPrecision_L: {:.4f}\tNpv_L: {:.4f} \tPpv_L:{:.4f}\tSpecificity_L:{:.4f}\tF1_score_L: {:.4f}\tFalse_alarm_rate_L: {:.4f}'.format(
                (recall_l), (precision_l), (npv_l), (ppv_l), (spe_l), (f1_score_l),
                (false_alarm_rate_l)
            ))
        accuracy = correct.cpu() / (len(CESMdata2))
        print('accuracy', accuracy)
        # 计算accuracy
        print('Test set: Accuracy: {:.4f}'.format(
            correct.float() / (len(CESMdata2))))

        precision1, recall1, thresholds1 = precision_recall_curve(label, pro)
        auc_precision_recall = auc(recall1, precision1)
        print(auc_precision_recall)
        fpr, tpr, threshold = metrics.roc_curve(label, pro)
        roc_auc = metrics.auc(fpr, tpr)

        print('label:', label)
        print('pro:', [f"{num:.4f}" for num in pro])

        plt.figure(figsize=(6, 6))
        lw = 2
        plt.title('ROC of Test')
        plt.plot(fpr, tpr, 'b', lw=lw, label='Test AUC = %0.3f' % roc_auc)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.legend(loc='lower right')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()

        plt.figure(figsize=(6, 6))
        plt.title('Precision-Recall Curve of Test')
        plt.plot(precision1, recall1, 'b', lw=lw, label='Test PRAUC = %0.3f' % auc_precision_recall)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('Precision')
        plt.xlabel('Recall')
        plt.show()

    import csv

    # 将数据整理成二维列表的形式，每一行作为一个子列表
    data = [
        ['accuracy', accuracy],
        ['auc_score', auc_score],
        ['Recall_L', recall_l],
        ['Precision_L', precision_l],
        ['Npv_L', npv_l.item()],
        ['ppv_l', ppv_l],
        ['Specificity_L', spe_l.item()],
        ['F1_score_L', f1_score_l.item()],
        ['False_alarm_rate_L', false_alarm_rate_l.item()],
        ['Test set: Accuracy', (correct.float() / (len(CESMdata2))).item()],
        ['auc_precision_recall', auc_precision_recall],
        ['label'] + label,  # 这里将 label 数据作为一行添加到 CSV 中
        ['pro'] + pro  # 这里将 pro 数据作为一行添加到 CSV 中
    ]

    # 指定保存路径和文件名
    file_path = file_path

    # 写入 CSV 文件
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

    print(f"Data has been written to {file_path}")

# 查看保存在csv中的数据
# import pandas as pd
#
# # 指定文件路径
# file_path = '/kaggle/working/testdata.csv'
# import csv
#
#
#
# # 读取 CSV 文件
# with open(file_path, mode='r') as file:
#     reader = csv.reader(file)
#     for row in reader:
#         print(row)


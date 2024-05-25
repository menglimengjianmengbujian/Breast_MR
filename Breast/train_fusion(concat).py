import os
import sys
import argparse
import time
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.nn import CosineEmbeddingLoss

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models.resnet50_2 import CustomResNet50 as model
from conf import settings
from utils import get_network,  WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights
from dataset import CESM

# 指定保存数据的文件夹路径
folder_path = r'C:\Users\Administrator\Desktop\Breast\MR_loss'

my_train_loss = []
my_eval_loss = []
my_train_correct= []
my_eval_correct = []
def train(epoch):
    start = time.time()
    train_loss = 0.0 # cost function error
    correct = 0.0
    net.train()
    p=0
    for i, x in enumerate(CESM_10_train_l):
        data_DCE = x['data_DCE'].to(device)
        data_T2 = x['data_T2'].to(device)
        csv = x['csv'].to(device).squeeze(2)

        labels=x['label']
        labels = torch.LongTensor(labels.numpy())



        labels=labels.to(device)
        optimizer.zero_grad()

        outputs= net(data_DCE,data_T2,csv)




        loss = loss_function(outputs, labels)
        print('loss:{}'.format(
            loss.item()
        )

        )



        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, preds = outputs.max(1)
        
        correct += preds.eq(labels).sum()
        n_iter = (epoch - 1) * len(CESM_10_train_l) + i + 1



        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            # correct.float() / len(CESMdata),
            epoch=epoch,
            trained_samples=i * args.b + len(data_DCE),
            total_samples=len(CESMdata)
        ))


        if epoch <= args.warm:
            warmup_scheduler.step()

    train_loss_ = train_loss / (len(CESMdata))
    train_correct = correct / (len(CESMdata))
    my_train_loss.append(train_loss_)
    my_train_correct.append(train_correct.cpu())

    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))
    print('Train Average loss: {:.4f}\tTrain Accuarcy: {:0.6f}'.format(
        train_loss / len(CESMdata),
        correct.float() / (len(CESMdata))
        ))


@torch.no_grad()
def eval_training(epoch=0, tb=True):

    start = time.time()
    net.eval()
    test_loss = 0.0  # cost function error
    correct = 0.0
    correct2=0.0

    for i, x in enumerate(CESM_10_test_l):

        data_DCE= x['data_DCE'].to(device)
        data_T2= x['data_T2'].to(device)
        csv = x['csv'].to(device).squeeze(2)

        labels=x['label']
        labels = torch.LongTensor(labels.numpy())


        labels=labels.to(device)

        optimizer.zero_grad()

        outputs = net(data_DCE,data_T2,csv)
        loss = loss_function(outputs, labels)


        n_iter = (epoch - 1) * len(CESM_10_test_l) + i + 1
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    test_loss_ = test_loss / (len(CESMdata2))
    my_eval_loss.append(test_loss_)
    eval_correct = correct / (len(CESMdata2))
    my_eval_correct.append(eval_correct.cpu())


    finish = time.time()


    #print('GPU INFO.....')
    #print(torch.cuda.memory_summary(), end='')
    print('Evaluating Network.....')
    print('Eval set: Epoch: {},Eval Average loss: {:.4f},Eval Accuracy: {:.4f} Time consumed:{:.2f}s'.format(
        epoch,
        test_loss / len(CESMdata2),
        correct.float() / (len(CESMdata2)),
        finish - start
    ))
    print()


    return correct.float() / (len(CESMdata2))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    train_root = r"C:\Users\Administrator\Desktop\Breast\data\train"
    val_root = r"C:\Users\Administrator\Desktop\Breast\data\valid"
    parser.add_argument('-train_data_path',type=str,default=train_root)
    parser.add_argument('-val_data_path', type=str, default=val_root)
    parser.add_argument('-net', type=str, default='Resnet50', help='net type')
    parser.add_argument('-device', action='store_true', default='cuda', help='device id （i.e.0or 0,1 or cpu)')
    parser.add_argument('-b', type=int, default=30, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=0, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    args = parser.parse_args()

    # 检查是否有可用的 CUDA 设备，如果没有，则使用 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        print("Device GPUid:{} is available".format(torch.cuda.current_device()))
    else:
        print("CUDA is not available, using CPU for training.")


    CESMdata = CESM(base_dir=args.train_data_path,transform=transforms.Compose([
                       # Random表示有可能做，所以也可能不做
                       transforms.RandomHorizontalFlip(p=0.5),# 水平翻转
                       transforms.RandomVerticalFlip(p=0.5), # 上下翻转
                       transforms.RandomRotation(10), # 随机旋转-10°~10°
                       # transforms.RandomRotation([90, 180]), # 随机在90°、180°中选一个度数来旋转，如果想有一定概率不旋转，可以加一个0进去
                       # transforms.ColorJitter(brightness=0.3, contrast=0.3, hue=0.3)
                       # transforms.Normalize(0, 1)
                       transforms.ToTensor(),

                   ]))
    CESM_10_train_l = DataLoader(CESMdata, batch_size=args.b, shuffle=True, drop_last=False,
                                 pin_memory=torch.cuda.is_available())

    CESMdata2 = CESM(base_dir=args.val_data_path,transform=transforms.Compose([
                       transforms.ToTensor(),
                   ]))
    CESM_10_test_l = DataLoader(CESMdata2, batch_size=args.b, shuffle=False, drop_last=False,
                                 pin_memory=torch.cuda.is_available())

    for i, x in enumerate(CESM_10_train_l):

        csv = x['csv'].squeeze(2)
        csv_shape = csv.shape[1]

        break  # 终止循环，只获取第一个批次的数据

    net = model( in_channels=3,num_classes=2,chunk=2,csv_shape = csv_shape,CSV=False)


    net = net.to(device)

    loss_function = nn.CrossEntropyLoss()
    loss_function.to(device)

    optimizer = optim.SGD([{"params":net.parameters()}], lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.5) #learning rate decay
    iter_per_epoch = len(CESM_10_train_l)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    # # 实例化SummaryWriter对象
    # tb_writer = SummaryWriter(log_dir=r"Breast/tensorboard")
    # if os.path.exists(r"./weights")is False:
    #     os.makedirs(r". /weights")





    if args.resume:
        recent_folder = most_recent_folder(os.path.join(settings.CHECKPOINT_PATH, args.net), fmt=settings.DATE_FORMAT)
        if not recent_folder:
            raise Exception('no recent folder were found')

        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder)

    else:
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)



    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    best_acc2=0.0
    if args.resume:
        best_weights = best_acc_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if best_weights:
            weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, best_weights)
            print('found best acc weights file:{}'.format(weights_path))
            print('load best training file to test acc...')
            net.load_state_dict(torch.load(weights_path))
            best_acc = eval_training(tb=False)
            print('best acc is {:0.2f}'.format(best_acc))

        recent_weights_file = most_recent_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if not recent_weights_file:
            raise Exception('no recent weights file were found')
        weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, recent_weights_file)
        print('loading weights file {} to resume training.....'.format(weights_path))
        net.load_state_dict(torch.load(weights_path))

        resume_epoch = last_epoch(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))


    for epoch in range(1, settings.EPOCH + 1):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        if args.resume:
            if epoch <= resume_epoch:
                continue


        train(epoch)


        acc= eval_training(epoch)


        np.savez(os.path.join(folder_path, 'loss.npz'),
                 my_train_loss = my_train_loss, my_eval_loss = my_eval_loss,
                 my_train_correct = my_train_correct,my_eval_correct = my_eval_correct )

        #start to save best performance model after learning rate decay to 0.01
        if epoch <= settings.MILESTONES[3] and best_acc < acc:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='best')

            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)

            best_acc = acc

            continue

    # 指定保存数据的文件夹路径
    folder_path = folder_path

    # 保存 label 和 pro 到一个文件
    np.savez(os.path.join(folder_path, 'loss1.npz'),
             my_train_loss = my_train_loss, my_eval_loss = my_eval_loss,
             my_train_correct = my_train_correct,my_eval_correct = my_eval_correct )
    print('my_train_loss:',my_train_loss)
    print('my_eval_loss:',my_eval_loss)




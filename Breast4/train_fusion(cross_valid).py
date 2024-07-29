import os
import sys
import argparse
import time
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.nn import CosineEmbeddingLoss

from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from models.Vit import CustomViT as model
# from models.resnet50_2 import CustomResNet50 as model
# from models.ConvNetT import CustomConvNeXtT as model
from conf import settings
from utils import get_network,  WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights
from dataset import CESM

# 指定保存数据的文件夹路径
folder_path = r"C:\Users\Administrator\Desktop\Breast\临时"

my_train_loss = []
my_eval_loss = []
my_train_correct= []
my_eval_correct = []
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold


my_train_loss = []
my_eval_loss = []
my_train_correct= []
my_eval_correct = []
class Trainer:
    def __init__(self, net, CESMdata, loss_function, optimizer, device, batch_size):
        self.net = net
        self.CESMdata = CESMdata
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.device = device
        self.batch_size = batch_size
        self.my_train_loss = []
        self.my_train_correct = []
        self.my_val_loss = []
        self.my_val_correct = []

    def train(self, epoch):
        start = time.time()
        self.my_train_loss = []
        self.my_train_correct = []
        self.my_val_loss = []
        self.my_val_correct = []

        self.net.train()
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(self.CESMdata)))):


            print(f'\nFold {fold + 1}/{kf.n_splits}')

            # 创建训练集和验证集
            train_subset = Subset(self.CESMdata, train_idx)
            val_subset = Subset(self.CESMdata, val_idx)

            train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=self.batch_size, shuffle=False)

            fold_train_loss = 0.0
            fold_correct = 0

            for i, x in enumerate(train_loader):
                data_DCE = x['data_DCE'].to(self.device)
                data_T2 = x['data_T2'].to(self.device)
                data_T1 = x['data_T1'].to(self.device)
                data_DWI = x['data_DWI'].to(self.device)
                csv = x['csv'].to(self.device).squeeze(2)

                labels = x['label']
                labels = torch.LongTensor(labels.numpy()).to(self.device)

                self.optimizer.zero_grad()

                outputs = self.net(data_DCE, data_T2, data_T1, data_DWI, csv)

                loss = self.loss_function(outputs, labels)
                loss.backward()
                self.optimizer.step()

                fold_train_loss += loss.item()
                _, preds = outputs.max(1)
                fold_correct += preds.eq(labels).sum().item()

                n_iter = (epoch - 1) * len(train_loader) + i + 1

                print(f'Training Epoch: {epoch} [{i * len(data_DCE)}/{len(train_loader.dataset)}]\t'
                      f'Loss: {loss.item():.4f}\tLR: {self.optimizer.param_groups[0]["lr"]:.6f}')

            train_loss = fold_train_loss / len(train_loader.dataset)
            train_correct = fold_correct / len(train_loader.dataset)
            self.my_train_loss.append(train_loss)
            self.my_train_correct.append(train_correct)



            # 验证阶段
            self.net.eval()
            val_loss = 0.0
            val_correct = 0
            with torch.no_grad():
                for x in val_loader:
                    data_DCE = x['data_DCE'].to(self.device)
                    data_T2 = x['data_T2'].to(self.device)
                    data_T1 = x['data_T1'].to(self.device)
                    data_DWI = x['data_DWI'].to(self.device)
                    csv = x['csv'].to(self.device).squeeze(2)

                    labels = x['label']
                    labels = torch.LongTensor(labels.numpy()).to(self.device)

                    outputs = self.net(data_DCE, data_T2, data_T1, data_DWI, csv)

                    loss = self.loss_function(outputs, labels)
                    val_loss += loss.item()
                    _, preds = outputs.max(1)
                    val_correct += preds.eq(labels).sum().item()

            val_loss /= len(val_loader.dataset)
            val_correct /= len(val_loader.dataset)
            self.my_val_loss.append(val_loss)
            self.my_val_correct.append(val_correct)

            print(f'Fold {fold + 1} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_correct:.4f}, '
                  f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_correct:.4f}')

        finish = time.time()
        print(f'Epoch {epoch} training time consumed: {finish - start:.2f}s')

        my_train_loss.append(np.mean(self.my_train_loss))
        my_eval_loss.append(np.mean(self.my_val_loss))
        my_train_correct.append(np.mean(self.my_train_correct))
        my_eval_correct.append(np.mean(self.my_val_correct))

        print(f'Average Validation Accuracy: {np.mean(self.my_val_correct):.4f}')

    @property
    def average_val_correct(self):
        return np.mean(self.my_val_correct)






if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    train_root = r"C:\Users\Administrator\Desktop\Breast\H5\train"
    parser.add_argument('-train_data_path',type=str,default=train_root)
    parser.add_argument('-net', type=str, default='Vit', help='net type')
    parser.add_argument('-device', action='store_true', default='cuda', help='device id （i.e.0or 0,1 or cpu)')
    parser.add_argument('-b', type=int, default=10, help='batcdh size for dataloader')
    parser.add_argument('-warm', type=int, default=0, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.0001, help='initial learning rate')
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



    for i, x in enumerate(CESM_10_train_l):

        csv = x['csv'].squeeze(2)
        csv_shape = csv.shape[1]

        break  # 终止循环，只获取第一个批次的数据

    net = model( in_channels=3,num_classes=2, Seq=5,csv_shape = csv_shape,CSV=True)


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

    trainer = Trainer(net, CESMdata, loss_function, optimizer, device, batch_size=args.b)


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
            best_acc = trainer.average_val_correct
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


        trainer.train(epoch)

        acc= trainer.average_val_correct

        np.savez(os.path.join(folder_path, 'loss.npz'),
                 my_train_loss=my_train_loss, my_eval_loss=my_eval_loss,
                 my_train_correct=my_train_correct, my_eval_correct=my_eval_correct)


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




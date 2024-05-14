import os
import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
from utils import GradCAM, show_cam_on_image, center_crop_img
import sys
sys.path.append('/CR-DLcode/CR_fussion/MF/models') # 即下图中标蓝的文件的路径
# sys.path.append('D:\AsciRes\AgraduationProject/test\EasyMocap2')t\EasyMocap2')
from resnet50 import *
import torch.nn as nn
from dataset import CESM
from torch.utils.data import DataLoader
def main():

    net = CustomResNet50( in_channels=1,num_classes=2,chunk=3)
    path1 = r'E:\pycharmproject\Breast\checkpoint\Resnet50\Thursday_21_March_2024_01h_58m_09s\Resnet50-98-best.pth'
    net.load_state_dict(torch.load(path1))

    model = net
    target_layers = [net.model.layer4]
    CESMdata2 = CESM(base_dir=r'C:\Users\Administrator\Desktop\custom\Breast\Test',
                     transform=transforms.Compose([
                         transforms.ToTensor(),
                     ]))

    CESM_10_test_l = DataLoader(CESMdata2, batch_size=1, shuffle=False, drop_last=True,
                                pin_memory=torch.cuda.is_available())




    for i, x in enumerate(CESM_10_test_l):

        low_energy = x['LOW_ENERGY']
        high_energy = x['HIGH_ENERGY']
        enhance = x['ENHANCE']
        inputs = torch.cat((low_energy, high_energy, enhance), 1)
        input_tensor=inputs
        data=low_energy.squeeze(0).numpy()

        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
        target_category = x['label']  # tabby, tabby cat
        grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
        grayscale_cam = grayscale_cam[0, :]


        visualization = show_cam_on_image(data / 255.,
                                          grayscale_cam,
                                          use_rgb=True)
        # plt.imshow(visualization)
        # plt.show()

        cv2.imshow('Image', visualization)
        cv2.waitKey(0)
        cv2.destroyAllWindows()




if __name__ == '__main__':
    main()

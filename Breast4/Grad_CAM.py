import os
import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
from utils import GradCAM, show_cam_on_image, center_crop_img
# from models.resnet50 import CustomResNet50 as model1
# from models.resnet50_2 import CustomResNet50 as model1
from models.Vit import CustomViT as model1
# from models.ConvNetT import CustomConvNeXtT as model
import torch.nn as nn
from dataset import CESM
from torch.utils.data import DataLoader
def main():

    net = model1(in_channels=3,num_classes=2, Seq=5,csv_shape = 428,CSV=True)
    path1 = r'E:\pycharmproject\Breast3\checkpoint\Vit\Monday_29_July_2024_16h_28m_43s\Vit-86-best.pth'
    net.load_state_dict(torch.load(path1))

    model = net
    target_layers = [net.model.layer4[-1]]
    #target_layers = [net.model2.layer4[-1]]
    CESMdata2 = CESM(base_dir=r'C:\Users\Administrator\Desktop\Breast\热力图数据集',
                     transform=transforms.Compose([
                         transforms.ToTensor(),
                     ]))

    CESM_10_test_l = DataLoader(CESMdata2, batch_size=1, shuffle=False, drop_last=True,
                                pin_memory=torch.cuda.is_available())




    for i, x in enumerate(CESM_10_test_l):

        data_DCE = x['data_DCE']
        data_T2 = x['data_T2']
        data_T1 = x['data_T1']
        data_DWI = x['data_DWI']
        csv = x['csv'].squeeze(2)

        channel = data_DCE.squeeze(0).numpy()[1, :, :]
        # import matplotlib.pyplot as plt
        # # 显示灰度图
        # plt.imshow(channel, cmap='gray')
        # plt.title('Channel 1 as Grayscale')
        # plt.axis('off')
        # plt.show()


        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
        target_category = x['label']  # tabby, tabby cat
        grayscale_cam = cam(data_DCE,data_T2,data_T1,data_DWI,csv, target_category=target_category)
        grayscale_cam = grayscale_cam[0, :]

        img = data_DCE.squeeze(0).numpy()
        img = img - np.min(img)
        img = img / np.max(img +1e-7)
        visualization = show_cam_on_image(img[1, :, :] ,
                                          grayscale_cam,
                                          use_rgb=True)

        cv2.imshow('Image', visualization)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

#保存heatmap为一张png图片

        heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        plt.imshow(heatmap)
        plt.show()
        #设置保存路径
        save_path = r"C:\Users\Administrator\Desktop\Breast\heatmap.png"

        # 使用Matplotlib保存图像
        plt.imsave(save_path,heatmap)

        # 提示保存成功
        print(f"图像已保存到 {save_path}")
        cv2.imshow('Image', visualization)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

#
#
#
if __name__ == '__main__':
    """
    先将一张h5的数据保存为一张png的图片，将梯度的heatmap保存为png图片，
    然后将两张图片合成一张图片，保存为一张png图片
    """
#     #将h5数据保存为一张png的图片
#     import h5py
#     import matplotlib.pyplot as plt
#
#     # 读取h5文件路径
#     file_path = r"C:\Users\Administrator\Desktop\临时\CAM\CHANG_JIN_ZHI_10.h5"
#
#     # 使用h5py库打开文件
#     with h5py.File(file_path, 'r') as f:
#         # 读取数据集
#         low_energy_data = f['data_DCE'][:]
#
#         # 获取要保存的图像数据
#         image_to_save = low_energy_data[1, :, :]  # 选择要保存的图像数据，假设是第二张图像
#
#         # 设置保存路径
#         save_path = r"C:\Users\Administrator\Desktop\临时\图片\low_energy_image.png"
#
#         # 使用Matplotlib保存图像
#         plt.imshow(image_to_save, cmap='gray')  # 显示灰度图像
#         plt.axis('off')  # 不显示坐标轴
#         plt.savefig(save_path, bbox_inches='tight', pad_inches=0)  # 保存图像，并确保边界适应紧密
#
#         # 提示保存成功
#         print(f"图像已保存到 {save_path}")
#
# #将heatmap图片与原始图片融合
#     import matplotlib.pyplot as plt
#     import numpy as np
#     from PIL import Image
#
#     # 加载并显示第一张图片
#     image_path1 = r"C:\Users\Administrator\Desktop\临时\图片\heatmap.png"
#     image1 = Image.open(image_path1)
#     plt.subplot(1, 2, 1)
#     plt.imshow(image1)
#     plt.title('Image 1')
#     plt.axis('off')
#
#     # 加载并显示第二张图片
#     image_path2 = r"C:\Users\Administrator\Desktop\临时\图片\low_energy_image.png"
#     image2 = Image.open(image_path2)
#     plt.subplot(1, 2, 2)
#     plt.imshow(image2)
#     plt.title('Image 2')
#     plt.axis('off')
#     # 将第一张图片裁剪到第二张图片的大小（或者相反，根据需要选择大小）
#     image1 = image1.resize(image2.size)  # 将第一张图片裁剪/缩放到与第二张图片相同的大小
#
#     # 将图片转换为数组，并进行归一化
#     image1_array = np.array(image1) / 255.0
#     image2_array = np.array(image2) / 255.0
#
#     # 执行加法操作
#     result_array = (image1_array * 1 + image2_array) * 255.0
#
#     # 将结果限制在0到255之间
#     result_array = np.clip(result_array, 0, 255)
#
#     # 将结果转换为整数类型
#     result_array = result_array.astype(np.uint8)
#
#     # 显示加法结果
#     plt.figure()
#     plt.imshow(result_array)
#     plt.title('Sum of Images (Normalized)')
#     plt.axis('off')
#     plt.show()
#
#
#
    main()

import glob
from random import random

import pandas as pd
from skimage import exposure
import h5py
import SimpleITK as sitk
from PIL import Image
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import numpy as np
import pydicom
import nibabel as nib
import random

"""
这个脚本需要将良性和恶性的数据路径修改好，然后会保存到h5文件中，修改label为0或1


可以将良性或恶性文件夹下的DCE和T2的数据，CSV的数据还有label数据保存到h5文件中

image_DCE_path文件夹存储着所有DCE序列原始图像的nii文件
image_T2_path文件夹存储着所有T2序列原始图像的nii文件
label_path文件夹存储着所有掩膜图像的nii文件
save_dir文件夹存储着所有处理后的图像h5文件
csv_file 为pyradiomics特征提取后的csv文件路径，注意要删除里面没意义的列
name_labbel = 0 或1 为良性或恶性的标签
一定要删除csv中字符串部分，不然会报错

在354-359行有两种图像保存方法，concat_image函数是将原始图像的边缘抠下来，然后利用插值算法缩放到224×224；
 one_three函数是在mask上找到中心点映射到原始图像上，以中心点为坐标裁剪为224×224
 使用其中一种即可
"""


# # 设置文件夹路径
image_benign_DCE_path = r'C:\Users\Administrator\Desktop\Breast\NIIdata\Low\DCE'
image_benign_T2_path = r'C:\Users\Administrator\Desktop\Breast\NIIdata\Low\T2'
image_benign_T1_path = r'C:\Users\Administrator\Desktop\Breast\NIIdata\Low\T1'
image_benign_DWI_path = r'C:\Users\Administrator\Desktop\Breast\NIIdata\Low\DWI'

label_benign_path = r'C:\Users\Administrator\Desktop\Breast\ROI\Low\DCE'
# CSV文件地址
csv_benign_file = r"C:\Users\Administrator\Desktop\Breast\Low_merged.csv"
label0 = 0


image_malignant_DCE_path = r'C:\Users\Administrator\Desktop\Breast\NIIdata\High\DCE'
image_malignant_T2_path = r'C:\Users\Administrator\Desktop\Breast\NIIdata\High\T2'
image_malignant_T1_path = r'C:\Users\Administrator\Desktop\Breast\NIIdata\High\T1'
image_malignant_DWI_path = r'C:\Users\Administrator\Desktop\Breast\NIIdata\High\DWI'

label_malignant_path = r'C:\Users\Administrator\Desktop\Breast\ROI\High\DCE'
csv_malignant_file = r"C:\Users\Administrator\Desktop\Breast\High_merged.csv"
label1 = 1

save_dir = r"C:\Users\Administrator\Desktop\Breast\临时\数据集"


# 定义一个函数来修剪图像，去除空白部分
def trim_image(image):
    # 转换为numpy数组
    image_array = np.array(image)

    # 找到非零像素的边界
    non_zero_indices = np.nonzero(image_array)
    min_row = np.min(non_zero_indices[0])
    max_row = np.max(non_zero_indices[0])
    min_col = np.min(non_zero_indices[1])
    max_col = np.max(non_zero_indices[1])
    min_depth = np.min(non_zero_indices[2])
    max_depth = np.max(non_zero_indices[2])

    # 裁剪图像
    cropped_image_array = image_array[min_row:max_row + 1, min_col:max_col + 1, min_depth:max_depth + 1]

    return cropped_image_array

# 数据预处理
def preprocess_data(image_array, window_width=1000, window_center=500):
    """
    对图像进行预处理，包括窗宽窗位变换。
    Args:
        image_array: 原始图像nii文件读取后的数据
        window_width:窗宽
        window_center: 窗位

    Returns:对图像进行了窗宽窗位变换和均质化后的图像

    """

    # 提取每张图像的像素值
    #img： 需要增强的图片
    #window_width:窗宽
    #window_center:中心
    minWindow = float(window_center)-0.5*float(window_width)
    new_img = (image_array -minWindow)/float(window_width)
    new_img[new_img<0] = 0
    new_img[new_img>1] = 1
    img = (new_img*255).astype('uint8')
    img_ = []
    for i in range(img.shape[0]):
        # Perform histogram equalization
        img_res = exposure.equalize_hist(img[i])
        img_.append(img_res)
    return np.array(img_)






def concat_image(label_path,image_path , map = False):

    # 读取标签NII文件
    image_label = sitk.ReadImage(label_path)
    # 读取原始NII文件
    image_origin = sitk.ReadImage(image_path)

    # 转换为NumPy数组
    origin_array = sitk.GetArrayFromImage(image_origin)
    label_array = sitk.GetArrayFromImage(image_label)

    # 提取像素值
    origin_array = np.array([origin_array[i] for i in range(origin_array.shape[0])])
    label_array = np.array([label_array[i] for i in range(label_array.shape[0])])
    origin_size = origin_array[0, :, :].shape
    if map == True:
        label_array = F.interpolate(torch.tensor(label_array, dtype=torch.float32).unsqueeze(0), size=origin_size, mode='nearest').squeeze().numpy().astype(np.uint8)

    #对数据进行均质化和窗宽窗位的调整
    #origin_array = preprocess_data(origin_array)

    # 遍历每张图片
    max_nonzero_pixels = 0
    max_nonzero_index = None

    for i in range(label_array.shape[0]):
        # 计算当前图片中非零像素的数量
        nonzero_pixels = np.count_nonzero(label_array[i])

        # 如果当前图片的非零像素数量比之前的最大值大，则更新最大值和对应的索引
        if nonzero_pixels > max_nonzero_pixels:
            max_nonzero_pixels = nonzero_pixels
            max_nonzero_index = i

    roi_array = np.array([label_array[max_nonzero_index] * origin_array[max_nonzero_index - 1],
                          label_array[max_nonzero_index] * origin_array[max_nonzero_index],
                          label_array[max_nonzero_index] * origin_array[max_nonzero_index + 1]])

    finish_array = trim_image(roi_array).astype(np.float64)

    image_tensor = torch.tensor(finish_array, dtype=torch.float32).unsqueeze(0)

    # 目标图像大小
    target_height, target_width = 224, 224

    # 使用双线性插值角对齐对图像进行缩放
    output_bilinear_corners_True = F.interpolate(image_tensor, size=(target_height, target_width), mode='bilinear',
                                                 align_corners=True)
    # 将张量转换回 numpy 数组
    output_bilinear_corners_True_array = output_bilinear_corners_True.squeeze().numpy().astype(np.uint8)

    return output_bilinear_corners_True_array

# 保存影像组特征
def get_features_by_name(name, csv_file):
    """
    根据姓名获取到csv文件中的同名那一行数据，保存这行第二列之后的数据
    Args:
        name: 为出入的姓名
        csv_file: 为pyradiomics特征提取后的csv文件路径，注意要删除里面没意义的列

    Returns:

    """
    # 读取CSV文件
    df = pd.read_csv(csv_file)

    # 查找姓名是否在第一列中
    if name in df['Name'].values:
        # 获取姓名所在行的索引
        index = df.index[df['Name'] == name].tolist()[0]

        # 获取特征（姓名后的列）
        features = df.iloc[index, 1:].tolist()
        return pd.DataFrame(features)
    else:
        print(f"Name '{name}' not found in CSV file.")
        return None


# 找到ROI的中心点坐标，映射到原始图像，向四周扩展为224×224的图像
class ImageProcessor:
    def __init__(self, label_path, image_path, map = False):
        self.label_path = label_path
        self.image_path = image_path
        self.map = map
        self.origin_array = None
        self.label_array = None
        self.max_index = None
        self.max_image = None
        self.max_label = None
        self.roi_center = None

    # 读取标签和原始图像
    def read_images(self):
        image_label = sitk.ReadImage(self.label_path)
        image_origin = sitk.ReadImage(self.image_path)

        # 转换为NumPy数组
        self.origin_array = sitk.GetArrayFromImage(image_origin)
        self.label_array = sitk.GetArrayFromImage(image_label)

        # 提取像素值
        self.origin_array = np.array([self.origin_array[i] for i in range(self.origin_array.shape[0])])
        self.label_array = np.array([self.label_array[i] for i in range(self.label_array.shape[0])])
        self.origin_size = self.origin_array[0, :, :].shape
        if map == True:
            self.label_array = F.interpolate(torch.tensor(self.label_array, dtype=torch.float32).unsqueeze(0), size=self.origin_size, mode='nearest').squeeze().numpy().astype(np.uint8)

    # 找出勾画ROI数值最多的那一层
    def find_max_index(self):
        # 将非零值变为 1
        label_array_binary = np.where(self.label_array != 0, 1, self.label_array)
        # 遍历每张图片
        max_nonzero_pixels = 0
        max_nonzero_index = None

        for i in range(label_array_binary.shape[0]):
            # 计算当前图片中非零像素的数量
            nonzero_pixels = np.count_nonzero(label_array_binary[i])

            # 如果当前图片的非零像素数量比之前的最大值大，则更新最大值和对应的索引
            if nonzero_pixels > max_nonzero_pixels:
                max_nonzero_pixels = nonzero_pixels
                max_nonzero_index = i

        self.max_index = max_nonzero_index

    # 根据找到的中心点坐标，以此为中心，映射到原图上，向四周延伸到224*224
    def crop_roi(self):
        # 找出标签中勾画ROI最多的那张图像
        self.max_image = self.origin_array[self.max_index]
        self.max_label = self.label_array[self.max_index]

        #找到ROI的中心点
        self.roi_center = self.find_roi_center(self.max_label)

        # 根据中心点坐标向四周延伸裁剪出224*224
        cropped_image = self._crop_roi(self.max_image, self.roi_center)

        return cropped_image

    # 根据勾画的ROI，找到其中心坐标
    def find_roi_center(self, max_label):
        # 找到所有 ROI 区域的索引
        indices = np.where(max_label == 1)
        # 计算中心点坐标
        center_x = int(np.round(np.mean(indices[0])))
        center_y = int(np.round(np.mean(indices[1])))
        return (center_x, center_y)

    def _crop_roi(self, max_image, roi_center):
        # 计算裁剪区域的边界坐标
        x_min = max(0, roi_center[0] - 112)
        x_max = min(max_image.shape[0], roi_center[0] + 112)
        y_min = max(0, roi_center[1] - 112)
        y_max = min(max_image.shape[1], roi_center[1] + 112)

        # 创建一个空的224x224大小的数组，并用0填充
        cropped_image = np.zeros((224, 224))

        # 计算裁剪区域在目标数组中的位置
        cropped_x_min = max(0, 112 - (roi_center[0] - x_min))
        cropped_x_max = cropped_x_min + min(x_max - x_min, 224)
        cropped_y_min = max(0, 112 - (roi_center[1] - y_min))
        cropped_y_max = cropped_y_min + min(y_max - y_min, 224)

        # 将ROI区域复制到裁剪图像中
        cropped_image[cropped_x_min:cropped_x_max, cropped_y_min:cropped_y_max] = max_image[x_min:x_max, y_min:y_max]

        return cropped_image

    # 保存裁剪后的图像
    def save_cropped_image(self, output_path):
        cropped_image = self.crop_roi()
        # 保存裁剪后的图像
        sitk.WriteImage(sitk.GetImageFromArray(cropped_image), output_path)

    # 保存三个层面的图像
    def save_three_images(self, output_path):
        images = self._three_image()
        # 保存三个层面的图像
        sitk.WriteImage(sitk.GetImageFromArray(images), output_path)

    # 根据最大值索引裁剪出三个层面的图像
    def _three_image(self):
        max_label0 = self.label_array[self.max_index - 1]
        max_label1 = self.label_array[self.max_index]
        max_label2 = self.label_array[self.max_index + 1]

        max_image0 = self.origin_array[self.max_index - 1]
        max_image1 = self.origin_array[self.max_index]
        max_image2 = self.origin_array[self.max_index + 1]

        # 找到ROI的中心点
        roi_center1 = self.find_roi_center(max_label1)

        # 根据中心点坐标向四周延伸裁剪出224*224
        cropped_image0 = self._crop_roi(max_image0, roi_center1)
        cropped_image1 = self._crop_roi(max_image1, roi_center1)
        cropped_image2 = self._crop_roi(max_image2, roi_center1)

        # 将三个图像数组堆叠成一个新的数组
        stacked_images = np.stack((cropped_image0, cropped_image1, cropped_image2))

        return stacked_images


def one_three(label_path, image_path, map=False):
# label_path = r"C:\Users\Administrator\Desktop\Breast\benign_label\AN_YU_MEI-label.nii"
# image_path = r"C:\Users\Administrator\Desktop\Breast\benign\DCE\AN_YU_MEI.nii"
# output_path = r"C:\Users\Administrator\Desktop\output_image.nii"

    processor = ImageProcessor(label_path, image_path, map = False)
    processor.read_images()
    processor.find_max_index()
    cropped_image = processor.crop_roi()
    three_images = processor._three_image()
    # processor.save_cropped_image(output_path)
    # processor.save_three_images(output_path)
    return three_images




def save_data(image_DCE_path,image_T2_path,image_T1_path,image_DWI_path, label_path,label,csv_file):
    r_num = 0
    image_DCE_files = glob.glob(f'{image_DCE_path}/*.nii')
    # image_T2_files = glob.glob(f'{image_T2_path}/*.nii')
    # label_files = glob.glob(f'{label_path}/*.nii')
    image_DCE_files = [x.split('.')[0] for x in os.listdir(image_DCE_path)]
    label_files = os.listdir(label_path)
    for i ,name_label in enumerate(label_files):
        name = name_label.split('-')[0]
        if name in image_DCE_files:
            label_path1 =  os.path.join(label_path, name_label)
            DCE_path =  os.path.join(image_DCE_path, name)
            T2_path =  os.path.join(image_T2_path, name)
            T1_path = os.path.join(image_T1_path, name)
            DWI_path = os.path.join(image_DWI_path, name)
            # 读取原始图像

            # image_DCE = concat_image(label_path1, DCE_path,map = False)
            # image_T2 = concat_image(label_path1, T2_path ,map =  True)

            image_DCE = one_three(label_path1, DCE_path)
            image_T2 = one_three(label_path1, T2_path , map = True)
            image_T1 = one_three(label_path1, T1_path, map=True)
            image_DWI = one_three(label_path1, DWI_path, map=True)

            name_labbel =  name.split('.')[0]
            csv_data =  get_features_by_name(name_labbel, csv_file)


            R = random.randint(1, 100)
            if R >= 0 and R <= 60:
                os.makedirs(save_dir+"/train", exist_ok=True)
                f = h5py.File(save_dir+"/train" + '/{}_{}.h5'.format(name_labbel, r_num), 'w')
            elif R > 60 and R <= 80:
                os.makedirs(save_dir + "/valid", exist_ok=True)
                f = h5py.File(save_dir+"/valid" + '/{}_{}.h5'.format(name_labbel, r_num), 'w')
            else:
                os.makedirs(save_dir + "/test", exist_ok=True)
                f = h5py.File(save_dir+"/test" + '/{}_{}.h5'.format(name_labbel, r_num), 'w')
            f.create_dataset('data_DCE', data=image_DCE, compression="gzip")
            f.create_dataset('data_T2', data=image_T2, compression="gzip")
            f.create_dataset('data_T1', data=image_T1, compression="gzip")
            f.create_dataset('data_DWI', data=image_DWI, compression="gzip")
            f.create_dataset('csv', data=csv_data)
            f.create_dataset('label', data=label)
            f.close()
            r_num += 1
            print("process {} uid = {} label={}".format(r_num, name_labbel,label))




            # plt.imshow(image_DCE[2], cmap='gray')
            #
            # plt.show()  # 显示图像

save_data(image_benign_DCE_path,image_benign_T2_path,image_benign_T1_path,image_benign_DWI_path,label_benign_path,label0,csv_benign_file)
save_data(image_malignant_DCE_path,image_malignant_T2_path,image_malignant_T1_path,image_malignant_DWI_path,label_malignant_path,label1,csv_malignant_file)
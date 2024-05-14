import glob
import os
import shutil
import numpy as np
import pandas as pd
import pydicom
import nibabel as nib
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

'''
在DCE图像上勾画的ROI转换为nii格式形成的mask，形状和DCE的原始图像一致，用于特征提取，形状不一致不能特征提取；
DCE图像和T2图像的形状不一致，所以mask不能用于T2的特征提取；
这个脚本根据DCE勾画的mask，和T2的图像，生成T2的mask，形状和T2一致，用于特征提取；

将mask插值为T2图像的大小，然后与T2图像进行相乘，得到T2的mask；
nii文件不但有图像的值信息还有位置等其他信息，所以将mask插值为T2图像的大小不能用于特征提取；而是在dcm文件转化为nii过程中今替换里面的图像的值，不改变其他信心。

dcms_root_path文件夹下是每一个患者的姓名，每个患者的姓名文件夹下是每张dcm文件
mask_dir_path 勾画好的mask文件夹，用DCE图像勾画的ROI，形状和DCE图像一致
savenii_path 存放新生成的nii文件

'''




def dcm2nii(dcmdir_path, mask_path, savenii_path):
    # 1.构建dicom序列文件阅读器，并执行（即将dicom序列文件“打包整合”）
    label_nii = sitk.ReadImage(mask_path)

    label_array = sitk.GetArrayFromImage(label_nii)
    label_array = F.interpolate(torch.tensor(label_array, dtype=torch.float32).unsqueeze(0), size=(672, 672),
                                mode='nearest').squeeze().numpy().astype(np.uint8)

    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dcmdir_path)
    reader.SetFileNames(dicom_names)
    image2 = reader.Execute()
    # 2.将整合后的数据转为array，并获取dicom文件基本信息
    image_array = sitk.GetArrayFromImage(image2)  # z, y, x
    # 将数组中所有的数值都变为1
    image_array = np.ones_like(image_array)
    label = image_array * label_array

    origin = image2.GetOrigin()  # x, y, z
    spacing = image2.GetSpacing()  # x, y, z
    direction = image2.GetDirection()  # x, y, z
    # 3.将array转为img，并保存为.nii.gz
    image3 = sitk.GetImageFromArray(label)
    image3.SetSpacing(spacing)
    image3.SetDirection(direction)
    image3.SetOrigin(origin)
    sitk.WriteImage(image3, os.path.join(savenii_path, ))


dcms_root_dir_path = r"F:\300多的乳腺MR图像\良性\T2"
mask_dir_path = r"C:\Users\Administrator\Desktop\Breast\benign_label"
savenii_dir = r"C:\Users\Administrator\Desktop\Breast\lianxi"


patients = os.listdir(dcms_root_dir_path)
mask_dir = os.listdir(mask_dir_path)
for mask_name in mask_dir:
    label_name = mask_name.split("-")[0]
    for i , p in enumerate(patients):
        name = p.split("-")[0]
        if name == label_name:
            dcms_path = os.path.join(dcms_root_dir_path, p)
            mask_path = os.path.join(mask_dir_path, mask_name)



            dcm2nii(dcms_path,mask_path, os.path.join(savenii_dir, f'{name}-T2label.nii'))
            print(f'正在处理第{i+1}个文件')



print("finished")
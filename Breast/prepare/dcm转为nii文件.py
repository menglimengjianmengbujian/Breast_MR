import os
import shutil
import numpy as np
import pandas as pd
import pydicom
import nibabel as nib
import SimpleITK as sitk

'''
dcms_root_path文件夹下是每一个患者的姓名，每个患者的姓名文件夹下是每张dcm文件
dcm2nii.py 读取dcm文件，并生成nii文件
nii_path是一个新的文件夹，存放新生成的nii文件
这个脚本只能处理一个序列
'''


def dcm2nii(dcms_path, nii_path):
    # 1.构建dicom序列文件阅读器，并执行（即将dicom序列文件“打包整合”）
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dcms_path)
    reader.SetFileNames(dicom_names)
    image2 = reader.Execute()
    # 2.将整合后的数据转为array，并获取dicom文件基本信息
    image_array = sitk.GetArrayFromImage(image2)  # z, y, x
    origin = image2.GetOrigin()  # x, y, z
    spacing = image2.GetSpacing()  # x, y, z
    direction = image2.GetDirection()  # x, y, z
    # 3.将array转为img，并保存为.nii.gz
    image3 = sitk.GetImageFromArray(image_array)
    image3.SetSpacing(spacing)
    image3.SetDirection(direction)
    image3.SetOrigin(origin)
    sitk.WriteImage(image3, os.path.join(nii_path, ))

dcms_root_path = r'F:\300多的乳腺MR图像\良性\T2'  # dicom序列文件所在路径
nii_path = r'C:\Users\Administrator\Desktop\Breast\benign\T2'  # 所需.nii.gz文件保存路径
if os.path.exists(nii_path):
    shutil.rmtree(nii_path)  # 如果 save_path 存在，则删除
os.makedirs(nii_path)  # 生成 save_path

patients = os.listdir(dcms_root_path)
for i , p in enumerate(patients):
    dcm_xulie_path = os.path.join(dcms_root_path, p)
    name = p.split("-")[0]
    save_path = os.path.join(nii_path)

    dcm2nii(dcm_xulie_path, os.path.join(save_path, f'{name}.nii'))
    print(f'正在处理第{i+1}个文件')

print("finished")
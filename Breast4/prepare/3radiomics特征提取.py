import os
import pandas as pd
import SimpleITK as sitk
import radiomics
from radiomics import featureextractor

''''
origin_path是存放原图nii的文件夹
label_path是存放标签nii的文件夹
csv_file_path是存放提取特征的csv文件
'''


origin_path = r"C:\Users\Administrator\Desktop\Breast\NIIdata\Low\T2"
label_path = r"C:\Users\Administrator\Desktop\Breast\ROI\Low\T2"
# 设置 CSV 文件路径
csv_file_path = r"C:\Users\Administrator\Desktop\Breast\临时\Low_T2.csv"
# 创建特征提取器
extractor = featureextractor.RadiomicsFeatureExtractor()
patients = os.listdir(label_path)
# 逐个处理每个病人
Name = []
run = 0
for p in patients:
    name = p.split("-")[0]
    Name.append(name)
    print(f"正在处理第{run}个病人，病人姓名：{name}")
    origin = os.path.join(origin_path, name + ".nii")
    label = os.path.join(label_path, p)

    # 读取原始NII文件
    origin_nii = sitk.ReadImage(origin)

    # 读取标签NII文件
    label_nii = sitk.ReadImage(label)

    # 提取特征
    feature = extractor.execute(origin_nii, label_nii)
    result = pd.DataFrame([feature])

    # 在DataFrame中添加Name列
    result.insert(0, 'Name', name)

    # 将特征追加到CSV文件中
    result.to_csv(csv_file_path, mode='a', index=False, header=not os.path.exists(csv_file_path))
    run += 1



print("特征提取完成并追加到CSV文件。")
# 一 安装相关依赖，按照requirements.txt来
# 二 数据集良性和恶性的数据放到 dataset里面
# 三 运行prepare文件夹下的脚本

prepare文件夹下脚本的流程图：
![流程图](https://github.com/menglimengjianmengbujian/Breast_MR/assets/122141677/1d3ffb92-4a82-480b-aa75-8d281a55bc5b)



### 3.1将每个序列的原始图像dcm格式的转化为nii格式（1.py），每次只能将一个文件夹下的dcm->nii；如果所序列或者良恶性类别是分开文件夹存放的，需要分别运行。
良性dcm文件夹

    ├─DCE
    │  ├─ZHANGE_SAN-220917114   #必须是通过“-”连接
                ├─ZHANGE_SAN.dcm
                ├─ZHANGE_SAN.dcm
                ├─ZHANGE_SAN.dcm
       ├─LI_SI-220219158
            .
            .
            .
新建一个保存转化后的nii文件夹,存放nii文件
良性转化后的nii文件夹

    ├─DCENII
       ├─ZHANGE_SAN.nii  #转化后的文件只有姓名通过“_"连接姓氏
       ├─LI_SI.nii]
第二个类别也一样，转化到另一个类别的文件夹中
### 3.2然后将在DCE序列上勾画好的ROI映射到T1WI和T2WI上（2.py）
### 3.3然后与勾画好的nii标签进行特征提取（3.py）：
良性文件夹

    ├─DCENII       #是通过dcm文件转化而来
       ├─ZHANGE_SAN.nii
       ├─LI_SI.nii
    
勾画好的ROI文件夹
    
    ├─label       #是通过3D Slicer勾画出来的，然后保存的文件夹
       ├─ZHANGE_SAN-label.nii   #中间必须通过”-“连接，表示label的nii
       ├─LI_SI-label.nii

----------------------------    
恶行文件夹

    ├─DCENII
       ├─ZHANGE_SAN.nii
       ├─LI_SI.nii
勾画好的ROI文件夹
    ├─label
       ├─ZHANGE_SAN-label.nii
       ├─LI_SI-label.nii
#### 3.3.2新建三个保存特征的csv文件，保存两个类别的特征，需要**删除掉不需要的特征**；如果做多序列任务，需要将多序列保存到各自类别的csv文件中。

    ├─benign
       ├─T1_csv
       ├─T2_csv
       ├─DCE_csv
----------------------------
    ├─malignant
       ├─T1_csv
       ├─T2_csv
       ├─DCE_csv
### 3.4对特征进行标准化和归一化（4.py），归一化后的特征缩放到0-1之间，然后保存到csv文件中，然后将良性文件夹和恶行文件夹的特征

    ├─benign_Stand_MinMaxScaler
       ├─T1_Stand_MinMaxScaler_csv
       ├─T2_Stand_MinMaxScaler_csv
       ├─DCE_Stand_MinMaxScaler_csv
----------------------------
    ├─malignant_Stand_MinMaxScaler
       ├─T1_Stand_MinMaxScaler_csv
       ├─T2_Stand_MinMaxScaler_csv
       ├─DCE_Stand_MinMaxScaler_csv
### 3.5将标准化和归一化后的三个序列合并到一个CSV当中（5.py）

    ├─benign_merged
       ├─T1_T2_DCE_csv
----------------------------
    ├─malignant_merged
       ├─T1_T2_DCE_csv
### 3.6再保存到h5文件中（6.py），将良性和恶行的路径和标签填好，多序列就填多个序列的路径，就可以运行了。

保存中用到了两种方法，一种是concat_image处理方式，将ROI的边界勾画出来，映射到原图中裁剪出224×224大小的图像
    #另一种是one_three处理方式，将ROI的中心点找出来，映射到原始图像上，作为中心点裁剪出224×224大小的图像，然后将标签也保存到h5文件中
    两种方法使用一种就可以
# 四 要训练就运行train_fusion(concat).py

流程图：
![流程图2](https://github.com/menglimengjianmengbujian/Breast_MR/assets/122141677/60258e95-9657-4be0-a51a-3f8ab75ebc82)

    修改好argparse中的参数
# 五 要测试就运行test(concat).py
    修改好保存的模型路径和测试集的路径 
如要测试不同的输入融合的结果，修改MF/dataset.py，注释掉相应的项就可以了



# 六 Grad-CAM

- Original Impl: [https://github.com/jacobgil/pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)
- Grad-CAM简介: [https://b23.tv/1kccjmb](https://b23.tv/1kccjmb)
- 使用Pytorch实现Grad-CAM并绘制热力图: [https://b23.tv/n1e60vN](https://b23.tv/n1e60vN)

## 使用流程(替换成自己的网络)

1. 将创建模型部分代码替换成自己创建模型的代码，并载入自己训练好的权重
2. 根据自己网络设置合适的`target_layers`
3. 根据自己的网络设置合适的预处理方法
4. 将要预测的图片路径赋值给`img_path`
5. 将感兴趣的类别id赋值给`target_category`


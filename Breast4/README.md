# Breast4是用于预测浸润性乳腺癌Ki-67的低表达和高表达，使用了T1WI、T2WI、DCE和DWI序列。在训练的过程中，由于数据量较少，使用了五折交叉验证。网络使用的还是Resnet50，Vit也跑通了。
# 一 安装相关依赖，按照requirements.txt来
# 二 数据集良性和恶性的数据放到 dataset里面
# 三 运行prepare文件夹下的脚本
这个流程还是和Breast3的一样，只不过是过了一个DWI序列，图就没有重新画了，能理解就行。
![流程图1](https://github.com/user-attachments/assets/f6f89202-1e30-44ee-8876-a886b1d9157e)




3.1先对勾画好的数据进行处理，将原始图像dcm格式的转化为nii格式（dcm转为nii文件.py），是一个类别转化到一个文件夹中。
良性dcm文件夹

    ├─DCE
    │  ├─ZHANGE_SAN-220917114   #必须是通过“-”连接
    │           ├─ZHANGE_SAN.dcm
    │           ├─ZHANGE_SAN.dcm
    │           ├─ZHANGE_SAN.dcm
    │  ├─LI_SI-220219158
            .
            .
            .
新建一个保存转化后的nii文件加,转化后的文件夹形式
处理后原始图像nii文件夹
良性转化后的文件夹

    ├─DCENII
    │  ├─ZHANGE_SAN.nii  #转化后的文件只有姓名通过“_"连接姓氏
    │  ├─LI_SI.nii]
第二个类别也一样，转化到另一个类别的文件夹中
3.2然后与勾画好的nii标签进行特征提取（radiomics特征提取.py）：
良性文件夹

    ├─DCENII       #是通过dcm文件转化而来
    │  ├─ZHANGE_SAN.nii
    │  ├─LI_SI.nii
    勾画好的ROI文件夹
    ├─label       #是通过3D Slicer勾画出来的，然后保存的文件夹
    │  ├─ZHANGE_SAN-label.nii   #中间必须通过”-“连接，表示label的nii
    │  ├─LI_SI-label.nii
恶行文件夹

    ├─DCENII
    │  ├─ZHANGE_SAN.nii
    │  ├─LI_SI.nii
    勾画好的ROI文件夹
    ├─label
    │  ├─ZHANGE_SAN-label.nii
    │  ├─LI_SI-label.nii
还得新建两个保存特征的csv文件，保存两个类别的特征，需要删除掉不需要的特征；如果做多序列任务，需要将多序列保存到各自类别的csv文件中。
3.3对特征进行归一化（归一化.py），归一化后的特征缩放到0-1之间，然后保存到csv文件中，然后将良性文件夹和恶行文件夹的特征
3.4再保存到h5文件中（存为H5文件.py），将良性和恶行的路径和标签填好，多序列就填多个序列的路径，就可以运行了。
3.4.1保存中用到了两种方法，一种是concat_image处理方式，将ROI的边界勾画出来，映射到原图中裁剪出224×224大小的图像
    #另一种是one_three处理方式，将ROI的中心点找出来，映射到原始图像上，作为中心点裁剪出224×224大小的图像，然后将标签也保存到h5文件中
    两种方法使用一种就可以
3.4.2如果做多序列任务，需要将多序列保存到各自类别的csv文件中，然后将良性文件夹和恶行
注：如果做多序列任务，勾画了一个序列的ROI，怎可以根据这个ROI序列，映射到其他序列生成mask标签（mask映射新的mask.py），然后与对应的原始图像进行特征提取，进行3.4步骤
# 四 要训练就运行train_fusion(concat).py
    修改好argparse中的参数
训练的流程图
![流程图](https://github.com/user-attachments/assets/bb140d3b-99cf-4915-9c33-5976fb1417c6)

# 五 要测试就运行test(concat).py
    修改好保存的模型路径和测试集的路径 
如要测试不同的输入融合的结果，修改MF/dataset.py，注释掉相应的项就可以了



## 六 Grad-CAM

- Original Impl: [https://github.com/jacobgil/pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)
- Grad-CAM简介: [https://b23.tv/1kccjmb](https://b23.tv/1kccjmb)
- 使用Pytorch实现Grad-CAM并绘制热力图: [https://b23.tv/n1e60vN](https://b23.tv/n1e60vN)

## 使用流程(替换成自己的网络)

1. 将创建模型部分代码替换成自己创建模型的代码，并载入自己训练好的权重
2. 根据自己网络设置合适的`target_layers`
3. 根据自己的网络设置合适的预处理方法
4. 将要预测的图片路径赋值给`img_path`
5. 将感兴趣的类别id赋值给`target_category`


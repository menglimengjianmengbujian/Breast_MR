import torch
import torch.nn as nn
import torchvision.models as models


class CustomResNet50(nn.Module):
    def __init__(self,  in_channels=1,num_classes=2, chunk=3):
        super(CustomResNet50, self).__init__()
        self.chunk = chunk
        # 加载预训练的ResNet-50模型
        resnet = models.resnet50(pretrained=True)

        # 将修改后的模型赋值给自定义的ResNet-50网络
        self.model = resnet

        # 修改第一个卷积层的输入通道数
        self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # 修改全连接层的输出特征数
        self.fc = nn.Linear(2048 * self.chunk, num_classes)


    def  cat(self,x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        return x


    def forward(self, x ):
        # 对输入的三通道图像进行分割
        self.data = []
        split_data = torch.split(x, 1, dim=1)
        for i in range(self.chunk):
            self.data.append(self.cat(split_data[i]))

        # 在全连接层之前进行拼接
        x = torch.cat(self.data, dim=1)


        # 全连接层输出
        output = self.fc(x)

        return output

if __name__ == '__main__':
    # 创建一个实例，并指定类别数
    net = CustomResNet50( in_channels=1,num_classes=2,chunk=3)

    # 输出网络结构
    print(net)
#
#
#
#     # 在输入数据上进行前向传播
#     input_data = torch.randn(64, 3, 224, 224)  # 假设输入数据尺寸为1x224x224
#     output = net(input_data)
#     print("前连接层特征尺寸:", output.size())

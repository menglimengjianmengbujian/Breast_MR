import torch
import torch.nn as nn
import torchvision.models as models


class CustomResNet50(nn.Module):
    def __init__(self,  in_channels=3,num_classes=2, chunk=2,csv_shape = 107,CSV=True):
        super(CustomResNet50, self).__init__()
        self.chunk = chunk
        self.num_classes=num_classes
        self.CSV = CSV
        # 加载预训练的ResNet-50模型
        resnet = models.resnet50(pretrained=True)

    #可以设置是否冻结权重，默认为不冻结权重，如果为False，则所有权重都会被更新
        for name, param in resnet.named_parameters():
            param.requires_grad = False

            #print(name, param.requires_grad)

        # 将修改后的模型赋值给自定义的ResNet-50网络
        self.model = resnet

        # 修改第一个卷积层的输入通道数
        self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # if CSV==False:
        #     self.fc = nn.Linear(2048 * self.chunk , num_classes)
        # else:

        self.model2 = resnet

        # 修改第一个卷积层的输入通道数
        self.model2.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.fc = nn.Linear(2048 * self.chunk + csv_shape , num_classes)
        # self.fc = nn.Linear(2048 * self.chunk , num_classes)


    def  cat1(self,x):
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

    def  cat2(self,x):
        x = self.model2.conv1(x)
        x = self.model2.bn1(x)
        x = self.model2.relu(x)
        x = self.model2.maxpool(x)

        x = self.model2.layer1(x)
        x = self.model2.layer2(x)
        x = self.model2.layer3(x)
        x = self.model2.layer4(x)
        x = self.model2.avgpool(x)
        x = torch.flatten(x, 1)
        return x
    def forward(self, data_DCE,data_T2,csv):


        data_DCE = self.cat1(data_DCE)
        data_T2 = self.cat2(data_T2)
        csv = csv
        if self.CSV==False:
            csv1 = torch.ones_like(csv )
            x = torch.cat((data_DCE, data_T2,csv1), dim=1)
            output = self.fc(x)
        else:

            # 在全连接层之前进行拼接
            x = torch.cat((data_DCE,data_T2,csv), dim=1)

            # 全连接层输出
            output = self.fc(x)

        return output

if __name__ == '__main__':
    # 创建一个实例，并指定类别数
    net = CustomResNet50( in_channels=3,num_classes=2,chunk=2,csv_shape = 107,CSV=False)
    # 打印每一层的名称以及其参数是否可训练
    for name, param in net.named_parameters():
        print(name, ":", param.requires_grad)
    # 输出网络结构
    print(net)
    #生成一个一行100列的随机张量


    #
    # # 在输入数据上进行前向传播
    # input_data = torch.randn(64, 3, 224, 224)  # 假设输入数据尺寸为1x224x224
    # output = net(input_data)
    # print("前连接层特征尺寸:", output.size())
    data_DCE=torch.randn(64,3, 224, 224)
    data_T2=torch.randn(64,3, 224, 224)
    csv= torch.randn(64,107)

    output = net(data_DCE,data_T2,csv)
    print("前连接层特征尺寸:", output.size())
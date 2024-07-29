import torch
import torch.nn as nn
import torchvision.models as models


class CustomResNet50(nn.Module):
    def __init__(self,  in_channels=3,num_classes=2, Seq=5,csv_shape = 428,CSV=True):
        super(CustomResNet50, self).__init__()
        self.Seq = Seq
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

        # 定义批量归一化层和ReLU层
        self.fc1 = nn.Linear(428 , 107)
        self.relu = nn.ReLU()
        if CSV==False:
            self.fc = nn.Linear(2048 * 4 , num_classes)
        else:

            self.fc = nn.Linear((2048 + csv_shape//4 ) * 4 , num_classes)
        # self.fc = nn.Linear(2048 * self.chunk , num_classes)


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


    def forward(self, data_DCE,data_T2,data_T1,data_DWI,csv):


        data_DCE = self.cat(data_DCE)
        data_T2 = self.cat(data_T2)
        data_T1 = self.cat(data_T1)
        data_DWI = self.cat(data_DWI)
        csv = csv
        # csv1 = csv[:, :107]
        # csv2 = csv[:, 214:]
        if self.Seq == 5:
            data_DCE = torch.cat((data_DCE, csv[:, :107]), dim=1)
            data_T1 = torch.cat((data_T1, csv[:, 107:214]), dim=1)
            data_T2 = torch.cat((data_T2, csv[:, 214:321]), dim=1)
            data_DWI = torch.cat((data_DWI, csv[:, 321:]), dim=1)
            x = torch.cat((data_DCE, data_T2, data_T1, data_DWI), dim=1)

            output = self.fc(x)
        if self.Seq == 1:
            data_DCE = torch.cat((data_DCE, csv[:, :107]), dim=1)
            data_T1 = torch.cat((data_T1, csv[:, :107]), dim=1)
            data_T2 = torch.cat((data_T2, csv[:, :107]), dim=1)
            data_DWI = torch.cat((data_DWI, csv[:, :107]), dim=1)
            x = torch.cat((data_DCE, data_T2, data_T1, data_DWI), dim=1)
            output = self.fc(x)
        if self.Seq == 2:
            data_DCE = torch.cat((data_DCE, csv[:, 107:214]), dim=1)
            data_T1 = torch.cat((data_T1, csv[:, 107:214]), dim=1)
            data_T2 = torch.cat((data_T2, csv[:, 107:214]), dim=1)
            data_DWI = torch.cat((data_DWI, csv[:, 107:214]), dim=1)
            x = torch.cat((data_DCE, data_T2, data_T1, data_DWI), dim=1)
            output = self.fc(x)
        if self.Seq == 3:
            data_DCE = torch.cat((data_DCE, csv[:, 214:321]), dim=1)
            data_T1 = torch.cat((data_T1, csv[:, 214:321]), dim=1)
            data_T2 = torch.cat((data_T2, csv[:, 214:321]), dim=1)
            data_DWI = torch.cat((data_DWI, csv[:, 214:321]), dim=1)
            x = torch.cat((data_DCE, data_T2, data_T1, data_DWI), dim=1)
            output = self.fc(x)
        if self.Seq == 4:
            data_DCE = torch.cat((data_DCE, csv[:, 321:]), dim=1)
            data_T1 = torch.cat((data_T1, csv[:, 321:]), dim=1)
            data_T2 = torch.cat((data_T2, csv[:, 321:]), dim=1)
            data_DWI = torch.cat((data_DWI, csv[:, 321:]), dim=1)
            x = torch.cat((data_DCE, data_T2, data_T1, data_DWI), dim=1)
            output = self.fc(x)

        return output
if __name__ == '__main__':
    # 创建一个实例，并指定类别数
    net = CustomResNet50( in_channels=3,num_classes=2,Seq=5,csv_shape = 428,CSV=True)
    # 打印每一层的名称以及其参数是否可训练
    for name, param in net.named_parameters():
        print(name, ":", param.requires_grad)
    # 输出网络结构
    # print(net)
    #生成一个一行100列的随机张量


    #
    # # 在输入数据上进行前向传播
    # input_data = torch.randn(64, 3, 224, 224)  # 假设输入数据尺寸为1x224x224
    # output = net(input_data)
    # print("前连接层特征尺寸:", output.size())
    data_DCE=torch.randn(64,3, 224, 224)
    data_T2=torch.randn(64,3, 224, 224)
    data_T1=torch.randn(64,3, 224, 224)
    data_DWI = torch.randn(64, 3, 224, 224)
    csv= torch.randn(64,428)

    output = net(data_DCE,data_T2,data_T2,data_DWI,csv)
    print("前连接层特征尺寸:", output.size())
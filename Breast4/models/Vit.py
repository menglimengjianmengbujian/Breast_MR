import torch
import torch.nn as nn
import torchvision.models as models

class CustomViT(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, Seq=5, csv_shape=428, CSV=True):
        super(CustomViT, self).__init__()
        self.Seq = Seq
        self.num_classes = num_classes
        self.CSV = CSV
        # 加载预训练的ViT模型
        vit = models.vit_b_16(pretrained=True)

        # 冻结权重
        for name, param in vit.named_parameters():
            param.requires_grad = False

        # 将修改后的模型赋值给自定义的ViT网络
        self.model = vit

        # 修改第一个卷积层的输入通道数
        self.model.conv_proj = nn.Conv2d(in_channels, 768, kernel_size=16, stride=16)

        # 调整位置嵌入的大小
        self.model.encoder.pos_embedding = nn.Parameter(self.model.encoder.pos_embedding[:, :196, :])

        # 设置全连接层
        self.fc = nn.Linear((768 + csv_shape // 4) * 4, num_classes)

    def cat(self, x):
        # ViT 不同于 ResNet，它不通过层级提取特征
        x = self.model.conv_proj(x)
        x = x.flatten(2).transpose(1, 2)

        # 确保输入张量的大小与位置嵌入的大小一致
        if x.size(1) != self.model.encoder.pos_embedding.size(1):
            # 调整位置嵌入的大小
            self.model.encoder.pos_embedding = nn.Parameter(
                self.model.encoder.pos_embedding[:, :x.size(1), :]
            )

        x = self.model.encoder(x)
        x = x.mean(dim=1)
        return x

    def forward(self, data_DCE, data_T2, data_T1, data_DWI, csv):
        data_DCE = self.cat(data_DCE)
        data_T2 = self.cat(data_T2)
        data_T1 = self.cat(data_T1)
        data_DWI = self.cat(data_DWI)

        if self.Seq == 5:
            data_DCE = torch.cat((data_DCE, csv[:, :107]), dim=1)
            data_T1 = torch.cat((data_T1, csv[:, 107:214]), dim=1)
            data_T2 = torch.cat((data_T2, csv[:, 214:321]), dim=1)
            data_DWI = torch.cat((data_DWI, csv[:, 321:]), dim=1)
            x = torch.cat((data_DCE, data_T2, data_T1, data_DWI), dim=1)
            output = self.fc(x)
        elif self.Seq == 1:
            data_DCE = torch.cat((data_DCE, csv[:, :107]), dim=1)
            data_T1 = torch.cat((data_T1, csv[:, :107]), dim=1)
            data_T2 = torch.cat((data_T2, csv[:, :107]), dim=1)
            data_DWI = torch.cat((data_DWI, csv[:, :107]), dim=1)
            x = torch.cat((data_DCE, data_T2, data_T1, data_DWI), dim=1)
            output = self.fc(x)
        elif self.Seq == 2:
            data_DCE = torch.cat((data_DCE, csv[:, 107:214]), dim=1)
            data_T1 = torch.cat((data_T1, csv[:, 107:214]), dim=1)
            data_T2 = torch.cat((data_T2, csv[:, 107:214]), dim=1)
            data_DWI = torch.cat((data_DWI, csv[:, 107:214]), dim=1)
            x = torch.cat((data_DCE, data_T2, data_T1, data_DWI), dim=1)
            output = self.fc(x)
        elif self.Seq == 3:
            data_DCE = torch.cat((data_DCE, csv[:, 214:321]), dim=1)
            data_T1 = torch.cat((data_T1, csv[:, 214:321]), dim=1)
            data_T2 = torch.cat((data_T2, csv[:, 214:321]), dim=1)
            data_DWI = torch.cat((data_DWI, csv[:, 214:321]), dim=1)
            x = torch.cat((data_DCE, data_T2, data_T1, data_DWI), dim=1)
            output = self.fc(x)
        elif self.Seq == 4:
            data_DCE = torch.cat((data_DCE, csv[:, 321:]), dim=1)
            data_T1 = torch.cat((data_T1, csv[:, 321:]), dim=1)
            data_T2 = torch.cat((data_T2, csv[:, 321:]), dim=1)
            data_DWI = torch.cat((data_DWI, csv[:, 321:]), dim=1)
            x = torch.cat((data_DCE, data_T2, data_T1, data_DWI), dim=1)
            output = self.fc(x)

        return output

if __name__ == '__main__':
    # 创建一个实例，并指定类别数
    net = CustomViT(in_channels=3, num_classes=2, Seq=5, csv_shape=428, CSV=True)
    # 打印每一层的名称以及其参数是否可训练
    for name, param in net.named_parameters():
        print(name, ":", param.requires_grad)

    # 减小批处理大小
    data_DCE = torch.randn(16, 3, 224, 224)
    data_T2 = torch.randn(16, 3, 224, 224)
    data_T1 = torch.randn(16, 3, 224, 224)
    data_DWI = torch.randn(16, 3, 224, 224)
    csv = torch.randn(16, 428)

    output = net(data_DCE, data_T2, data_T1, data_DWI, csv)
    print("前连接层特征尺寸:", output.size())

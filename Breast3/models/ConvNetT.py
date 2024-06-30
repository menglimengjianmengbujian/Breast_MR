import torch
import torch.nn as nn
import torchvision.models as models


class CustomConvNeXtT(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, chunk=2, csv_shape=107, CSV=True):
        super(CustomConvNeXtT, self).__init__()
        self.chunk = chunk
        self.num_classes = num_classes
        self.CSV = CSV

        # 加载预训练的ConvNeXt-Tiny模型
        convnext = models.convnext_tiny(pretrained=True)

        # 冻结预训练模型的所有参数
        for name, param in convnext.named_parameters():
            param.requires_grad = False

        # 将修改后的模型赋值给自定义的ConvNeXt-T网络
        self.model = convnext

        # 修改第一个卷积层的输入通道数
        self.model.features[0][0] = nn.Conv2d(in_channels, 96, kernel_size=4, stride=4)

        # 获取特征提取器的输出特征维度
        num_ftrs = self.model.classifier[2].in_features

        # 修改分类头部
        self.model.classifier = nn.Sequential(
            nn.LayerNorm(num_ftrs * self.chunk + (csv_shape if CSV else 0), eps=1e-6, elementwise_affine=True),
            nn.Linear(num_ftrs * self.chunk + (csv_shape if CSV else 0), num_classes)
        )

    def extract_features(self, x):
        x = self.model.features(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, data_DCE, data_T2, csv):
        data_DCE = self.extract_features(data_DCE)
        data_T2 = self.extract_features(data_T2)

        if not self.CSV:
            csv = torch.ones_like(csv)

        x = torch.cat((data_DCE, data_T2, csv), dim=1)
        print(f"Feature size after concatenation: {x.size()}")  # 打印特征拼接后的尺寸

        output = self.model.classifier(x)
        return output


if __name__ == '__main__':
    net = CustomConvNeXtT(in_channels=3, num_classes=2, chunk=2, csv_shape=107, CSV=True)
    for name, param in net.named_parameters():
        print(name, ":", param.requires_grad)

    data_DCE = torch.randn(64, 3, 224, 224)
    data_T2 = torch.randn(64, 3, 224, 224)
    csv = torch.randn(64, 107)

    output = net(data_DCE, data_T2, csv)
    print("输出特征尺寸:", output.size())

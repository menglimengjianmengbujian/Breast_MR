import torch
import torch.nn as nn
import timm


class CustomSwinTransformer(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, chunk=3, csv_shape=107, CSV=True, pretrained_weights_path=None):
        super(CustomSwinTransformer, self).__init__()
        self.chunk = chunk
        self.num_classes = num_classes
        self.CSV = CSV

        # 加载预训练的Swin Transformer模型
        self.swin = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False)

        if pretrained_weights_path:
            # 加载预训练权重
            state_dict = torch.load(pretrained_weights_path)
            # 过滤掉与当前模型结构不匹配的键
            filtered_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items() if
                                   k.replace('module.', '') in self.swin.state_dict()}
            self.swin.load_state_dict(filtered_state_dict, strict=False)

        # 冻结权重
        for name, param in self.swin.named_parameters():
            param.requires_grad = False

        # 修改第一个卷积层的输入通道数
        self.patch_embed = nn.Conv2d(in_channels, self.swin.embed_dim, kernel_size=4, stride=4, padding=0)

        # 保留Swin Transformer模型的中间部分
        self.model = nn.Sequential(
            self.swin.patch_embed,
            self.swin.layers,
            self.swin.norm
        )

        if CSV == False:
            self.fc = nn.Linear(self.swin.num_features * self.chunk, num_classes)
        else:
            self.fc = nn.Linear(self.swin.num_features * self.chunk + csv_shape, num_classes)

    def cat(self, x):
        # 使用自定义的 patch_embed
        x = self.patch_embed(x)
        x = x.view(-1, 96, int(224 / 4), int(224 / 4))  # 调整维度以适应 patch_embed 方法的期望输入格式
        x = x.flatten(2).transpose(1, 2)  # (B, N, C) where N = H*W/p^2, C = embed_dim
        # 通过模型的中间部分
        for layer in self.model:
            x = layer(x)
        x.view(-1, 56, 56, 96)
        x = x.mean(dim=1)
        return x

    def forward(self, data_DCE, data_T2, data_T1, csv):
        data_DCE = self.cat(data_DCE)
        data_T2 = self.cat(data_T2)
        data_T1 = self.cat(data_T1)
        if self.CSV == False:
            x = torch.cat((data_DCE, data_T2, data_T1), dim=1)
            output = self.fc(x)
        else:
            # 在全连接层之前进行拼接
            x = torch.cat((data_DCE, data_T2, data_T1, csv), dim=1)
            # 全连接层输出
            output = self.fc(x)
        return output


if __name__ == '__main__':
    # 指定预训练权重的路径
    pretrained_weights_path = 'E:/pycharmproject/swin_transformer/swin_tiny_patch4_window7_224.pth'

    # 创建一个实例，并指定类别数
    net = CustomSwinTransformer(in_channels=3, num_classes=2, chunk=3, csv_shape=107, CSV=False,
                                pretrained_weights_path=pretrained_weights_path)

    # 打印每一层的名称以及其参数是否可训练
    for name, param in net.named_parameters():
        print(name, ":", param.requires_grad)

    data_DCE = torch.randn(64, 3, 224, 224)
    data_T2 = torch.randn(64, 3, 224, 224)
    data_T1 = torch.randn(64, 3, 224, 224)
    csv = torch.randn(64, 107)

    output = net(data_DCE, data_T2, data_T1, csv)
    print("前连接层特征尺寸:", output.size())

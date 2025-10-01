import torch
import torch.nn as nn


class RABM(nn.Module):
    def __init__(self, in_channels):
        super(RABM, self).__init__()

        # 分支1：不同尺寸卷积
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,dilation=1,groups=in_channels),
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
        )
        self.conv5x5 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=2, dilation=2,groups=in_channels),
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
        )
        self.conv7x7 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=3, dilation=3,groups=in_channels),
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
        )
        self.conv9x9 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=4, dilation=4,groups=in_channels),
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
        )

        # 分支2：最大池化 + 1x1卷积
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, W, H = x.shape

        # 分支1：通过 3x3, 5x5, 7x7, 9x9 卷积
        out_3x3 = self.conv3x3(x)
        out_5x5 = self.conv5x5(x)
        out_7x7 = self.conv7x7(x)
        out_9x9 = self.conv9x9(x)

        # 将它们沿着通道维度拼接 [B, C/4, W, H] -> [B, C, W, H]
        out_branch1 = torch.cat([out_3x3, out_5x5, out_7x7, out_9x9], dim=1)

        # 与输入特征图相乘，得到特征1
        feature1 = out_branch1 * x

        # 分支2：最大池化 -> 1x1卷积 -> sigmoid
        pooled1 = self.maxpool(x)  # [B, C, W/2, H/2]
        pooled2 = self.avgpool(x)
        pooled = pooled1+pooled2
        out_branch2 = nn.functional.interpolate(pooled, size=(W, H), mode='bilinear',
                                           align_corners=False)  # 上采样回 [B, C, W, H]
        out_branch2 = self.conv1x1(out_branch2)
        out_branch2 = self.sigmoid(out_branch2)


        # 与输入特征图相乘，得到特征2
        feature2 = out_branch2 * x

        # 最终输出：特征1 + 特征2
        out = feature1 + feature2

        return out


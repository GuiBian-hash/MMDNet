import torch
import torch.nn as nn

from Model_Code.RABM import RABM


class DetectHead(nn.Module):
    def __init__(self, in_channels_deep=256, in_channels_mid=128, in_channels_shallow=64, out_channels=6):
        super(DetectHead, self).__init__()

        # 处理最深层特征 (8, 256, 32, 32)
        self.RABM1 = RABM(in_channels_deep)
        self.RABM2 = RABM(in_channels_mid)
        self.RABM3 = RABM(in_channels_shallow)
        self.deep_conv = nn.Sequential(
            nn.Conv2d(in_channels_deep, in_channels_deep, kernel_size=3, padding=1, groups=in_channels_deep),  # 深度可分离卷积
            nn.Conv2d(256, 128, kernel_size=1),  # 逐点卷积降维
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # 上采样至 (8, 128, 64, 64)
        )

        # 处理中间层特征 (8, 128, 64, 64)
        self.mid_conv = nn.Sequential(
            nn.Conv2d(in_channels_mid*2, in_channels_mid*2, kernel_size=3, padding=1, groups=in_channels_mid*2),  # 深度可分离卷积
            nn.Conv2d(128*2, 64*2, kernel_size=1),  # 逐点卷积降维
            nn.BatchNorm2d(64*2),
            nn.GELU(),)
        self.mid_conv1 = nn.Sequential(
            nn.Conv2d(in_channels_mid , in_channels_mid, kernel_size=3, padding=1, groups=in_channels_mid),
            # 深度可分离卷积
            nn.Conv2d(128 , 64 , kernel_size=1),  # 逐点卷积降维
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # 上采样至 (8, 64, 128, 128)
        )

        # 处理最浅层特征 (8, 64, 128, 128)
        self.shallow_conv = nn.Sequential(
            nn.Conv2d(in_channels_shallow*2, 64*2, kernel_size=3, padding=1, groups=in_channels_shallow*2),  # 深度可分离卷积
            nn.Conv2d(64*2, 64, kernel_size=1),  # 逐点卷积保持通道不变
            nn.BatchNorm2d(64),
            nn.GELU()
        )

        # 特征融合


        # 上采样至目标大小 (8, 64, 256, 256)
        self.upsample_to_256 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # (8, 32, 128, 128)

            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # (8, 16, 256, 256)

            nn.Conv2d(16, out_channels, kernel_size=3, padding=1),  # 输出通道为目标通道数
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x_deep,x_mid,x_shallow):
        x_deep = self.RABM1(x_deep)
        x_mid = self.RABM2(x_mid)
        x_shallow = self.RABM3(x_shallow)
        # 逐级处理深层、中间层和浅层特征
        x_deep = self.deep_conv(x_deep)  # (8, 128, 64, 64)
        x_fusion_mid = torch.cat([x_deep, x_mid], dim=1)
        x_mid_out = self.mid_conv(x_fusion_mid)  # (8, 64, 128, 128)
        x_mid = self.mid_conv1(x_mid_out)
        x_fusion_mid = torch.cat([x_shallow, x_mid], dim=1)
        x_shallow = self.shallow_conv(x_fusion_mid)  # (8, 64, 128, 128)
        # 深层特征上采样后与中间层特征融合


        # 上采样至目标大小
        x_output = self.upsample_to_256(x_shallow)  # (8, 6, 256, 256)

        return x_output

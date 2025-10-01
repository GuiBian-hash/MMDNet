
import torch
import torch.nn as nn
import torch.nn.functional as F
class RIFM(nn.Module):
    def __init__(self, low_dim_channels, high_dim_channels):
        super(RIFM, self).__init__()
        self.high_dim_channels = high_dim_channels
        self.conv = nn.Conv2d(high_dim_channels, low_dim_channels, kernel_size=1)
        self.low_conv1 = nn.Sequential(
            nn.Conv2d(low_dim_channels,low_dim_channels, kernel_size=3, padding=1,stride=1,groups=low_dim_channels),
            nn.BatchNorm2d(low_dim_channels),
            nn.GELU()
        )
        self.low_conv2 = nn.Sequential(
            nn.Conv2d(low_dim_channels,low_dim_channels, kernel_size=3, padding=1,stride=1,groups=low_dim_channels),
            nn.BatchNorm2d(low_dim_channels),
            nn.GELU()
        )
        self.low_conv_dilation = nn.Sequential(
            nn.Conv2d(low_dim_channels,low_dim_channels, kernel_size=3, padding=1,stride=1,groups=low_dim_channels),
            nn.BatchNorm2d(low_dim_channels),
            nn.GELU()
        )
        self.high_attention1 = nn.Sequential(
            nn.Conv2d(low_dim_channels, low_dim_channels, kernel_size=3, padding=1,stride=1,groups=low_dim_channels),
            nn.Conv2d(low_dim_channels,low_dim_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.high_attention2 = nn.Sequential(
            nn.Conv2d(low_dim_channels, low_dim_channels, kernel_size=3, padding=2,stride=1,dilation=2,groups=low_dim_channels),
            nn.Conv2d(low_dim_channels,low_dim_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.high_attention3 = nn.Sequential(
            nn.Conv2d(low_dim_channels, low_dim_channels, kernel_size=3, padding=3,stride=1,dilation=3,groups=low_dim_channels),
            nn.Conv2d(low_dim_channels,low_dim_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.high_conv = nn.Conv2d(low_dim_channels, low_dim_channels, kernel_size=3, padding=1,stride=1,groups=low_dim_channels)
        self.final_conv_1 = nn.Sequential(
            nn.Conv2d(low_dim_channels*2,low_dim_channels*2, kernel_size=3, padding=1,stride=1,groups=low_dim_channels*2,dilation=1),
            nn.BatchNorm2d(low_dim_channels*2)
        )
        self.final_conv_2 = nn.Sequential(
            nn.Conv2d(low_dim_channels*2, low_dim_channels*2, kernel_size=3, padding=2, stride=1,
                                      groups=low_dim_channels*2, dilation=2),
            nn.BatchNorm2d(low_dim_channels*2)
        )
        self.final_conv_4 = nn.Sequential(
            nn.Conv2d(low_dim_channels*2, low_dim_channels*2, kernel_size=3, padding=4, stride=1,
                                      groups=low_dim_channels*2, dilation=4),
            nn.BatchNorm2d(low_dim_channels*2)
        )
        self.final_conv_8 = nn.Sequential(
            nn.Conv2d(low_dim_channels*2, low_dim_channels*2, kernel_size=3, padding=8, stride=1,
                                      groups=low_dim_channels*2, dilation=8),
            nn.BatchNorm2d(low_dim_channels*2)
        )
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(low_dim_channels*8, low_dim_channels,1),
            nn.BatchNorm2d(low_dim_channels),
            nn.GELU()
        )
    def forward(self, low_features,high_features):
        if self.high_dim_channels == 256:
            high_features = high_features.reshape(high_features.size(0),256,32,32)
            high_features = self.conv(high_features)
            high_features = F.interpolate(high_features, size=(low_features.shape[2], low_features.shape[3]), mode='bilinear', align_corners=False)
        low_conv1 = self.low_conv1(low_features)
        low_conv2 = self.low_conv2(low_features)
        low_conv3 = self.low_conv_dilation(low_features)
        low_features = low_conv2*low_conv3
        low_features = low_conv1+low_features
        high_attention1 = self.high_attention1(high_features)
        high_attention2 = self.high_attention2(high_features)
        high_attention3 = self.high_attention3(high_features)
        high_conv1 = self.high_conv(high_features)
        high_features = high_conv1*high_attention1*high_attention2*high_attention3
        feature = torch.cat((low_features,high_features),dim=1)
        x1 = self.final_conv_1(feature)
        x2 = self.final_conv_2(feature)
        x3 = self.final_conv_4(feature)
        x4 = self.final_conv_8(feature)
        x = torch.cat((x1,x2,x3,x4),dim=1)
        x = self.conv1x1(x)

        return x


# 测试模块
# if __name__ == "__main__":
#     # 模拟输入
#     B, C_low, H_low, W_low = 32, 64, 64, 64
#     B, C_high, H_high, W_high = 32, 256, 16, 16
#
#     # 定义模块
#     module = FeatureSwapModule(C_low, C_high, H_low, W_low, H_high, W_high,2)
#
#     # 创建模拟输入
#     low_features = torch.randn(B, C_low, H_low, W_low)
#     high_features = torch.randn(B, C_high, H_high, W_high)
#
#     # 前向传播
#     low_to_high, high_to_low = module(low_features, high_features)
#
#     # 输出特征图大小
#     print(f"低维特征（调整后）大小: {low_to_high.shape}")  # 应为 (32, 256, 16, 16)
#     print(f"高维特征（调整后）大小: {high_to_low.shape}")  # 应为 (32, 128, 32, 32)

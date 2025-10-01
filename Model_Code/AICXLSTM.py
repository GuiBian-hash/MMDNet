import torch
import torch.nn as nn

from Model_Code import VisionLSTM
class Mutil_Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Mutil_Conv, self).__init__()
        self.batch_norm = nn.BatchNorm2d(in_channels)
        self.conv_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, dilation=1,groups=in_channels, bias=False),
            nn.Conv2d(in_channels, 2 * in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(2 * in_channels),
        )
        self.conv_5x5 = nn.Sequential(
            nn.Conv2d(in_channels,in_channels,kernel_size=3, stride=2, padding=2, dilation=2,groups=in_channels),
            nn.Conv2d(in_channels, 2 * in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(2 * in_channels),
        )
        self.conv_7x7 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=3, dilation=3,groups=in_channels, bias=False),
            nn.Conv2d(in_channels, 2 * in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(2 * in_channels),
        )
        self.conv = nn.Conv2d(6 * in_channels, out_channels, kernel_size=1)
        self.gelu = nn.GELU()

    def forward(self, x):
        x_batch = self.batch_norm(x)
        x1 = self.conv_1x1(x_batch)
        x1 = self.pool(x1)
        x2 = self.conv_3x3(x_batch)
        x3 = self.conv_5x5(x_batch)
        x4 = self.conv_7x7(x_batch)
        x_conv = torch.cat((x2, x3,x4), dim=1)
        x_conv = self.conv(x_conv)
        x_conv = x1 + x_conv
        x_conv = self.gelu(x_conv)


        return x_conv
class Upsample(nn.Module):
    def __init__(self,input_channels,output_channels):
        super(Upsample, self).__init__()
        self.deep_conv = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, groups=input_channels),  # 深度可分离卷积
            nn.Conv2d(input_channels, output_channels, kernel_size=1),  # 逐点卷积降维
            nn.BatchNorm2d(output_channels),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # 上采样
        )
    def forward(self,x):
        x = self.deep_conv(x)
        return x
class AICXLSTM1(nn.Module):
    def __init__(self, in_channels, out_channels,input_shape,patch_size):
        super(AICXLSTM1, self).__init__()
        self.conv1 = Mutil_Conv(in_channels,out_channels)
        self.conv2 = Mutil_Conv(out_channels, out_channels*2)
        self.conv3 = Mutil_Conv(out_channels*2, out_channels*4)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=2)
        self.xlstm = VisionLSTM(dim=256, depth=1,input_shape=input_shape,patch_size=patch_size)
        self.output_channels = out_channels
        self.upsample1 = Upsample(out_channels * 8, out_channels * 4)
        self.upsample2 = Upsample(out_channels*4,out_channels*2)
        self.upsample3 = Upsample(out_channels*2,out_channels)
    def forward(self, x):
        x_conv = self.conv1(x)
        x_1 = self.conv2(x_conv)
        x_2 = self.conv3(x_1)
        x = self.xlstm(x)
        x = self.upsample1(x)+x_2
        x = self.upsample2(x)+x_1
        x = self.upsample3(x)+x_conv


        return x
class AICXLSTM2(nn.Module):
    def __init__(self, in_channels, out_channels,input_shape,patch_size):
        super(AICXLSTM2, self).__init__()
        self.conv1 = Mutil_Conv(in_channels, out_channels)
        self.conv2 = Mutil_Conv(out_channels, out_channels * 2)
        self.xlstm = VisionLSTM(dim=256, depth=1, input_shape=input_shape, patch_size=patch_size)
        self.output_channels = out_channels

        self.upsample1 = Upsample(out_channels * 4, out_channels * 2)
        self.upsample2 = Upsample(out_channels * 2, out_channels)

    def forward(self, x):
        x_conv = self.conv1(x)
        x_1 = self.conv2(x_conv)
        a, b, c, d = x.shape
        x = self.xlstm(x)
        x = self.upsample1(x) + x_1
        x = self.upsample2(x) + x_conv


        return x
class AICXLSTM3(nn.Module):
    def __init__(self, in_channels, out_channels,input_shape,patch_size):
        super(AICXLSTM3, self).__init__()
        self.conv1 = Mutil_Conv(in_channels, out_channels)
        self.xlstm = VisionLSTM(dim=256, depth=1, input_shape=input_shape, patch_size=patch_size)
        self.output_channels = out_channels
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsample1 = Upsample(out_channels * 2, out_channels)

    def forward(self, x):
        x_conv = self.conv1(x)
        x = self.xlstm(x)
        x = self.upsample1(x) + x_conv


        return x
class AICXLSTM4(nn.Module):
    def __init__(self, in_channels, out_channels,input_shape,patch_size):
        super(AICXLSTM4, self).__init__()
        # Step 1: Batch Normalization
        self.conv1 = Mutil_Conv(in_channels, out_channels)
        self.xlstm = VisionLSTM(dim=256, depth=1, input_shape=input_shape, patch_size=patch_size)

    def forward(self, x):
        x_conv = self.conv1(x)
        x = self.xlstm(x)
        x = x + x_conv


        return x

# 示例使用
# 输入为 1x64x128x128 的特征图
# input_tensor = torch.randn(1, 64, 128, 128)
# model = AICXLSTM(in_channels=64, out_channels=128)
# output = model(input_tensor)
#
# # 输出特征图的尺寸: 1x128x64x64
# print("Output shape:", output.shape)

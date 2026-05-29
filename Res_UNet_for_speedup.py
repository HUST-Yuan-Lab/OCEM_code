import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def pixel_shuffle(tensor, scale_factor):
    """
    自定义 pixel_shuffle：只在 height 方向上做上采样
    输入 tensor.shape = (B, C*scale, H, W)
    输出 shape = (B, C, H*scale, W)
    """
    B, Cmul, H, W = tensor.shape
    C = Cmul // scale_factor
    # reshape + permute 来重排像素
    x = tensor.view(B, C, scale_factor, 1, H, W)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    return x.view(B, C, H * scale_factor, W)

def convt2x2(in_ch, out_ch):
    return nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2, bias=False)

def conv1x1(in_ch, out_ch):
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=True)

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=True),
            nn.Dropout2d(dropout),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)

class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch, ch, 3),
            nn.InstanceNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch, ch, 3),
            nn.InstanceNorm2d(ch),
        )
    def forward(self, x):
        return x + self.block(x)

# class PostResidual(nn.Module):
#     """pixel_shuffle 后的残差细化块"""
#     def __init__(self, ch):
#         super().__init__()
#         self.block = nn.Sequential(
#             nn.Conv2d(ch, ch, 3, padding=1, bias=True),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(ch, ch, 3, padding=1, bias=True),
#         )
#     def forward(self, x):
#         return x + self.block(x)

class SDUnet_start(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.):
        super(SDUnet_start, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, padding=1, bias=True),
            nn.ReLU(inplace=True))
    def forward(self, x):
        x = self.conv(x)
        return x

class ResUNet_PixelShuffle(nn.Module):
    def __init__(self,
                 n_channel_in=12,
                 n_channel_out=5,
                 n_filter_base=48,
                 scale_factor=4,
                 post_blocks=2):
        """
        输入: (B, n_channel_in, H, W)
        输出: (B, n_channel_out, H*scale_factor, W)
        """
        super().__init__()
        self.scale = scale_factor

        # ----- encoder -----
        self.conv1 = DoubleConv(n_channel_in,   n_filter_base)
        self.res1  = ResidualBlock(n_filter_base)
        self.conv2 = DoubleConv(n_filter_base, n_filter_base*2)
        self.res2  = ResidualBlock(n_filter_base*2)
        self.conv3 = DoubleConv(n_filter_base*2, n_filter_base*4)
        self.res3  = ResidualBlock(n_filter_base*4)
        self.conv4 = DoubleConv(n_filter_base*4, n_filter_base*8)
        self.res4  = ResidualBlock(n_filter_base*8)
        self.conv5 = DoubleConv(n_filter_base*8, n_filter_base*16, dropout=0.5)
        self.res5  = ResidualBlock(n_filter_base*16)

        # ----- decoder -----
        self.up1 = convt2x2(n_filter_base*16, n_filter_base*8)
        self.conv6 = DoubleConv(n_filter_base*16, n_filter_base*8)
        self.res6  = ResidualBlock(n_filter_base*8)

        self.up2 = convt2x2(n_filter_base*8, n_filter_base*4)
        self.conv7 = DoubleConv(n_filter_base*8, n_filter_base*4)
        self.res7  = ResidualBlock(n_filter_base*4)

        self.up3 = convt2x2(n_filter_base*4, n_filter_base*2)
        self.conv8 = DoubleConv(n_filter_base*4, n_filter_base*2)
        self.res8  = ResidualBlock(n_filter_base*2)

        self.up4 = convt2x2(n_filter_base*2, n_filter_base*1)
        self.conv9 = DoubleConv(n_filter_base*2, n_filter_base*1)
        self.res9  = ResidualBlock(n_filter_base*1)

        # 最终映射到 n_channel_out * scale，以便 pixel_shuffle
        self.conv10 = conv1x1(n_filter_base*1, n_channel_out * scale_factor)
        self.conv11 = conv1x1(n_filter_base * 1, n_channel_out)
        self.conv0 = SDUnet_start(n_filter_base // scale_factor, n_filter_base)
        # pixel_shuffle 后的细化残差块
        # self.post_res = nn.Sequential(
        #     *[PostResidual(n_channel_out) for _ in range(post_blocks)]
        # )

        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        # ---- encoder ----
        x1 = self.res1(self.conv1(x))
        x2 = self.res2(self.conv2(self.maxpool(x1)))
        x3 = self.res3(self.conv3(self.maxpool(x2)))
        x4 = self.res4(self.conv4(self.maxpool(x3)))
        x5 = self.res5(self.conv5(self.maxpool(x4)))

        # ---- decoder ----
        u1 = self.up1(x5)
        u1 = torch.cat([u1, x4], dim=1)
        u1 = self.res6(self.conv6(u1))

        u2 = self.up2(u1)
        u2 = torch.cat([u2, x3], dim=1)
        u2 = self.res7(self.conv7(u2))

        u3 = self.up3(u2)
        u3 = torch.cat([u3, x2], dim=1)
        u3 = self.res8(self.conv8(u3))

        u4 = self.up4(u3)
        u4 = torch.cat([u4, x1], dim=1)
        u4 = self.res9(self.conv9(u4))

        # 映射到 (B, n_channel_out*scale, H, W)
        # out = self.conv10(u4)
        # 自定义 pixel shuffle → (B, n_channel_out, H*scale, W)
        out = pixel_shuffle(u4, scale_factor=self.scale)
        out = self.conv0(out)
        out = self.conv11(out)
        # 后处理残差细化
        # out = self.post_res(out)
        return out

# ------------------------
# 测试 shape
# if __name__ == "__main__":
#     from torchinfo import summary
#     net = ResUNet_PixelShuffle(
#         n_channel_in=12,
#         n_channel_out=5,
#         n_filter_base=64,
#         scale_factor=4,
#         post_blocks=2
#     )
#     x = torch.randn(1, 12, 64, 256)
#     y = net(x)
#     print("input shape:", x.shape)
#     print("output shape:", y.shape)  # 应为 (1,5,256,256)
#     summary(net, input_size=(1, 12, 64, 256), col_names=("input_size", "output_size", "num_params", "mult_adds"),
#             depth=4)

def measure_gpu_speed(model, input_tensor, runs=100, warmup=10):
    assert torch.cuda.is_available(), "需要可用的 CUDA 设备"
    device = torch.device('cuda')
    model.to(device).eval()
    x = input_tensor.to(device)

    # 预热
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)

    # CUDA 事件计时
    starter = torch.cuda.Event(enable_timing=True)
    ender   = torch.cuda.Event(enable_timing=True)
    times = []

    with torch.no_grad():
        for _ in range(runs):
            starter.record()
            _ = model(x)
            ender.record()
            torch.cuda.synchronize()
            times.append(starter.elapsed_time(ender))  # ms

    return sum(times) / len(times)

if __name__ == "__main__":
    from torchinfo import summary
    net = ResUNet_PixelShuffle(
        n_channel_in=12,
        n_channel_out=5,
        n_filter_base=64,
        scale_factor=4,
        post_blocks=2
    )
    x = torch.randn(1, 12, 64, 256)
    y = net(x)
    print("input shape:", x.shape)
    print("output shape:", y.shape)  # 应为 (1,5,256,256)
    summary(net, input_size=(1, 12, 64, 256), col_names=("input_size", "output_size", "num_params", "mult_adds"),
            depth=4)
    avg_gpu_ms = measure_gpu_speed(net, x, runs=100, warmup=10)
    print(f"Average inference time on GPU: {avg_gpu_ms:.2f} ms over 100 runs")

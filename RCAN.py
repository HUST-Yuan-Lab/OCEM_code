import torch
from torch import nn
import torch.nn.functional as F

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="true"


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel=32, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        # self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                # nn.Conv3d(channel, channel // reduction, kernel_size=(1, 1, 1), padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                # nn.Conv3d(channel // reduction, channel, kernel_size=(1, 1, 1), padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):

    def __init__(self, channel=32):
        super(RCAB, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1),
            # nn.Conv3d(channel, channel, (3, 3, 3), 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 3, 1, 1),
            # nn.Conv3d(channel, channel, (3, 3, 3), 1, 1),
        )
        self.ca = CALayer(channel)

    def forward(self, x):
        y1 = self.conv(x)
        y2 = self.ca(y1)
        return x + y2


## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, B_RCAB, channel=32):
        super(ResidualGroup, self).__init__()
        module_body = []
        for i in range(B_RCAB):
            module_body.append(RCAB(channel))
        module_body.append(nn.Conv2d(channel, channel, 3, 1, 1))
        # module_body.append(nn.Conv3d(channel, channel, (3, 3, 3), 1, 1))
        self.body = nn.Sequential(*module_body)

    def forward(self, x):
        res = self.body(x)

        res += x
        return res


class SDUnet_start(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.):
        super(SDUnet_start, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, padding=1, bias=True),
            nn.ReLU(inplace=True))
    def forward(self, x):
        x = self.conv(x)
        return x


def conv1x1(in_channels, out_channels):
    """1x1 convolution"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=True)


def pixel_shuffle(tensor, scale_factor):
    num, ch, height, width = tensor.shape
    new_ch = ch // (scale_factor)
    new_height = height * scale_factor
    input_view = tensor.contiguous().view(
        num, new_ch, scale_factor, 1, height, width)

    shuffle_out = input_view.permute(0, 1, 4, 2, 5, 3).contiguous()

    return shuffle_out.view(num, new_ch, new_height, width)


class RCAN(nn.Module):
    def __init__(self, G_RG, channel=32):
        super(RCAN, self).__init__()
        self.conv0 = SDUnet_start(channel // 4, channel)
        self.conv10 = conv1x1(channel * 1, 5)

        modules_head = [nn.Conv2d(12, channel, 3, 1, 1)]
        # modules_head = [nn.Conv3d(1, channel, (3, 3, 3), 1, 1)]
        modules_body = []
        for i in range(G_RG):
            modules_body.append(ResidualGroup(5, channel))  #RCAB的个数4
        modules_body.append(nn.Conv2d(channel, channel, 3, 1, 1))
        # modules_body.append(nn.Conv3d(channel, channel, (3, 3, 3), 1, 1))
        # modules_tail = [nn.Conv2d(channel, 3, 3, 1, 1)]

        # modules_tail = [nn.ConvTranspose2d(channel, channel, 2, 2, 0)]

        modules_tail = []

        modules_tail.append(nn.Conv2d(channel, 5, 3, 1, 1))
        # modules_tail.append(nn.Conv3d(32, 1, (3, 3, 3), 1, 1))
        # modules_tail.append(nn.Sigmoid())

        self.head = nn.Sequential(*modules_head)

        self.body = nn.Sequential(*modules_body)

        self.tail = nn.Sequential(*modules_tail)

        self.conv_before_upsample = nn.Sequential(nn.Conv2d(channel, channel, 3, 1, 1),
                                                  nn.LeakyReLU(inplace=True))
        self.conv1_upsample = nn.Conv2d(channel, channel * 2, 3, 1, 1)
        self.conv2_upsample = nn.Conv2d(channel, channel * 2, 3, 1, 1)
        self.conv_last = nn.Conv2d(channel, 5, 3, 1, 1)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x
        # res = self.tail(res)
        # res = pixel_shuffle(res, scale_factor=4)
        # res = self.conv0(res)
        # res = self.conv10(res)
        out = self.conv_before_upsample(res)
        out = self.conv1_upsample(out)
        out = pixel_shuffle(out, scale_factor=2)
        out = self.conv2_upsample(out)
        out = pixel_shuffle(out, scale_factor=2)
        res = self.conv_last(out)
        return res

#
# if __name__ == '__main__':
#     from torchinfo import summary
#     # from thop import profile
#     x = torch.rand(1, 12, 64, 256)
#     net = RCAN(5, 64) #RG的个数
#     print(x.shape)
#     print(net(x).shape)
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

if __name__ == '__main__':
    from torchinfo import summary
    # 模拟输入，shape=(1, 12, 64, 256)
    x = torch.rand(1, 12, 64, 256)

    # 实例化模型
    net = RCAN(5, 64)  # RG 的个数

    # 打印形状验证
    print("Input shape :", x.shape)
    print("Output shape:", net(x).shape)

    # 打印模型信息（含 mult-adds）
    summary(net,
            input_size=(1, 12, 64, 256),
            col_names=("input_size", "output_size", "num_params", "mult_adds"),
            depth=4)

    # 测量 GPU 推理速度
    avg_gpu_ms = measure_gpu_speed(net, x, runs=100, warmup=10)
    print(f"Average inference time on GPU: {avg_gpu_ms:.2f} ms over 100 runs")
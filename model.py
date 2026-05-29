import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import numpy as np
import torch.nn.init as init
#import torchvision.models as models
from collections import OrderedDict
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# 基于pytorch实现CARE网络结构
def conv3x3(in_channels, out_channels):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)


def conv6x6():
    pixel_shift = torch.nn.Conv2d(6, 6, kernel_size=(9, 1), bias=0, padding=[4,0])
    ones = np.zeros((6,6,9,1))
    ones[0,0,:,0] = np.array([0, 0, 0.5, 0.5, 0, 0, 0, 0, 0])
    ones[1,1,:,0] = np.array([0, 0, 0, 0.75, 0.25, 0, 0, 0, 0])
    ones[2,2,:,0] = np.array([0, 0, 0, 0, 1., 0, 0, 0, 0])
    ones[3,3,:,0] = np.array([0, 0, 0, 0.5, 0.5, 0, 0, 0, 0])
    ones[4,4,:,0] = np.array([0, 0.25, 0.75, 0, 0, 0, 0, 0, 0])
    ones[5,5,:,0] = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0])

    # weight = torch.Tensor(ones[:,i:i+1,:,:])
    weight = torch.Tensor(ones[:, :, :, :])
    pixel_shift.weight = torch.nn.Parameter(weight, requires_grad=False)
    return pixel_shift


def conv12x12():
    pixel_shift = torch.nn.Conv2d(12, 12, kernel_size=(9, 1), bias=0, padding=[4,0])
    ones = np.zeros((12,12,9,1))
    ones[0,0,:,0] = np.array([0, 0, 0.5, 0.5, 0, 0, 0, 0, 0])
    ones[1,1,:,0] = np.array([0, 0, 0, 0.75, 0.25, 0, 0, 0, 0])
    ones[2,2,:,0] = np.array([0, 0, 0, 0, 1., 0, 0, 0, 0])
    ones[3,3,:,0] = np.array([0, 0, 0, 0.5, 0.5, 0, 0, 0, 0])
    ones[4,4,:,0] = np.array([0, 0.25, 0.75, 0, 0, 0, 0, 0, 0])
    ones[5,5,:,0] = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0])
    ones[6,6,:,0] = np.array([0, 0, 0.5, 0.5, 0, 0, 0, 0, 0])
    ones[7,7,:,0] = np.array([0, 0, 0, 0.75, 0.25, 0, 0, 0, 0])
    ones[8,8,:,0] = np.array([0, 0, 0, 0, 1., 0, 0, 0, 0])
    ones[9,9,:,0] = np.array([0, 0, 0, 0.5, 0.5, 0, 0, 0, 0])
    ones[10,10,:,0] = np.array([0, 0.25, 0.75, 0, 0, 0, 0, 0, 0])
    ones[11,11,:,0] = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0])
    # weight = torch.Tensor(ones[:,i:i+1,:,:])
    weight = torch.Tensor(ones[:, :, :, :])
    pixel_shift.weight = torch.nn.Parameter(weight, requires_grad=False)
    return pixel_shift


def conv1x1(in_channels, out_channels):
    """1x1 convolution"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=True)


def convt2x1(in_channels, out_channels):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(2, 1), stride=(2, 1), bias=False)


def convt2x2(in_channels, out_channels):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False)


def convt3x1(in_channels, out_channels):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(3, 1), stride=(3, 1), padding=0, bias=False)

def pixel_shuffle(tensor, scale_factor):
    num, ch, height, width = tensor.shape
    new_ch = ch // (scale_factor)
    new_height = height * scale_factor
    input_view = tensor.contiguous().view(
        num, new_ch, scale_factor, 1, height, width)

    shuffle_out = input_view.permute(0, 1, 4, 2, 5, 3).contiguous()

    return shuffle_out.view(num, new_ch, new_height, width)


class Conv2d1(nn.Conv2d):
    '''
    shape:
    input: (Batch_size, in_channels, H_in, W_in)
    output: ((Batch_size, out_channels, H_out, W_out))
    '''

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d1, self).__init__(in_channels, out_channels, kernel_size, stride,
                                     padding, dilation, groups, bias)
    def forward(self, x):
        weight = self.weight  # self.weight 的shape为(out_channels, in_channels, kernel_size_w, kernel_size_h)
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


def SeparableConv2d_ws(inp, oup):      # 其它层的depthwise convolution：conv3*3+BN+ReLU+conv1*1+BN+ReLU
    return nn.Sequential(
        Conv2d1(inp, inp, 3, 1, 1, bias=True),
        nn.Conv2d(inp, oup, 1, 1, 0, bias=True))


class SeparableConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.):
        super(SeparableConv2d, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, 3, 1, 1, bias=True),
                nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=True),
                nn.Dropout2d(dropout, inplace=False))
    def forward(self, x):
        x = self.conv(x)
        return x


class SDUnet_start(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.):
        super(SDUnet_start, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, padding=1, bias=True),
            nn.ReLU(inplace=True))
    def forward(self, x):
        x = self.conv(x)
        return x


class SDUnet_block(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.):
        super(SDUnet_block, self).__init__()
        self.conv = nn.Sequential(
                SeparableConv2d_ws(in_ch, out_ch),    # in_ch、out_ch是通道数
                nn.ReLU(inplace=True),
                SeparableConv2d_ws(out_ch, out_ch),
                nn.ReLU(inplace=True))
    def forward(self, x):
        x = self.conv(x)
        return x


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.):
        super(DoubleConv, self).__init__()
        self.conv1 = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=True),    # in_ch、out_ch是通道数
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout, inplace=False),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=True),
                nn.ReLU(inplace=True))
        self.in_ch = in_ch
        self.out_ch = out_ch
    def forward(self, x):
        [num, w, h, d] = np.shape(x)
        y = torch.cat((x, torch.zeros([num, (self.out_ch - self.in_ch), h, d]).cuda()), dim=1)
        x = self.conv1(x)+y
        return x

class DoubleConv1(nn.Module):
    def __init__(self, in_ch, out_ch,dropout=0.):
        super(DoubleConv1, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=True),    # in_ch、out_ch是通道数
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout, inplace=False),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=True),
                nn.ReLU(inplace=True))
        self.in_ch = in_ch
        self.out_ch = out_ch
    def forward(self, x):
        x = self.conv(x)
        return x


class LW_UNet(nn.Module):

    def __init__(self, n_channel_in=12, n_channel_out=5, n_filter_base=64,
                 residual=True, prob_out=False, eps_scale=0.001, scale=4):
        super().__init__()
        self.conv0 = SDUnet_start(n_filter_base//scale, n_filter_base)
        self.conv1 = DoubleConv1(n_channel_in, n_filter_base)
        self.conv2 = DoubleConv1(n_filter_base, n_filter_base * 2)
        self.conv3 = DoubleConv1(n_filter_base * 2, n_filter_base * 4)
        self.conv4 = DoubleConv1(n_filter_base * 4, n_filter_base * 8)
        self.conv5 = DoubleConv1(n_filter_base * 8, n_filter_base * 16, dropout=0.5)

        self.up1 = convt2x2(n_filter_base * 16, n_filter_base * 8)
        self.conv6 = DoubleConv1(n_filter_base * 16, n_filter_base * 8)

        self.up2 = convt2x2(n_filter_base * 8, n_filter_base * 4)
        self.conv7 = DoubleConv1(n_filter_base * 8, n_filter_base * 4)

        self.up3 = convt2x2(n_filter_base * 4, n_filter_base * 2)
        self.conv8 = DoubleConv1(n_filter_base * 4, n_filter_base * 2)

        self.up4 = convt2x2(n_filter_base * 2, n_filter_base * 1)
        self.conv9 = DoubleConv1(n_filter_base * 2, n_filter_base * 1)

        self.conv10 = conv1x1(n_filter_base * 1, n_channel_out)

        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(scale_factor=2)
        self.n_channel_in = n_channel_in
        self.residual = residual
        self.prob_out = prob_out
        self.eps_scale = eps_scale
        self.scale = scale

    def forward(self, x):
        out = self.conv1(x)
        skip1 = out
        out = self.maxpool(out)

        out = self.conv2(out)
        skip2 = out
        out = self.maxpool(out)

        out = self.conv3(out)
        skip3 = out
        out = self.maxpool(out)

        out = self.conv4(out)
        skip4 = out
        out = self.maxpool(out)

        out = self.conv5(out)
        #out = channel_shuffle(out, 2)
        out = self.up1(out)
        out = torch.cat((skip4, out), dim=1)

        out = self.conv6(out)
        out = self.up2(out)
        out = torch.cat((skip3, out), dim=1)
        out = self.conv7(out)
        out = self.up3(out)
        out = torch.cat((skip2, out), dim=1)

        out = self.conv8(out)
        out = self.up4(out)
        out = torch.cat((skip1, out), dim=1)

        out = self.conv9(out)
        out = pixel_shuffle(out, scale_factor=self.scale)
        out = self.conv0(out)
        out = self.conv10(out)
        return out


# class LW_UNet(nn.Module):
#
#     def __init__(self, n_channel_in=12, n_channel_out=5, n_filter_base=64,
#                  residual=True, prob_out=False, eps_scale=0.001, scale=4):
#         super().__init__()
#         self.conv0 = SDUnet_start(n_filter_base//scale, n_filter_base)
#         self.conv1 = DoubleConv1(n_channel_in, n_filter_base)
#         self.conv2 = DoubleConv1(n_filter_base, n_filter_base * 2)
#         self.conv3 = DoubleConv1(n_filter_base * 2, n_filter_base * 4)
#         self.conv4 = DoubleConv1(n_filter_base * 4, n_filter_base * 8)
#         self.conv5 = DoubleConv1(n_filter_base * 8, n_filter_base * 16)
#         self.conv6 = DoubleConv1(n_filter_base * 16, n_filter_base * 32, dropout=0.5)
#         self.up1 = convt2x2(n_filter_base * 32, n_filter_base * 16)
#         self.conv7 = DoubleConv1(n_filter_base * 32, n_filter_base * 16)
#
#         self.up2 = convt2x2(n_filter_base * 16, n_filter_base * 8)
#         self.conv8 = DoubleConv1(n_filter_base * 16, n_filter_base * 8)
#
#         self.up3 = convt2x2(n_filter_base * 8, n_filter_base * 4)
#         self.conv9 = DoubleConv1(n_filter_base * 8, n_filter_base * 4)
#
#         self.up4 = convt2x2(n_filter_base * 4, n_filter_base * 2)
#         self.conv10 = DoubleConv1(n_filter_base * 4, n_filter_base * 2)
#
#         self.up5 = convt2x2(n_filter_base * 2, n_filter_base * 1)
#         self.conv11 = DoubleConv1(n_filter_base * 2, n_filter_base * 1)
#
#         self.conv12 = conv1x1(n_filter_base * 1, n_channel_out)
#
#         self.maxpool = nn.MaxPool2d(kernel_size=2)
#         self.relu = nn.ReLU(inplace=True)
#         self.upsample = nn.Upsample(scale_factor=2)
#         self.n_channel_in = n_channel_in
#         self.residual = residual
#         self.prob_out = prob_out
#         self.eps_scale = eps_scale
#         self.scale = scale
#
#     def forward(self, x):
#         out = self.conv1(x)
#         skip1 = out
#         out = self.maxpool(out)
#
#         out = self.conv2(out)
#         skip2 = out
#         out = self.maxpool(out)
#
#         out = self.conv3(out)
#         skip3 = out
#         out = self.maxpool(out)
#
#         out = self.conv4(out)
#         skip4 = out
#         out = self.maxpool(out)
#
#         out = self.conv5(out)
#         skip5 = out
#         out = self.maxpool(out)
#
#         out = self.conv6(out)
#         out = self.up1(out)
#         out = torch.cat((skip5, out), dim=1)
#
#         out = self.conv7(out)
#         out = self.up2(out)
#         out = torch.cat((skip4, out), dim=1)
#
#         out = self.conv8(out)
#         out = self.up3(out)
#         out = torch.cat((skip3, out), dim=1)
#         out = self.conv9(out)
#         out = self.up4(out)
#         out = torch.cat((skip2, out), dim=1)
#
#         out = self.conv10(out)
#         out = self.up5(out)
#         out = torch.cat((skip1, out), dim=1)
#
#         out = self.conv11(out)
#         out = pixel_shuffle(out, scale_factor=self.scale)
#         out = self.conv0(out)
#         out = self.conv12(out)
#         return out



class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # 下采样的gating signal 卷积
        g1 = self.W_g(g)
        # 上采样的 l 卷积
        x1 = self.W_x(x)
        # concat + relu
        psi = self.relu(x1+g1)
        # channel 减为1，并Sigmoid,得到权重矩阵
        psi = self.psi(psi)
        # 返回加权的 x
        x1 = x * psi
        return x1

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(CBAM, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.Conv2d(2, 1, 3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))

        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        out = self.sigmoid(out)
        avg_out = torch.mean(out, dim=1, keepdim=True)
        max_out, _ = torch.max(out, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        x = x * self.sigmoid(out)
        return x


class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()
        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None
        self.relu = nn.ReLU(inplace=True)
        rep = []
        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters
        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, bias=False))
            rep.append(nn.BatchNorm2d(filters))
        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)
        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)
        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
        x += skip
        return x


class UNet(nn.Module):

    def __init__(self, n_channel_in=1, n_channel_out=5, n_filter_base=64,
                 residual=True, prob_out=False, eps_scale=0.001):
        super().__init__()
        self.conv1 = DoubleConv1(n_channel_in, n_filter_base)
        self.conv2 = DoubleConv1(n_filter_base, n_filter_base * 2)
        self.conv3 = DoubleConv1(n_filter_base * 2, n_filter_base * 4)
        self.conv4 = DoubleConv1(n_filter_base * 4, n_filter_base * 8)
        self.conv5 = DoubleConv1(n_filter_base * 8, n_filter_base * 16, dropout=0.5)

        self.up1 = convt2x2(n_filter_base * 16, n_filter_base * 8)
        self.conv6 = DoubleConv1(n_filter_base * 16, n_filter_base * 8)

        self.up2 = convt2x2(n_filter_base * 8, n_filter_base * 4)
        self.conv7 = DoubleConv1(n_filter_base * 8, n_filter_base * 4)

        self.up3 = convt2x2(n_filter_base * 4, n_filter_base * 2)
        self.conv8 = DoubleConv1(n_filter_base * 4, n_filter_base * 2)

        self.up4 = convt2x2(n_filter_base * 2, n_filter_base * 1)
        self.conv9 = DoubleConv1(n_filter_base * 2, n_filter_base * 1)

        self.conv10 = conv1x1(n_filter_base * 1, n_channel_out)

        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(scale_factor=2)
        self.n_channel_in = n_channel_in
        self.residual = residual
        self.prob_out = prob_out
        self.eps_scale = eps_scale

    def forward(self, x):
        # 有无横向加速
        #x = F.interpolate(x, scale_factor=(4, 1), mode='bilinear', align_corners=True)
        out = self.conv1(x)
        skip1 = out
        out = self.maxpool(out)

        out = self.conv2(out)
        skip2 = out
        out = self.maxpool(out)

        out = self.conv3(out)
        skip3 = out
        out = self.maxpool(out)

        out = self.conv4(out)
        skip4 = out
        out = self.maxpool(out)

        out = self.conv5(out)
        out = self.up1(out)
        out = torch.cat((skip4, out), dim=1)

        out = self.conv6(out)
        out = self.up2(out)
        out = torch.cat((skip3, out), dim=1)

        out = self.conv7(out)
        out = self.up3(out)
        out = torch.cat((skip2, out), dim=1)

        out = self.conv8(out)
        out = self.up4(out)
        out = torch.cat((skip1, out), dim=1)

        out = self.conv9(out)
        out = self.conv10(out)
        return out


class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()

        def conv_bn(inp, oup, stride):    # 第一层传统的卷积：conv3*3+BN+ReLU
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def convt2x2(in_channels, out_channels):
            return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False)

        def conv_dw(inp, oup, stride):      # 其它层的depthwise convolution：conv3*3+BN+ReLU+conv1*1+BN+ReLU
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(6,  32, 2),   # 第一层传统的卷积
            conv_dw(32,  64, 1),   # 其它层depthwise convolution
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            #nn.AvgPool2d(7),
            nn.ConvTranspose2d(1024, 512, kernel_size=1, stride=2, bias=False),
            convt2x2(512, 256),
            convt2x2(256, 128),
            convt2x2(128, 64),
            convt2x2(64, 32),
            nn.Conv2d(32, 6, kernel_size=1, stride=1, bias=True)
       )
        self.fc = nn.Linear(1024, 1000)   # 全连接层
        self.shift = conv6x6()
    def forward(self, x):
        x = F.interpolate(x, scale_factor=(6, 1), mode='bilinear')
        x = self.shift(x)
        x = self.model(x)
        #x = x.view(-1, 1024)
        #x = self.fc(x)
        return x

class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)


class SqueezeNet(nn.Module):

    def __init__(self, version='1_1', num_classes=1000):
        super(SqueezeNet, self).__init__()
        self.num_classes = num_classes
        if version == '1_0':
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=7, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256),
            )
        elif version == '1_1':
            self.features = nn.Sequential(
                nn.Conv2d(6, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
                nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, bias=False),
                nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, bias=False),
                nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, bias=False),
                nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, bias=False),
                nn.Conv2d(32, 6, kernel_size=1, stride=1, bias=True)
            )
            self.shift = conv6x6()
        else:
            # FIXME: Is this needed? SqueezeNet should only be called from the
            # FIXME: squeezenet1_x() functions
            # FIXME: This checking is not done for the other models
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1_0 or 1_1 expected".format(version=version))

        # Final convolution is initialized differently from the rest
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


    def forward(self, x):
        out = F.interpolate(x, scale_factor=(6, 1), mode='bilinear')
        x = self.shift(out)
        x = self.features(x)
        #x = self.classifier(x)
        return x

class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """
    def __init__(self, num_classes=1000):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(6, 32, 3,2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32,64,3,bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        #do relu here

        self.block1=Block(64,128,2,2,start_with_relu=False,grow_first=True)
        self.block2=Block(128,256,2,2,start_with_relu=True,grow_first=True)
        self.block3=Block(256,728,2,2,start_with_relu=True,grow_first=True)

        self.block4=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block5=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block6=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block7=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block8=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block9=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block10=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block11=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block12=Block(728,1024,2,2,start_with_relu=True,grow_first=False)

        self.conv3 = SeparableConv2d(1024,1536,3,1,1)
        self.bn3 = nn.BatchNorm2d(1536)
        self.relu3 = nn.ReLU(inplace=True)

        #do relu here
        self.conv4 = SeparableConv2d(1536,2048,3,1,1)
        self.bn4 = nn.BatchNorm2d(2048)
        self.shift = conv6x6()
        self.fc = nn.Linear(2048, num_classes)
        self.convt1 = nn.ConvTranspose2d(2048, 1024, kernel_size=1, stride=2, bias=False)
        self.convt2 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2, bias=False)
        self.convt3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, bias=False)
        self.convt4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, bias=False)
        self.convt5 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, bias=False)
        self.convt6 = nn.Conv2d(64, 6, kernel_size=1, stride=1, bias=True)
        # ------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                #m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        # -----------------------------

    def forward(self, x):
        #out = F.interpolate(x, scale_factor=(6, 1), mode='bilinear')
        #x = self.shift(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.convt1(x)
        x = self.convt2(x)
        x = self.convt3(x)
        x = self.convt4(x)
        x = self.convt5(x)
        x = self.convt6(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, n_channel_in=3, n_channel_out=1, n_filter_base=64):
        super(Discriminator, self).__init__()
        self.conv1 = DoubleConv(n_channel_in, n_filter_base)
        self.conv2 = DoubleConv(n_filter_base, n_filter_base * 2)
        self.conv3 = DoubleConv(n_filter_base * 2, n_filter_base * 4)
        self.conv4 = DoubleConv(n_filter_base * 4, n_filter_base * 8)
        self.conv5 = DoubleConv(n_filter_base * 8, n_filter_base * 8)
        self.conv6 = DoubleConv(n_filter_base * 8, n_filter_base * 8)
        self.conv7 = DoubleConv(n_filter_base * 8, n_filter_base * 8)
        self.conv8 = DoubleConv(n_filter_base * 8, n_filter_base * 8)
        self.fc = nn.Sequential(
            nn.Linear(n_filter_base * 8 * 9, n_filter_base * 64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5, inplace=False),
            nn.Linear(n_filter_base * 64, n_filter_base * 16),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5, inplace=False),
            nn.Linear(n_filter_base * 16, n_channel_out),
            nn.Sigmoid()
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(3, 3))
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpool(out)

        out = self.conv2(out)
        out = self.maxpool(out)

        out = self.conv3(out)
        out = self.maxpool(out)

        out = self.conv4(out)
        out = self.maxpool(out)

        out = self.conv5(out)
        out = self.maxpool1(out)

        out = self.conv6(out)
        out = self.maxpool(out)

        out = self.conv7(out)
        out = self.maxpool(out)

        out = self.avgpool(out)

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def summary_test(model, input_size, batch_size=-1, device="cuda"):

    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            if len(input) != 0:
                summary[m_key]["input_shape"] = list(input[0].size())
                summary[m_key]["input_shape"][0] = batch_size
            else:
                summary[m_key]["input_shape"] = input
            #summary[m_key]["input_shape"] = list(input[0].size())
            #summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    # print(type(x[0]))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    print("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
    print(line_new)
    print("================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        print(line_new)

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    print("================================================================")
    print("Total params: {0:,}".format(total_params))
    print("Trainable params: {0:,}".format(trainable_params))
    print("Non-trainable params: {0:,}".format(total_params - trainable_params))
    print("----------------------------------------------------------------")
    print("Input size (MB): %0.2f" % total_input_size)
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    print("Params size (MB): %0.2f" % total_params_size)
    print("Estimated Total Size (MB): %0.2f" % total_size)
    print("----------------------------------------------------------------")
    # return summary


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
    import torch
    x = torch.rand(1, 12, 64, 256)
    net = LW_UNet()
    print(x.shape)
    print(net(x).shape)
    summary(net, input_size=(1, 12, 64, 256), col_names=("input_size", "output_size", "num_params", "mult_adds"),
            depth=4)
    avg_gpu_ms = measure_gpu_speed(net, x, runs=100, warmup=10)
    print(f"Average inference time on GPU: {avg_gpu_ms:.2f} ms over 100 runs")







import torch
# from model import UNet, LW_UNet
# from URCANsub import URCANsub
# from RCAN import RCAN
from Res_UNet_for_speedup import ResUNet_PixelShuffle
import tifffile as tiff
import time
import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from image_split_and_merge import *

start = time.time()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 可视化0,1这两块GPU
device = torch.device('cuda:0')
# model = UNet(n_channel_in=12, n_channel_out=5, n_filter_base=64).to(device)
# ch = 1
speed = 4
depth = 5
scale = 65535
layers = range(3200, 3399 - depth + 1, depth - 1)
strips = range(35, 36, 1)
# array_range = range(2, 14)
# patch_size = 288
# stride = 144
patch_size = 256
stride = 128
# model = LW_UNet(n_channel_in=12, n_channel_out=depth, n_filter_base=64).to(device)
# model = RCAN(5, 64).to(device)
model = ResUNet_PixelShuffle(n_channel_in=12, n_channel_out=depth, n_filter_base=64, scale_factor=4).to(device)


model.load_state_dict(
    torch.load(r'F:\20250510thy1_ResUNet\train\depth5_4v_resunet_LiMo_2025_5_10_17\weights_99_nline.pth'))
model.eval()

file = r'I:\196617_thy1\test_datasets_CH1\ori_datasets_final\nline_4v'
file_write = os.path.join(r'F:\20250510thy1_ResUNet\test\depth%d_LiMo_%dv_resunet' % (depth, speed))
os.makedirs(file_write, exist_ok=True)


def get_image(ilayer, istrip):
    # 根据文件命名规则改
    image_name = os.path.join(file, 'nline_%05d_%02d.tif' % (ilayer, istrip))
    print(image_name)
    image = tiff.imread(image_name)
    # image = image[:, 0:1952]
    if ilayer % 2 == 1:
        dislocation = 0  # 肝脏4  细胞计数2  对比实验1
    if ilayer % 2 == 0:
        # 奇偶层之间是否存在上下翻转
        dislocation = 0
    # roi：行起点，列起点，行终点，列终点
    # 是否裁剪
    if image.ndim == 2:
        image = image[np.newaxis, :, :]
    image = image[:, dislocation:, :]
    print(image.shape)
    return image


def image2data(image):
    if image.ndim == 2:
        image = image[np.newaxis, :, :]
    # image_stack.shape (32, 7950, 1960)
    data, _, r, c, before1, before2, after1, after2 = image2patch(image, patch_size=patch_size, stride=stride, speed=speed)
    return data, r, c, before1, before2, after1, after2


class MyDataset(Dataset):
    def __init__(self, inputs, scale):
        self.input = inputs
        self.scale = scale

    def __len__(self):
        length, _, _, _ = self.input.shape
        return length

    def __getitem__(self, idx):
        return self.input[idx, :, :, :].astype('float32') / self.scale


def get_data(ilayer, istrip):
    # 2层6线重建5层
    # 上下两层恢复中间 or 中间一层恢复上下
    image_stack1 = get_image(ilayer, istrip)
    image_stack2 = get_image(ilayer + depth - 1, istrip)
    image_stack = np.vstack([image_stack1, image_stack2])
    image_stack1 = 0
    image_stack2 = 0
    # image_stack = image_stack1
    print('image_stack.shape', image_stack.shape)
    data, r, c, before1, before2, after1, after2 = image2data(image_stack)
    print('data.shape', data.shape)
    image_stack = 0
    data_loader = DataLoader(MyDataset(data, scale), batch_size=80)
    return data_loader, r, c, before1, before2, after1, after2


def image_pre(model, data, device):
    model.eval()
    with torch.no_grad():
        data_list = []
        for x in data:
            x_in = x.to(device)
            x_out = model(x_in)
            x_numpy = x_out.cpu().detach().numpy()
            data_list.append(x_numpy)
        data_pre = np.vstack(data_list)
        # x_in = data.to(device)
        # x_out = model(x_in)
        # x_numpy = x_out.cpu().detach().numpy()
        # data_pre = x_numpy
    return data_pre


def data2image(data, r, c, before1, before2, after1, after2):
    # data.shape (32, 5, 256, 1952)
    # 去边缘冗余拼接
    ch = data.shape[1]
    data[data >= 1] = 1
    data[data <= 0] = 0
    # 注意重建数据亮度
    data = data * scale
    image = patch2image(data, ch, r, c, before1, before2, after1, after2, patch_size=patch_size, stride=stride, speed=speed)
    print('image.shape', image.shape)
    # image.shape (5, 7950, 1952)
    return image


def output_stack():
    for ilayer in layers:
        for istrip in strips:
            data_loader, r, c, before1, before2, after1, after2 = get_data(ilayer, istrip)
            tstar = time.time()
            data_pre = image_pre(model, data_loader, device)
            tend = time.time()
            print('总:%s ' % (tend - tstar))
            print('data_pre.shape', data_pre.shape)
            image = data2image(data_pre, r, c, before1, before2, after1, after2)
            print('image_pre.shape', image.shape)
            # 保存重建结果路径
            # 只保存了四张
            # 上下两层恢复中间 or 中间一层恢复上下
            for k in range(depth):
                file_write1 = os.path.join(file_write, 'rec_our_%05d_%02d_%dlayer.tif' % (ilayer + k, istrip, depth))
                tiff.imwrite(file_write1, image[k, :, :])
            data_loader = 0
            data_pre = 0
            image = 0


output_stack()
end = time.time()
print('Time:{:4f}s'.format(end-start))
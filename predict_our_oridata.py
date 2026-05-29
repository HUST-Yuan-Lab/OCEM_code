import torch
# from model import UNet
from URCANsub import URCANsub
import tifffile as tiff
import time
import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from image_split_and_merge import *

start = time.time()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 可视化0,1这两块GPU
device = torch.device('cuda:0')
# model = UNet(n_channel_in=24, n_channel_out=5, n_filter_base=64).to(device)
model = URCANsub(2, 5, 64, 20).to(device)

ch = 1
speed = 1
depth = 5
scale = 2000
layers = range(1, 43 - depth, depth - 1)
strips = range(40, 52, 1)
array_range = range(2, 14)
patch_size = 288
stride = 144

model.load_state_dict(
    torch.load(r'H:\230701bingli\train_datasets_CH2\ours\unet_results_2023_7_11_14_nline\weights_2023_7_11_15_nline.pth'))
model.eval()

file_root = r'I:\230701病理\CH%d' % ch
file_write = os.path.join(r'H:\230701bingli\Rec_CH%d\Rec_our_CH%d_%dv_5layer' % (ch, speed, ch))
os.makedirs(file_write, exist_ok=True)


def get_image(ilayer, istrip):
    image_list = []
    # 每32线探测一个image_list
    for iarray in array_range:
        image_name = '230701_' + str('%05d' % ilayer) + '_' + str(istrip) + '_CH%d_' % ch + str(iarray) + '.tif'
        full_name = os.path.join(file_root, '230701_CH%d_' % ch + str('%05d' % ilayer), image_name)
        print(full_name)
        image = tiff.imread(full_name)
        # 奇偶层错位位移dislocation
        if ilayer % 2 == 1:
            dislocation = 0  # 肝脏4  细胞计数2  对比实验1
        if ilayer % 2 == 0:
            # 奇偶层之间是否存在上下翻转
            # image = np.flip(image, axis=0)
            dislocation = 0
        image = image.astype('float32')
        # 模拟四倍加速，配准同1倍速，真实四倍加速配准得根据实际图像特点改写
        if speed > 1:
            [w1, h1] = image.shape
            for i in range(0, w1 - speed + 1, speed):
                image[i // speed, :] = np.mean(image[i:i + speed, :], 0)
            image1 = image[:w1 // speed, :]
            image1 = np.clip(image1, 0, 65535)
        elif speed == 1:
            image1 = image
        image_list.append(image1)

    # 多线原速结果合成一个stack
    nimage = np.stack(image_list, axis=0)
    nimage = np.clip(nimage - 100, 0, 65535)
    image = nimage.astype('uint16')
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
                file_write1 = os.path.join(file_write, 'rec_our_%05d_%02d_CH%d.tif' % (ilayer + k, istrip, ch))
                tiff.imwrite(file_write1, image[k, :, :])
            data_loader = 0
            data_pre = 0
            image = 0


output_stack()
end = time.time()
print('Time:{:4f}s'.format(end-start))
# -*- coding: utf-8 -*-
import numpy as np
import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import tifffile as tiff

def load_data(file_name, depth, layer=199, strip=1, speed=4):
    # two layers restore 5 limo layers;
    tr_i = os.path.join(file_name, 'nline_%dv.npy'%speed)
    tr_o = os.path.join(file_name, 'LiMo.npy')              #拼接数据路径。tr_i指的是输入数据，tr_o是输出数据

    input0 = np.load(tr_i)
    output0 = np.load(tr_o)             #加载 .npy 文件中的输入数据 input0 和输出数据 output0
    print('inputs:', input0.shape)
    print('outputs:', output0.shape)
    input0 = input0
    output0 = output0
    [n, d, w, h] = input0.shape
    [n1, d, w, h] = output0.shape      #解析 input0 和 output0 的维度信息。n 和 n1 是样本数，d 是深度，w 和 h 分别是宽度和高度
    nums = n // layer // strip
    print('nums:', nums)
    nums1 = n1 // layer // strip
    print('nums1:', nums1)
    assert nums == nums1, "错误，数目应相等"  #计算每个分割的数量 nums，并确保 nums 和 nums1 一致

    input_list = []
    for j in range(strip):
        for i in range(0, strip * (layer - 1) + 1, strip):
            # 五维，layer*nums*d*w*h
            input_list.append(input0[(i + j) * nums:(i + j) * nums + nums, :, :, :])
    print('input_list shape:', np.array(input_list).shape)      #按照 strip 和 layer 参数，处理输入数据，将它们按块分割，并存入 input_list

    output_list = []
    for j in range(strip):
        for i in range(0, strip * (layer - 1) + 1, strip):
            output_list.append(output0[(i + j) * nums1:(i + j) * nums1 + nums1, :, :, :])
    print('output_list shape:', np.array(output_list).shape)

    in_list = []
    out_list = []
    input0 = 0
    output0 = 0
    # 上下两层恢复中间 or 中间一层恢复上下
    for j in range(strip):
        for i in range(0, layer - depth + 1, max(depth, 1)):
            temp = np.hstack(output_list[i + layer * j:i + layer * j + depth])
            # print('temp', temp.shape)
            out_list.append(temp)
            temp1 = np.hstack((input_list[i + layer * j], input_list[i + layer * j + depth - 1]))
            # temp1 = np.hstack((input_list[i + layer * j:i + layer * j + depth]))
            # print('temp1', temp1.shape)
            in_list.append(temp1)

    print('in_list shape:', np.array(in_list).shape)
    print('out_list shape:', np.array(out_list).shape)
    inputs = np.vstack(in_list)
    in_list = 0
    outputs = np.vstack(out_list)
    out_list = 0
    print('inputs:', inputs.shape)
    print('outputs:', outputs.shape)
    # 筛选有信号的
    indexs = []
    for m in range(outputs.shape[0]):
        output_index = outputs[m]
        input_index = inputs[m]
        if output_index.max() > 60 and output_index.mean() > 0.01 and input_index.max() > 0:
            indexs.append(m)
    input_choose = inputs[indexs]
    inputs = 0
    output_choose = outputs[indexs]
    outputs = 0
    print('input_choose shape:', input_choose.shape)
    print('output_choose shape:', output_choose.shape)
    return input_choose, output_choose
    # return input_choose.astype('float32') / 65535, output_choose.astype('float32') / 65535


class MyDataset(Dataset):
    def __init__(self, inputs, outputs, transform=''):
        self.transform = transform
        self.input = inputs
        self.output = outputs

    def __len__(self):
        length, _, _, _ = self.output.shape
        return length

    def __getitem__(self, idx):
        if self.input[idx, :, :, :].max()==0 or self.output[idx, :, :, :].max()==0:
            print(self.input[idx, :, :, :].max(), self.output[idx, :, :, :].max())
        # return self.input[idx, :, :, :].astype('float32') / self.input[idx, :, :, :].max(), self.output[idx, :, :, :].astype('float32') / self.output[idx, :, :, :].max()
        return self.input[idx, :, :, :].astype('float32') / 65535, self.output[idx, :, :, :].astype('float32') / 65535


# 改层数
def my_dataloader(file, ratio, depth, batchsize=4, shuffle=True):
    """ the sum of ratio must equal to 1"""
    inputs, outputs = load_data(file, depth)

    # 不打乱顺序
    print('inputs:', inputs.shape)
    print('outputs:', outputs.shape)

    # 打乱顺序
    # datasets = np.hstack((inputs, outputs))
    # print(datasets.shape)
    # np.random.shuffle(datasets)
    # inputs = datasets[:, 0:12]
    # outputs = datasets[:, 12:12+depth]
    # print('inputs:', inputs.shape)
    # print('outputs:', outputs.shape)

    # inputs[inputs <= 0] = 0
    # inputs[inputs >= 1] = 1
    # outputs[outputs <= 0] = 0
    # outputs[outputs >= 1] = 1
    length, _, _, _ = outputs.shape

    num_train = int(length * ratio[0])
    num_test = int(length * ratio[1])
    num_val = int(length * ratio[2])

    train_inputs = inputs[0:num_train, :, :, :]
    train_outputs = outputs[0:num_train, :, :, :]
    # 保存两张图看看数据集效果，改路径
    # nline or wide
    for i in range(10):
        os.makedirs(r'F:\20250510thy1_ResUNet\train\datalodercheck\our_%dlayer_4v/' % depth, exist_ok=True)
        tiff.imwrite(r'F:\20250510thy1_ResUNet\train\datalodercheck\our_%dlayer_4v\input_nline_%d.tif' % (depth,i), (train_inputs[100*i]).astype('uint16'))
        tiff.imwrite(r'F:\20250510thy1_ResUNet\train\datalodercheck\our_%dlayer_4v\output_limo_%d.tif' % (depth,i), (train_outputs[100*i]).astype('uint16'))
    print('done')

    test_inputs = inputs[num_train: num_train+num_test, :, :, :]
    test_outputs = outputs[num_train: num_train+num_test, :, :, :]

    val_inputs = inputs[num_train+num_test: num_train+num_test+num_val, :, :, :]
    val_outputs = outputs[num_train+num_test: num_train+num_test+num_val, :, :, :]
    train_dataloader = DataLoader(MyDataset(train_inputs, train_outputs), batch_size=batchsize, drop_last=True,
                                  shuffle=shuffle)
    test_dataloader = DataLoader(MyDataset(test_inputs, test_outputs), batch_size=batchsize, drop_last=False,
                                 shuffle=False)
    val_dataloader = DataLoader(MyDataset(val_inputs, val_outputs), batch_size=batchsize, drop_last=False,
                                shuffle=False)
    # test_dataloader = DataLoader(MyDataset(test_inputs, test_outputs), batch_size=batchsize, drop_last=False,
    #                              shuffle=shuffle)
    # val_dataloader = DataLoader(MyDataset(val_inputs, val_outputs), batch_size=batchsize, drop_last=False,
    #                             shuffle=shuffle)
    loader = {}
    loader['train'] = train_dataloader
    loader['val'] = val_dataloader
    loader['test'] = test_dataloader
    print('Train shape:', train_inputs.shape, train_outputs.shape)
    print('test shape:', test_inputs.shape, test_outputs.shape)
    print('val shape:', val_inputs.shape, val_outputs.shape)

    return loader

# if __name__ == '__main__':
#     # 加载数据集路径
#     depth = 5
#     data_dir = r'I:\196617_thy1\train_datasets_CH1\ori_datasets_final\Data'
#     loader = my_dataloader(data_dir, [0.8, 0.1, 0.1], depth, batchsize=4)
#     a = loader['train'].dataset.input.shape


# if __name__ == '__main__':
#     # 加载数据集路径
#     depth = 5
#     data_dir = r'I:\196617_thy1\train_datasets_CH1\ori_datasets_final\Data'
#     loader = my_dataloader(data_dir, [0.8, 0.1, 0.1], depth, batchsize=4)
#     a = loader['train'].dataset.input.shape

# def load_data(file_name, depth, layer=199, strip=1, speed=4):
#     # two layers restore 5 limo layers;
#     tr_i = os.path.join(file_name, 'nline_%dv.npy'%speed)
#     tr_o = os.path.join(file_name, 'LiMo.npy')              #拼接数据路径。tr_i指的是输入数据，tr_o是输出数据
#
#     input0 = np.load(tr_i)
#     output0 = np.load(tr_o)             #加载 .npy 文件中的输入数据 input0 和输出数据 output0
#     print('inputs:', input0.shape)
#     print('outputs:', output0.shape)
#     input0 = input0
#     output0 = output0
#     [n, d, w, h] = input0.shape
#     [n1, d, w, h] = output0.shape      #解析 input0 和 output0 的维度信息。n 和 n1 是样本数，d 是深度，w 和 h 分别是宽度和高度
#     nums = n // layer // strip
#     print('nums:', nums)
#     nums1 = n1 // layer // strip
#     print('nums1:', nums1)
#     assert nums == nums1, "错误，数目应相等"  #计算每个分割的数量 nums，并确保 nums 和 nums1 一致
#
#     input_list = []
#     for j in range(strip):
#         for i in range(0, strip * (layer - 1) + 1, strip):
#             # 五维，layer*nums*d*w*h
#             input_list.append(input0[(i + j) * nums:(i + j) * nums + nums, :, :, :])
#     print('input_list shape:', np.array(input_list).shape)      #按照 strip 和 layer 参数，处理输入数据，将它们按块分割，并存入 input_list
#
#     output_list = []
#     for j in range(strip):
#         for i in range(0, strip * (layer - 1) + 1, strip):
#             output_list.append(output0[(i + j) * nums1:(i + j) * nums1 + nums1, :, :, :])
#     print('output_list shape:', np.array(output_list).shape)
#
#     in_list = []
#     out_list = []
#     input0 = 0
#     output0 = 0
#     # 上下两层恢复中间 or 中间一层恢复上下
#     for j in range(strip):
#         for i in range(0, layer - depth + 1, max(depth - 1, 1)):
#             temp = np.hstack(output_list[i + layer * j:i + layer * j + depth])
#             # temp = np.concatenate(output_list[i + layer * j:i + layer * j + depth], axis=1)
#             # # print('temp', temp.shape)
#             out_list.append(temp)
#             temp1 = np.hstack((input_list[i + layer * j], input_list[i + layer * j + depth - 1]))
#             # temp1 = np.concatenate(
#             #     (input_list[i + layer * j], input_list[i + layer * j + depth - 1]),
#             #     axis=1
#             # )
#             # # temp1 = np.hstack((input_list[i + layer * j:i + layer * j + depth]))
#             # # print('temp1', temp1.shape)
#
#             in_list.append(temp1)
#
#     print('in_list shape:', np.array(in_list).shape)
#     print('out_list shape:', np.array(out_list).shape)
#     inputs = np.vstack(in_list)
#     in_list = 0
#     outputs = np.vstack(out_list)
#     out_list = 0
#     print('inputs:', inputs.shape)
#     print('outputs:', outputs.shape)
#     # 筛选有信号的
#     indexs = []
#     for m in range(outputs.shape[0]):
#         output_index = outputs[m]
#         input_index = inputs[m]
#         if output_index.max() > 60 and output_index.mean() > 0.01 and input_index.max() > 0:
#             indexs.append(m)
#     input_choose = inputs[indexs]
#     inputs = 0
#     output_choose = outputs[indexs]
#     outputs = 0
#     print('input_choose shape:', input_choose.shape)
#     print('output_choose shape:', output_choose.shape)
#     return input_choose, output_choose
#     # return input_choose.astype('float32') / 65535, output_choose.astype('float32') / 65535
#
#
# class MyDataset(Dataset):
#     def __init__(self, inputs, outputs, transform=''):
#         self.transform = transform
#         self.input = inputs
#         self.output = outputs
#
#     def __len__(self):
#         length, _, _, _ = self.output.shape
#         return length
#
#     def __getitem__(self, idx):
#         if self.input[idx, :, :, :].max()==0 or self.output[idx, :, :, :].max()==0:
#             print(self.input[idx, :, :, :].max(), self.output[idx, :, :, :].max())
#         # return self.input[idx, :, :, :].astype('float32') / self.input[idx, :, :, :].max(), self.output[idx, :, :, :].astype('float32') / self.output[idx, :, :, :].max()
#         return self.input[idx, :, :, :].astype('float32') / 65535, self.output[idx, :, :, :].astype('float32') / 65535
#
#
# # 改层数
# def my_dataloader(file, ratio, depth, batchsize=4, shuffle=True):
#     """ the sum of ratio must equal to 1"""
#     inputs, outputs = load_data(file, depth)
#
#     # 不打乱顺序
#     print('inputs:', inputs.shape)
#     print('outputs:', outputs.shape)
#
#     # 打乱顺序
#     # datasets = np.hstack((inputs, outputs))
#     # print(datasets.shape)
#     # np.random.shuffle(datasets)
#     # inputs = datasets[:, 0:12]
#     # outputs = datasets[:, 12:12+depth]
#     # print('inputs:', inputs.shape)
#     # print('outputs:', outputs.shape)
#
#     # inputs[inputs <= 0] = 0
#     # inputs[inputs >= 1] = 1
#     # outputs[outputs <= 0] = 0
#     # outputs[outputs >= 1] = 1
#     length, _, _, _ = outputs.shape
#
#     num_train = int(length * ratio[0])
#     num_test = int(length * ratio[1])
#     num_val = int(length * ratio[2])
#
#     train_inputs = inputs[0:num_train, :, :, :]
#     train_outputs = outputs[0:num_train, :, :, :]
#     # 保存两张图看看数据集效果，改路径
#     # nline or wide
#     for i in range(10):
#         os.makedirs(r'F:\20250421thy1_LWUnet\train\datalodercheck\our_%dlayer_4v_b16_test/' % depth, exist_ok=True)
#         # tiff.imwrite(r'F:\20250421thy1_LWUnet\train\datalodercheck\our_%dlayer\input_nline_%d.tif' % (depth, i), (train_inputs[100*i]).astype('uint16'))
#         # tiff.imwrite(r'F:\20250421thy1_LWUnet\train\datalodercheck\our_%dlayer\output_limo_%d.tif' % (depth, i), (train_outputs[100*i]).astype('uint16'))
#         tiff.imwrite(
#              r'F:\20250421thy1_LWUnet\train\datalodercheck\our_%dlayer_4v_b16_test\input_nline_%d.tif' % (depth, i),
#              (train_inputs[120 * i]).astype('uint16'),
#              imagej = True,
#              photometric = 'minisblack',
#              metadata = {'axes': 'ZYX'}
#              )
#         tiff.imwrite(
#             r'F:\20250421thy1_LWUnet\train\datalodercheck\our_%dlayer_4v_b16_test\output_limo_%d.tif' % (depth, i),
#             (train_outputs[120 * i]).astype('uint16'),
#             imagej = True,
#             photometric = 'minisblack',
#             metadata = {'axes': 'ZYX'}
#             )
#     print('done')
#
#     test_inputs = inputs[num_train: num_train+num_test, :, :, :]
#     test_outputs = outputs[num_train: num_train+num_test, :, :, :]
#
#     val_inputs = inputs[num_train+num_test: num_train+num_test+num_val, :, :, :]
#     val_outputs = outputs[num_train+num_test: num_train+num_test+num_val, :, :, :]
#     train_dataloader = DataLoader(MyDataset(train_inputs, train_outputs), batch_size=batchsize, drop_last=True,
#                                   shuffle=shuffle)
#     test_dataloader = DataLoader(MyDataset(test_inputs, test_outputs), batch_size=batchsize, drop_last=False,
#                                  shuffle=False)
#     val_dataloader = DataLoader(MyDataset(val_inputs, val_outputs), batch_size=batchsize, drop_last=False,
#                                 shuffle=False)
#     # test_dataloader = DataLoader(MyDataset(test_inputs, test_outputs), batch_size=batchsize, drop_last=False,
#     #                              shuffle=shuffle)
#     # val_dataloader = DataLoader(MyDataset(val_inputs, val_outputs), batch_size=batchsize, drop_last=False,
#     #                             shuffle=shuffle)
#     loader = {}
#     loader['train'] = train_dataloader
#     loader['val'] = val_dataloader
#     loader['test'] = test_dataloader
#     print('Train shape:', train_inputs.shape, train_outputs.shape)
#     print('test shape:', test_inputs.shape, test_outputs.shape)
#     print('val shape:', val_inputs.shape, val_outputs.shape)
#
#     return loader
#
#
# # if __name__ == '__main__':
# #     # 加载数据集路径
# #     depth = 5
# #     data_dir = r'I:\196617_thy1\train_datasets_CH1\ori_datasets_final\567Data'
# #     loader = my_dataloader(data_dir, [0.8, 0.1, 0.1], depth, batchsize=4)
# #     a = loader['train'].dataset.input.shape















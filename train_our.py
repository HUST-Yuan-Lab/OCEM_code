# from myloss import *
from torch import optim
from data_loader_our import my_dataloader
import argparse
from myfun import *
import torch
import datetime
import os
from collections import OrderedDict
from Res_UNet_for_speedup import ResUNet_PixelShuffle
# from model import UNet, LW_UNet
# from LW_UNet import LW_UNet
# from RCAN import RCAN
# from URCANsub import URCANsub
import tifffile as tiff
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_ids = [0]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


class UNetTrain:
    def __init__(self, device='gpu', file_result='beads', dtype='uint16', m=None, data=None):
        now = datetime.datetime.now()
        self.file_result = '%s_LiMo_%d_%d_%d_%d' % (file_result, now.year, now.month, now.day, now.hour)
        self.args = get_args()      #通过 get_args() 函数获取训练参数
        os.makedirs(self.file_result, exist_ok=True)
        self.device = device
        self.dtype = dtype
        self.model = m
        self.data = data
        self.log = Logger(os.path.join(self.file_result, 'train.log'), level='info')
        self.args = get_args()
        self.i = 0

    def train(self, criterion, optimizer, dataload):
        dataset_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0
        length = len(dataload)
        cycle = length//10 + 1
        self.model.train()

        for x, y in dataload:  # 分100次遍历数据集，每次遍历batch_size=4
            optimizer.zero_grad()  # 每次minibatch都要将梯度(dw,db,...)清零
            inputs = x.to(self.device)
            labels = y.to(self.device)
            outputs = self.model(inputs)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 梯度下降,计算出梯度
            optimizer.step()  # 更新参数一次：所有的优化器Optimizer都实现了step()方法来对所有的参数进行更新
            epoch_loss += loss.item()
            step += 1
            if step % cycle == 0:
                print("%d/%d,train_loss:%0.5f" % (step, dataset_size // dataload.batch_size, loss.item()))
        cur = datetime.datetime.now()
        year, month, day, hour = [cur.year, cur.month, cur.day, cur.hour]
        # file_write = os.path.join(self.file_result, 'weights_%d_%d_%d_%d_nline.pth' % (year, month, day, hour))
        file_write = os.path.join(self.file_result, 'weights_%d_nline.pth' % (self.i))
        if len(gpu_ids) > 1:
            torch.save(model.module.state_dict(), file_write)
        else:
            torch.save(model.state_dict(), file_write)
        return epoch_loss/step

    # 测试
    def test(self, data, criterion):
        loss_sum = 0
        step = 0
        self.model.eval()
        with torch.no_grad():
            for x, y in data:
                step += 1
                x_in, y_out = x.to(self.device), y.to(self.device)
                loss = criterion(model(x_in), y_out)
                loss_sum += loss.item()
        return loss_sum / step

    def pre(self, data, criterion):
        loss_sum = 0
        step = 0
        self.i = self.i + 1
        self.model.eval()
        with torch.no_grad():
            for x, y in data:
                step += 1
                x_in, y_out = x.to(self.device), y.to(self.device)
                loss = criterion(self.model(x_in), y_out)
                loss_sum += loss.item()
            for x, y in data:
                x_in, y_out = x.to(self.device), y.to(self.device)
                predict = model(x_in)
                break
            x_data = x_in.cpu().detach().numpy()
            y_data = y_out.cpu().detach().numpy()
            pre_data = predict.cpu().detach().numpy()
            y_image = self.data2image(y_data)
            pre_image = self.data2image(pre_data)
            name_out = os.path.join(self.file_result, 'real_%i.tif' % self.i)
            name_pre = os.path.join(self.file_result, 'predict_%i.tif' % self.i)
            if y_image.shape[0] == 3 or y_image.shape[0] == 4:
                black_image = np.zeros((5-y_image.shape[0], y_image.shape[1], y_image.shape[2]), np.uint16)
                y_image = np.concatenate((y_image, black_image), axis=0)
                pre_image = np.concatenate((pre_image, black_image), axis=0)
            tiff.imwrite(name_out, y_image)
            tiff.imwrite(name_pre, pre_image)
            # tiff.imwrite(
            #     name_out,
            #     y_image.astype('uint16'),
            #     imagej=True,
            #     photometric='minisblack',
            #     metadata={'axes': 'ZYX'}
            # )
            # tiff.imwrite(
            #     name_pre,
            #     pre_image.astype('uint16'),
            #     imagej=True,
            #     photometric='minisblack',
            #     metadata={'axes': 'ZYX'}
            # )
            if self.args.save_in:
                x_image = self.data2image(x_data)
                name_in = os.path.join(self.file_result, 'input_%i.tif' % self.i)
                tiff.imwrite(name_in, x_image)
            #     tiff.imwrite(
            #         name_in,
            #         x_image.astype('uint16'),
            #         imagej=True,
            #         photometric='minisblack',
            #         metadata={'axes': 'ZYX'}
            #     )
            name_log = os.path.join(self.file_result, 'log.txt')
            f = open(name_log, 'a+')
            print('val loss: %f' % (loss_sum/step), file=f)
            f.close()
        return loss_sum/step, [x_data, y_data, pre_data]



    def data2image(self, data, num_x=4):
        [num, w, h, d] = np.shape(data)
        # num_x=2，因为batch_size=4，所以将四张图合成一张大图输出
        # 之前通道数为4的时候，不加1会自动存为rgb格式，所以统一加1
        # data_zeros = np.zeros((w+1, num_x*h, num//num_x*d))
        data_zeros = np.zeros((w, num // num_x * h, num_x * d))
        # print(data_zeros.shape, data.shape)
        data[data <= 0] = 0
        data[data >= 1] = 1
        if self.dtype == 'uint8':
            scale = 255
        elif self.dtype == 'uint16':
            scale = 65535
            # scale = 2000
        else:
            scale = 1
        data = (data) * scale
        for i in range(num):
            irow = i // num_x
            icol = i % num_x
            data_zeros[0:w, h*irow: h*irow+h, d*icol:d*icol+d] = data[i, :, :, :]
        image = data_zeros
        if self.dtype == 'uint8':
            image = image.astype('uint8')
        elif self.dtype == 'uint16':
            image = image.astype('uint16')
        else:
            image = image
        image = image.squeeze()
        # print(image.shape)
        return image

    # 训练模型
    def start(self, optimizer, criterion, num_epochs):
        self.log.logger.info(self.args.train_info)
        self.log.logger.info(f'''
            Starting training:
            Epochs:                 {self.args.num_epochs}
            Batch size:             {self.args.batchsize}
            Training input shape:   {self.data["train"].dataset.input.shape}
            Training output shape:  {self.data["train"].dataset.output.shape}
            Device:                 {device}
        ''')
        model_loss = [[], []]
        data_pre = []
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch + 1, num_epochs))
            print('-' * 10)
            train_loss = self.train(criterion, optimizer, self.data['train'])
            print("epoch %d train loss:%0.5f" % (epoch + 1, train_loss))
            test_loss = self.test(self.data['test'], criterion)
            print("epoch %d test loss:%0.5f" % (epoch + 1, test_loss))
            model_loss[0].append(train_loss)
            model_loss[1].append(test_loss)

            # 验证集
            val_loss,  data_pre= self.pre(self.data['val'], criterion)
            [x_in, y_out, y_pre] = data_pre
            print("epoch %d val loss:%0.5f" % (epoch + 1, val_loss))

            plot_loss(model_loss, self.file_result)
            if epoch == 30:
                params = filter(lambda p: p.requires_grad, self.model.parameters())
                args.lr = args.lr/2
                optimizer = optim.Adam(params, lr=args.lr)

            if epoch == 60:
                params = filter(lambda p: p.requires_grad, self.model.parameters())
                args.lr = args.lr/4
                optimizer = optim.Adam(params, lr=args.lr)

            if epoch == 80:
                params = filter(lambda p: p.requires_grad, self.model.parameters())
                args.lr = args.lr / 8
                optimizer = optim.Adam(params, lr=args.lr)

        losses = np.array(model_loss)
        np.save(os.path.join(self.file_result, 'loss.npy'), losses)
        self.log.logger.info(f'Stopping training:')
        return losses, data_pre


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train_information', dest='train_info', type=str, default='''
        This network is used to reconstruct 8Line%d层, thy1 Barin.
        ''' % depth)
    # 迁移学习
    parser.add_argument('--load', dest='load', type=bool, default=False)
    # parser.add_argument('--load', dest='load', type=bool, default=True)
    parser.add_argument('--model_file', dest='file_model', type=str, default=
    r'F:\20250421thy1_LWUnet\train\depth5_4v_batch16_LiMo_2025_4_29_16\weights_99_nline.pth')
    # 文件保存路径
    # parser.add_argument('--train_save_file', dest='file_save', type=str,
                        # default=r'H:\230701bingli\train\train_datasets_CH1\ours\urcansub_ori_mseloss_results')
    parser.add_argument('--train_save_file', dest='file_save', type=str,
                        default=r'F:\20250510thy1_ResUNet\train\depth5_4v_resunet')
    parser.add_argument('--data_type', dest='dtype', type=str, help='uint16 or uint8', default='uint16')
    # 训练数据路径
    parser.add_argument('--data_file', dest='file_data', type=str, default=r'I:\196617_thy1\train_datasets_CH1\ori_datasets_final\Data')
    # 分配训练/验证/测试集比例，一般不改
    parser.add_argument('--train_test_val', dest='data_scale', type=tuple, default=(14/16, 1/16, 1/16))
    parser.add_argument('--channel_in', dest='c_in', type=int, default=12)
    parser.add_argument('--channel_out', dest='c_out', type=int, default=depth)
    parser.add_argument('--filter_base', dest='filter_b', type=int, default=64)
    parser.add_argument('--batch_size', dest='batchsize', type=int, default=16)
    parser.add_argument('--num_epochs', dest='num_epochs', type=int, default=100)
    parser.add_argument('--learning-rate', dest='lr', type=float, default=0.0002)
    parser.add_argument('--save_input', dest='save_in', type=bool, default=True)
    return parser.parse_args()


if __name__ == '__main__':
    depth = 5
    # 2GPU分别跑
    args = get_args()
    dataloader = my_dataloader(args.file_data, ratio=args.data_scale, depth=depth, batchsize=args.batchsize)
    # net = UNet(n_channel_in=args.c_in, n_channel_out=args.c_out, n_filter_base=args.filter_b)
    # net = LW_UNet(n_channel_in=args.c_in, n_channel_out=args.c_out, n_filter_base=args.filter_b)
    # net = URCANsub(12, 5, 64, 20)
    # net = RCAN(5, 64)
    net = ResUNet_PixelShuffle(n_channel_in=args.c_in, n_channel_out=args.c_out, n_filter_base=args.filter_b, scale_factor=4)
    if len(gpu_ids) > 1:
        net = torch.nn.DataParallel(net, device_ids=gpu_ids)
    model = net.to(device)
    if args.load:
        checkpoint = torch.load(args.file_model)
        new_state_dict = OrderedDict()
        for key, value in checkpoint.items():
            # name = 'module.' + key
            # new_state_dict[name] = value
            new_key = key[len("module."):] if key.startswith("module.") else key
            new_state_dict[new_key] = value
        model.load_state_dict(new_state_dict)

    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(params, lr=args.lr)
    criterion = torch.nn.MSELoss().to(device)
    # criterion = torch.nn.L1Loss().to(device)
    # mse_loss = torch.nn.MSELoss().to(device)
    # mae_loss = torch.nn.L1Loss().to(device)
    # alpha, beta = 1, 1  # 设置L1L2权重
    # criterion = lambda output, target: alpha * mse_loss(output, target) + beta * mae_loss(output, target)
    t = UNetTrain(device=device, file_result=args.file_save, dtype=args.dtype, m=model, data=dataloader)
    losses, data_pre = t.start(optimizer, criterion, args.num_epochs)
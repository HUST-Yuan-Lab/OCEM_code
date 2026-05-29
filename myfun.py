import logging
from logging import handlers
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, savefig
import os
import math
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset


class Logger (object):
    # 日志级别关系映射
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }

    def __init__(self, filename, level='info', when='D', backCount=3,
                 fmt='%(asctime)s ----- %(levelname)s: %(message)s'):
        # fmt = '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)
        # 设置日志格式
        self.logger.setLevel(self.level_relations.get(level))
        # 设置日志级别
        sh = logging.StreamHandler()
        # 往屏幕上输出
        sh.setFormatter(format_str)
        # 设置屏幕上显示的格式
        th = handlers.TimedRotatingFileHandler(filename=filename, when=when, backupCount=backCount, encoding='utf-8')
        # 往文件里写入#指定间隔时间自动生成文件的处理器
        # 实例化TimedRotatingFileHandler
        # interval是时间间隔，backupCount是备份文件的个数，如果超过这个个数，就会自动删除，when是间隔的时间单位
        th.setFormatter(format_str)
        # 设置文件里写入的格式
        self.logger.addHandler(sh)
        # 把对象加到logger里
        self.logger.addHandler(th)


def plot_loss(loss, file_save=''):
    # print(loss)
    train_loss = np.array(loss[0][3:])
    test_loss = np.array(loss[1][3:])
    x = np.linspace(1, len(loss[0]), len(train_loss))
    fig = plt.figure()
    plt.title('Training and Testing loss')
    plt.xlabel('epoch', fontsize=12, color='black')
    plt.ylabel('loss', fontsize=12, color='black')
    line_train = plt.plot(x, train_loss, '-', label='train')
    line_test = plt.plot(x, test_loss, 'o--', label='test')
    # plt.title('G and D loss')
    # plt.xlabel('epoch', fontsize=12, color='black')
    # plt.ylabel('loss', fontsize=12, color='black')
    # line_train = plt.plot(x, train_loss, '-', label='g_loss')
    # line_test = plt.plot(x, test_loss, 'o--', label='d_loss')
    plt.legend(loc=1)
    name = os.path.join(file_save, 'loss.jpg')
    savefig(name)
    np_loss = np.array(loss)
    name_npy = os.path.join(file_save, 'loss.npy')
    np.save(name_npy, np_loss)
    plt.close()


# 删除指定目录，首先删除指定目录下的文件和子文件夹，然后再删除该文件夹
def delete_file(file):

    if os.path.isfile(file):
        try:
            os.remove(file)
        except:
            pass
    elif os.path.isdir(file):
        # 如果是文件夹，则首先删除文件夹下文件和子文件夹，再删除文件夹
        for item in os.listdir(file):
            tf = os.path.join(file, item)   # 递归调用
            delete_file(tf)
        try:
           os.rmdir(file)
        except:
            pass


def get_ssim(image1, image2):
    k1 = 0.01
    k2 = 0.03
    l = 1

    if image1.dtype == 'uint16':
        image1 = image1/65535
        image2 = image2 / 65535
    elif image1.dtype == 'uint8':
        image1 = image1/255
        image2 = image2 / 255

    mean1 = np.mean(image1)
    std1 = np.std(image1)
    mean2 = np.mean(image2)
    std2 = np.std(image2)
    # image1 = (image1-mean1)/std1
    # image2 = (image2 - mean2) / std2
    image = ((image1-mean1)*(image2 - mean2)+(k2*l)**2/2)/(std1*std2+(k2*l)**2/2)
    lxy = (2*mean1*mean2 + (k1*l)**2)/(mean1**2 + mean2**2 + (k1*l)**2)
    cxy = (2*std1*std2 + (k2*l)**2)/(std1**2 + std2**2 + (k2*l)**2)
    sxy = np.mean(image)
    return lxy*cxy*sxy


# 获取图像的PSNR
def get_psnr(image1, image2):
    if image1.dtype == 'uint16':
        image1 = image1/65535
        image2 = image2 / 65535
    elif image1.dtype == 'uint8' :
        image1 = image1/255
        image2 = image2 / 255
    image = (image1-image2)
    mse = np.var(image)
    psnr = 10*math.log10(1/mse)
    return psnr


def image_predict(model, image_in):
    nums = 65535 / 6

    def image2torch_data(image):
        data = np.expand_dims (image, axis=0)
        # data = np.expand_dims(data, axis=0)
        data = tranform_2data (data.astype ('float32'), nums)
        data_tensor = torch.from_numpy (data)
        data_tensor = TensorDataset(data_tensor)
        data_loader = DataLoader(data_tensor, batch_size=1)
        return data_loader

    def tranform_2data(image, nums):
        return image / nums

    def tranform_2img(data, nums):
        return data * nums

    def model_pre(model, image):
        data_tensor = image2torch_data (image)
        model.val ()
        with torch.no_grad ():
            data_tensor = data_tensor.dataset.tensors[0]
            predict = model (data_tensor.cuda ())
        image = predict.cpu ().detach ().numpy ()
        image = tranform_2img (image, nums)
        image = image.squeeze ()
        image[image < 0] = 0
        image[image > 65535] = 65535
        image = image.astype ('uint16')
        return image

    data_in = image2torch_data(image_in)
    image_pre = model_pre(model, data_in)
    return image_pre


if __name__ == '__main__':
    log = Logger('all.log', level='info')
    log.logger.info('info')







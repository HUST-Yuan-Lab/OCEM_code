#!/usr/bin/python
# -*- coding: UTF-8 -*-
#对奇偶层进行配准并产生训练所用原始数据，可直接转换成npy文件
import numpy as np
import os.path
import os
import tifffile as tiff
import cv2


class Processor:
    def __init__(self, file_root='', sample_number='', channels=(1,), strip_range=range(40, 41),
                 file_write='', point1=(0, 0), point2=(0, 0)):
        self.file_root = file_root
        self.sample_number = sample_number
        self.channels = channels
        self.strip_range = strip_range
        self.file_write = file_write
        self.p1 = point1
        self.p2 = point2
        os.makedirs(self.file_write, exist_ok=True)
        self.file_list = []
        self.roi_enable = 1

    # 得到1倍速（原速）扫描的多线、宽场、LiMo图(轴向1umLiMo图)
    def get_1v_1uimage(self, layers, arrays):
        file_root = self.file_root
        sample_number = self.sample_number
        channels = self.channels
        strip_range = self.strip_range
        layer_range = layers
        file_write = self.file_write
        # 几线探测或者要用几线
        array_range = arrays
        # 裁剪想要的尺寸，去掉边缘
        roi = (self.p1[0], self.p1[1], self.p1[0] + self.p1[2], self.p1[1] + self.p1[3])
        print('1vimage_roi', roi)
        os.makedirs(os.path.join(file_write, 'LiMo'), exist_ok=True)

        for ichannal in channels:
            for ilayer in layer_range:
                for istrip in strip_range:
                    image_list = []
                    # 每32线探测一个image_list
                    for iarray in array_range:
                        # 220526_00001(00030)_40(51)_CH1_0(31).tif 30层,12个条带,1个通道,32线探测
                        image_name = sample_number + '_' + str('%05d' % ilayer) + '_' + str(istrip) + '_CH' + str(
                            ichannal) + '_' + str(iarray) + '.tif'
                        full_name = os.path.join(file_root, sample_number + '_CH' + str(ichannal)
                                                 + '_' + str('%05d' % ilayer), image_name)
                        print(full_name)
                        image = tiff.imread(full_name)
                        # 奇偶层错位位移dislocation
                        if ilayer % 2 == 1:
                            dislocation = 0  # 肝脏4  细胞计数2  对比实验1
                        if ilayer % 2 == 0:
                            # 奇偶层之间是否存在上下翻转
                            # image = np.flip(image, axis=0)
                            dislocation = 0
                       # 线之间是否存在错位
                        # roi：行起点，列起点，行终点，列终点
                        # 是否裁剪
                        image = image[roi[0] + dislocation:roi[2] + dislocation, roi[1]:roi[3]]
                        image = image.astype('float32')
                        ##image = image.astype('uint16')
                        image_list.append(image)

                    # 得到limo数据
                    limo = self.get_limo(image_list)
                    # 中值滤波去噪
                    # limo_median = cv2.medianBlur(limo, 3)
                    # LiMo_00001_40.tif
                    name_subline = (str('LiMo_%05d_%02d_CH%d.tif' % (ilayer, istrip, ichannal)))
                    name_write_limo = os.path.join(file_write, 'LiMo', name_subline)
                    tiff.imwrite(name_write_limo, limo)
                    # tiff.imwrite(name_write_limo, limo_median)

    # 得到1倍速（原速）扫描的多线、宽场、LiMo图(轴向4um多线、宽场图)
    def get_nv_4uimage(self, layers, arrays, speed=1):
        file_root = self.file_root
        sample_number = self.sample_number
        channels = self.channels
        strip_range = self.strip_range
        layer_range = layers
        file_write = self.file_write
        # 几线探测或者要用几线
        array_range = arrays
        # 裁剪想要的尺寸，去掉边缘
        roi = (self.p2[0] * speed, self.p2[1], (self.p2[0] + self.p2[2]) * speed, self.p2[1] + self.p2[3])
        print('%dvimage_roi' % speed, roi)
        os.makedirs(os.path.join(file_write, 'nline_%dv' % speed), exist_ok=True)
        os.makedirs(os.path.join(file_write, 'wide_%dv' % speed), exist_ok=True)
        os.makedirs(os.path.join(file_write, 'line_%dv' % speed), exist_ok=True)

        for ichannal in channels:
            for ilayer in layer_range:
                for istrip in strip_range:
                    image_list = []
                    # 每32线探测一个image_list
                    for iarray in array_range:
                        # 220526_00001(00030)_40(51)_CH1_0(31).tif 30层,12个条带,1个通道,32线探测
                        image_name = sample_number + '_' + str('%05d' % ilayer) + '_' + str(istrip) + '_CH' + str(
                            ichannal) + '_' + str(iarray) + '.tif'
                        full_name = os.path.join(file_root, sample_number + '_CH' + str(ichannal)
                                                 + '_' + str('%05d' % ilayer), image_name)
                        print(full_name)
                        image = tiff.imread(full_name)
                        # 奇偶层错位位移dislocation
                        if ilayer % 2 == 1:
                            dislocation = 0  # 肝脏4  细胞计数2  对比实验1
                        if ilayer % 2 == 0:
                            # 奇偶层之间是否存在上下翻转
                            # image = np.flip(image, axis=0)
                            dislocation = 0
                        # 线之间是否存在错位
                        # roi：行起点，列起点，行终点，列终点
                        # 是否裁剪
                        image = image[roi[0] + dislocation:roi[2] + dislocation, roi[1]:roi[3]]
                        image = image.astype('float32')
                        # image -= 100
                        [h1, w1] = image.shape
                        # 模拟四倍加速，配准同1倍速，真实四倍加速配准得根据实际图像特点改写
                        if speed > 1:
                            for i in range(0, h1 - speed + 1, speed):
                                image[i // speed, :] = np.mean(image[i:i + speed, :], 0)
                            image1 = image[:h1 // speed, :]
                            image1 = np.clip(image1, 0, 65535)
                        elif speed == 1:
                            image1 = image
                            image1 = np.clip(image1, 0, 65535)
                        image_list.append(image1)

                    # 多线原速结果合成一个stack
                    nimage = np.stack(image_list, axis=0)
                    # nimage = np.clip(nimage-100, 0, 65535)
                    print(nimage.shape)
                    nimage = nimage.astype('uint16')
                    # #Line%dv存六线数据stack
                    name_nline = (str('nline_%05d_%02d_CH%d.tif' % (ilayer, istrip, ichannal)))
                    name_write_nline = os.path.join(file_write, str('nline_%dv'%speed), name_nline)
                    tiff.imwrite(name_write_nline, nimage)

                    # 验证加速图和原速图配准问题（LiMo和Line4v）
                    # 单线数据
                    line = nimage[3]
                    name_line = (str('line_%05d_%02d_CH%d.tif' % (ilayer, istrip, ichannal)))
                    name_write_line = os.path.join(file_write, str('line_%dv' % speed), name_line)
                    tiff.imwrite(name_write_line, line)
                    if speed > 1:
                        os.makedirs(os.path.join(file_write, 'line_%dv_upsample' % speed), exist_ok=True)
                        line_up = cv2.resize(line, (w1, h1))
                        name_line_up = (str('line_upsample_%05d_%02d_CH%d.tif' % (ilayer, istrip, ichannal)))
                        name_write_line_up = os.path.join(file_write, str('line_%dv_upsample' % speed), name_line_up)
                        tiff.imwrite(name_write_line_up, line_up)

                    # 32线平均得到宽场结果
                    wide = np.mean(nimage, axis=0)
                    wide = wide.astype('uint16')
                    name_wide = (str('wide_%05d_%02d_CH1.tif' % (ilayer, istrip)))
                    name_write_wide = os.path.join(file_write, str('wide_%dv' % speed), name_wide)
                    tiff.imwrite(name_write_wide, wide)

    def image2data(self, speed=1, file_name='test', roi=(), shape=(240, 240), red=4, sigma=0):
        roi = (roi[0]//speed, roi[1], roi[2]//speed, roi[3])
        print('roi', roi)
        sh = (shape[0]//speed, shape[1])
        print('sh', sh)
        step = (sh[0]-sh[0]//red, sh[1]-sh[1]//red)
        print('step', step)
        file = os.path.join(self.file_write, file_name)
        print(file)
        name_list = self.get_name(file)
        print(len(name_list))
        list_range = range(len(name_list))
        data_list = []
        for i in list_range:
            name = os.path.join(file, name_list[i])
            image = tiff.imread(name)
            print(name)
            print(image.shape)
            # ndim返回image维度，三维还是二维
            if image.ndim == 2:
                # np.newaxis的作用是增加一个维度，统一变为三维图像
                image = image[np.newaxis, :, :]
            image = image[:, roi[0]:roi[2]+roi[0], roi[1]:roi[3]+roi[1]]
            #把图像切割成小图
            temp = self.image_split(image, sh, step)
            data_list.append(temp)
        print(len(data_list))
        data = np.vstack(data_list)
        #做成npy文件存入Data
        os.makedirs(os.path.join(self.file_write, 'Data'), exist_ok=True)
        name = os.path.join(self.file_write, 'Data', file_name+'.npy')
        np.save(name, data)
        return data

    def get_name(self, file_dir):
        files = []
        for root, dirs, files in os.walk(file_dir):
            print(str('该文件下有%d个文件' % len(files)))  # 当前路径下所有非目录子文件
        return files

    def get_limo(self, image_list):
        i = image_list
        limo = (i[2]+i[3])*2-i[0]-i[1]-i[4]-i[5]
        # limo = (i[4] + i[5] + i[6] + i[7]) * 2 - i[0] - i[1] - i[2] - i[3] - i[8] - i[9] - i[10] - i[11]
        limo = (abs(limo) + limo)/2
        limo = ((65535 + limo)-abs(65535 - limo))/2
        limo = limo.astype('uint16')
        return limo

    #切割图像
    def image_split(self, image, shape=(1, 1), step=(1, 1)):
        num_x = shape[0]
        num_y = shape[1]
        step_x = step[0]
        step_y = step[1]
        # 定义一个图像分割函数，把图片分割成任意大小的图片存放在四维图像中
        if np.ndim(image) == 2:
            [w, h] = np.shape(image)
            image = np.reshape(image, (1, w, h))
            print(image.shape)
        [d, w, h] = np.shape(image)
        imagelist = []
        for i in range(0, w - num_x + 1, step_x):
            for j in range(0, h - num_y + 1, step_y):
                image_temp = image[:, i:i + num_x, j:j + num_y]
                imagelist.append(image_temp)
        data = np.stack(imagelist, axis=0)
        print(data.shape)
        return data
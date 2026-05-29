from Preprocess import *


def preprocess_main():
    file_root = r'D:\Data\230701病理\CH1'
    sample_number = '230701'
    channels = (1,)
    arrays = range(2, 14)
    # 训练一般只使用一个条带
    # 训练
    strip_range = range(45, 48, 1)
    # 测试
    # strip_range = range(40, 41, 1)
    # 一般和ROI保持一致，只截取自己想要的部分，一般大小为2048*8000
    #3000层
    # 训练
    point_1 = (11000, 0, 12000, 2000)
    # point_2 = (11000, 0, 12000, 2000)
    # 测试
    # point_1 = (0, 0, 62889, 2048)
    # # point_2 = (0, 0, 62889, 2048)
    # 训练层数1+4n
    layers = range(1, 6, 1)
    # 测试层数1+4n
    # layers = range(1, 41, 1)

    # 保存文件夹路径
    # save_file = r'H:\230701bingli\train_datasets_CH2\ori_datasets_final'
    save_file = r'D:\OneDrive - hust.edu.cn\project\多线编码解码标准代码\数据集\train\train_datasets_CH1\ori_datasets_final'
    # speed = 1
    speed = 4
    p1 = (point_1[0], point_1[1], point_1[2], point_1[3])
    p2 = (point_1[0] // speed, point_1[1], point_1[2] // speed, point_1[3])
    print(p1, p2)
    get_image = 10
    # get_image = 1
    # get_data = [1, 1, 0, 0]
    # get_data = [0, 0, 0, 0]
    get_data = [1, 1, 1, 1]

    # p1 原速， p2 横向4倍加速
    point1 = p1[0:4]
    point2 = p2[0:4]
    pp1 = Processor(file_root, sample_number, channels, strip_range, save_file, point1, point2)

    if get_image == 10:
        pp1.get_1v_1uimage(layers, arrays)
        pp1.get_nv_4uimage(layers, arrays, speed)

# 将前面得到的图片变为npy格式
    # 二次裁剪，不需要的话，前两个起点均始终设为0，后两个长度与前面第一次裁剪保持一致
    # 训练
    roi = (0, 0, 12000, 2000)
    # 测试
    # roi = (0, 0, 62889, 2048)

    shape = (256, 256)
    red = 8  # shape/red部分重叠

    if get_data[0] == 1:
        speed = 1
        sigma = 0
        file_name = 'LiMo'
        data = pp1.image2data(speed, file_name, roi, shape=shape, red=red, sigma=sigma)
        print(data.shape)
        print('LiMo_data', data.shape)

    if get_data[1] == 1:
        # speed = 1
        speed = 4
        sigma = 1
        file_name = 'nline_%dv'%speed
        data = pp1.image2data(speed, file_name, roi, shape=shape, red=red, sigma=sigma)
        print('nline_data', data.shape)

    if get_data[2] == 1:
        # speed = 1
        # speed = 4
        sigma = 2
        file_name = 'wide_%dv'%speed
        data = pp1.image2data(speed, file_name, roi, shape=shape, red=red, sigma=sigma)
        print('wide_data', data.shape)

    if get_data[3] == 1:
        # speed = 1
        # speed = 4
        sigma = 3
        file_name = 'line_%dv'%speed
        data = pp1.image2data(speed, file_name, roi, shape=shape, red=red, sigma=sigma)
        print('line_data', data.shape)


preprocess_main()




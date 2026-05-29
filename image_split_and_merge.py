import numpy as np
import tifffile as tf


def image2patch(image, patch_size=128, stride=64, speed=1):
    [ch, w, h] = image.shape
    image_list = []
    patch_size_x = patch_size // speed
    stride_x = stride // speed
    patch_size_y = patch_size
    stride_y = stride

    if (w - patch_size_x) % stride_x != 0:
        m = (w - patch_size_x) // stride_x
        r = (m + 1) * stride_x + patch_size_x + stride_x
    else:
        r = w

    if (h - patch_size_y) % stride_y != 0:
        n = (h - patch_size_y) // stride_y
        c = (n + 1) * stride_y + patch_size_y + stride_y
    else:
        c = h

    before1, before2 = (r - w) // 2, (c - h) // 2
    after1, after2 = (r - w) - before1, (c - h) - before2
    print(w, h, r, c, before1, before2, after1, after2)
    image_pad = np.pad(image, ((0, 0), (before1, after1), (before2, after2)), 'constant')
    image = 0
    for m in range(0, r - patch_size_x + 1, stride_x):
        for n in range(0, c - patch_size_y + 1, stride_y):
            image_list.append(image_pad[:, m:m + patch_size_x, n:n + patch_size_y])  # 数据分块
    image_pad = 0
    data = np.stack(image_list, axis=0)
    image_list = 0
    print('data', data.shape)
    # return data.astype('float32'), ch, r, c, before1, before2, after1, after2
    return data, ch, r, c, before1, before2, after1, after2


def patch2image(data, ch, r, c, before1, before2, after1, after2, patch_size=128, stride=64, speed=1):
    print('data.shape', data.shape)
    output_img = np.zeros((ch, r*speed, c), dtype='uint16')  # 32
    print('output_img.shape', output_img.shape)
    i = 0
    for s in range(0, r*speed - patch_size + 1, stride):
        for t in range(0, c - patch_size + 1, stride):
            output_img[:, s + stride//2:s + patch_size - stride//2, t + stride//2:t + patch_size - stride//2] =\
                data[i, :, stride//2:patch_size - stride//2, stride//2:patch_size - stride//2]
            i = i + 1
    if after1 == 0 and after2 > 0:
        image = output_img[:, before1*speed:, before2:-after2]
    elif after2 == 0 and after1 > 0:
        image = output_img[:, before1*speed:-after1*speed, before2:]
    elif after2 == 0 and after1 == 0:
        image = output_img[:, before1*speed:, before2:]
    else:
        image = output_img[:, before1*speed:-after1*speed, before2:-after2]
    print('image.shape', image.shape)
    return image.astype('uint16')
    # return image.astype('float32')


# image = tf.imread(r'E:\wanghe\221028\test_datasets\ori_datasets_final\nline\nline_00017_50.tif')
# data = image2patch(image)
# print(data.shape, type(data), data.dtype)
# image = patch2image(image, 6, 3072, 2112, 3000, 2000, 36, 56, 36, 56)
# tf.imwrite(r'E:\wanghe\221028\test_datasets\nline_00017_50.tif', image)
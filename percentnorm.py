import numpy as np
from tifffile import imread, imwrite


# 定义基于分位数的归一化函数
def percentile_normalization(image, p_low, p_high):
    """
    基于分位数的归一化，将像素值缩放到 [0, 1] 范围。
    :param image: 输入的图像数组
    :param p_low: 下限分位数（通常为1~3之间）
    :param p_high: 上限分位数（通常为99.5~99.9之间）
    :return: 归一化后的图像
    """
    # 计算图像中对应分位数的像素值
    perc_low = np.percentile(image, p_low)
    perc_high = np.percentile(image, p_high)

    # 防止分母为0的情况（即上下限相等）
    if perc_high - perc_low == 0:
        perc_high += 1e-6

    # # 使用分位数进行归一化
    # normalized_image = (image - perc_low) / (perc_high - perc_low)
    #
    # # 将归一化值限制在 [0, 1] 范围内
    # normalized_image = np.clip(normalized_image, 0, 1)

    # return perc_low, perc_high, normalized_image
    return perc_high

# # 设置输入和输出路径
input_path = r'I:\196617_thy1\train_datasets_CH1\ori_datasets_final\LiMo\LiMo_03045_35.tif'  # 输入TIFF文件路径
# output_path = r'H:\FISH0314\LiMo\norm220314_00037_34_CH1.tif'   # 输出归一化后的TIFF文件路径
#
# 读取TIFF图像
image = imread(input_path)

# 定义分位数范围
p_low = 1.0  # 下限分位数
p_high = 99.9  # 上限分位数

# 对图像进行归一化
high = percentile_normalization(image, p_low, p_high)
print(f"low与high分别为: {high}")
#
# # 保存归一化后的图像
# imwrite(output_path, (normalized_image*high).astype(np.uint16))  # 保存为16位TIFF图像
# # low, high = percentile_normalization(image, p_low, p_high)
# print(f"low与high分别为: {low},{high}")

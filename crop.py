import tifffile
import os
import numpy as np

# 用户配置参数
input_dir = r'I:\196617_thy1\train_datasets_CH1\ori_datasets_final\LiMo'  # 输入目录路径
output_dir = r'I:\196617_thy1\train_datasets_CH1\ori_datasets_final\567Data\LiMo_3v'  # 输出目录路径
start_num = 3001  # 起始编号
end_num = 3199  # 结束编号
crop_region = (0, 0, 1600, 3999)  # (left, top, right, bottom)


def batch_crop_2d_tiffs():
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 解包裁剪区域参数
    left, top, right, bottom = crop_region

    # 遍历所有指定编号的文件
    for num in range(start_num, end_num + 1):
        # 生成文件名
        filename = f"LiMo_{num:05d}_35.tif"
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # 检查文件是否存在
        if not os.path.isfile(input_path):
            print(f"警告：文件 {filename} 不存在，跳过。")
            continue

        try:
            # 读取TIFF文件
            with tifffile.TiffFile(input_path) as tif:
                image = tif.asarray()
                # 获取元数据（包含所有TIFF标签）
                metadata = {}
                for tag in tif.pages[0].tags.values():
                    metadata[tag.name] = tag.value

            # 验证图像维度
            if image.ndim not in [2, 3]:
                print(f"文件 {filename} 不是2D或伪彩色图像，跳过。")
                continue

            # 执行裁剪操作（自动处理2D和伪彩色3D图像）
            if image.ndim == 3 and image.shape[2] in [3, 4]:  # 处理RGB/RGBA
                cropped = image[top:bottom, left:right, :]
            else:  # 标准2D或灰度堆栈
                cropped = image[..., top:bottom, left:right]  # 自动处理多维度

            # 保存裁剪后的图像（保留原始元数据）
            tifffile.imwrite(
                output_path,
                cropped,
                dtype=image.dtype,  # 保持原始数据类型
                metadata=metadata,  # 继承原始元数据
            )

            print(f"成功处理：{filename}")

        except Exception as e:
            print(f"处理文件 {filename} 时出错：{str(e)}")


if __name__ == "__main__":
    batch_crop_2d_tiffs()
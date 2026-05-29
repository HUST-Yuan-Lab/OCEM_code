import os
import numpy as np
import tifffile as tiff


class NPYConverter:
    def __init__(self, input_dir, output_dir):
        """
        初始化转换器
        :param input_dir: 降采样后的TIFF文件目录
        :param output_dir: 输出.npy文件的目录
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def image_split(self, image, target_shape, step):
        """
        改进版图像分割函数（支持三维堆栈）
        :param image: 输入图像堆栈 (n_frames, H, W)
        :param target_shape: 目标切片尺寸 (h, w)
        :param step: 滑动步长 (step_h, step_w)
        :return: 四维数组 (n_slices, n_frames, h, w)
        """
        if image.ndim == 2:
            image = image[np.newaxis, :, :]

        n_frames, H, W = image.shape
        h, w = target_shape
        step_h, step_w = step

        slices = []
        for i in range(0, H - h + 1, step_h):
            for j in range(0, W - w + 1, step_w):
                slice_data = image[:, i:i + h, j:j + w]
                slices.append(slice_data)

        return np.stack(slices, axis=0)

    def convert_to_npy(self, start_num, end_num,
                       target_shape=(32, 256),
                       red=8,
                       output_name='dataset'):
        """
        主转换函数
        :param start_num: 起始文件编号
        :param end_num: 结束文件编号
        :param target_shape: 目标切片尺寸
        :param red: 步长计算系数
        :param output_name: 输出文件名
        """
        # 计算动态步长
        step = (target_shape[0] - target_shape[0] // red,
                target_shape[1] - target_shape[1] // red)

        all_data = []

        # 批量处理文件
        for num in range(start_num, end_num + 1):
            filename = f"nline_{num:05d}_35.tif"
            file_path = os.path.join(self.input_dir, filename)

            try:
                # 读取三维堆栈 (n_frames, H, W)
                stack = tiff.imread(file_path)
                print(f"处理中: {filename} | 原始形状: {stack.shape}")

                # 自动处理二维单帧图像
                if stack.ndim == 2:
                    stack = stack[np.newaxis, :, :]

                # 执行分割
                slices = self.image_split(stack, target_shape, step)
                all_data.append(slices)

                print(f"成功分割: {slices.shape} → 累计数据量: {len(all_data)}")

            except Exception as e:
                print(f"处理 {filename} 出错: {str(e)}")

        # 合并所有数据
        final_data = np.vstack(all_data)
        print(f"最终数据集形状: {final_data.shape}")

        # 保存为npy文件
        output_path = os.path.join(self.output_dir, f"{output_name}.npy")
        np.save(output_path, final_data)
        print(f"数据集已保存至: {output_path}")


# 使用示例
if __name__ == "__main__":
    # 配置参数
    INPUT_DIR = r'I:\196617_thy1\train_datasets_CH1\ori_datasets_final\567Data\nline_3v'  # 降采样后的TIFF目录
    OUTPUT_DIR = r'I:\196617_thy1\train_datasets_CH1\ori_datasets_final\567Data'  # 输出目录

    CONVERTER = NPYConverter(INPUT_DIR, OUTPUT_DIR)

    CONVERTER.convert_to_npy(
        start_num=3001,  # 起始编号
        end_num=3199,  # 结束编号
        target_shape=(96, 288),  # 切片尺寸
        red=8,  # 步长系数
        output_name="nline_3v"  # 输出文件名
    )
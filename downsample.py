import os
import numpy as np
import tifffile


class AcceleratedDownsampler:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _downsample_speed(self, stack, speed):
        """
        加速采集模拟下采样核心函数
        :param stack: 输入堆栈 (n_frames, H, W) 或单帧 (H, W)
        :param speed: 加速倍数（相邻像素合并数）
        :return: 下采样后的堆栈 (n_frames, H//speed, W)
        """
        # 统一处理维度
        if stack.ndim == 2:
            stack = stack[np.newaxis, ...]

        n_frames, H, W = stack.shape

        # 计算可下采样的最大高度
        new_H = H // speed
        trimmed_H = new_H * speed

        # 初始化输出数组
        downsampled = np.zeros((n_frames, new_H, W), dtype=np.float32)

        # 执行加速采集模拟
        for frame in range(n_frames):
            # 裁剪无法整除的部分
            frame_data = stack[frame, :trimmed_H, :]

            # 相邻像素合并
            for row in range(new_H):
                start = row * speed
                end = start + speed
                downsampled[frame, row] = np.mean(frame_data[start:end], axis=0)

        # 处理16-bit数据范围
        downsampled = np.clip(downsampled, 0, 65535).astype(stack.dtype)
        return downsampled.squeeze()  # 移除单帧时增加的维度

    def batch_downsample(self, start_num, end_num, speed):
        """
        批量下采样执行函数
        :param start_num: 起始编号（包含）
        :param end_num: 结束编号（包含）
        :param speed: 加速倍数（下采样率）
        """
        for num in range(start_num, end_num + 1):
            filename = f"nline_{num:05d}_35.tif"
            input_path = os.path.join(self.input_dir, filename)
            output_path = os.path.join(self.output_dir, filename)

            try:
                # 读取原始堆栈
                stack = tifffile.imread(input_path)
                print(f"处理中: {filename} | 原始形状: {stack.shape}")

                # 执行下采样
                downsampled = self._downsample_speed(stack, speed)

                # 保存结果
                tifffile.imwrite(
                    output_path,
                    downsampled,
                    metadata={'axes': 'TYX'},
                    imagej=True
                )
                print(f"下采样完成: {downsampled.shape} → 保存至 {output_path}")

            except Exception as e:
                print(f"处理 {filename} 时出错: {str(e)}")


# 使用示例
if __name__ == "__main__":
    # 配置参数
    INPUT_DIR = r'I:\196617_thy1\test_datasets_CH1\ori_datasets_final\nline_1v'
    OUTPUT_DIR = r'I:\196617_thy1\test_datasets_CH1\ori_datasets_final\nline_3v'
    SPEED = 3  # 加速倍数（下采样率）
    START_NUM = 3200  # 起始编号
    END_NUM = 3399  # 结束编号

    processor = AcceleratedDownsampler(INPUT_DIR, OUTPUT_DIR)
    processor.batch_downsample(
        start_num=START_NUM,
        end_num=END_NUM,
        speed=SPEED
    )
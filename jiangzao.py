import cv2
import numpy as np
import time
import os


def adaptive_median_filter_fast(img, max_window_size=7):
    """高性能自适应中值滤波"""
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    out_img = img.copy()
    current_window_size = 3
    already_processed = np.zeros(img.shape, dtype=bool)

    while current_window_size <= max_window_size:
        kernel = np.ones((current_window_size, current_window_size), np.uint8)
        z_med = cv2.medianBlur(out_img, current_window_size)
        z_min = cv2.erode(out_img, kernel)
        z_max = cv2.dilate(out_img, kernel)
        z_xy = out_img

        mask_a = (z_med > z_min) & (z_med < z_max)
        process_mask = mask_a & (~already_processed)
        replace_with_med = (z_xy <= z_min) | (z_xy >= z_max)

        update_condition = process_mask & replace_with_med
        out_img[update_condition] = z_med[update_condition]

        already_processed[process_mask] = True
        if np.all(already_processed):
            break
        current_window_size += 2

    return out_img


def batch_process(input_folder, output_folder):
    # 1. 如果输出文件夹不存在，则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"创建输出文件夹: {output_folder}")

    # 2. 定义 CLAHE 对象
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # 3. 遍历输入文件夹
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.bmp', '.jpg', '.png', '.jpeg', '.tif'))]

    if not files:
        print("未在文件夹中找到有效图像文件。")
        return

    print(f"开始处理，共找到 {len(files)} 张图片...")
    total_start = time.time()

    for filename in files:
        img_path = os.path.join(input_folder, filename)

        # 读取图像 (灰度模式)
        image = cv2.imread(img_path, 0)
        if image is None:
            print(f"跳过无法读取的文件: {filename}")
            continue

        # --- 核心处理步骤 ---
        # 1. 滤波
        denoised = adaptive_median_filter_fast(image, max_window_size=7)
        # 2. CLAHE 增强
        enhanced = clahe.apply(denoised)

        # 4. 保存最终图像
        save_path = os.path.join(output_folder, f"processed_{filename}")
        cv2.imwrite(save_path, enhanced)
        print(f"已保存: {filename} -> processed_{filename}")

    total_end = time.time()
    print("-" * 30)
    print(f"批量处理完成！总耗时: {total_end - total_start:.2f} 秒")


# --- 配置路径 ---
if __name__ == "__main__":
    # 输入文件夹路径 (请确保路径正确)
    input_dir = r"E:\29230\1\111\volunteer01"

    # 处理后保存的文件夹路径
    output_dir = r"E:\29230\1\111\volunteer01_processed"

    batch_process(input_dir, output_dir)
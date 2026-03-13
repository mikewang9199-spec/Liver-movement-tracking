import cv2
import numpy as np


def apply_clahe_on_grayscale(image_path, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    对灰度图像直接进行 CLAHE 增强，不经过降噪处理。
    适用于已经加载的灰度图或指定路径的灰度图。

    :param image_path: 待处理的图像文件路径。
    :param clip_limit: 限制对比度的阈值，越高对比度越剧烈。
                       建议范围 2.0-4.0，过高可能导致噪声被放大。
    :param tile_grid_size: 局部增强的网格大小。
                           通常为 (8, 8)，可根据图像内容调整。
                           越小对局部细节增强越明显，越大越平滑。
    """
    # 1. 读取图像，强制以灰度模式加载
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"错误：无法读取图像文件 '{image_path}'。请检查路径是否正确或文件是否存在。")
        return

    # 2. 创建 CLAHE 对象并应用增强
    clahe_processor = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced_img = clahe_processor.apply(img)

    # 3. 结果展示
    # 使用 np.hstack 将原图和增强后的图像水平拼接，方便对比
    comparison_img = np.hstack((img, enhanced_img))

    cv2.imshow('Original Grayscale | CLAHE Enhanced', comparison_img)

    # 4. 保存增强后的图像
    output_filename = "grayscale_clahe_enhanced.jpg"
    cv2.imwrite(output_filename, enhanced_img)
    print(f"增强后的灰度图像已保存为 '{output_filename}'。")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# --- 使用示例 ---
if __name__ == "__main__":
    # --- 请将 'your_grayscale_image.jpg' 替换为你实际的灰度图像路径 ---
    # 例如: 'D:/images/my_grayscale_photo.png'
    # 或者用我帮你生成一个低对比度灰度图来测试

    # 如果你手头没有灰度图，可以使用下面这段代码生成一个模拟图进行测试
    # 生成一个低对比度模拟灰度图
    # mock_image_path = "mock_low_contrast_grayscale.png"
    # if not cv2.haveImageWriter(mock_image_path):  # 避免重复写入，但通常不会检查
    #     mock_img_data = (np.random.rand(480, 640) * 80 + 80).astype(np.uint8)  # 模拟低对比度灰度图
    #     # 可以在这里选择性添加一点微弱的噪声，如果你想看噪声被放大的效果
    #     # noise_mask = np.random.rand(*mock_img_data.shape)
    #     # mock_img_data[noise_mask < 0.005] = 0
    #     # mock_img_data[noise_mask > 0.995] = 255
    #     cv2.imwrite(mock_image_path, mock_img_data)
    #     print(f"生成模拟灰度图 '{mock_image_path}' 用于演示。")
    #
    # # 调用函数进行 CLAHE 增强
    # apply_clahe_on_grayscale(mock_image_path, clip_limit=3.0, tile_grid_size=(10, 10))

    # 你也可以直接用你自己的图片路径来测试
    apply_clahe_on_grayscale(r"E:\29230\1\111\volunteer01\Image_120910_102034_00580.bmp", clip_limit=2.5, tile_grid_size=(8, 8))

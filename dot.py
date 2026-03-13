import os
import cv2
import numpy as np
import pandas as pd

# =========================
# 配置区
# =========================
IMAGE_DIR = r"E:\29230\1\111\1"  # 请确保路径正确
BLOCK_SIZE = 21  # 稍微增大块大小以包含更多纹理特征 (建议奇数)
SEARCH_RADIUS = 30  # 扩大搜索范围，防止快速运动跑丢
TEMPLATE_UPDATE_RATE = 0.1  # 模板更新率 (0.0=不更新, 1.0=每帧全更新)
# 0.1 表示 90% 保留旧特征，10% 融合新特征

FPS = 25
WAIT_MS = int(1000 / FPS)
SPATIAL_RESOLUTION = 0.71
SAVE_EXCEL = "liver_tracking_enhanced.xlsx"


# =========================
# 工具函数
# =========================
def get_subpixel_shift(corr_map, max_loc):
    """
    使用泰勒展开/抛物线拟合计算亚像素偏移
    """
    x, y = max_loc
    h, w = corr_map.shape

    # 边界检查
    if x <= 0 or x >= w - 1 or y <= 0 or y >= h - 1:
        return 0.0, 0.0

    # 获取中心及周围 4 邻域的值
    c = corr_map[y, x]
    l = corr_map[y, x - 1]
    r = corr_map[y, x + 1]
    u = corr_map[y - 1, x]
    d = corr_map[y + 1, x]

    # 简化的抛物线拟合公式
    dx = (l - r) / (2 * (l + r - 2 * c) + 1e-10)
    dy = (u - d) / (2 * (u + d - 2 * c) + 1e-10)

    return dx, dy


def track_template(ref_template, cur_img, center_prev):
    """
    使用 cv2.matchTemplate 进行快速搜索
    """
    cx, cy = center_prev
    h_t, w_t = ref_template.shape

    # 定义搜索区域 (ROI)
    x_min = max(0, cx - SEARCH_RADIUS - w_t // 2)
    y_min = max(0, cy - SEARCH_RADIUS - h_t // 2)
    x_max = min(cur_img.shape[1], cx + SEARCH_RADIUS + w_t // 2)
    y_max = min(cur_img.shape[0], cy + SEARCH_RADIUS + h_t // 2)

    # 提取 ROI (此时它是 uint8 类型)
    roi = cur_img[y_min:y_max, x_min:x_max]

    # 如果 ROI 比模板还小，说明到了边缘，无法追踪
    if roi.shape[0] < h_t or roi.shape[1] < w_t:
        return None, 0.0

    # ==========================================
    # 🔥 关键修复：将 ROI 转为与 Template 一致的 float32 类型
    # ==========================================
    roi = roi.astype(np.float32)

    # 1. 快速匹配 (NCC)
    res = cv2.matchTemplate(roi, ref_template, cv2.TM_CCOEFF_NORMED)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # 2. 亚像素精修
    sub_dx, sub_dy = get_subpixel_shift(res, max_loc)

    # 3. 计算在原图中的绝对坐标
    best_x = x_min + max_loc[0] + w_t // 2 + sub_dx
    best_y = y_min + max_loc[1] + h_t // 2 + sub_dy

    return np.array([best_x, best_y]), max_val

# =========================
# 鼠标回调
# =========================
clicked_point = None


def mouse_callback(event, x, y, flags, param):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)


# =========================
# 主程序
# =========================
def main():
    global clicked_point

    # 读取图像
    try:
        img_files = sorted([os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR)
                            if f.lower().endswith((".bmp", ".png", ".jpg"))])
    except FileNotFoundError:
        print(f"❌ 目录不存在: {IMAGE_DIR}")
        return

    if not img_files:
        print("❌ 未找到图像文件")
        return

    first_frame = cv2.imread(img_files[0], cv2.IMREAD_GRAYSCALE)

    # --- 选点 ---
    cv2.namedWindow("Select")
    cv2.setMouseCallback("Select", mouse_callback)
    print("👉 请在图像上点击要追踪的肝脏特征点...")

    while clicked_point is None:
        vis = cv2.cvtColor(first_frame, cv2.COLOR_GRAY2BGR)
        cv2.putText(vis, "Click target point", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("Select", vis)
        if cv2.waitKey(20) == 27: return
    cv2.destroyWindow("Select")

    # --- 初始化 ---
    x0, y0 = clicked_point
    cur_x, cur_y = float(x0), float(y0)

    # 提取初始模板
    r = BLOCK_SIZE // 2
    template = first_frame[y0 - r:y0 + r + 1, x0 - r:x0 + r + 1].astype(np.float32)

    records = []

    cv2.namedWindow("Tracking")

    for i, fpath in enumerate(img_files):
        frame = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)

        # 第一帧直接记录
        if i == 0:
            records.append({"frame": 0, "dy_mm": 0.0, "score": 1.0})
            continue

        # --- 核心追踪 ---
        new_pos, score = track_template(template, frame, (int(cur_x), int(cur_y)))

        if new_pos is None or score < 0.4:  # 增加相关性阈值判断
            print(f"⚠️ Frame {i}: Lost tracking (Score: {score:.2f})")
            # 策略：如果丢了，保持上一帧位置，或者标记错误
            # 这里选择保持上一帧位置
        else:
            dx = new_pos[0] - cur_x
            dy = new_pos[1] - cur_y

            # --- 竖直位移软约束 (Optional) ---
            # 如果不是为了强制从物理上限制，通常不建议强行截断，
            # 而是相信 matchTemplate 的结果。如果确实需要限制：
            # if abs(dy) > 10: dy = np.sign(dy) * 10

            cur_x = new_pos[0]
            cur_y = new_pos[1]

            # --- 关键：动态模板更新 (防漂移) ---
            # 提取当前位置的新图像块
            cur_x_int, cur_y_int = int(cur_x), int(cur_y)
            if (cur_y_int - r >= 0 and cur_y_int + r + 1 < frame.shape[0] and
                    cur_x_int - r >= 0 and cur_x_int + r + 1 < frame.shape[1]):

                new_block = frame[cur_y_int - r:cur_y_int + r + 1, cur_x_int - r:cur_x_int + r + 1].astype(np.float32)

                # 只有当匹配度较高时才更新模板，防止把错误的纹理学进去
                if score > 0.6:
                    template = (1 - TEMPLATE_UPDATE_RATE) * template + TEMPLATE_UPDATE_RATE * new_block

        # 计算相对位移
        dy_pixel = cur_y - y0
        dy_mm = dy_pixel * SPATIAL_RESOLUTION

        records.append({
            "frame": i,
            "x": cur_x,
            "y": cur_y,
            "dy_mm": dy_mm,
            "score": score
        })

        # --- 可视化 ---
        vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # 画搜索框范围 (示意)
        cv2.rectangle(vis, (int(cur_x) - SEARCH_RADIUS, int(cur_y) - SEARCH_RADIUS),
                      (int(cur_x) + SEARCH_RADIUS, int(cur_y) + SEARCH_RADIUS), (0, 255, 255), 1)
        # 画追踪点
        cv2.circle(vis, (int(cur_x), int(cur_y)), 4, (0, 0, 255), -1)

        cv2.putText(vis, f"dy: {dy_mm:.2f}mm | Score: {score:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Tracking", vis)
        if cv2.waitKey(WAIT_MS) == 27:
            break

    pd.DataFrame(records).to_excel(SAVE_EXCEL, index=False)
    cv2.destroyAllWindows()
    print("✅ 完成")


if __name__ == "__main__":
    main()
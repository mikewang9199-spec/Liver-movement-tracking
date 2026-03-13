import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# =========================
# 配置区
# =========================
IMAGE_DIR = r"E:\29230\1\111\volunteer01"  # bmp 序列路径
BLOCK_SIZE = 16              # BMA block size
SEARCH_RADIUS = 12           # 搜索半径
POINT_STEP = 8               # ROI 内点间距
SAVE_EXCEL = "bma_roi_tracking_weighted.xlsx"

# =========================
# 工具函数
# =========================
def ncc(a, b):
    """Normalized Cross Correlation"""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    a -= a.mean()
    b -= b.mean()
    denom = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2)) + 1e-8
    return np.sum(a * b) / denom

def get_block(img, center, block_size):
    """安全获取 block（防越界）"""
    x, y = center
    half = block_size // 2
    h, w = img.shape
    if x - half < 0 or y - half < 0 or x + half >= w or y + half >= h:
        return None
    return img[y - half:y + half, x - half:x + half]

def block_matching(ref_img, cur_img, center):
    """基础 BMA"""
    ref_block = get_block(ref_img, center, BLOCK_SIZE)
    if ref_block is None:
        return np.array([0, 0]), -1

    best_score = -1
    best_disp = np.array([0, 0])
    cx, cy = center

    for dx in range(-SEARCH_RADIUS, SEARCH_RADIUS + 1):
        for dy in range(-SEARCH_RADIUS, SEARCH_RADIUS + 1):
            nx, ny = cx + dx, cy + dy
            cur_block = get_block(cur_img, (nx, ny), BLOCK_SIZE)
            if cur_block is None:
                continue
            score = ncc(ref_block, cur_block)
            if score > best_score:
                best_score = score
                best_disp = np.array([dx, dy])
    return best_disp, best_score

# =========================
# OpenCV 选择 ROI
# =========================
def select_roi_cv2(img):
    """用 OpenCV GUI 选择 ROI"""
    roi = cv2.selectROI("Select ROI", img, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    x, y, w, h = roi
    if w == 0 or h == 0:
        raise RuntimeError("❌ 未选择 ROI")
    return x, y, x + w, y + h

def generate_points_in_roi(x_min, y_min, x_max, y_max, step=POINT_STEP):
    points = []
    for x in range(x_min, x_max, step):
        for y in range(y_min, y_max, step):
            points.append((x, y))
    return points

def multi_point_bma_weighted(ref_img, cur_img, points):
    """多点 BMA + NCC 加权平均"""
    disps = []
    scores = []
    for (x, y) in points:
        disp, score = block_matching(ref_img, cur_img, (x, y))
        if score > 0:  # 过滤极低 NCC
            disps.append(disp)
            scores.append(score)
    if len(disps) == 0:
        return np.array([0, 0]), -1
    disps = np.array(disps)
    scores = np.array(scores)
    weights = scores / np.sum(scores)
    weighted_disp = np.sum(disps * weights[:, None], axis=0)
    weighted_ncc = np.sum(scores * weights)
    return weighted_disp, weighted_ncc

# =========================
# 主程序
# =========================
def main():
    # ---------- 读取 bmp 文件 ----------
    img_files = sorted([os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if f.lower().endswith(".bmp")])
    assert len(img_files) > 1, "❌ bmp 文件数量不足"

    images = []
    for f in img_files:
        img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"⚠️ 无法读取，跳过: {f}")
            continue
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        images.append(img)
    assert len(images) > 1, "❌ 有效图像不足"

    h, w = images[0].shape
    print(f"✅ 读取图像数量: {len(images)}, 尺寸: {w}x{h}")

    # ---------- 手动选择 ROI ----------
    x_min, y_min, x_max, y_max = select_roi_cv2(images[0])
    print(f"📍 初始 ROI: ({x_min}, {y_min}) → ({x_max}, {y_max})")
    points = generate_points_in_roi(x_min, y_min, x_max, y_max, step=POINT_STEP)

    # ---------- BMA 主循环 ----------
    positions = [(x_min, y_min)]
    ncc_scores = [1.0]
    cur_x, cur_y = x_min, y_min

    for i in tqdm(range(1, len(images))):
        ref_img = images[i - 1]
        cur_img = images[i]
        disp, avg_ncc = multi_point_bma_weighted(ref_img, cur_img, points)
        cur_x += int(round(disp[0]))
        cur_y += int(round(disp[1]))
        positions.append((cur_x, cur_y))
        ncc_scores.append(avg_ncc)
        # 更新 ROI 内点位置
        points = [(x+int(round(disp[0])), y+int(round(disp[1]))) for x, y in points]

    # ---------- 保存 Excel ----------
    df = pd.DataFrame({
        "frame": np.arange(len(positions)),
        "x": [p[0] for p in positions],
        "y": [p[1] for p in positions],
        "ncc": ncc_scores
    })
    df.to_excel(SAVE_EXCEL, index=False)
    print(f"📁 结果已保存: {SAVE_EXCEL}")

    # ---------- 可视化 ----------
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]

    # 1️⃣ 轨迹曲线
    plt.figure(figsize=(6,6))
    plt.plot(xs, ys, '-o', markersize=2, linewidth=1)
    plt.gca().invert_yaxis()
    plt.xlabel("x (pixel)")
    plt.ylabel("y (pixel)")
    plt.title("Weighted ROI-BMA Tracked Trajectory")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 2️⃣ ROI 轨迹叠加在最后一帧
    plt.figure(figsize=(6,6))
    plt.imshow(images[-1], cmap='gray')
    plt.plot(xs, ys, 'r-', linewidth=1)
    plt.plot(xs[-1], ys[-1], 'ro')
    plt.title("Weighted ROI-BMA Trajectory Overlay")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

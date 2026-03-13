import os
import cv2
import numpy as np
import pandas as pd

# =========================
# 配置区
# =========================
IMAGE_DIR = r"E:\29230\1\111\volunteer01_processed"  # 请确认路径正确
SAVE_EXCEL = "lk_optical_flow_result5-geji.xlsx"

# 物理参数
SPATIAL_RESOLUTION = 0.71  # mm/pixel
FPS = 25
WAIT_MS = int(1000 / FPS)

# --- LK 光流参数 (关键) ---
# winSize: 搜索窗口大小。对于肝脏这种大块组织，21x21 或 31x31 比较稳。
# maxLevel: 金字塔层数。3 表示 1/8, 1/4, 1/2, 原图 四层。层数越高越能捕捉快速运动。
lk_params = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
)

# =========================
# 交互与初始化
# =========================
clicked_point = None


def mouse_callback(event, x, y, flags, param):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)


def main():
    global clicked_point

    # 1. 读取文件列表
    try:
        img_files = sorted([
            os.path.join(IMAGE_DIR, f)
            for f in os.listdir(IMAGE_DIR)
            if f.lower().endswith((".bmp", ".png", ".jpg", ".jpeg"))
        ])
    except FileNotFoundError:
        print(f"❌ 路径错误: {IMAGE_DIR}")
        return

    if len(img_files) < 2:
        print("❌ 图像数量不足")
        return

    # 2. 读取第一帧并选点
    old_frame = cv2.imread(img_files[0])
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    cv2.namedWindow("Select Target")
    cv2.setMouseCallback("Select Target", mouse_callback)

    print("👉 请点击肝脏上纹理明显的特征点（如血管分叉、边缘）...")
    print("   不要点击纯色平滑区域，光流法依赖梯度！")

    while clicked_point is None:
        cv2.imshow("Select Target", old_frame)
        if cv2.waitKey(20) == 27: return

    cv2.destroyWindow("Select Target")

    # 3. 初始化追踪点
    # LK 需要 float32 类型的 numpy 数组，形状为 (n, 1, 2)
    p0 = np.array([[clicked_point]], dtype=np.float32)

    # 记录初始位置
    start_y = p0[0, 0, 1]

    # 用于绘制轨迹
    mask = np.zeros_like(old_frame)
    color = (0, 255, 0)

    records = []
    # 存入第0帧数据
    records.append({
        "frame": 0,
        "x": p0[0, 0, 0],
        "y": p0[0, 0, 1],
        "disp_y_pixel": 0.0,
        "disp_y_mm": 0.0,
        "status": 1
    })

    print(f"🚀 开始追踪... (按 ESC 退出)")

    # 4. 循环处理
    for i in range(1, len(img_files)):
        frame = cv2.imread(img_files[i])
        if frame is None: break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ====================================================
        # 🔥 核心：计算光流
        # p0: 上一帧位置
        # p1: 当前帧预测位置
        # st: 状态 (1=找到, 0=丢失)
        # err: 误差估计
        # ====================================================
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # 检查是否跟丢
        if st[0] == 1:
            # 获取新坐标
            new_x, new_y = p1[0].ravel()
            old_x, old_y = p0[0].ravel()

            # --- 运动约束逻辑 (可选) ---
            # 光流法不需要强制截断位移，因为它能处理大位移。
            # 这里我们只计算数据，不干预 p1 的位置，保证算法连贯性。

            dy_pixel = new_y - start_y
            dy_mm = dy_pixel * SPATIAL_RESOLUTION

            # 记录数据
            records.append({
                "frame": i,
                "x": new_x,
                "y": new_y,
                "disp_y_pixel": dy_pixel,
                "disp_y_mm": dy_mm,
                "status": 1
            })

            # --- 可视化 ---
            # 1. 画轨迹线
            mask = cv2.line(mask, (int(old_x), int(old_y)), (int(new_x), int(new_y)), color, 2)
            vis_img = cv2.add(frame, mask)

            # 2. 画当前点
            cv2.circle(vis_img, (int(new_x), int(new_y)), 5, (0, 0, 255), -1)

            # 3. 显示信息
            info_text = f"Frame: {i} | dY: {dy_mm:.2f} mm"
            cv2.putText(vis_img, info_text, (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow("LK Optical Flow Tracking", vis_img)

            # 更新上一帧数据
            old_gray = frame_gray.copy()
            p0 = p1.reshape(-1, 1, 2)

        else:
            print(f"⚠️ Frame {i}: 追踪点丢失 (特征不明显或移出画面)")
            records.append({
                "frame": i, "x": np.nan, "y": np.nan,
                "disp_y_pixel": np.nan, "disp_y_mm": np.nan, "status": 0
            })
            # 如果丢了，可以选择由用户重新选点，或者直接把 p0 保持不变试试运气
            # 这里简单处理：保持上一帧位置不动
            pass

        k = cv2.waitKey(WAIT_MS) & 0xff
        if k == 27:  # ESC
            break

    cv2.destroyAllWindows()

    # 5. 保存数据
    df = pd.DataFrame(records)
    df.to_excel(SAVE_EXCEL, index=False)
    print(f"✅ 数据已保存: {SAVE_EXCEL}")


if __name__ == "__main__":
    main()
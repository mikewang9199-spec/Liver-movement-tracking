import os
import cv2
import numpy as np
import pandas as pd

# =========================
# 配置区
# =========================
IMAGE_DIR = r"E:\29230\1\111\1"  # 请确认路径正确
SAVE_EXCEL = "lk_dual_tracking_result.xlsx"  # 修改文件名以示区别

# 物理参数
SPATIAL_RESOLUTION = 0.71  # mm/pixel
FPS = 25
WAIT_MS = int(1000 / FPS)

# --- LK 光流参数 ---
lk_params = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
)

# =========================
# 交互部分：多点选择
# =========================
selected_points = []


def mouse_callback(event, x, y, flags, param):
    global selected_points
    if event == cv2.EVENT_LBUTTONDOWN:
        # 限制只能选2个点
        if len(selected_points) < 2:
            selected_points.append((x, y))
            print(f"✅ 已选中第 {len(selected_points)} 个点: ({x}, {y})")


def main():
    global selected_points

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

    cv2.namedWindow("Select Targets")
    cv2.setMouseCallback("Select Targets", mouse_callback)

    print("=======================================================")
    print("👉 操作提示：")
    print("   1. 请点击 【肝脏】 特征点 (Point 1)")
    print("   2. 请点击 【膈肌】 特征点 (Point 2)")
    print("   选完两个点后，程序自动开始追踪。")
    print("=======================================================")

    while len(selected_points) < 2:
        display_img = old_frame.copy()
        # 画出已选的点
        for i, pt in enumerate(selected_points):
            color = (0, 0, 255) if i == 0 else (255, 0, 0)  # 0=红(肝), 1=蓝(膈)
            cv2.circle(display_img, pt, 5, color, -1)
            cv2.putText(display_img, f"P{i + 1}", (pt[0] + 10, pt[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("Select Targets", display_img)
        if cv2.waitKey(20) == 27: return  # ESC 退出

    cv2.destroyWindow("Select Targets")

    # 3. 初始化追踪点
    # 将列表转换为 numpy 数组，形状为 (2, 1, 2)
    p0 = np.array(selected_points, dtype=np.float32).reshape(-1, 1, 2)

    # 记录初始位置
    start_y_liver = p0[0, 0, 1]
    start_y_diaph = p0[1, 0, 1]

    # 初始化轨迹画布 (两个独立的 mask)
    mask_liver = np.zeros_like(old_frame)
    mask_diaph = np.zeros_like(old_frame)

    # 颜色定义
    color_liver = (0, 0, 255)  # 红色
    color_diaph = (255, 0, 0)  # 蓝色

    records = []

    print(f"🚀 开始双目标追踪... (按 ESC 退出)")

    # 4. 循环处理
    for i in range(1, len(img_files)):
        frame = cv2.imread(img_files[i])
        if frame is None: break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ====================================================
        # 🔥 计算光流 (一次性计算两个点)
        # ====================================================
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # 准备当前帧数据容器
        current_data = {"frame": i}

        # --- 处理肝脏 (Index 0) ---
        if st[0] == 1:
            new_x0, new_y0 = p1[0].ravel()
            old_x0, old_y0 = p0[0].ravel()

            dy_pixel_0 = new_y0 - start_y_liver
            dy_mm_0 = dy_pixel_0 * SPATIAL_RESOLUTION

            # 画轨迹 (Mask 1)
            mask_liver = cv2.line(mask_liver, (int(old_x0), int(old_y0)), (int(new_x0), int(new_y0)), color_liver, 2)

            current_data["liver_x"] = new_x0
            current_data["liver_y"] = new_y0
            current_data["liver_disp_mm"] = dy_mm_0
            current_data["liver_status"] = 1
        else:
            current_data["liver_x"] = np.nan
            current_data["liver_y"] = np.nan
            current_data["liver_disp_mm"] = np.nan
            current_data["liver_status"] = 0

        # --- 处理膈肌 (Index 1) ---
        if st[1] == 1:
            new_x1, new_y1 = p1[1].ravel()
            old_x1, old_y1 = p0[1].ravel()

            dy_pixel_1 = new_y1 - start_y_diaph
            dy_mm_1 = dy_pixel_1 * SPATIAL_RESOLUTION

            # 画轨迹 (Mask 2)
            mask_diaph = cv2.line(mask_diaph, (int(old_x1), int(old_y1)), (int(new_x1), int(new_y1)), color_diaph, 2)

            current_data["diaph_x"] = new_x1
            current_data["diaph_y"] = new_y1
            current_data["diaph_disp_mm"] = dy_mm_1
            current_data["diaph_status"] = 1
        else:
            current_data["diaph_x"] = np.nan
            current_data["diaph_y"] = np.nan
            current_data["diaph_disp_mm"] = np.nan
            current_data["diaph_status"] = 0

        records.append(current_data)

        # ====================================================
        # 🎨 可视化：生成两个窗口
        # ====================================================

        # 窗口 1：肝脏 (叠加 mask_liver)
        vis_liver = cv2.add(frame, mask_liver)
        if st[0] == 1:
            cv2.circle(vis_liver, (int(p1[0, 0, 0]), int(p1[0, 0, 1])), 5, color_liver, -1)
            cv2.putText(vis_liver, f"Liver dY: {current_data['liver_disp_mm']:.2f}mm",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_liver, 2)

        # 窗口 2：膈肌 (叠加 mask_diaph)
        vis_diaph = cv2.add(frame, mask_diaph)
        if st[1] == 1:
            cv2.circle(vis_diaph, (int(p1[1, 0, 0]), int(p1[1, 0, 1])), 5, color_diaph, -1)
            cv2.putText(vis_diaph, f"Diaphragm dY: {current_data['diaph_disp_mm']:.2f}mm",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_diaph, 2)

        # 显示双窗口
        cv2.imshow("Window 1: Liver Tracking", vis_liver)
        cv2.imshow("Window 2: Diaphragm Tracking", vis_diaph)

        # 更新
        old_gray = frame_gray.copy()
        p0 = p1.reshape(-1, 1, 2)

        k = cv2.waitKey(WAIT_MS) & 0xff
        if k == 27:  # ESC
            break

    cv2.destroyAllWindows()

    # 5. 保存数据
    if records:
        df = pd.DataFrame(records)
        df.to_excel(SAVE_EXCEL, index=False)
        print(f"✅ 双点数据已保存: {SAVE_EXCEL}")
    else:
        print("无数据记录")


if __name__ == "__main__":
    main()
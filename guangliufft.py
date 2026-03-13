import os
import cv2
import numpy as np
import pandas as pd
from collections import deque

# =========================
# 1. 配置参数区
# =========================
IMAGE_DIR = r"E:\29230\1\111\volunteer01"  # 🔴 请修改为你的图片文件夹路径
SAVE_EXCEL = "image_fft_breathing_result.xlsx"

# 物理参数
SPATIAL_RESOLUTION = 0.71  # mm/pixel

# 🔴这是关键：因为是读取图片，程序不知道拍摄时的帧率。
# 请根据你的超声设备设置填写 (通常是 25, 30 或 60)
FPS = 25
WAIT_MS = int(1000 / FPS)

# 光流参数
lk_params = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
)


# =========================
# 2. 呼吸频率计算类 (FFT 版本)
# =========================
class RespiratoryCounter:
    def __init__(self, buffer_duration_sec, fps):
        self.fps = fps
        # 缓冲区长度 (帧数 = 秒数 * FPS)
        self.buffer_size = int(buffer_duration_sec * fps)
        self.buffer = deque(maxlen=self.buffer_size)
        self.last_bpm = 0.0

        # 频率范围限制 (对应 6 BPM - 40 BPM)
        self.min_freq = 6.0 / 60.0  # 0.1 Hz
        self.max_freq = 40.0 / 60.0  # 0.66 Hz

    def update(self, dy_val):
        self.buffer.append(dy_val)

        # 缓冲区未满时不计算，返回0或上一次的值
        if len(self.buffer) < self.buffer_size:
            return 0.0

        # 1. 准备数据并去直流
        data = np.array(self.buffer)
        data = data - np.mean(data)

        # 2. 加窗 (Hamming)
        window = np.hamming(len(data))
        data_windowed = data * window

        # 3. FFT 变换
        fft_spectrum = np.fft.rfft(data_windowed)
        fft_freqs = np.fft.rfftfreq(len(data), d=1.0 / self.fps)
        fft_magnitude = np.abs(fft_spectrum)

        # 4. 提取有效频段 (6-40 BPM)
        valid_mask = (fft_freqs >= self.min_freq) & (fft_freqs <= self.max_freq)
        valid_freqs = fft_freqs[valid_mask]
        valid_magnitudes = fft_magnitude[valid_mask]

        # 5. 寻找主峰
        if len(valid_magnitudes) > 0:
            peak_idx = np.argmax(valid_magnitudes)
            dominant_freq = valid_freqs[peak_idx]
            bpm = dominant_freq * 60.0

            # 平滑更新 (0.1 的更新率)
            if self.last_bpm == 0:
                self.last_bpm = bpm
            else:
                self.last_bpm = 0.9 * self.last_bpm + 0.1 * bpm

        return self.last_bpm

    def draw_waveform(self, img, x, y, w, h):
        """可视化波形"""
        if len(self.buffer) < 2: return

        overlay = img.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 1)

        data = list(self.buffer)
        min_val, max_val = min(data), max(data)
        if max_val - min_val == 0: return

        prev_pt = None
        for i, val in enumerate(data):
            px = x + int((i / len(self.buffer)) * w)
            norm_y = (val - min_val) / (max_val - min_val)
            py = y + h - int(norm_y * h) - 2

            curr_pt = (px, py)
            if prev_pt:
                cv2.line(img, prev_pt, curr_pt, (0, 255, 0), 1, cv2.LINE_AA)
            prev_pt = curr_pt


# =========================
# 3. 交互与主程序
# =========================
clicked_point = None


def mouse_callback(event, x, y, flags, param):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)


def main():
    global clicked_point

    # --- 1. 读取文件列表 ---
    try:
        img_files = sorted([
            os.path.join(IMAGE_DIR, f)
            for f in os.listdir(IMAGE_DIR)
            if f.lower().endswith((".bmp", ".png", ".jpg", ".jpeg"))
        ])
    except FileNotFoundError:
        print(f"❌ 路径不存在: {IMAGE_DIR}")
        return

    if len(img_files) < 2:
        print("❌ 文件夹内图像数量不足")
        return

    print(f"✅ 找到 {len(img_files)} 张图像，假设帧率 FPS={FPS}")

    # --- 2. 第一帧选点 ---
    old_frame = cv2.imread(img_files[0])
    if old_frame is None:
        print("❌ 无法读取第一帧")
        return

    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    cv2.namedWindow("Image Sequence Tracking")
    cv2.setMouseCallback("Image Sequence Tracking", mouse_callback)

    print("👉 请点击肝脏特征点 (按 ESC 退出)...")
    while clicked_point is None:
        cv2.imshow("Image Sequence Tracking", old_frame)
        if cv2.waitKey(20) == 27: return

    # --- 3. 初始化 ---
    p0 = np.array([[clicked_point]], dtype=np.float32)
    start_y = p0[0, 0, 1]

    # 初始化呼吸计算器 (缓冲15秒)
    respiratory_calc = RespiratoryCounter(buffer_duration_sec=15, fps=FPS)

    mask = np.zeros_like(old_frame)
    records = []

    # 记录第0帧
    records.append({
        "frame": 0, "time_sec": 0.0,
        "disp_y_mm": 0.0, "BPM": 0.0
    })

    print("🚀 开始处理图像序列 (前15秒正在积累数据，BPM可能为0)...")

    # --- 4. 循环处理 ---
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

        if st[0] == 1:
            new_x, new_y = p1[0].ravel()

            # 计算位移和时间
            dy_mm = (new_y - start_y) * SPATIAL_RESOLUTION
            current_time = i / FPS

            # 更新呼吸频率
            current_bpm = respiratory_calc.update(dy_mm)

            records.append({
                "frame": i,
                "time_sec": current_time,
                "disp_y_mm": dy_mm,
                "BPM": current_bpm
            })

            # 可视化
            vis_img = frame.copy()
            cv2.circle(vis_img, (int(new_x), int(new_y)), 5, (0, 0, 255), -1)

            # 绘制波形和文字
            respiratory_calc.draw_waveform(vis_img, 20, vis_img.shape[0] - 120, 200, 100)

            cv2.putText(vis_img, f"Frame: {i}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(vis_img, f"BPM (FFT): {current_bpm:.1f}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255),
                        2)

            cv2.imshow("Image Sequence Tracking", vis_img)

            old_gray = frame_gray.copy()
            p0 = p1.reshape(-1, 1, 2)
        else:
            print(f"⚠️ Frame {i}: 追踪丢失")
            records.append({"frame": i, "time_sec": i / FPS, "disp_y_mm": np.nan, "BPM": 0})

        # 按 ESC 退出
        if cv2.waitKey(WAIT_MS) & 0xff == 27:
            break

    cv2.destroyAllWindows()

    # --- 5. 保存结果 ---
    df = pd.DataFrame(records)
    df.to_excel(SAVE_EXCEL, index=False)
    print(f"✅ 处理完成，数据已保存至: {SAVE_EXCEL}")


if __name__ == "__main__":
    main()
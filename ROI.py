import sys
import os
import cv2
import numpy as np
import pandas as pd
# 将所有 PyQt5 替换为 PySide6 即可，其余代码几乎不用动
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *

# =========================
# 核心处理类（集成降噪与追踪）
# =========================
class LiverTracker:
    def __init__(self):
        # 物理参数
        self.spatial_res = 0.71
        self.fps = 25

        # 算法状态
        self.max_window_size = 7  # 自适应滤波窗口
        self.clip_limit = 2.0  # CLAHE 强度
        self.lk_params = dict(winSize=(21, 21), maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        self.feature_params = dict(maxCorners=100, qualityLevel=0.1, minDistance=7, blockSize=7)

        # 数据缓存
        self.old_gray = None
        self.p0 = None
        self.initial_avg_y = 0
        self.records = []
        self.roi = None

    def adaptive_median_fast(self, img, window_size):
        """高性能自适应中值滤波"""
        out_img = img.copy()
        current_k = 3
        processed = np.zeros(img.shape, dtype=bool)
        while current_k <= window_size:
            kernel = np.ones((current_k, current_k), np.uint8)
            z_med = cv2.medianBlur(out_img, current_k)
            z_min = cv2.erode(out_img, kernel)
            z_max = cv2.dilate(out_img, kernel)
            mask_a = (z_med > z_min) & (z_med < z_max)
            process_mask = mask_a & (~processed)
            replace = (out_img <= z_min) | (out_img >= z_max)
            out_img[process_mask & replace] = z_med[process_mask & replace]
            processed[process_mask] = True
            if np.all(processed): break
            current_k += 2
        return out_img

    def process_frame(self, frame, frame_idx):
        """单帧处理管道：降噪 -> 增强 -> 追踪"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. 实时降噪
        denoised = self.adaptive_median_fast(gray, self.max_window_size)

        # 2. 实时增强
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        # 3. 光流追踪
        vis_frame = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        if self.p0 is not None:
            # 计算光流
            p1, st, _ = cv2.calcOpticalFlowPyrLK(self.old_gray, enhanced, self.p0, None, **self.lk_params)

            # 状态筛选 (st == 1 表示追踪成功)
            if p1 is not None and len(p1) > 0:
                good_new = p1[st == 1]

                if len(good_new) > 0:
                    # 关键修复：统一索引方式。good_new 筛选后通常是 (N, 2)
                    # 我们直接通过列索引访问 x (0) 和 y (1)
                    avg_x = np.mean(good_new[:, 0])
                    avg_y = np.mean(good_new[:, 1])

                    dy_mm = (avg_y - self.initial_avg_y) * self.spatial_res
                    self.records.append({"frame": frame_idx, "disp_y_mm": dy_mm, "pts": len(good_new)})

                    # 绘制点
                    for pt in good_new:
                        cv2.circle(vis_frame, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)

                    # 绘制重心
                    cv2.circle(vis_frame, (int(avg_x), int(avg_y)), 5, (0, 0, 255), -1)

                    # 更新特征点用于下一帧，并保持 (N, 1, 2) 的标准格式
                    self.p0 = good_new.reshape(-1, 1, 2)
                else:
                    self.p0 = None  # 点全部丢失

            self.old_gray = enhanced.copy()

        return vis_frame


# =========================
# GUI 主界面
# =========================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.tracker = LiverTracker()
        self.img_files = []
        self.current_idx = 0
        self.is_playing = False
        self.initUI()

        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)

    def initUI(self):
        self.setWindowTitle("肝脏运动实时追踪分析系统 - 毕设版")
        self.setGeometry(100, 100, 1000, 800)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # --- 图像显示区 ---
        self.image_label = QLabel("请加载文件夹并选择 ROI")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: black; color: white;")
        self.image_label.setMinimumSize(640, 480)
        layout.addWidget(self.image_label)

        # --- 控制面板 ---
        ctrl_layout = QHBoxLayout()

        # 降噪强度滑动条
        self.add_slider(ctrl_layout, "降噪强度 (Window)", 3, 15, 7, self.update_denoise)
        # 增强强度滑动条
        self.add_slider(ctrl_layout, "增强强度 (CLAHE)", 1, 10, 2, self.update_clahe)

        layout.addLayout(ctrl_layout)

        # --- 按钮区 ---
        btn_layout = QHBoxLayout()
        self.btn_load = QPushButton("加载文件夹")
        self.btn_load.clicked.connect(self.load_dir)
        self.btn_roi = QPushButton("框选 ROI 并开始")
        self.btn_roi.clicked.connect(self.select_roi)
        self.btn_export = QPushButton("导出 Excel")
        self.btn_export.clicked.connect(self.export_data)

        btn_layout.addWidget(self.btn_load)
        btn_layout.addWidget(self.btn_roi)
        btn_layout.addWidget(self.btn_export)
        layout.addLayout(btn_layout)

    def add_slider(self, layout, label, min_v, max_v, def_v, callback):
        v_box = QVBoxLayout()
        lbl = QLabel(f"{label}: {def_v}")
        v_box.addWidget(lbl)
        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_v, max_v)
        slider.setValue(def_v)
        slider.valueChanged.connect(lambda v: [callback(v), lbl.setText(f"{label}: {v}")])
        v_box.addWidget(slider)
        layout.addLayout(v_box)

    def update_denoise(self, v):
        # 确保窗口大小为奇数
        self.tracker.max_window_size = v if v % 2 != 0 else v + 1

    def update_clahe(self, v):
        self.tracker.clip_limit = float(v)

    def load_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择图像文件夹")
        if path:
            self.img_files = sorted([os.path.join(path, f) for f in os.listdir(path)
                                     if f.lower().endswith(('.bmp', '.jpg', '.png'))])
            if self.img_files:
                self.show_image(cv2.imread(self.img_files[0]))

    def select_roi(self):
        if not self.img_files: return
        first_frame = cv2.imread(self.img_files[0])
        roi = cv2.selectROI("ROI Selection", first_frame, False)
        cv2.destroyWindow("ROI Selection")

        if roi[2] > 0 and roi[3] > 0:
            x, y, w, h = roi
            gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
            mask = np.zeros_like(gray)
            mask[y:y + h, x:x + w] = 255
            self.tracker.p0 = cv2.goodFeaturesToTrack(gray, mask=mask, **self.tracker.feature_params)
            self.tracker.old_gray = gray
            self.tracker.initial_avg_y = np.mean(self.tracker.p0[:, 0, 1])
            self.is_playing = True
            self.timer.start(int(1000 / self.tracker.fps))

    def next_frame(self):
        if self.current_idx < len(self.img_files):
            frame = cv2.imread(self.img_files[self.current_idx])
            processed = self.tracker.process_frame(frame, self.current_idx)
            self.show_image(processed)
            self.current_idx += 1
        else:
            self.timer.stop()
            self.is_playing = False

    def show_image(self, img):
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(q_img).scaled(self.image_label.width(),
                                                                   self.image_label.height(), Qt.KeepAspectRatio))

    def export_data(self):
        save_path, _ = QFileDialog.getSaveFileName(self, "保存数据", "", "Excel Files (*.xlsx)")
        if save_path and self.tracker.records:
            pd.DataFrame(self.tracker.records).to_excel(save_path, index=False)
            print("数据导出成功！")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
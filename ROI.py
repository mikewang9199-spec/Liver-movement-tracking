import sys
import os
import cv2
import numpy as np
import pandas as pd
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *


# =========================
# 算法核心类
# =========================
class LiverTracker:
    def __init__(self):
        # 物理与基础参数
        self.spatial_res = 0.71
        self.max_window_size = 7
        self.clip_limit = 2.0

        # --- 沿用你认为很好的约束参数 ---
        self.max_drift_threshold = 30  # 离群值阈值

        # --- LK 光流参数 ---
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        # --- 特征点提取参数 ---
        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=0.1,
            minDistance=7,
            blockSize=7
        )

        # 运行时变量
        self.old_gray = None
        self.p0 = None
        self.initial_avg_y = 0
        self.records = []
        self.roi = None

    def adaptive_median_fast(self, img, window_size):
        """自适应中值滤波"""
        if window_size < 3: return img
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

    def get_enhanced_frame(self, frame):
        """处理流水线"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        denoised = self.adaptive_median_fast(gray, self.max_window_size)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=(8, 8))
        return clahe.apply(denoised)


# =========================
# GUI 主界面
# =========================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.tracker = LiverTracker()
        self.img_files = []
        self.current_idx = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.initUI()

    def initUI(self):
        self.setWindowTitle("肝脏追踪系统")
        self.setMinimumSize(1000, 850)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # 图像显示
        self.image_label = QLabel("请加载数据")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: black; border: 1px solid #333;")
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.image_label, stretch=1)

        # 滑动条面板
        ctrl_panel = QHBoxLayout()
        self.add_slider(ctrl_panel, "降噪强度", 1, 15, 7, self.update_params)
        self.add_slider(ctrl_panel, "增强对比度", 1, 10, 2, self.update_params)
        layout.addLayout(ctrl_panel)

        # 按钮
        btn_layout = QHBoxLayout()
        btns = [("加载文件夹", self.load_dir), ("框选ROI并追踪", self.select_roi),
                ("重置", self.reset_tracker), ("导出数据", self.export_data)]
        for text, func in btns:
            btn = QPushButton(text)
            btn.setFixedHeight(40)
            btn.clicked.connect(func)
            btn_layout.addWidget(btn)
        layout.addLayout(btn_layout)

    def add_slider(self, layout, label, min_v, max_v, def_v, callback):
        v_box = QVBoxLayout()
        lbl = QLabel(f"{label}: {def_v}")
        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_v, max_v)
        slider.setValue(def_v)
        slider.valueChanged.connect(lambda v: [callback(v, label), lbl.setText(f"{label}: {v}")])
        v_box.addWidget(lbl)
        v_box.addWidget(slider)
        layout.addLayout(v_box)

    def update_params(self, v, label):
        if "降噪" in label:
            self.tracker.max_window_size = v if v % 2 != 0 else v + 1
        else:
            self.tracker.clip_limit = float(v)
        if self.img_files and not self.timer.isActive():
            self.update_preview()

    def load_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if path:
            self.img_files = sorted([os.path.join(path, f) for f in os.listdir(path)
                                     if f.lower().endswith(('.bmp', '.jpg', '.png'))])
            self.current_idx = 0
            if self.img_files: self.update_preview()

    def update_preview(self):
        raw = cv2.imread(self.img_files[self.current_idx])
        processed = self.tracker.get_enhanced_frame(raw)
        vis = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        if self.tracker.roi:
            x, y, w, h = self.tracker.roi
            cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), 2)
        self.show_image(vis)

    def select_roi(self):
        if not self.img_files: return
        self.timer.stop()
        raw = cv2.imread(self.img_files[self.current_idx])
        processed = self.tracker.get_enhanced_frame(raw)
        temp_vis = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

        roi = cv2.selectROI("ROI", temp_vis, False)
        cv2.destroyWindow("ROI")

        if roi[2] > 0 and roi[3] > 0:
            self.tracker.roi = roi
            x, y, w, h = roi
            mask = np.zeros_like(processed)
            mask[y:y + h, x:x + w] = 255
            p0 = cv2.goodFeaturesToTrack(processed, mask=mask, **self.tracker.feature_params)

            if p0 is not None:
                # 统一维度为 (N, 2)
                self.tracker.p0 = p0.reshape(-1, 1, 2)
                self.tracker.old_gray = processed
                # 初始化平均 Y 值
                temp_p0 = p0.reshape(-1, 2)
                self.tracker.initial_avg_y = np.mean(temp_p0[:, 1])
                self.tracker.records = []
                self.timer.start(40)

    def next_frame(self):
        if self.current_idx >= len(self.img_files):
            self.timer.stop()
            return

        raw = cv2.imread(self.img_files[self.current_idx])
        processed = self.tracker.get_enhanced_frame(raw)
        vis = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

        x, y, w, h = self.tracker.roi
        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), 1)

        if self.tracker.p0 is not None:
            # A. 计算光流
            p1, st, _ = cv2.calcOpticalFlowPyrLK(
                self.tracker.old_gray, processed, self.tracker.p0, None, **self.tracker.lk_params
            )

            if p1 is not None:
                good_new = p1[st == 1]
                if good_new.ndim == 3: good_new = good_new.reshape(-1, 2)

                if len(good_new) > 0:
                    # --- 核心约束：离群值剔除 (Outlier Removal) ---
                    temp_avg_x = np.mean(good_new[:, 0])
                    temp_avg_y = np.mean(good_new[:, 1])

                    # 计算每个点到当前集群中心的距离
                    distances = np.sqrt((good_new[:, 0] - temp_avg_x) ** 2 + (good_new[:, 1] - temp_avg_y) ** 2)
                    # 只保留距离小于阈值的点
                    valid_mask = distances < self.tracker.max_drift_threshold
                    valid_new = good_new[valid_mask]

                    if len(valid_new) > 0:
                        # 计算最终平均位置
                        final_avg_x = np.mean(valid_new[:, 0])
                        final_avg_y = np.mean(valid_new[:, 1])
                        dy_mm = (final_avg_y - self.tracker.initial_avg_y) * self.tracker.spatial_res

                        self.tracker.records.append({"frame": self.current_idx, "dy_mm": dy_mm, "pts": len(valid_new)})

                        # 绘制
                        for pt in valid_new:
                            cv2.circle(vis, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)
                        cv2.circle(vis, (int(final_avg_x), int(final_avg_y)), 6, (0, 255, 255), 2)

                        # 更新状态
                        self.tracker.p0 = valid_new.reshape(-1, 1, 2)
                        self.tracker.old_gray = processed
                    else:
                        self.tracker.p0 = None
                else:
                    self.tracker.p0 = None

        self.show_image(vis)
        self.current_idx += 1

    def show_image(self, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        self.image_label.setPixmap(pix.scaled(self.image_label.width(), self.image_label.height(),
                                              Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def reset_tracker(self):
        self.timer.stop()
        self.current_idx = 0
        self.tracker.p0 = None
        self.tracker.roi = None
        if self.img_files: self.update_preview()

    def export_data(self):
        path, _ = QFileDialog.getSaveFileName(self, "保存", "", "Excel (*.xlsx)")
        if path and self.tracker.records:
            pd.DataFrame(self.tracker.records).to_excel(path, index=False)
            QMessageBox.information(self, "成功", "导出成功")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
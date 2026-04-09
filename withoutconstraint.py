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
        self.spatial_res = 0.71  # 空间分辨率 (mm/pixel)
        self.max_window_size = 7
        self.clip_limit = 2.0

        # 光流参数
        self.lk_params = dict(
            winSize=(31, 31),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        # 特征点检测参数
        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=0.1,
            minDistance=7,
            blockSize=7
        )

        self.old_gray = None
        self.p0 = None
        self.initial_avg_y = 0
        self.records = []
        self.roi = None
        self.mode = "ROI"  # "ROI" 或 "Point"

    def adaptive_median_fast(self, img, window_size):
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
        if frame is None: return None
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
        self.cap = None
        self.current_frame = None
        self.current_idx = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.initUI()

    def initUI(self):
        self.setWindowTitle("肝脏追踪分析系统 (视频版 - 无约束)")
        self.setMinimumSize(1100, 850)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # --- 工具栏 ---
        top_bar = QHBoxLayout()
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["ROI 区域集群追踪", "单点手动追踪"])
        self.mode_selector.currentIndexChanged.connect(self.change_mode)
        top_bar.addWidget(QLabel("选择追踪模式:"))
        top_bar.addWidget(self.mode_selector)
        top_bar.addStretch()
        layout.addLayout(top_bar)

        # 图像显示
        self.image_label = QLabel("请加载视频文件")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: black; border: 1px solid #333;")
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.image_label, stretch=1)

        # 控制面板
        ctrl_panel = QHBoxLayout()
        self.add_slider(ctrl_panel, "降噪强度", 1, 15, 7, self.update_params)
        self.add_slider(ctrl_panel, "增强对比度", 1, 10, 2, self.update_params)
        layout.addLayout(ctrl_panel)

        # 按钮
        btn_layout = QHBoxLayout()
        self.btn_load = QPushButton("加载视频")
        self.btn_load.clicked.connect(self.load_video)
        self.btn_select = QPushButton("选择追踪目标")
        self.btn_select.clicked.connect(self.start_selection)
        self.btn_reset = QPushButton("重置视频")
        self.btn_reset.clicked.connect(self.reset_tracker)
        self.btn_export = QPushButton("导出数据")
        self.btn_export.clicked.connect(self.export_data)

        for b in [self.btn_load, self.btn_select, self.btn_reset, self.btn_export]:
            b.setFixedHeight(40)
            btn_layout.addWidget(b)
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

    def change_mode(self, index):
        self.tracker.mode = "ROI" if index == 0 else "Point"
        self.reset_tracker()

    def update_params(self, v, label):
        if "降噪" in label:
            self.tracker.max_window_size = v if v % 2 != 0 else v + 1
        else:
            self.tracker.clip_limit = float(v)
        if self.current_frame is not None and not self.timer.isActive():
            self.update_preview(self.current_frame)

    def load_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "Video Files (*.mp4 *.avi *.mkv)")
        if path:
            if self.cap is not None: self.cap.release()
            self.cap = cv2.VideoCapture(path)
            self.current_idx = 0
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                self.update_preview(frame)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def update_preview(self, frame):
        processed = self.tracker.get_enhanced_frame(frame)
        vis = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        if self.tracker.roi and self.tracker.mode == "ROI":
            x, y, w, h = self.tracker.roi
            cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), 2)
        self.show_image(vis)

    def start_selection(self):
        if self.cap is None: return
        self.timer.stop()

        # 获取当前帧
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_idx)
        ret, frame = self.cap.read()
        if not ret: return

        processed = self.tracker.get_enhanced_frame(frame)
        temp_vis = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

        if self.tracker.mode == "ROI":
            win_name = "ROI_Selection"
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            roi = cv2.selectROI(win_name, temp_vis, False, False)
            cv2.destroyWindow(win_name)
            if roi[2] > 0 and roi[3] > 0:
                self.tracker.roi = roi
                x, y, w, h = roi
                mask = np.zeros_like(processed)
                mask[y:y + h, x:x + w] = 255
                p0 = cv2.goodFeaturesToTrack(processed, mask=mask, **self.tracker.feature_params)
                if p0 is not None: self.init_tracking(processed, p0)
        else:
            win_name = "Point_Selection"
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            point_data = []

            def on_mouse(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    point_data.append([x, y])
                    cv2.circle(temp_vis, (x, y), 5, (0, 255, 255), -1)
                    cv2.imshow(win_name, temp_vis)

            cv2.imshow(win_name, temp_vis)
            cv2.setMouseCallback(win_name, on_mouse)
            cv2.waitKey(0)
            cv2.destroyWindow(win_name)
            if point_data:
                p0 = np.array(point_data, dtype=np.float32).reshape(-1, 1, 2)
                self.init_tracking(processed, p0)

    def init_tracking(self, processed_img, p0):
        self.tracker.p0 = p0
        self.tracker.old_gray = processed_img
        self.tracker.initial_avg_y = np.mean(p0.reshape(-1, 2)[:, 1])
        self.tracker.records = []
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.timer.start(int(1000 / fps) if fps > 0 else 33)

    def next_frame(self):
        if self.cap is None: return
        ret, raw = self.cap.read()
        if not ret:
            self.timer.stop()
            return

        self.current_frame = raw
        processed = self.tracker.get_enhanced_frame(raw)
        vis = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

        if self.tracker.roi is not None:
            x, y, w, h = self.tracker.roi
            cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), 1)

        if self.tracker.p0 is not None:
            # 计算光流
            p1, st, _ = cv2.calcOpticalFlowPyrLK(
                self.tracker.old_gray, processed, self.tracker.p0, None, **self.tracker.lk_params
            )

            # 仅保留成功追踪到的点 (st==1)
            if p1 is not None and len(p1[st == 1]) > 0:
                good_new = p1[st == 1].reshape(-1, 2)

                # --- 取消了距离中值/平均值的离群约束，直接使用所有有效点 ---
                final_avg_y = np.mean(good_new[:, 1])
                dy_mm = (final_avg_y - self.tracker.initial_avg_y) * self.tracker.spatial_res

                # 记录数据
                self.tracker.records.append({
                    "frame": int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)),
                    "dy_mm": dy_mm
                })

                # 绘制所有点
                for pt in good_new:
                    cv2.circle(vis, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)

                # 更新追踪状态
                self.tracker.p0 = good_new.reshape(-1, 1, 2)
                self.tracker.old_gray = processed
            else:
                self.tracker.p0 = None

        self.show_image(vis)
        self.current_idx += 1

    def show_image(self, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        if self.image_label.width() > 0:
            self.image_label.setPixmap(pix.scaled(self.image_label.width(), self.image_label.height(),
                                                  Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def reset_tracker(self):
        self.timer.stop()
        self.current_idx = 0
        self.tracker.p0 = None
        self.tracker.roi = None
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if ret:
                self.update_preview(frame)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def export_data(self):
        path, _ = QFileDialog.getSaveFileName(self, "导出Excel数据", "", "Excel (*.xlsx)")
        if path and self.tracker.records:
            pd.DataFrame(self.tracker.records).to_excel(path, index=False)
            QMessageBox.information(self, "成功", "数据导出成功")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
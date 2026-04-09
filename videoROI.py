import sys
import cv2
import numpy as np
import pandas as pd
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *


# =========================
# 算法核心类：Supporter
# =========================
class LiverTrackerSupporter:
    def __init__(self):
        self.spatial_res = 0.71
        self.max_window_size = 7
        self.clip_limit = 2.0
        self.alpha = 0.95
        self.max_drift_threshold = 50

        self.lk_params = dict(winSize=(31, 31), maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        self.feature_params = dict(maxCorners=80, qualityLevel=0.05, minDistance=10, blockSize=7)

        self.old_gray = None
        self.p0_supporters = None
        self.u_s = None
        self.target_pos = None
        self.initial_target_y = 0
        self.raw_dy_history = []
        self.detrend_window = 100
        self.records = []
        self.mode = "ROI"  # "ROI" 或 "Point"

    def adaptive_median_fast(self, img, window_size):
        if window_size < 3: return img
        k = int(window_size)
        if k % 2 == 0: k += 1
        out_img = img.copy()
        current_k = 3
        processed = np.zeros(img.shape, dtype=bool)
        while current_k <= k:
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
# 独立视频追踪组件
# =========================
class VideoTrackingUnit(QGroupBox):
    def __init__(self, title, unit_id, parent=None):
        super().__init__(title, parent)
        self.unit_id = unit_id
        self.tracker = LiverTrackerSupporter()
        self.cap = None
        self.current_frame = None
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)

        # 模式切换下拉框
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("追踪模式:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["ROI 区域追踪", "单点追踪"])
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)
        mode_layout.addWidget(self.mode_combo)
        layout.addLayout(mode_layout)

        # 图像显示
        self.image_label = QLabel("未加载视频")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: black; border: 1px solid #555;")
        self.image_label.setMinimumSize(400, 300)
        layout.addWidget(self.image_label, stretch=1)

        # 独立控制滑块
        self.add_slider(layout, "降噪强度", 1, 15, 7, self.update_denoise)
        self.add_slider(layout, "增强对比度", 1, 10, 2, self.update_contrast)

        # 按钮组
        btn_layout = QHBoxLayout()
        self.btn_load = QPushButton("加载视频")
        self.btn_load.clicked.connect(self.load_video)
        self.btn_select = QPushButton("选取目标")
        self.btn_select.clicked.connect(self.start_selection)
        btn_layout.addWidget(self.btn_load)
        btn_layout.addWidget(self.btn_select)
        layout.addLayout(btn_layout)

    def add_slider(self, layout, label, min_v, max_v, def_v, callback):
        h_box = QHBoxLayout()
        lbl = QLabel(f"{label}: {def_v}")
        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_v, max_v)
        slider.setValue(def_v)

        def val_changed(v):
            callback(v)
            lbl.setText(f"{label}: {v}")
            if self.current_frame is not None:
                self.show_frame(self.tracker.get_enhanced_frame(self.current_frame))

        slider.valueChanged.connect(val_changed)
        h_box.addWidget(lbl)
        h_box.addWidget(slider)
        layout.addLayout(h_box)

    def on_mode_changed(self, index):
        self.tracker.mode = "ROI" if index == 0 else "Point"
        self.reset()

    def update_denoise(self, v):
        self.tracker.max_window_size = v

    def update_contrast(self, v):
        self.tracker.clip_limit = float(v)

    def load_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "Videos (*.mp4 *.avi *.mkv)")
        if path:
            if self.cap: self.cap.release()
            self.cap = cv2.VideoCapture(path)
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                self.show_frame(self.tracker.get_enhanced_frame(frame))
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def start_selection(self):
        if self.cap is None: return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self.cap.read()
        if not ret: return

        win_name = f"Selector_{self.unit_id}"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        processed = self.tracker.get_enhanced_frame(frame)

        if self.tracker.mode == "ROI":
            # --- ROI 模式 ---
            roi = cv2.selectROI(win_name, frame, False, False)
            if roi[2] > 0 and roi[3] > 0:
                x, y, w, h = roi
                self.tracker.target_pos = np.array([x + w / 2, y + h / 2], dtype=np.float32)
                mask = np.zeros_like(processed)
                mask[y:y + h, x:x + w] = 255
                p0 = cv2.goodFeaturesToTrack(processed, mask=mask, **self.tracker.feature_params)
                if p0 is not None:
                    self.tracker.p0_supporters = p0.reshape(-1, 2)
                    self.init_tracking_params()
        else:
            # --- 单点模式 ---
            point_data = []
            temp_vis = frame.copy()

            def on_mouse(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    point_data.append([x, y])
                    cv2.circle(temp_vis, (x, y), 5, (0, 255, 255), -1)
                    cv2.imshow(win_name, temp_vis)

            cv2.setMouseCallback(win_name, on_mouse)
            cv2.imshow(win_name, temp_vis)
            cv2.waitKey(0)
            if point_data:
                self.tracker.target_pos = np.array(point_data[0], dtype=np.float32)
                # 单点模式下，支持者就是这一个点
                self.tracker.p0_supporters = self.tracker.target_pos.reshape(1, 2)
                self.init_tracking_params()

        cv2.destroyWindow(win_name)

    def init_tracking_params(self):
        # 统一初始化追踪所需的参数
        self.tracker.initial_target_y = self.tracker.target_pos[1]
        self.tracker.u_s = self.tracker.target_pos - self.tracker.p0_supporters
        self.tracker.old_gray = self.tracker.get_enhanced_frame(self.current_frame)
        self.tracker.raw_dy_history = []
        self.tracker.records = []
        QMessageBox.information(self, "成功", f"{self.title()} 目标已锁定")

    def process_step(self):
        if not self.cap or self.tracker.p0_supporters is None: return
        ret, raw = self.cap.read()
        if not ret: return

        processed = self.tracker.get_enhanced_frame(raw)
        vis = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

        p1, st, _ = cv2.calcOpticalFlowPyrLK(self.tracker.old_gray, processed,
                                             self.tracker.p0_supporters.reshape(-1, 1, 2), None,
                                             **self.tracker.lk_params)

        if p1 is not None and np.any(st == 1):
            st = st.flatten()
            curr_s = p1[st == 1].reshape(-1, 2)
            old_u = self.tracker.u_s[st == 1]
            preds = curr_s + old_u

            # 使用中值预测目标位置
            self.tracker.target_pos = np.median(preds, axis=0)
            self.tracker.p0_supporters = curr_s
            self.tracker.u_s = self.tracker.alpha * old_u + (1 - self.tracker.alpha) * (
                        self.tracker.target_pos - curr_s)

            # 去趋势位移
            raw_dy = (self.tracker.target_pos[1] - self.tracker.initial_target_y) * self.tracker.spatial_res
            self.tracker.raw_dy_history.append(raw_dy)
            if len(self.tracker.raw_dy_history) > self.tracker.detrend_window: self.tracker.raw_dy_history.pop(0)
            final_dy = raw_dy - np.mean(self.tracker.raw_dy_history)

            # 记录数据
            self.tracker.records.append({
                "frame": int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)),
                "dy_mm": round(final_dy, 4)
            })

            # 可视化
            cv2.drawMarker(vis, (int(self.tracker.target_pos[0]), int(self.tracker.target_pos[1])), (0, 0, 255),
                           cv2.MARKER_CROSS, 20, 2)
            for pt in self.tracker.p0_supporters:
                cv2.circle(vis, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1)

        self.tracker.old_gray = processed
        self.show_frame(vis)

    def show_frame(self, img):
        if img is None: return
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.image_label.setPixmap(
            QPixmap.fromImage(qimg).scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio))

    def reset(self):
        self.tracker.p0_supporters = None
        self.tracker.raw_dy_history = []
        self.tracker.records = []
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if ret: self.show_frame(self.tracker.get_enhanced_frame(frame))
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)


# =========================
# 主窗口
# =========================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("双通道肝脏-膈肌追踪系统")
        self.setMinimumSize(1100, 800)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        video_area = QHBoxLayout()
        self.unit_a = VideoTrackingUnit("视频通道 A", "A")
        self.unit_b = VideoTrackingUnit("视频通道 B", "B")
        video_area.addWidget(self.unit_a)
        video_area.addWidget(self.unit_b)
        layout.addLayout(video_area)

        # 全局控制按钮
        global_btn_layout = QHBoxLayout()

        self.btn_run = QPushButton("开始同步追踪")
        self.btn_run.setFixedHeight(45)
        self.btn_run.setStyleSheet("background-color: #2E7D32; color: white; font-weight: bold;")
        self.btn_run.clicked.connect(self.toggle_run)

        self.btn_reset = QPushButton("全部重置")
        self.btn_reset.setFixedHeight(45)
        self.btn_reset.clicked.connect(self.global_reset)

        self.btn_export = QPushButton("导出数据 ")
        self.btn_export.setFixedHeight(45)
        self.btn_export.clicked.connect(self.export_combined_data)

        global_btn_layout.addWidget(self.btn_run)
        global_btn_layout.addWidget(self.btn_reset)
        global_btn_layout.addWidget(self.btn_export)
        layout.addLayout(global_btn_layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.sync_step)

    def toggle_run(self):
        if self.timer.isActive():
            self.timer.stop()
            self.btn_run.setText("恢复追踪")
            self.btn_run.setStyleSheet("background-color: #2E7D32; color: white;")
        else:
            self.timer.start(30)
            self.btn_run.setText("停止追踪")
            self.btn_run.setStyleSheet("background-color: #C62828; color: white;")

    def sync_step(self):
        self.unit_a.process_step()
        self.unit_b.process_step()

    def global_reset(self):
        self.timer.stop()
        self.btn_run.setText("开始同步追踪")
        self.unit_a.reset()
        self.unit_b.reset()

    def export_combined_data(self):
        if not self.unit_a.tracker.records and not self.unit_b.tracker.records:
            QMessageBox.warning(self, "警告", "没有可导出的追踪数据")
            return

        path, _ = QFileDialog.getSaveFileName(self, "保存数据", "", "Excel (*.xlsx)")
        if path:
            try:
                with pd.ExcelWriter(path) as writer:
                    if self.unit_a.tracker.records:
                        df_a = pd.DataFrame(self.unit_a.tracker.records)
                        df_a.to_excel(writer, sheet_name='通道 A 数据', index=False)

                    if self.unit_b.tracker.records:
                        df_b = pd.DataFrame(self.unit_b.tracker.records)
                        df_b.to_excel(writer, sheet_name='通道 B 数据', index=False)

                QMessageBox.information(self, "成功", "数据已成功导出至 Excel")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"导出失败: {str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
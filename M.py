import sys
import cv2
import numpy as np
import csv
import os
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *


# --- 低通滤波器类 ---
class LowPassFilter:
    def __init__(self, alpha=0.15):
        self.alpha = alpha
        self.last_value = None

    def apply(self, current_value):
        if self.last_value is None:
            self.last_value = current_value
            return current_value
        filtered_value = self.alpha * current_value + (1 - self.alpha) * self.last_value
        self.last_value = filtered_value
        return filtered_value


class VideoUnit(QGroupBox):
    def __init__(self, title, mode, depth_mm, v_res, parent=None):
        super().__init__(title, parent)
        self.mode = mode
        self.cap = None
        self.total_frames = 0
        self.fps = 0

        # --- 核心物理参数校准 ---
        # 自动计算像素物理精度 (mm/pixel)
        # 通道 A: 110/530 ≈ 0.2075 | 通道 B: 220/1584 ≈ 0.1389
        self.spatial_res = depth_mm / v_res
        self.scale_factor = 0.75
        self.denoise_level = 3
        self.smooth_alpha = 0.15

        # --- 追踪与滤波器 ---
        self.lp_filter = LowPassFilter(alpha=self.smooth_alpha)
        self.detrend_window = 100
        self.history_disp = []
        self.csv_file = None
        self.csv_writer = None
        self.tracker_ready = False

        # 算法参数
        self.lk_params = dict(winSize=(31, 31), maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        self.old_gray = None
        self.p0_supporters = None
        self.target_pos = None
        self.initial_x = 0
        self.u_s = None
        self.m_pts = []
        self.line_len = 0
        self.m_initial_y = None
        self.last_edge_idx = None
        self.m_buffer = []
        self.m_history_len = 300

        self.initUI()

    def initUI(self):
        self.setFixedWidth(580)
        layout = QVBoxLayout(self)

        # 显示当前的像素分辨率信息
        info_str = f"通道: {self.mode} | 精度: {self.spatial_res:.4f} mm/px"
        self.title_lbl = QLabel(info_str)
        self.title_lbl.setStyleSheet("font-weight: bold; color: #1B5E20;")
        layout.addWidget(self.title_lbl)

        self.img_label = QLabel("等待视频加载...")
        self.img_label.setFixedSize(540, 400)
        self.img_label.setStyleSheet("background: #000; border: 1px solid #CCC;")
        self.img_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.img_label, alignment=Qt.AlignCenter)

        if self.mode == "Diaphragm":
            filter_box = QHBoxLayout()
            self.smooth_lbl = QLabel(f"平滑度(Alpha): {self.smooth_alpha:.2f}")
            self.smooth_lbl.setFixedWidth(120)
            self.slider_alpha = QSlider(Qt.Horizontal)
            self.slider_alpha.setRange(1, 100)
            self.slider_alpha.setValue(int(self.smooth_alpha * 100))
            self.slider_alpha.valueChanged.connect(self.update_alpha)
            filter_box.addWidget(self.smooth_lbl)
            filter_box.addWidget(self.slider_alpha)
            layout.addLayout(filter_box)

            self.m_plot_label = QLabel()
            self.m_plot_label.setFixedSize(540, 120)
            self.m_plot_label.setStyleSheet("background: #000; border: 1px solid #2E7D32;")
            layout.addWidget(QLabel("M-Mode 采样线时空轨迹预览:"))
            layout.addWidget(self.m_plot_label, alignment=Qt.AlignCenter)
        else:
            self.m_plot_label = QLabel()
            self.m_plot_label.hide()

        btns = QHBoxLayout()
        self.btn_load = QPushButton("加载视频")
        self.btn_select = QPushButton("划定目标/取样线")
        self.btn_load.clicked.connect(self.load_video)
        self.btn_select.clicked.connect(self.start_selection)
        btns.addWidget(self.btn_load)
        btns.addWidget(self.btn_select)
        layout.addLayout(btns)

    def update_alpha(self, val):
        self.smooth_alpha = val / 100.0
        self.lp_filter.alpha = self.smooth_alpha
        self.smooth_lbl.setText(f"平滑度(Alpha): {self.smooth_alpha:.2f}")

    def preprocess(self, frame):
        if frame is None: return None
        resized = cv2.resize(frame, (0, 0), fx=self.scale_factor, fy=self.scale_factor)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        denoised = cv2.medianBlur(gray, self.denoise_level)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(denoised)

    def load_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "打开视频", "", "Video (*.mp4 *.avi *.mov)")
        if path:
            if self.cap: self.cap.release()
            self.cap = cv2.VideoCapture(path)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)

            ret, frame = self.cap.read()
            if ret:
                proc = self.preprocess(frame)
                self.display_frame(cv2.cvtColor(proc, cv2.COLOR_GRAY2BGR))
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

                v_name = os.path.splitext(os.path.basename(path))[0]
                out_path = os.path.join(os.path.dirname(path), f"{v_name}_{self.mode}_Data.csv")
                if self.csv_file: self.csv_file.close()
                self.csv_file = open(out_path, 'w', newline='')
                self.csv_writer = csv.writer(self.csv_file)
                self.csv_writer.writerow(['Original_Frame_Idx', 'Physical_Displacement_mm'])
                self.tracker_ready = False

    def start_selection(self):
        if self.cap is None: return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self.cap.read()
        if not ret: return
        sf = self.scale_factor
        win = f"Selection {self.mode}"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

        if self.mode == "Liver":
            roi = cv2.selectROI(win, frame, False)
            if roi[2] > 0:
                sx, sy, sw, sh = [int(v * sf) for v in roi]
                proc = self.preprocess(frame)
                self.target_pos = np.array([sx + sw / 2, sy + sh / 2], dtype=np.float32)
                self.initial_x = self.target_pos[0]
                mask = np.zeros_like(proc)
                mask[sy:sy + sh, sx:sx + sw] = 255
                p0 = cv2.goodFeaturesToTrack(proc, mask=mask, maxCorners=60, qualityLevel=0.03, minDistance=10)
                if p0 is not None:
                    self.p0_supporters = p0.reshape(-1, 2)
                    self.u_s = self.target_pos - self.p0_supporters
                    self.old_gray = proc
                    self.tracker_ready = True
        else:
            pts = []
            temp = frame.copy()

            def cb(e, x, y, f, p):
                nonlocal pts
                if e == cv2.EVENT_LBUTTONDOWN:
                    pts = [(x, y)]
                elif e == cv2.EVENT_LBUTTONUP:
                    pts.append((x, y))
                    cv2.line(temp, pts[0], pts[1], (0, 255, 0), 2)

            cv2.setMouseCallback(win, cb)
            while len(pts) < 2:
                cv2.imshow(win, temp)
                if cv2.waitKey(1) in [13, 27]: break
            if len(pts) == 2:
                self.m_pts = [(p[0] * sf, p[1] * sf) for p in pts]
                self.line_len = int(np.linalg.norm(np.array(self.m_pts[1]) - np.array(self.m_pts[0])))
                self.tracker_ready = True
        cv2.destroyWindow(win)

    def jump_to_and_process(self, ratio):
        if not self.cap or not self.tracker_ready: return

        target_f = int(ratio * (self.total_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_f)
        ret, raw = self.cap.read()
        if not ret: return

        proc = self.preprocess(raw)
        sf = self.scale_factor
        vis = cv2.cvtColor(proc, cv2.COLOR_GRAY2BGR)
        raw_disp_mm = 0.0

        if self.mode == "Liver":
            p1, st, _ = cv2.calcOpticalFlowPyrLK(self.old_gray, proc, self.p0_supporters.reshape(-1, 1, 2), None,
                                                 **self.lk_params)
            if p1 is not None and np.any(st == 1):
                st = st.flatten() == 1
                curr_supporters = p1[st].reshape(-1, 2)
                active_u_s = self.u_s[st]
                predictions = curr_supporters + active_u_s
                rough_consensus = np.median(predictions, axis=0)
                distances = np.linalg.norm(predictions - rough_consensus, axis=1)
                sigma = np.std(distances) + 1e-5
                weights = np.exp(-(distances ** 2) / (2 * sigma ** 2))
                self.target_pos = np.average(predictions, axis=0, weights=weights)
                self.p0_supporters = curr_supporters
                self.u_s = active_u_s
                self.old_gray = proc

                # 计算水平位移并转换为mm
                raw_disp_mm = ((self.target_pos[0] - self.initial_x) / sf) * self.spatial_res

                for pt in curr_supporters:
                    cv2.circle(vis, (int(pt[0]), int(pt[1])), 2, (255, 0, 0), -1)
                cv2.drawMarker(vis, (int(self.target_pos[0]), int(self.target_pos[1])), (0, 0, 255), cv2.MARKER_CROSS,
                               10, 2)

        elif self.mode == "Diaphragm":
            x_c = np.linspace(self.m_pts[0][0], self.m_pts[1][0], self.line_len).astype(np.float32)
            y_c = np.linspace(self.m_pts[0][1], self.m_pts[1][1], self.line_len).astype(np.float32)
            profile = cv2.remap(proc, x_c, y_c, cv2.INTER_LINEAR).flatten()
            self.m_buffer.append(profile)
            if len(self.m_buffer) > self.m_history_len: self.m_buffer.pop(0)

            curr_data = profile.astype(np.float32)
            thresh = np.min(curr_data) + (np.max(curr_data) - np.min(curr_data)) * 0.7
            mask = curr_data > thresh
            raw_idx = np.sum(np.where(mask)[0] * curr_data[mask]) / np.sum(curr_data[mask]) if np.any(mask) else (
                        self.last_edge_idx or self.line_len / 2)

            filtered_idx = self.lp_filter.apply(raw_idx)
            if self.m_initial_y is None: self.m_initial_y = filtered_idx
            self.last_edge_idx = filtered_idx

            # 沿采样线的位移并转换为mm
            raw_disp_mm = ((filtered_idx - self.m_initial_y) / sf) * self.spatial_res

            self.update_m_view(filtered_idx)
            r = filtered_idx / self.line_len
            cx = int(self.m_pts[0][0] + r * (self.m_pts[1][0] - self.m_pts[0][0]))
            cy = int(self.m_pts[0][1] + r * (self.m_pts[1][1] - self.m_pts[0][1]))
            cv2.line(vis, (int(self.m_pts[0][0]), int(self.m_pts[0][1])),
                     (int(self.m_pts[1][0]), int(self.m_pts[1][1])), (0, 255, 255), 1)
            cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)

        self.history_disp.append(raw_disp_mm)
        if len(self.history_disp) > self.detrend_window: self.history_disp.pop(0)
        final_disp = raw_disp_mm - np.mean(self.history_disp)

        if self.csv_writer: self.csv_writer.writerow([target_f, round(final_disp, 4)])
        self.display_frame(vis)

    def update_m_view(self, current_idx):
        if not self.m_buffer: return
        h, w = self.m_plot_label.height(), self.m_plot_label.width()
        m_map = np.array(self.m_buffer).T
        m_img = cv2.normalize(m_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        m_vis = cv2.applyColorMap(m_img, cv2.COLORMAP_JET)
        cv2.circle(m_vis, (m_vis.shape[1] - 1, int(current_idx)), 3, (255, 255, 255), -1)
        m_resized = cv2.resize(m_vis, (w, h))
        qimg = QImage(m_resized.data, w, h, w * 3, QImage.Format_RGB888).rgbSwapped()
        self.m_plot_label.setPixmap(QPixmap.fromImage(qimg))

    def display_frame(self, frame):
        h, w = self.img_label.height(), self.img_label.width()
        resized = cv2.resize(frame, (w, h))
        qimg = QImage(resized.data, w, h, w * 3, QImage.Format_RGB888).rgbSwapped()
        self.img_label.setPixmap(QPixmap.fromImage(qimg))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("肝脏-膈肌呼吸同步分析系统")
        self.progress = 0.0
        self.playback_speed = 0.001

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        view_layout = QHBoxLayout()

        # --- 针对不同分辨率和深度的关键实例化 ---
        # 通道 A (Liver): 深度 110mm, 垂直分辨率 530px
        self.unit_liver = VideoUnit("Liver", "Liver", depth_mm=110, v_res=530)

        # 通道 B (Diaphragm): 深度 220mm, 垂直分辨率 1584px
        self.unit_diaph = VideoUnit("Diaphragm", "Diaphragm", depth_mm=220, v_res=1584)

        view_layout.addWidget(self.unit_liver)
        view_layout.addWidget(self.unit_diaph)
        layout.addLayout(view_layout)

        self.btn_run = QPushButton("开始同步分析")
        self.btn_run.setFixedHeight(50)
        self.btn_run.setStyleSheet("background-color: #2E7D32; color: white; font-weight: bold; font-size: 16px;")
        self.btn_run.clicked.connect(self.toggle)
        layout.addWidget(self.btn_run)

        self.timer = QTimer()
        self.timer.timeout.connect(self.step)

    def toggle(self):
        if self.timer.isActive():
            self.timer.stop()
            self.btn_run.setText("继续分析")
        else:
            if not (self.unit_liver.tracker_ready and self.unit_diaph.tracker_ready):
                QMessageBox.warning(self, "提示", "请先在两个窗口划定ROI或采样线")
                return
            max_f = max(self.unit_liver.total_frames, self.unit_diaph.total_frames)
            self.playback_speed = 1.0 / max_f if max_f > 0 else 0.001
            self.timer.start(1)
            self.btn_run.setText("分析中... (点击暂停)")
            self.btn_run.setStyleSheet("background-color: #C62828; color: white; font-weight: bold;")

    def step(self):
        if self.progress > 1.0:
            self.timer.stop()
            self.btn_run.setText("分析完成")
            return
        self.unit_liver.jump_to_and_process(self.progress)
        self.unit_diaph.jump_to_and_process(self.progress)
        self.progress += self.playback_speed


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
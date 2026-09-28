# -*- coding: utf-8 -*-
"""
M2_Mmode_GUI.py
原生 M 超位移实时分析 GUI

位移计算逻辑：
  1) 滚动检测自动定位 M 超条带
  2) 呼吸频段 FFT 自动定位膈肌亮线深度带
  3) 右缘逐帧采样 + 亮度加权质心追踪膈肌线
  4) 按条带高度与设定深度换算 mm/px
  5) 峰谷检测提取逐呼吸位移 + 质量门控

运行方式（注意使用 PyPy 解释器）：
  D:\\anaconda3\\envs\\Bishe\\pypy3.exe  M2_Mmode_GUI.py
"""
import sys
import os
import csv
from collections import deque

import numpy as np
import cv2
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QLabel, QPushButton, QSpinBox,
                               QGroupBox, QListWidget, QFileDialog, QMessageBox,
                               QDialog)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def centered_moving_mean(x, win):
    """居中滑动均值（uniform_filter1d，无常数填充边界畸变）"""
    from scipy.ndimage import uniform_filter1d
    x = np.asarray(x, dtype=np.float64)
    if win < 2 or len(x) < win:
        return np.zeros_like(x)
    return uniform_filter1d(x, size=int(win), mode="nearest")


def bandpass_zero(x, fs, lo=0.08, hi=0.7):
    """零相位 FFT 带通（去漂移去噪，边界不产生数值畸变）"""
    x = np.asarray(x, dtype=np.float64)
    fft = np.fft.rfft(x - x.mean())
    f = np.fft.rfftfreq(len(x), d=1.0 / fs)
    fft[(f < lo) | (f > hi)] = 0
    return np.fft.irfft(fft, len(x))


def linear_detrend(x):
    """一阶线性去漂移：只去掉整体趋势，保留波形的平台/细节形状。"""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 3:
        return x - x.mean()
    t = np.arange(n, dtype=np.float64)
    tc = t - t.mean()
    xc = x - x.mean()
    k = float(np.sum(tc * xc) / np.sum(tc * tc))
    b = float(x.mean() - k * t.mean())
    return x - (k * t + b)


def resample_to_grid(t, cols, target_fs=60.0):
    """把不等间隔/低帧率曲线线性重采样到 target_fs(默认60Hz)统一网格。
    返回 (t_new, [各列重采样结果])；数值列插值，0/1 列按 >0.5 取整。"""
    t = np.asarray(t, dtype=np.float64)
    t_new = np.arange(t.min(), t.max(), 1.0 / target_fs)
    if t_new[-1] < t.max():
        t_new = np.append(t_new, t.max())
    out = []
    for col in cols:
        col = np.asarray(col, dtype=np.float64)
        v = np.interp(t_new, t, col)
        if col.min() >= 0 and col.max() <= 1 and np.all(np.isin(col, (0, 1))):
            v = (v > 0.5).astype(np.float64)
        out.append(v)
    return t_new, out


def detect_line_position(profile, y_abs_offset):
    """在剖面中找最亮峰，用峰邻域亮度加权质心返回亚像素位置"""
    if profile is None or len(profile) < 8:
        return None, 0.0
    seg = gaussian_filter1d(np.asarray(profile, np.float64), sigma=3, mode="reflect")
    pk = int(np.argmax(seg))
    half = 10
    lo, hi = max(0, pk - half), min(len(seg), pk + half + 1)
    win = seg[lo:hi]
    base = win.min()
    wgt = win - base + 1e-6
    y = y_abs_offset + float(np.average(np.arange(lo, hi), weights=wgt))
    return y, float(seg.max() - np.percentile(seg, 30))


def find_strip_top(cap, n, fps, w, h):
    """通过相邻帧行带互相关找横向滚动的 M 超条带顶部"""
    f0 = int(0.25 * n)
    f1 = min(n - 3, f0 + 3)

    def grab(f):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, fr = cap.read()
        if not ok or fr is None:
            return None
        return cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY) if fr.ndim == 3 else fr

    g1, g2 = grab(f0), grab(f1)
    if g1 is None or g2 is None:
        return None
    shifts = []
    for y0 in range(int(h * 0.35), h - 30, 12):
        a = g1[y0:y0 + 12].mean(axis=0)
        b = g2[y0:y0 + 12].mean(axis=0)
        a, b = a - a.mean(), b - b.mean()
        corr = np.correlate(a, b, mode="full")
        shifts.append((y0, int(np.argmax(corr)) - (len(a) - 1)))
    runs = []
    cur = None
    for y0, s in shifts:
        if abs(s) > 15:
            cur = [y0, y0] if cur is None else [cur[0], y0]
        else:
            if cur is not None:
                runs.append(tuple(cur))
                cur = None
    if cur is not None:
        runs.append(tuple(cur))
    if not runs:
        return None
    merged = [list(runs[0])]
    for y0, y1 in runs[1:]:
        if y0 - merged[-1][1] <= 90:
            merged[-1][1] = y1
        else:
            merged.append([y0, y1])
    return max(merged, key=lambda r: r[1] - r[0])[0]


def find_diaphragm_band(cap, n, fps, w, strip_top, strip_bottom):
    """呼吸频段 FFT：定位条带下半部最强的连续呼吸功率簇（膈肌亮线深度带）"""
    scale = 0.25
    t0 = int(0.15 * n)
    t1 = min(n - 10, t0 + int(15 * fps))
    frames = list(range(t0, t1, 3))
    if len(frames) < 60:
        frames = list(range(max(5, t0), max(t0 + 60, min(n - 5, t0 + 180)), 3))
    arr = []
    for f in frames[::2]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, fr = cap.read()
        if not ok or fr is None:
            continue
        if fr.ndim == 3:
            fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        fr = fr[strip_top:strip_bottom, :]
        fr = cv2.resize(fr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        arr.append(fr.astype(np.float64))
    if len(arr) < 40:
        return None, None
    arr = np.stack(arr)
    H, _ = arr.shape[1], arr.shape[2]
    fft = np.fft.rfft(arr - arr.mean(axis=0, keepdims=True), axis=0)
    freqs = np.fft.rfftfreq(len(arr), d=3 / fps)
    mband = (freqs >= 0.12) & (freqs <= 0.9)
    p = np.abs(fft[mband]).sum(axis=0)
    total = np.abs(fft).mean(axis=0) + 1e-9
    ratio = p / total
    hist = np.bincount(np.where(ratio > np.percentile(ratio, 99))[0], minlength=H)
    strong = hist >= max(5, hist.max() * 0.15)
    clusters = []
    in_c = False
    for i in range(H):
        if strong[i] and not in_c:
            c0, in_c = i, True
        elif not strong[i] and in_c:
            clusters.append((c0, i - 1))
            in_c = False
    if in_c:
        clusters.append((c0, H - 1))
    if not clusters:
        return None, None
    scored = []
    for c0, c1 in clusters:
        power = int(hist[c0:c1 + 1].sum())
        center = strip_top + int(((c0 + c1) / 2) / scale)
        scored.append((c0, c1, power, center))
    lower = [s for s in scored if s[3] > strip_top + 0.45 * (strip_bottom - strip_top)]
    pick = max(lower or scored, key=lambda s: s[2])
    b0 = max(0, pick[0] - 4)
    b1 = min(H - 1, pick[1] + 5)
    return strip_top + int(b0 / scale), strip_top + int(b1 / scale)


class FrameSelector(QWidget):
    """纯 Qt 的鼠标拖框选择器：显示整帧，拖出矩形后自动返回原图像素坐标。"""
    MAX_W, MAX_H = 1100, 720

    def __init__(self, bgr_frame):
        super().__init__()
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        self.orig_h, self.orig_w = rgb.shape[:2]
        self.scale = min(self.MAX_W / self.orig_w, self.MAX_H / self.orig_h)
        disp_w = max(1, int(self.orig_w * self.scale))
        disp_h = max(1, int(self.orig_h * self.scale))
        self.setFixedSize(disp_w, disp_h)
        resized = cv2.resize(rgb, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        qimg = QImage(resized.data, disp_w, disp_h, disp_w * 3,
                      QImage.Format_RGB888).copy()
        self.pixmap = QPixmap.fromImage(qimg)
        self.start = None
        self.end = None
        self.rect_orig = None
        self.on_finished = None
        self.setMouseTracking(True)

    def paintEvent(self, _event):
        p = QPainter(self)
        p.drawPixmap(0, 0, self.pixmap)
        if self.start and self.end:
            x0 = min(self.start.x(), self.end.x())
            y0 = min(self.start.y(), self.end.y())
            x1 = max(self.start.x(), self.end.x())
            y1 = max(self.start.y(), self.end.y())
            p.setPen(QPen(QColor(0, 255, 255), 2))
            p.drawRect(x0, y0, x1 - x0, y1 - y0)
        p.end()

    def _to_orig(self, pos):
        x = int(max(0, min(self.orig_w - 1, pos.x() / self.scale)))
        y = int(max(0, min(self.orig_h - 1, pos.y() / self.scale)))
        return x, y

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.start = event.position().toPoint()
            self.end = self.start
            self.update()

    def mouseMoveEvent(self, event):
        if self.start is not None:
            self.end = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.start is not None:
            self.end = event.position().toPoint()
            x0, y0 = self._to_orig(self.start)
            x1, y1 = self._to_orig(self.end)
            self.rect_orig = (min(x0, x1), min(y0, y1),
                              max(x0, x1), max(y0, y1))
            if self.on_finished is not None:
                self.on_finished()


def select_frame_region(title, bgr_frame, parent=None):
    """弹出一个 Qt 对话框，返回用户拖框选出的原图像素区域 (x0,y0,x1,y1)；取消返回 None。"""
    dlg = QDialog(parent)
    dlg.setWindowTitle(title)
    lay = QVBoxLayout(dlg)
    tip = QLabel("按住鼠标左键拖动，框出目标区域后松开即完成")
    lay.addWidget(tip)
    sel = FrameSelector(bgr_frame)
    sel.on_finished = dlg.accept
    lay.addWidget(sel)
    dlg.setModal(True)
    dlg.exec()
    return sel.rect_orig


class M2Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("M 超膈肌位移实时分析")
        self.resize(1280, 860)

        self.cap = None
        self.video_path = ""
        self.fps = 0.0
        self.total_frames = 0
        self.frame_w = 0
        self.frame_h = 0
        self.mmpx = 0.0
        self.strip_top = 0
        self.strip_bottom = 0
        self.band0 = 0
        self.band1 = 0

        # 实时缓冲
        self.times = []
        self.ys = []
        self.mm_rel = []
        self.contrasts = deque(maxlen=200)
        self.shifts = deque(maxlen=200)
        self.valid_flags = []
        self.last_small = None
        self.m_columns = deque(maxlen=500)
        self.m_track = deque(maxlen=500)
        self.breath_rows = []
        self.last_record_t = -99.0
        self.running = False

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.step_once)
        self._build_ui()

    # ---------------- UI ----------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        top = QHBoxLayout()
        self.btn_open = QPushButton("加载视频")
        self.btn_detect = QPushButton("自动检测条带/膈肌带")
        self.btn_strip_manual = QPushButton("手动选条带")
        self.btn_band_manual = QPushButton("手动选膈肌带")
        self.btn_run = QPushButton("开始实时分析")
        self.btn_save = QPushButton("停止并保存 CSV")
        self.btn_open.clicked.connect(self.open_video)
        self.btn_detect.clicked.connect(self.auto_detect)
        self.btn_strip_manual.clicked.connect(self.manual_select_strip)
        self.btn_band_manual.clicked.connect(self.manual_select_band)
        self.btn_run.clicked.connect(self.toggle_run)
        self.btn_save.clicked.connect(self.save_and_stop)
        for b in (self.btn_open, self.btn_detect, self.btn_strip_manual,
                  self.btn_band_manual, self.btn_run, self.btn_save):
            top.addWidget(b)
        root.addLayout(top)

        param = QHBoxLayout()
        param.addWidget(QLabel("深度(cm):"))
        self.sp_depth = QSpinBox()
        self.sp_depth.setRange(5, 40)
        self.sp_depth.setValue(21)
        param.addWidget(self.sp_depth)
        param.addSpacing(12)
        param.addWidget(QLabel("条带顶:"))
        self.sp_st = QSpinBox(); self.sp_st.setRange(0, 3000); param.addWidget(self.sp_st)
        param.addWidget(QLabel("条带底:"))
        self.sp_sb = QSpinBox(); self.sp_sb.setRange(1, 4000); param.addWidget(self.sp_sb)
        param.addSpacing(12)
        param.addWidget(QLabel("膈肌带:"))
        self.sp_b0 = QSpinBox(); self.sp_b0.setRange(0, 4000); param.addWidget(self.sp_b0)
        self.sp_b1 = QSpinBox(); self.sp_b1.setRange(1, 4000); param.addWidget(self.sp_b1)
        self.lbl_mmpx = QLabel("mm/px: --")
        param.addWidget(self.lbl_mmpx)
        param.addStretch(1)
        root.addLayout(param)

        body = QHBoxLayout()
        col1 = QVBoxLayout()
        self.video_label = QLabel("视频预览")
        self.video_label.setFixedSize(560, 400)
        self.video_label.setStyleSheet("background:#111;color:#aaa;border:1px solid #555;")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.m_label = QLabel("M 超重建 + 追踪轨迹")
        self.m_label.setFixedSize(560, 220)
        self.m_label.setStyleSheet("background:#000;border:1px solid #2E7D32;")
        self.m_label.setAlignment(Qt.AlignCenter)
        col1.addWidget(self.video_label)
        col1.addWidget(self.m_label)
        col1.addWidget(QLabel("实时位移波形 (mm)"))
        self.wave_label = QLabel()
        self.wave_label.setFixedSize(560, 180)
        self.wave_label.setStyleSheet("background:#000;border:1px solid #1565C0;")
        col1.addWidget(self.wave_label)
        body.addLayout(col1)

        col2 = QVBoxLayout()
        self.status_label = QLabel("请加载视频并自动检测")
        self.status_label.setWordWrap(True)
        col2.addWidget(self.status_label)
        col2.addWidget(QLabel("逐呼吸位移（有效呼吸）:"))
        self.breath_list = QListWidget()
        col2.addWidget(self.breath_list, stretch=1)
        body.addLayout(col2, stretch=1)
        root.addLayout(body, stretch=1)

    # ---------------- 基础 ----------------
    def open_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "打开 M 超视频", "", "Video (*.mp4 *.avi *.mov)")
        if not path:
            return
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(path)
        self.video_path = path
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 20.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.status_label.setText(f"已加载: {os.path.basename(path)}\n"
                                  f"{self.frame_w}x{self.frame_h} @{self.fps:.1f}fps, "
                                  f"{self.total_frames}帧 ({self.total_frames / self.fps:.1f}s)")
        self._clear_buffers()
        self._show_first_frame()

    def _clear_buffers(self):
        self.times = []
        self.ys = []
        self.mm_rel = []
        self.contrasts.clear()
        self.shifts.clear()
        self.valid_flags = []
        self.m_columns.clear()
        self.m_track.clear()
        self.breath_rows = []
        self.last_record_t = -99.0
        self.breath_list.clear()

    def _show_first_frame(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, fr = self.cap.read()
        if ok:
            self._show_video(fr, mark=None)

    def _current_params(self):
        self.strip_top = self.sp_st.value()
        self.strip_bottom = self.sp_sb.value()
        self.band0 = self.sp_b0.value()
        self.band1 = self.sp_b1.value()
        hpx = max(1, self.strip_bottom - self.strip_top)
        self.mmpx = self.sp_depth.value() * 10.0 / hpx
        self.lbl_mmpx.setText(f"mm/px: {self.mmpx:.4f} ({self.sp_depth.value()}cm/{hpx}px)")

    # ---------------- 自动检测 ----------------
    def auto_detect(self):
        if self.cap is None:
            QMessageBox.warning(self, "提示", "请先加载视频")
            return
        self.status_label.setText("正在自动检测条带...")
        QApplication.processEvents()
        top = find_strip_top(self.cap, self.total_frames, self.fps,
                             self.frame_w, self.frame_h)
        if top is None:
            QMessageBox.warning(self, "提示",
                                "未检测到滚动的 M 超条带，请确认视频包含原生 M 超，或手动设置参数")
            return
        self.sp_st.setValue(top)
        self.sp_sb.setValue(self.frame_h)
        self.status_label.setText(f"条带: y {top}-{self.frame_h}，正在定位膈肌带...")
        QApplication.processEvents()
        b0, b1 = find_diaphragm_band(self.cap, self.total_frames, self.fps,
                                     self.frame_w, top, self.frame_h)
        if b0 is None or b1 is None or b1 - b0 < 8:
            fall = top + int((self.frame_h - top) * 0.6)
            self.sp_b0.setValue(fall - 25)
            self.sp_b1.setValue(fall + 25)
            QMessageBox.information(self, "提示", "膈肌带自动定位失败，已按条带下部默认值设置，可手动调整")
        else:
            self.sp_b0.setValue(b0)
            self.sp_b1.setValue(b1)
        self._current_params()
        self.status_label.setText(f"自动检测完成\n条带 y[{top},{self.frame_h}]\n"
                                  f"膈肌带 y[{self.sp_b0.value()},{self.sp_b1.value()}]\n"
                                  f"{self.mmpx:.4f} mm/px")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._show_first_frame()

    # ---------------- 手动选择 ----------------
    def _read_frame0(self):
        if self.cap is None:
            return None
        if self.running:
            self.stop_common(save=False)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, fr = self.cap.read()
        return fr if ok else None

    def manual_select_strip(self):
        fr = self._read_frame0()
        if fr is None:
            QMessageBox.warning(self, "提示", "请先加载视频")
            return
        rect = select_frame_region(
            "手动选择 M 超条带：框住整个滚动条带后松开鼠标", fr, self)
        if rect is not None and (rect[3] - rect[1]) > 10:
            self.sp_st.setValue(rect[1])
            self.sp_sb.setValue(rect[3])
            self._current_params()
            self.status_label.setText(
                f"手动条带已设置: y[{rect[1]},{rect[3]}]\n"
                f"{self.mmpx:.4f} mm/px（{self.sp_depth.value()}cm/"
                f"{self.sp_sb.value() - self.sp_st.value()}px）")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._show_first_frame()

    def manual_select_band(self):
        fr = self._read_frame0()
        if fr is None:
            QMessageBox.warning(self, "提示", "请先加载视频")
            return
        rect = select_frame_region(
            "手动选择膈肌亮线区间：框住上下起伏的亮线带后松开鼠标", fr, self)
        if rect is not None and (rect[3] - rect[1]) > 5:
            b0, b1 = rect[1], rect[3]
            # 若还没有有效条带范围，按视频底部为条带底、框上方为条带顶兜底
            if self.sp_sb.value() <= self.sp_st.value():
                self.sp_sb.setValue(self.frame_h)
                self.sp_st.setValue(max(0, b0 - 40))
            self.sp_b0.setValue(b0)
            self.sp_b1.setValue(b1)
            self._current_params()
            self.status_label.setText(
                f"手动膈肌带已设置: y[{b0},{b1}]\n"
                f"条带 y[{self.sp_st.value()},{self.sp_sb.value()}], "
                f"{self.mmpx:.4f} mm/px")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._show_first_frame()

    # ---------------- 实时循环 ----------------
    def toggle_run(self):
        if self.cap is None:
            QMessageBox.warning(self, "提示", "请先加载视频")
            return
        if not self.running:
            self._current_params()
            if self.strip_bottom <= self.strip_top:
                self.auto_detect()
                self._current_params()
            if self.band1 <= self.band0:
                self.status_label.setText("正在自动定位膈肌带...")
                QApplication.processEvents()
                b0, b1 = find_diaphragm_band(
                    self.cap, self.total_frames, self.fps,
                    self.frame_w, self.strip_top, self.strip_bottom)
                if b0 is None or b1 is None or b1 - b0 < 8:
                    fall = self.strip_top + int((self.strip_bottom - self.strip_top) * 0.6)
                    b0, b1 = fall - 25, fall + 25
                    self.status_label.setText("膈肌带自动定位失败，使用条带下部默认带，可手动调整")
                self.sp_b0.setValue(b0)
                self.sp_b1.setValue(b1)
                self._current_params()
            self._clear_buffers()
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.running = True
            self.btn_run.setText("暂停")
            interval = max(1, int(1000.0 / self.fps)) if self.fps <= 60 else 16
            self.timer.start(interval)
        else:
            self.running = False
            self.timer.stop()
            self.btn_run.setText("继续")

    def step_once(self):
        if self.cap is None:
            return
        ok, fr = self.cap.read()
        if not ok or fr is None:
            self.stop_common(save=False)
            self.status_label.setText(self.status_label.text() + "\n[视频播放完毕]")
            return
        self._current_params()
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY) if fr.ndim == 3 else fr
        t = self.cap.get(cv2.CAP_PROP_POS_FRAMES) / self.fps

        # 探头/画面位移（降采样帧差）
        small = cv2.resize(gray, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        shift = float(np.abs(small.astype(np.int16) - self.last_small).mean()) \
            if self.last_small is not None else 0.0
        self.last_small = small
        self.shifts.append(shift)

        # 右缘剖面
        y0 = max(0, self.band0 - 12)
        y1 = min(self.frame_h, self.band1 + 13)
        x0 = max(0, self.frame_w - 70)
        prof = gray[y0:y1, x0:self.frame_w].mean(axis=1)
        y, contrast = detect_line_position(prof, y0)
        if y is None or (prof.max() if len(prof) else 0) < 8:
            y = float(self.ys[-1]) if self.ys else (self.band0 + self.band1) / 2.0
            contrast = 0.0
        self.contrasts.append(contrast)

        # 质量阈值（自适应）
        if len(self.contrasts) >= 50:
            c_thr = float(np.percentile(self.contrasts, 40))
            q75 = np.percentile(self.shifts, 75)
            q95 = np.percentile(self.shifts, 95)
            s_thr = float(q75 + 1.5 * (q95 - q75))
        else:
            c_thr, s_thr = 0.0, 1e18
        valid = contrast >= c_thr and shift <= s_thr

        # 去趋势位移 mm
        self.times.append(t)
        self.ys.append(y)
        self.valid_flags.append(valid)
        mm_now = y * self.mmpx
        win = int(8 * self.fps)
        mm_arr = np.array(self.ys) * self.mmpx
        if len(mm_arr) > win:
            mm_rel = mm_arr - centered_moving_mean(mm_arr, win)
        else:
            mm_rel = mm_arr - mm_arr.mean()
        self.mm_rel = mm_rel.tolist()

        # M 超重建列 + 轨迹
        self.m_columns.append(prof.astype(np.float64))
        self.m_track.append(y - y0)

        # 呼吸检测
        self._update_breaths()

        # 显示
        self._show_video(fr, mark=int(y))
        self._show_m_view()
        self._show_wave()
        n_ok = sum(1 for r in self.breath_rows if r[2])
        self.status_label.setText(
            f"t={t:.1f}s  当前 y={y:.0f}px  位移≈{mm_rel[-1] if len(mm_rel) else 0:+.2f}mm\n"
            f"对比度={contrast:.0f}  画面位移={shift:.2f}  状态={'有效' if valid else '无效'}\n"
            f"帧 {int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))}/{self.total_frames}  "
            f"呼吸 {len(self.breath_rows)}（有效 {n_ok}）")

    def _update_breaths(self):
        if len(self.mm_rel) < int(2.5 * self.fps):
            return
        # 幅度直接用原始 y(px)，不去趋势：避免 8s 滑动均值削掉慢/深呼吸
        sig = np.asarray(self.ys, dtype=np.float64)
        p10, p90 = np.percentile(sig, 10), np.percentile(sig, 90)
        if p90 - p10 < 2.0:
            return
        prom = max((p90 - p10) * 0.3, 2.0)
        peaks, _ = find_peaks(sig, prominence=prom, distance=int(1.1 * self.fps))
        troughs, _ = find_peaks(-sig, prominence=prom, distance=int(1.1 * self.fps))
        for p in peaks:
            if p >= len(sig) - int(0.5 * self.fps):
                continue
            tp = self.times[p]
            if tp - self.last_record_t < 0.7:
                continue
            lt = troughs[troughs < p]
            rt = troughs[troughs > p]
            cand = []
            if len(lt) and self.times[p] - self.times[lt[-1]] < 6.0:
                cand.append(abs(self.ys[p] - self.ys[lt[-1]]) * self.mmpx)
            if len(rt) and self.times[rt[0]] - self.times[p] < 6.0:
                cand.append(abs(self.ys[p] - self.ys[rt[0]]) * self.mmpx)
            if not cand:
                continue
            amp = max(cand)
            ok = self.valid_flags[p] and (self.valid_flags[max(0, lt[-1])] if len(lt) else True) \
                and (self.valid_flags[rt[0]] if len(rt) else True)
            self.breath_rows.append((round(tp, 2), round(amp, 2), bool(ok)))
            self.last_record_t = tp
            tag = "✓" if ok else "✗"
            self.breath_list.addItem(f"{tag} t={tp:6.2f}s  幅度={amp:6.2f} mm")
            self.breath_list.scrollToBottom()

    # ---------------- 绘制 ----------------
    def _show_video(self, frame, mark=None):
        disp = frame.copy()
        if mark is not None and self.band1 > self.band0:
            cv2.line(disp, (0, self.band0), (self.frame_w - 1, self.band0), (0, 0, 255), 1)
            cv2.line(disp, (0, self.band1), (self.frame_w - 1, self.band1), (0, 0, 255), 1)
            cv2.circle(disp, (self.frame_w - 35, int(mark)), 5, (0, 255, 0), -1)
        self._set_pixmap(self.video_label, disp, fit=(560, 400))

    def _show_m_view(self):
        if not self.m_columns:
            return
        mat = np.array(self.m_columns).T
        Hm, Wm = mat.shape
        if Hm < 2 or Wm < 2:
            return
        lo, hi = np.percentile(mat, 2), np.percentile(mat, 98)
        img8 = np.clip((mat - lo) * 255.0 / max(hi - lo, 1), 0, 255).astype(np.uint8)
        vis = cv2.applyColorMap(img8, cv2.COLORMAP_JET)
        if len(self.m_track) >= 2:
            track = list(self.m_track)[-Wm:]
            pts = np.array([[x, min(Hm - 1, max(0, int(v)))] for x, v in enumerate(track)],
                           dtype=np.int32)
            cv2.polylines(vis, [pts], False, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(vis, (Wm - 1, min(Hm - 1, max(0, int(self.m_track[-1])))), 3, (255, 255, 255), -1)
        self._set_pixmap(self.m_label, vis, fit=(560, 220))

    def _show_wave(self):
        w, h = 560, 180
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        if len(self.mm_rel) >= 2:
            n = min(len(self.mm_rel), int(30 * self.fps))
            vals = np.asarray(self.mm_rel[-n:])
            vmax = max(abs(vals.min()), abs(vals.max()), 0.1)
            xs = np.linspace(5, w - 5, len(vals))
            ys_px = h // 2 - vals / vmax * (h // 2 - 8)
            pts = np.column_stack([xs, ys_px]).astype(np.int32)
            cv2.polylines(canvas, [pts], False, (255, 255, 255), 1, cv2.LINE_AA)
            # 有效段绿色刻度
            vv = np.asarray(self.valid_flags[-n:])
            for i in range(1, len(vals)):
                if vv[i] and vv[i - 1]:
                    cv2.line(canvas, (int(xs[i-1]), int(ys_px[i-1])),
                             (int(xs[i]), int(ys_px[i])), (0, 200, 0), 1)
            cv2.line(canvas, (5, h // 2), (w - 5, h // 2), (90, 90, 90), 1)
            cv2.putText(canvas, f"±{vmax:.1f} mm  (绿色=有效)", (8, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        self._set_pixmap(self.wave_label, canvas, fit=(w, h))

    def _set_pixmap(self, label, bgr, fit):
        resized = cv2.resize(bgr, fit, interpolation=cv2.INTER_AREA)
        h, w = resized.shape[:2]
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888).copy()
        label.setPixmap(QPixmap.fromImage(qimg))

    # ---------------- 停止与保存 ----------------
    def stop_common(self, save=False):
        self.running = False
        self.timer.stop()
        self.btn_run.setText("开始实时分析")
        if save and self.video_path:
            self._save_csv()

    def save_and_stop(self):
        self.stop_common(save=True)

    def _save_csv(self):
        base = os.path.splitext(os.path.basename(self.video_path))[0]
        folder = os.path.dirname(self.video_path)
        track_csv = os.path.join(folder, f"{base}_Mmode_track1.csv")
        breath_csv = os.path.join(folder, f"{base}_Mmode_breaths1.csv")
        try:
            # ---- 原生采样数据 -> 统一重采样到 60Hz 后再保存 ----
            if len(self.times) >= 2:
                native_fs = 1.0 / float(np.median(np.diff(np.asarray(self.times))))
                if native_fs < 59.0:      # 只对低于 60Hz 的原始数据做重采样
                    t_new, cols = resample_to_grid(
                        self.times, [np.asarray(self.ys),
                                     np.asarray(self.valid_flags, dtype=np.float64)])
                    y60, valid60 = cols
                else:
                    t_new = np.asarray(self.times)
                    y60 = np.asarray(self.ys)
                    valid60 = np.asarray(self.valid_flags, dtype=np.float64)
                # 相对位移: 只去线性漂移，保留原始波形形状(平台/快变)，不过度平滑
                disp60 = linear_detrend(y60) * self.mmpx
                with open(track_csv, "w", newline="") as f:
                    wr = csv.writer(f)
                    wr.writerow(["Time_s", "Y_px", "Disp_mm", "Valid"])
                    edge = min(240, max(1, len(y60) // 8))  # 首尾缓冲段标为无效
                    for i in range(len(t_new)):
                        wr.writerow([round(float(t_new[i]), 4),
                                     round(float(y60[i]), 3),
                                     round(float(disp60[i]), 4),
                                     0 if (i < edge or i >= len(t_new) - edge) else int(valid60[i])])
            else:
                with open(track_csv, "w", newline="") as f:
                    f.write("Time_s,Y_px,Disp_mm,Valid\n")
            # ---- 逐呼吸幅度 ----
            with open(breath_csv, "w", newline="") as f:
                wr = csv.writer(f)
                wr.writerow(["Peak_Time_s", "Amplitude_mm", "Valid"])
                for t0, amp, ok in self.breath_rows:
                    wr.writerow([t0, amp, int(ok)])
            QMessageBox.information(self, "保存", f"已保存:\n{track_csv}\n{breath_csv}")
        except Exception as e:
            QMessageBox.warning(self, "保存失败", str(e))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = M2Window()
    win.show()
    sys.exit(app.exec())

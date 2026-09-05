import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, butter, filtfilt


# ==========================
# Butterworth低通滤波函数
# ==========================
def butter_lowpass(data, cutoff, fs, order=4):
    """
    data   : 输入信号
    cutoff : 截止频率(Hz)
    fs     : 采样频率(Hz)
    order  : 滤波器阶数
    """
    nyquist = fs * 0.5
    normal_cutoff = cutoff / nyquist

    b, a = butter(order, normal_cutoff, btype='low')
    filtered = filtfilt(b, a, data)

    return filtered


# ==========================
# 读取CSV
# ==========================
input_csv = r"F:\0430ICU\杨兴国-膈肌_Diaphragm_Data.csv"

df = pd.read_csv(input_csv)

# 第二列数据
y = df.iloc[:, 1].values.ravel()


# ==========================
# Step1：SG滤波
# ==========================
sg = savgol_filter(
    y,
    window_length=21,
    polyorder=3
)


# ==========================
# Step2：低通滤波
# ==========================
fs = 25        # 采样频率
cutoff = 0.7   # 截止频率

smooth = butter_lowpass(
    sg,
    cutoff=cutoff,
    fs=fs,
    order=4
)


# ==========================
# 导出结果CSV
# ==========================
result_df = pd.DataFrame({
    'Raw': y,
    'Savitzky_Golay': sg,
    'Final_Filtered': smooth
})

output_csv = r"F:\0430ICU\杨兴国-膈肌_Diaphragm_Data_Filter.csv"

result_df.to_csv(output_csv, index=False, encoding='utf-8-sig')

print(f'滤波结果已保存至:\n{output_csv}')


# ==========================
# 绘图
# ==========================
plt.figure(figsize=(12, 6))

plt.plot(y,
         label='Raw',
         alpha=0.35,
         linewidth=1)

plt.plot(sg,
         label='Savitzky-Golay',
         linewidth=2)

plt.plot(smooth,
         label='SG + Lowpass',
         linewidth=3)

plt.xlabel('Frame')
plt.ylabel('Displacement')
plt.title('Signal Filtering')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()
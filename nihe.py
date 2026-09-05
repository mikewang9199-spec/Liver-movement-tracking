import pandas as pd
import numpy as np
import os
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy import stats, signal


def run_displacement_modeling():
    # ---------------------------------------------------------
    # 1. 数据读取与基础处理
    # ---------------------------------------------------------
    df_a = pd.read_csv(r"F:\本科毕设\毕设数据\gooddata\wtq_深呼吸_肝脏_Liver_Data.csv")
    df_b = pd.read_csv(r"F:\本科毕设\毕设数据\gooddata\wtq_深呼吸_膈肌_Diaphragm_Data.csv")

    disp_a = df_a.iloc[:, 1].values.ravel()
    disp_b = df_b.iloc[:, 1].values.ravel()
    min_len = min(len(disp_a), len(disp_b))

    x_raw = disp_a[:min_len]
    y_raw = disp_b[:min_len]

    # ---------------------------------------------------------
    # 2. 核心逻辑：自动校正正负相关性
    # ---------------------------------------------------------
    def get_best_fit(x_input, y_input):
        """内部函数：执行滞后校正和线性回归，现在返回 p 值"""
        x_norm = (x_input - np.mean(x_input)) / (np.std(x_input) * len(x_input))
        y_norm = (y_input - np.mean(y_input)) / (np.std(y_input))
        corr = signal.correlate(x_norm, y_norm, mode='full')
        lags = signal.correlation_lags(len(x_input), len(y_input), mode='full')

        max_allowed_lag = 100
        lag_mask = (lags >= -max_allowed_lag) & (lags <= max_allowed_lag)
        restricted_lags = lags[lag_mask]
        restricted_corr = corr[lag_mask]

        if len(restricted_corr) > 0:
            best_l = restricted_lags[np.argmax(np.abs(restricted_corr))]
        else:
            best_l = 0

        if best_l > 0:
            x_ali = x_input[best_l:]
            y_ali = y_input[:-best_l]
        elif best_l < 0:
            x_ali = x_input[:best_l]
            y_ali = y_input[-best_l:]
        else:
            x_ali, y_ali = x_input, y_input

        slp, inter, r_v, p_v, std_e = stats.linregress(x_ali, y_ali)
        return slp, inter, r_v, p_v, best_l, x_ali, y_ali   # 现在返回 p 值

    # --- 第一次拟合尝试 ---
    slope, intercept, r_value, p_value, best_lag, x_aligned, y_aligned = get_best_fit(x_raw, y_raw)

    is_inverted = False
    if slope < 0:
        print("检测到负相关，正在对通道 A 进行取反修正以获得正相关模型...")
        x_raw = -x_raw
        is_inverted = True
        slope, intercept, r_value, p_value, best_lag, x_aligned, y_aligned = get_best_fit(x_raw, y_raw)

    r2 = r_value ** 2
    y_predict = slope * x_aligned + intercept

    # ---------------------------------------------------------
    # 【高鲁棒增强版】利用波峰波谷解算呼吸运动幅度 Mean & SD
    # ---------------------------------------------------------
    def calculate_amplitude_stats(signal_data):
        sig_range = np.max(signal_data) - np.min(signal_data)
        if sig_range == 0:
            return 0.0, 0.0, np.array([]), np.array([])

        prominence_thr = sig_range * 0.15  # 显著度调谐为15%，提高ICU浅快呼吸的捕获率

        # 1. 寻找波峰与波谷
        peaks, _ = signal.find_peaks(signal_data, prominence=prominence_thr, distance=25)
        valleys, _ = signal.find_peaks(-signal_data, prominence=prominence_thr, distance=25)

        amplitudes = []
        # 2. 稳健的双向极值对距离匹配
        for p in peaks:
            # 寻找该波峰前 150 帧内的最近波谷
            pre_valleys = valleys[(valleys < p) & (valleys >= p - 150)]
            if len(pre_valleys) > 0:
                amp = signal_data[p] - signal_data[pre_valleys[-1]]
                amplitudes.append(abs(amp))
            else:
                # 如果前面没有波谷，寻找后面 150 帧内的最近波谷
                post_valleys = valleys[(valleys > p) & (valleys <= p + 150)]
                if len(post_valleys) > 0:
                    amp = signal_data[p] - signal_data[post_valleys[0]]
                    amplitudes.append(abs(amp))

        # 3. 极度恶劣噪声工况下的【强制防崩盘兜底机制】
        if len(amplitudes) == 0:
            # 若由于基线漂移未配对成功，则退化为局部滑动窗口统计最大形变，确保代码绝不报错
            window_size = min(150, len(signal_data))
            for i in range(0, len(signal_data) - window_size, window_size // 2):
                sub_seg = signal_data[i:i + window_size]
                amplitudes.append(np.max(sub_seg) - np.min(sub_seg))

        mean_amp = np.mean(amplitudes) if len(amplitudes) > 0 else 0.0
        sd_amp = np.std(amplitudes, ddof=1) if len(amplitudes) > 1 else 0.0

        return mean_amp, sd_amp, peaks, valleys

    # 执行幅度统计
    liver_mean, liver_sd, liver_peaks, liver_valleys = calculate_amplitude_stats(x_aligned)
    diaph_mean, diaph_sd, diaph_peaks, diaph_valleys = calculate_amplitude_stats(y_aligned)

    # ---------------------------------------------------------
    # 2.5 Bland-Altman 一致性指标计算
    # ---------------------------------------------------------
    ba_mean = (y_predict + y_aligned) / 2.0
    ba_diff = y_predict - y_aligned

    mean_bias = np.mean(ba_diff)
    std_bias = np.std(ba_diff, ddof=1)
    upper_loa = mean_bias + 1.96 * std_bias
    lower_loa = mean_bias - 1.96 * std_bias

    within_loa = np.sum((ba_diff >= lower_loa) & (ba_diff <= upper_loa))
    proportion_within = (within_loa / len(ba_diff)) * 100.0 if len(ba_diff) > 0 else 0.0

    # ---------------------------------------------------------
    # 3. 绘图配置（全部原始画图功能完好保留）
    # ---------------------------------------------------------
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # --- 图 1：原始 A/B 通道位移曲线 ---
    plt.figure(figsize=(10, 4))
    plt.plot(x_raw, label='通道 A', color='#1f77b4', linewidth=2)
    plt.plot(y_raw, label='通道 B', color='#ff7f0e', linewidth=2)
    plt.title('原始位移曲线')
    plt.xlabel('时间步 (Frames)')
    plt.ylabel('位移 (mm)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # --- 图 1.5：消除滞后后的曲线（同步包含波峰波谷的圆点标记可视化） ---
    plt.figure(figsize=(10, 4))
    plt.plot(x_aligned, label='通道 A', color='#1f77b4', linewidth=2)
    plt.plot(y_aligned, label='通道 B', color='#ff7f0e', linewidth=2)

    # 仅在成功提取出极值点时才进行 scatter 绘制，防止因空列表报错
    if len(liver_peaks) > 0:
        plt.scatter(liver_peaks, x_aligned[liver_peaks], color='blue', marker='^', s=35, label='肝脏波峰')
    if len(liver_valleys) > 0:
        plt.scatter(liver_valleys, x_aligned[liver_valleys], color='cyan', marker='v', s=35, label='肝脏波谷')
    if len(diaph_peaks) > 0:
        plt.scatter(diaph_peaks, y_aligned[diaph_peaks], color='red', marker='^', s=35, label='膈肌波峰')
    if len(diaph_valleys) > 0:
        plt.scatter(diaph_valleys, y_aligned[diaph_valleys], color='darkred', marker='v', s=35, label='膈肌波谷')

    plt.title(f'消除滞后效应后的两路曲线')
    plt.xlabel('时间步 (Frames)')
    plt.ylabel('位移 (mm)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # --- 图 2：回归散点图 ---
    plt.figure(figsize=(8, 6))
    plt.scatter(x_aligned, y_aligned, s=15, color='blue', alpha=0.3, label='对齐采样点')
    line_x = np.array([np.min(x_aligned), np.max(x_aligned)])
    plt.plot(line_x, slope * line_x + intercept, color='red', linewidth=2.5,
             label=r'拟合直线: $y=%.4fx+%.4f$' % (slope, intercept))
    plt.title(r'线性相关性分析 ($R^2=%.4f$)' % r2)
    plt.xlabel('通道 A 位移 (mm)')
    plt.ylabel('通道 B 位移 (mm)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # --- 图 3：预测对比图 ---
    plt.figure(figsize=(10, 5))
    plt.plot(y_aligned, label='实际观测值 (Channel B)', color='black', linewidth=1.2)
    plt.plot(y_predict, label='模型预测值 (Predicted B)', color='red', linestyle='--', linewidth=1.8)
    plt.title('基于肝脏位移预测膈肌位移的效果对比')
    plt.xlabel('时间步 (Frames)')
    plt.ylabel('位移 (mm)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # --- 新增图 4：Bland-Altman 一致性分析散点图 ---
    plt.figure(figsize=(9, 6.5))
    plt.scatter(ba_mean, ba_diff, s=20, alpha=0.4, color='purple', edgecolors='none', label='联动观测点')
    plt.axhline(mean_bias, color='black', linestyle='-', linewidth=2, label=f'平均偏倚 (Mean Bias): {mean_bias:.4f} mm')
    plt.axhline(upper_loa, color='red', linestyle='--', linewidth=1.8,
                label=f'95%% 一致性上限 (Upper LoA): {upper_loa:.4f} mm')
    plt.axhline(lower_loa, color='red', linestyle='--', linewidth=1.8,
                label=f'95%% 一致性下限 (Lower LoA): {lower_loa:.4f} mm')

    x_text_pos = np.max(ba_mean) - (np.max(ba_mean) - np.min(ba_mean)) * 0.25
    plt.text(x_text_pos, upper_loa + std_bias * 0.15, f'+1.96 SD: {upper_loa:.3f}', color='red', fontweight='bold',
             fontsize=10)
    plt.text(x_text_pos, mean_bias + std_bias * 0.15, f'Mean: {mean_bias:.3f}', color='black', fontweight='bold',
             fontsize=10)
    plt.text(x_text_pos, lower_loa - std_bias * 0.25, f'-1.96 SD: {lower_loa:.3f}', color='red', fontweight='bold',
             fontsize=10)

    plt.title(f'预测值与观测值的 Bland-Altman 一致性分析\n(95% LoA 内点占比: {proportion_within:.2f}%)', fontsize=12,
              fontweight='bold')
    plt.xlabel('位移平均值 (预测值 + 实际值) / 2 (mm)', fontsize=10)
    plt.ylabel('位移差值 (预测值 - 实际值) (mm)', fontsize=10)
    plt.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#CCC')
    plt.grid(True, alpha=0.25, linestyle=':')

    y_range = max(abs(upper_loa), abs(lower_loa)) + std_bias
    plt.ylim(mean_bias - y_range, mean_bias + y_range)
    plt.tight_layout()

    # ---------------------------------------------------------
    # 4. 终端数据控制台输出
    # ---------------------------------------------------------
    print("\n" + "=" * 50)
    if is_inverted:
        print("注意：通道 A 的原始数据已被乘以 -1 以实现正拟合")
    print(f"回归参数：斜率 = {slope:.4f}, 截距 = {intercept:.4f}, 时轴滞后 = {best_lag} 帧")
    print(f"决定系数 R² = {r2:.4f}")
    print(f"线性回归 p 值 = {p_value:.3e}")          # 新增：p 值输出（科学记数法）
    print("-" * 50)
    print("📋 呼吸运动幅度分析 (波峰减波谷法):")
    print(f"  肝脏随动位移 (通道 A) -> 平均幅度: {liver_mean:.4f} mm, 标准差(SD): {liver_sd:.4f} mm")
    print(f"  膈肌实际位移 (通道 B) -> 平均幅度: {diaph_mean:.4f} mm, 标准差(SD): {diaph_sd:.4f} mm")
    print("-" * 50)
    print("Bland-Altman 一致性检验报告：")
    print(f"  平均系统偏倚 (Mean Bias) : {mean_bias:.4f} mm")
    print(f"  差值标准差 (SD of Diff)  : {std_bias:.4f} mm")
    print(f"  95% 一致性上限 (Upper LoA): {upper_loa:.4f} mm")
    print(f"  95% 一致性下限 (Lower LoA): {lower_loa:.4f} mm")
    print(f"  LoA 内部点占比 (Proportion): {proportion_within:.2f}%")
    print("=" * 50 + "\n")

    try:
        plt.show()
    except:
        pass


if __name__ == "__main__":
    run_displacement_modeling()
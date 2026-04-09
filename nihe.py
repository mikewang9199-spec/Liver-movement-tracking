import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy import stats, signal


def run_displacement_modeling():
    # ---------------------------------------------------------
    # 第一步：配置文件路径（建议使用绝对路径防止 FileNotFoundError）
    # ---------------------------------------------------------
    file_path = r"E:\剪映\数据\1.xlsx"

    if not os.path.exists(file_path):
        print(f"❌ 错误：找不到文件 {file_path}")
        print("请确认：1. 文件名是 1.xlsx 还是 1.csv；2. 路径中是否有中文字符导致识别失败。")
        return

    try:
        # 读取 Excel 的两个 Sheet
        print("正在读取数据...")
        df_a = pd.read_excel(file_path, sheet_name='通道 A 数据')
        df_b = pd.read_excel(file_path, sheet_name='通道 B 数据')
    except Exception as e:
        print(f"❌ 读取 Excel 失败: {e}")
        return

    # 提取第二列位移数据，ravel() 确保是一维数组，防止 ValueError
    disp_a = df_a.iloc[:, 1].values.ravel()
    disp_b = df_b.iloc[:, 1].values.ravel()

    # 确保长度一致
    min_len = min(len(disp_a), len(disp_b))
    x_raw = disp_a[:min_len]
    y_raw = disp_b[:min_len]

    # ---------------------------------------------------------
    # 第二步：自动寻找最佳滞后量 (Time Lag Correction)
    # ---------------------------------------------------------
    # 归一化后计算互相关，用于寻找位移同步的最佳偏移点
    x_norm = (x_raw - np.mean(x_raw)) / (np.std(x_raw) * len(x_raw))
    y_norm = (y_raw - np.mean(y_raw)) / (np.std(y_raw))
    corr = signal.correlate(x_norm, y_norm, mode='full')
    lags = signal.correlation_lags(len(x_raw), len(y_raw), mode='full')
    best_lag = lags[np.argmax(np.abs(corr))]  # 使用绝对值因为数据是负相关的

    print(f"✅ 检测到最佳滞后量: {best_lag} 帧")

    # 根据滞后量对齐
    if best_lag > 0:
        x_aligned = x_raw[best_lag:]
        y_aligned = y_raw[:-best_lag]
    elif best_lag < 0:
        x_aligned = x_raw[:best_lag]
        y_aligned = y_raw[-best_lag:]
    else:
        x_aligned, y_aligned = x_raw, y_raw

    # ---------------------------------------------------------
    # 第三步：数学建模 (线性回归)
    # ---------------------------------------------------------
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_aligned, y_aligned)
    r2 = r_value ** 2

    print("\n" + "=" * 40)
    print(f"【最终数学模型】")
    print(f"  公式: B(t) = {slope:.4f} * A(t + {best_lag}) + {intercept:.4f}")
    print(f"  拟合优度 R²: {r2:.4f} (越接近1越精准)")
    print(f"  P值: {p_value:.4e}")
    print("=" * 40)

    # ---------------------------------------------------------
    # 第四步：可视化分析报告
    # ---------------------------------------------------------
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文乱码
    plt.rcParams['axes.unicode_minus'] = False

    fig = plt.figure(figsize=(12, 5))

    # 1. 时域对齐效果图
    plt.subplot(1, 2, 1)
    plt.plot(x_aligned, label='通道 A (对齐后)', alpha=0.8)
    plt.plot(y_aligned, label='通道 B (对齐后)', alpha=0.8)
    plt.title('位移-时间对齐曲线')
    plt.xlabel('时间步 (Frames)')
    plt.ylabel('位移 (mm)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    # 2. 数量关系散点图
    plt.subplot(1, 2, 2)
    plt.scatter(x_aligned, y_aligned, s=10, color='gray', alpha=0.3, label='实验数据点')
    line_x = np.array([np.min(x_aligned), np.max(x_aligned)])
    plt.plot(line_x, slope * line_x + intercept, color='red', label='线性拟合模型')
    plt.title(f'相关性分析 (R²={r2:.3f})')
    plt.xlabel('通道 A 位移 (mm)')
    plt.ylabel('通道 B 位移 (mm)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()

    # 保存结果图片
    plt.savefig('displacement_model_results.png', dpi=300)
    print("\n📊 分析图表已保存为: displacement_model_results.png")

    try:
        plt.show()
    except Exception as e:
        print(f"⚠️ 无法弹出显示窗口 (可能是环境后端问题)，但图片已保存成功。")


if __name__ == "__main__":
    run_displacement_modeling()
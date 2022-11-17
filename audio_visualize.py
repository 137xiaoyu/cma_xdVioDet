# - * - coding: utf-8 - * -
import librosa
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from scipy.fft import fft

matplotlib.rc("font", family='Times New Roman')  # 显示中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 显示符号
matplotlib.rcParams['font.size'] = 20.0

chinese_font_properties = FontProperties(fname='C:/Windows/Fonts/simsun.ttc')

if __name__ == '__main__':
    samples, sr = librosa.load(r'D:/Users/Chase/Desktop/Bullet.in.the.Head.1990__#00-17-20_00-18-55_label_B1-0-0_R_1.wav', sr=48000)

    print(len(samples), sr)
    time = np.arange(0, len(samples)) * (1.0 / sr)

    plt.figure(figsize=(16, 6))
    plt.plot(time, samples)
    # plt.title("语音信号时域波形", fontproperties=FontProperties(fname='C:/Windows/Fonts/simsun.ttc'))
    # plt.xlabel("时间（秒）", fontproperties=chinese_font_properties)
    # plt.ylabel("振幅", fontproperties=chinese_font_properties)
    plt.tight_layout()
    plt.show()

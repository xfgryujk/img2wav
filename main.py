# -*- coding: utf-8 -*-

import wave

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.mlab import window_none


def img2wav(img_path, wav_path, fft_size=1024):
    """
    把图片写到音频频域
    :param img_path: 输入图片路径
    :param wav_path: 输出音频路径
    :param fft_size: 图片每列代表的音频长度，也是频域长度的两倍
    """

    # 读取图片，转为灰度图
    img = Image.open(img_path).convert('L')
    # 缩放到高度 = fft_size / 2 （令负频率全为0）
    img = img.resize((img.width * fft_size // 2 // img.height, fft_size // 2),
                     Image.BICUBIC)

    # 转为numpy数组
    img = np.array(img, 'float')
    # 变换到-100~0分贝，太大了转到时域时可能会溢出
    img = img * (100 / 255) - 100
    # 单位从分贝转成1
    # amp_dB = 20 * ln(amp / 32767) / ln(10)
    # amp = exp(amp_dB / 20 * ln(10)) * 32767
    img = np.exp(img * (np.log(10) / 20)) * 32767
    # 翻转（索引小的频率小）然后转置（要迭代列）
    img = img[::-1].T

    with wave.open(wav_path, 'wb') as f:
        # (nchannels, sampwidth, framerate, nframes, comptype, compname)
        f.setparams((1, 2, 44100, len(img) * fft_size, 'NONE', ''))
        for col in img:
            # 傅里叶反变换
            data = np.fft.ifft(col, fft_size).real

            # 限制范围在-32768~32767
            for index in np.where(data < -32768):
                data[index] = -32768
            for index in np.where(data > 32767):
                data[index] = 32767
            data = data.astype('short')

            # 写到wav文件
            f.writeframesraw(data)


def draw_spectrum(wav_path, fft_size=1024):
    """
    画音频频谱图
    :param wav_path: 输入音频路径
    :param fft_size: 傅里叶变换用的长度
    """

    # 读wav
    with wave.open(wav_path, 'rb') as f:
        n_samples = f.getnframes()
        data = f.readframes(n_samples)
        n_channels = f.getnchannels()
        sample_rate = f.getframerate()

    # 转为numpy数组
    data = np.fromstring(data, 'short')
    # 取第一个声道
    data.shape = (n_samples, n_channels)
    data = data.T[0]

    # 画频谱，无加窗和重叠
    plt.specgram(data / 32767, fft_size, sample_rate, window=window_none,
                 noverlap=0, scale='dB')
    plt.show()


if __name__ == '__main__':
    img2wav('test.jpg', 'test.wav')
    draw_spectrum('test.wav')

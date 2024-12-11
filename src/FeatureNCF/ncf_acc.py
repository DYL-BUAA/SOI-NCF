import math
import os
import re
import sys
import yaml
import copy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Union
import numpy as np
import matplotlib.pyplot as plt
import quaternion
from scipy import signal
from scipy.spatial.transform import Rotation

from input_output import read_sca_file
from input_output import read_orbit_file
from input_output import read_numerical_differentiation_file
from input_output import read_conservative_force_file, read_gforcemodel_file, output_conservative_force_file
from input_output import read_non_conservative_force_file
from input_output import output_non_conservative_force_file, plot_non_conservative_force_acc

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import ConfigSOI

config_soi = ConfigSOI()
# 目录
output_dir = config_soi.output_dir
cf_model_dir = config_soi.cf_model_dir
# 轨道推算时间间隔
time_interval = config_soi.time_interval
# 时间格式
time_in_out_format = config_soi.time_in_out_format
# 窗口长度
win_size = config_soi.num_diff_win_size


def calculate_ncf_acc(diff_acc_file: str, cf_acc_file: str, save_txt: Union[str, bool] = False, save_jpg: Union[str, bool] = False):
    print('-'*25+' '+'Non-Conservative Force'+' '+'-'*25)

    # 读取Num_Diff文件和CF文件
    times, pos_2order_diffs = read_numerical_differentiation_file(diff_acc_file)
    times_, cf_accs = read_conservative_force_file(cf_acc_file)

    pos_2order_diffs = np.array(pos_2order_diffs)
    cf_accs = np.array(cf_accs)
    ncf_accs = pos_2order_diffs - cf_accs

    # 初步处理NCF(去除切换tle时导致的极大NCF)
    ncf_accs = remove_mutation_after_tle_update(ncf_accs=ncf_accs, mode='percentile')

    # 滤波
    ncf_accs = filter_noise(ncf_accs)

    # 输出
    if save_txt:
        output_non_conservative_force_file(times, ncf_accs, savefile=save_txt)
    if save_jpg:
        times_r = []
        time_ref = datetime.strptime(times[0], time_in_out_format)
        for i in range(len(times)):
            delta_t = (datetime.strptime(times[i], time_in_out_format) - time_ref).total_seconds()
            times_r.append(int(delta_t))
        plot_non_conservative_force_acc(times_value=times_r, ncf_accs=ncf_accs, savefile=save_jpg)


def remove_mutation_after_tle_update(ncf_accs, **kwargs):
    """
    两种模式进行异常值去除：标准差和百分位数
    :param ncf_accs:
    :param kwargs:
    :return:
    """
    ncf_x = ncf_accs[:, 0]
    ncf_y = ncf_accs[:, 1]
    ncf_z = ncf_accs[:, 2]

    if kwargs['mode'] == 'normal':
        mean = np.mean(ncf_accs, axis=0)
        std_dev = np.std(ncf_accs, axis=0)
        threshold = 2 * std_dev

        size = int(win_size / 2)
        mutation_flag = []
        for i in range(ncf_accs.shape[1]):
            mutation_index = np.where(np.abs(ncf_accs[:, i] - mean[i]) > threshold[i])[0]
            for m in mutation_index:
                for s in range(-size, size, 1):
                    ms = m + s
                    if ms < 0 or ms >= ncf_accs.shape[0]:
                        continue
                    if ms not in mutation_flag:
                        mutation_flag.append(ms)

    elif kwargs['mode'] == 'percentile':
        mutation = np.percentile(ncf_accs, 99.8, axis=0)

        size = int(win_size / 2)
        mutation_flag = []
        for i in range(ncf_accs.shape[1]):
            mutation_index = np.where(np.abs(ncf_accs[:, i]) > mutation[i])[0]
            for m in mutation_index:
                for s in range(-size, size, 1):
                    ms = m + s
                    if ms < 0 or ms >= ncf_accs.shape[0]:
                        continue
                    if ms not in mutation_flag:
                        mutation_flag.append(ms)
    else:
        raise ValueError("mode should be in ['normal', 'percentile']")

    mutation_flag = np.array(mutation_flag)
    ncf_x[mutation_flag] = 0
    ncf_y[mutation_flag] = 0
    ncf_z[mutation_flag] = 0
    ncf_accs = np.vstack((ncf_x, ncf_y, ncf_z)).T

    return ncf_accs


def filter_noise(ncf_accs):
    """
    去除NCF中的高频噪声
    :param ncf_accs: [[x,y,z]]
    :return: 去除噪声后的NCF: [[x,y,z]]
    """
    if isinstance(ncf_accs, list):
        ncf_accs = np.array(ncf_accs)

    if isinstance(ncf_accs, np.ndarray):
        # 长度
        length = len(ncf_accs)

        # 巴特沃斯低通滤波器
        b, a = signal.butter(N=3, Wn=[3e-6, 2e-4], btype='bandpass', fs=1/time_interval)

        filtered_acc_x = signal.filtfilt(b, a, ncf_accs[:, 0])
        filtered_acc_y = signal.filtfilt(b, a, ncf_accs[:, 1])
        filtered_acc_z = signal.filtfilt(b, a, ncf_accs[:, 2])

        ncf_accs = []
        for i in range(length):
            ncf_accs.append([filtered_acc_x[i], filtered_acc_y[i], filtered_acc_z[i]])
        return ncf_accs
    else:
        raise TypeError("ncf_accs should be in format like this: [[x,y,z]] or Array([[x,y,z]])")


def get_ncf_acc_srf(sca_txt: str, ncf_txt: str, plot_flag=False):
    """
    计算GRACE-FO 卫星(gni1b,sca1b产品) SBF坐标系下NCF
    :param sca_txt:
    :param ncf_txt:
    :param plot_flag:
    :return:
    """
    from input_output import read_sca_file, read_non_conservative_force_file

    # 读取数据
    times, ncf_accs = read_non_conservative_force_file(ncf_txt)
    _, quaternions = read_sca_file(sca_txt)  # 单位四元数

    times_value = np.linspace(0, len(times) - 1, len(times))
    ncf_accs_eci = np.array(ncf_accs)
    quaternions = np.array(quaternions)

    # 将四元数转换为旋转矩阵
    rotations = Rotation.from_quat(quaternions)
    # 将加速度由ECI转至SBF（SRF）
    ncf_accs_srf = np.array([rotations[i].as_matrix().T.dot(ncf_accs_eci[i]) for i in range(len(ncf_accs_eci))])

    if plot_flag:
        colors = ['#1f77b4', '#ff7f0e', 'g']
        # 绘制非保守力加速度
        plt.subplots(nrows=3, ncols=1, sharex='col')
        labels = ['ncf_acc_x', 'ncf_acc_y', 'ncf_acc_z']
        plt.subplot(3, 1, 1)
        plt.plot(times_value, ncf_accs_srf[:, 0], color=colors[0], label=labels[0])
        plt.xlabel('Time (s)')
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(times_value, ncf_accs_srf[:, 1], color=colors[1], label=labels[1])
        plt.xlabel('Time (s)')
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(times_value, ncf_accs_srf[:, 2], color=colors[2], label=labels[2])
        plt.xlabel('Time (s)')
        plt.legend()
        plt.ylabel('Non-conservative Force Acceleration (nm/s$^2$)', labelpad=10, y=1.75, rotation=90)

        plt.suptitle('Components of Non-conservative Force Acceleration in XYZ direction in Satellite Body Fixed Frame')
        plt.show()

    return ncf_accs_srf



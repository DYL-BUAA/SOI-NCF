# -*- coding = utf-8 -*-
# @Time:2023/9/12,16:34
# @Author:邓云龙
# @File:GP.py
# @Software:PyCharm

import os
import re
import sys
import copy
import yaml
from pathlib import Path
import math
import datetime
from datetime import datetime, timedelta
from typing import Optional, Union
import numpy as np
from numpy.polynomial.chebyshev import Chebyshev
from scipy.spatial import transform
from scipy.fftpack import fft, ifft
from scipy.interpolate import lagrange, polyint, interp1d
import matplotlib.pyplot as plt

from SpaceObjectGP import SOGP
from input_output import read_orbit_file, output_numerical_differentiation_file, plot_numerical_differentiation_acc
from coordinate_frame_transform import teme_to_j2000_skyfield

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import ConfigSOI

# time_in_out_format = "%Y-%m-%dT%H:%M:%S.%f"
#
# output_dir = "E:/Program/Pycharm2022/projects/SOI-NCF/output"
config_soi = ConfigSOI()

# 目录
output_dir = config_soi.output_dir
# 时间格式
time_in_out_format = config_soi.time_in_out_format
time_filename_format = config_soi.time_filename_format
# 轨道推算时间间隔
time_interval = config_soi.time_interval


def lagrange_interpolation_n_order_diff(t, x, t_diff, n=1):
    """
    lagrange数值微分法
    估计中心点x的导数
    :param t: N dim list
    :param x: N dim list
    :param t_diff: 待差分的t
    :param n: n阶微分
    :return: t处的lagrange差分值
    """
    # 创建拉格朗日插值函数
    t = np.array(t)
    f_lagrange = lagrange(t, x)   # poly_1d
    # print('lagrange插值函数：', f_lagrange)
    derive_lagrange = np.polyder(f_lagrange, n)  # poly_1d
    # print('lagrange插值函数导数:', derive_lagrange)
    return derive_lagrange(t_diff)


def acc_lagrange(orbit_file: str, save_txt: Union[bool, str] = False, save_jpg: Union[bool, str] = False, win_size=11):
    """
    lagrange数值微分法计算空间目标合加速度
    :param orbit_file:
    :param win_size: 文件保存名字，仅名字
    :param save_txt: 是否保存txt
    :param save_jpg: 是否保存jpg
    :return:
    """
    print('-'*25 + ' ' + 'Numerical Differentiation' + ' ' + '-'*25)
    half_win_size = int(win_size / 2) + 1

    # 读取时间、位置、速度
    times, poss, vels = read_orbit_file(orbit_file)
    poss, vels = np.array(poss), np.array(vels)

    # utc时间转为相对起始时刻的时间
    times_datetime, times_r = [], []
    for i in range(len(times)):
        times_datetime.append(datetime.strptime(times[i], time_in_out_format))
        # Time relative to the starting point, 单位：s
        times_r.append((times_datetime[i] - times_datetime[0]).total_seconds())
    times_r = np.array(times_r)

    # 截取中心点的N_point_lagrange长度的序列片段，求数值微分加速度
    pos_2order_diffs = []
    total_num = len(times)
    for i in range(total_num):
        if 0 <= i < half_win_size:  # 开始的N/2个点
            t = times_r[0: win_size] - times_r[i]
            p = poss[0: win_size]
        elif half_win_size <= i < total_num-half_win_size:  # 中间部分
            t = times_r[i - half_win_size: i + half_win_size + 1] - times_r[i]
            p = poss[i - half_win_size: i + half_win_size + 1]
        else:  # 结尾的N/2个点
            t = times_r[-win_size:] - times_r[i]
            p = poss[-win_size:]

        pos_2order_diff_x = lagrange_interpolation_n_order_diff(t, p[:, 0], 0, n=2)
        pos_2order_diff_y = lagrange_interpolation_n_order_diff(t, p[:, 1], 0, n=2)
        pos_2order_diff_z = lagrange_interpolation_n_order_diff(t, p[:, 2], 0, n=2)
        pos_2order_diffs.append([pos_2order_diff_x, pos_2order_diff_y, pos_2order_diff_z])

    '''保存数值微分结果'''
    if save_txt:
        output_numerical_differentiation_file(times, pos_2order_diffs, savefile=save_txt)
    if save_jpg:
        plot_numerical_differentiation_acc(list(times_r), pos_2order_diffs, savefile=save_jpg)


if __name__ == "__main__":
    refine_f = r'orbit.txt'
    diff_txt = False
    diff_pic = True
    acc_lagrange(orbit_file=refine_f, save_txt=diff_txt, save_jpg=diff_pic)  # 数值微分计算合加速度






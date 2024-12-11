import os
import math
import sys
import yaml
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from astropy.time import Time
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from SpaceObjectGP import SOGP
from predict_orbit import predict_orbit, predict_orbit_gps
from diff_acc import acc_lagrange
from cf_acc import calculate_cf_acc
from ncf_acc import calculate_ncf_acc
from input_output import output_gni_to_tleorbit

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import ConfigSOI


def del_dir(dir_to_del):
    for root, dirs, files in os.walk(dir_to_del):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    os.rmdir(dir_to_del)


def get_ncf_from_norad(norad_id: int, time_start: datetime, time_end: datetime) -> None:
    """
    计算并提取目标在采样时间点上的非保守力特征
    :param norad_id:
    :param time_start:
    :param time_end:
    :return:
    """
    # 配置参数
    config_soi = ConfigSOI()
    output_dir = config_soi.output_dir
    time_filename_format = config_soi.time_filename_format
    time_interval = config_soi.time_interval

    # # 1 推算轨道
    name = f"{str(norad_id).zfill(5)}_J2000_{time_start.strftime(time_filename_format)}-{time_end.strftime(time_filename_format)}"

    name_orbit = name + config_soi.orbit_suffix
    name_orbit_txt = name_orbit + '.txt'
    name_orbit_jpg = name_orbit + '.jpg'

    orbit_txt_dir = os.path.join(output_dir, 'orbit', f'{time_interval}', 'txt')
    orbit_pic_dir = os.path.join(output_dir, 'orbit', f'{time_interval}', 'pic')
    os.makedirs(orbit_txt_dir, exist_ok=True)  # 创建txt路径
    os.makedirs(orbit_pic_dir, exist_ok=True)  # 创建jpg路径

    orbit_txt = os.path.join(orbit_txt_dir, name_orbit_txt)
    orbit_pic = os.path.join(orbit_pic_dir, name_orbit_jpg)

    predict_orbit(norad_id, time_start, time_end, teme2j2000=True, save_txt=orbit_txt, save_jpg=orbit_pic)  # 推算轨道

    # # 2 数值微分计算合加速度
    win_size = config_soi.num_diff_win_size
    name_diff = name + config_soi.diff_suffix
    name_diff_txt = name_diff + '.txt'
    name_diff_jpg = name_diff + '.jpg'
    diff_txt_dir = os.path.join(output_dir, 'num_diff', f'{time_interval}', 'txt')
    diff_pic_dir = os.path.join(output_dir, 'num_diff', f'{time_interval}', 'pic')
    os.makedirs(diff_txt_dir, exist_ok=True)  # 创建txt路径
    os.makedirs(diff_pic_dir, exist_ok=True)  # 创建jpg路径

    diff_txt = os.path.join(diff_txt_dir, name_diff_txt)
    diff_pic = os.path.join(diff_pic_dir, name_diff_jpg)

    acc_lagrange(orbit_file=orbit_txt, save_txt=diff_txt, save_jpg=diff_pic, win_size=win_size)  # 数值微分计算合加速度

    # # 3 精密保守力模型计算保守力加速度
    name_cf = name + config_soi.cf_suffix
    name_cf_txt = name_cf + '.txt'
    # name_cf_jpg = name_cf + '.jpg'
    cf_txt_dir = os.path.join(output_dir, 'cf', f'{time_interval}', 'txt')
    # cf_pic_dir = os.path.join(output_dir, 'cf', f'{time_interval}', 'pic')
    os.makedirs(cf_txt_dir, exist_ok=True)  # 创建txt路径
    # os.makedirs(cf_pic_dir, exist_ok=True)  # 创建jpg路径

    cf_txt = os.path.join(cf_txt_dir, name_cf_txt)
    # diff_pic = os.path.join(diff_pic_dir, name_diff_jpg)

    calculate_cf_acc(orbit_file=orbit_txt, cf_acc_file=cf_txt)  # 精密保守力模型计算保守力加速度

    # # 4 计算非保守力加速度
    name_ncf = name + config_soi.ncf_suffix
    name_ncf_txt = name_ncf + '.txt'
    name_ncf_jpg = name_ncf + '.jpg'
    ncf_txt_dir = os.path.join(output_dir, 'ncf', f'{time_interval}', 'txt')
    ncf_pic_dir = os.path.join(output_dir, 'ncf', f'{time_interval}', 'pic')
    os.makedirs(ncf_txt_dir, exist_ok=True)  # 创建txt路径
    os.makedirs(ncf_pic_dir, exist_ok=True)  # 创建jpg路径

    ncf_txt = os.path.join(ncf_txt_dir, name_ncf_txt)
    ncf_pic = os.path.join(ncf_pic_dir, name_ncf_jpg)
    calculate_ncf_acc(diff_acc_file=diff_txt, cf_acc_file=cf_txt, save_txt=ncf_txt, save_jpg=ncf_pic)  # 计算非保守力加速度


def get_ncf_from_gps(gps: SOGP, time_start: datetime, time_end: datetime, output_dir: str) -> None:
    """
    计算并提取目标在采样时间点上的非保守力特征
    :param gps:
    :param time_start:
    :param time_end:
    :param output_dir:
    :return:
    """
    # 配置参数
    config_soi = ConfigSOI()
    time_filename_format = config_soi.time_filename_format
    time_interval = config_soi.time_interval

    # # 1 推算轨道
    name = f"{str(gps[0]['NORAD_CAT_ID']).zfill(5)}_J2000_{time_start.strftime(time_filename_format)}-{time_end.strftime(time_filename_format)}"
    name_orbit = name + config_soi.orbit_suffix
    name_orbit_txt = name_orbit + '.txt'
    orbit_txt_dir = os.path.join(output_dir, 'orbit')
    os.makedirs(orbit_txt_dir, exist_ok=True)  # 创建txt路径
    orbit_txt = os.path.join(orbit_txt_dir, name_orbit_txt)

    predict_orbit_gps(gps, time_start, time_end, teme2j2000=True, save_txt=orbit_txt, save_jpg=False)  # 推算轨道

    # # 2 数值微分计算合加速度
    win_size = config_soi.num_diff_win_size
    name_diff = name + config_soi.diff_suffix
    name_diff_txt = name_diff + '.txt'
    diff_txt_dir = os.path.join(output_dir, 'num_diff')
    os.makedirs(diff_txt_dir, exist_ok=True)  # 创建txt路径
    diff_txt = os.path.join(diff_txt_dir, name_diff_txt)

    acc_lagrange(orbit_file=orbit_txt, save_txt=diff_txt, save_jpg=False, win_size=win_size)  # 数值微分计算合加速度

    # # 3 精密保守力模型计算保守力加速度
    name_cf = name + config_soi.cf_suffix
    name_cf_txt = name_cf + '.txt'
    cf_txt_dir = os.path.join(output_dir, 'cf')
    os.makedirs(cf_txt_dir, exist_ok=True)  # 创建txt路径
    cf_txt = os.path.join(cf_txt_dir, name_cf_txt)

    calculate_cf_acc(orbit_file=orbit_txt, cf_acc_file=cf_txt)  # 精密保守力模型计算保守力加速度

    # # 4 计算非保守力加速度
    name_ncf = name + config_soi.ncf_suffix
    name_ncf_txt = name_ncf + '.txt'
    ncf_txt_dir = os.path.join(output_dir, 'ncf')
    os.makedirs(ncf_txt_dir, exist_ok=True)  # 创建txt路径
    ncf_txt = os.path.join(ncf_txt_dir, name_ncf_txt)

    calculate_ncf_acc(diff_acc_file=diff_txt, cf_acc_file=cf_txt, save_txt=ncf_txt, save_jpg=False)  # 计算非保守力加速度

    # 删除num_diff,cf,gfm文件
    del_dir(diff_txt_dir)
    del_dir(cf_txt_dir)
    gfm_txt_dir = orbit_txt_dir.replace('orbit', 'gfm')
    del_dir(gfm_txt_dir)




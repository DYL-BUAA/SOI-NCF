# -*- coding = utf-8 -*-
# @Time:2023/9/26,16:24
# @Author:邓云龙
# @File:getGPhistoryDatasets.py
# @Software:PyCharm
import os
import sys
from datetime import timedelta, datetime
from typing import Union


from GP import GP
from SpaceObjectGP import SOGP
from input_output import read_orbit_file, output_orbit_file, plot_predict_orbit
from coordinate_frame_transform import teme_to_j2000_skyfield, teme_to_j2000_series_skyfield

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import ConfigSOI


config_soi = ConfigSOI()


def save_orbit_txt(times, poss, vels, save_txt: Union[str, bool]):
    if save_txt:
        times_string = []
        for i in range(len(times)):
            times_string.append(times[i].strftime(config_soi.time_in_out_format))
        # 指定文件名？
        if isinstance(save_txt, bool):
            save_txt = f'orbit.txt'
        output_orbit_file(times_string, poss, vels, savefile=save_txt)


def save_orbit_jpg(times, poss, vels, save_jpg: Union[str, bool]):
    if save_jpg:
        times_value = []
        for i in range(len(times)):
            times_value.append((times[i] - times[0]).total_seconds())

        plot_predict_orbit(times_value, poss, vels, savefile=save_jpg)


def predict_orbit_gps(gps: SOGP, time_start: Union[str, datetime], time_end: Union[str, datetime], teme2j2000=True, save_txt: Union[str, bool] = False, save_jpg: Union[str, bool] = False):
    """
    由SGP4每隔一段时间根据空间目标TLE，更新一次位置、速度信息（UTC+TEME），并输出相关文件。
    当teme2j2000=True时输出UTC+J2000下的结果
    :param norad_id: 空间目标，以GRACE-FO 1, norad_id：43476为例
    :param time_start: 据此选择一条GP数据，该条数据记录时刻最接近 start time to predict
    :param time_end: 据此选择一条GP数据，该条数据记录时刻最接近 end time to predict
    :param teme2j2000: 是否将结果转至J2000
    :param save_txt: 是否保存txt？是，保存为 save_txt
    :param save_jpg: 是否保存jpg？是，保存为 save_jpg
    :return:
    """
    print('-' * 25 + ' ' + 'Predict Orbit' + ' ' + '-' * 25)

    # 将时间字符串转为datetime对象
    if isinstance(time_start, str):
        try:
            time_start = datetime.strptime(time_start, config_soi.time_in_out_format)
        except TypeError:
            raise TypeError('time_start must be string or datetime.')
    if isinstance(time_end, str):
        try:
            time_end = datetime.strptime(time_end, config_soi.time_in_out_format)
        except TypeError:
            raise TypeError('time_end must be string or datetime.')

    # 搜索最接近time的一条GP数据
    gp_start, index_start = gps.search_nearest(search_time=time_start)
    gp_end, index_end = gps.search_nearest(search_time=time_end)

    print(f"predicting: from {time_start} to {time_end} {gps[0]['NORAD_CAT_ID']}'s orbit...")

    # 把时间转为相对于某一时刻的相对数值时间，单位：秒(s)
    time_reference = datetime(1950, 1, 1, 0, 0, 0)
    epoch_first_sec = (gps[index_start]['EPOCH'] - time_reference).total_seconds()
    epoch_last_sec = (gps[index_end]['EPOCH'] - time_reference).total_seconds()
    time_s = (time_start - time_reference).total_seconds()
    time_e = (time_end - time_reference).total_seconds()

    total_time = time_e - time_s

    """ ---------------------------------------------- """
    # 与时刻对应的 轨道位置、速度
    times, poss, vels = [], [], []

    # 根据index_start和index_end的取值，分三种情况：
    delta_index = index_end - index_start

    total_num = int(total_time / config_soi.time_interval)
    # 循环变量初始化
    delta_t = timedelta(seconds=0)
    # a.离开始、结束时间最近记录的GP数据为同一条
    if delta_index == 0:
        for i in range(total_num):
            delta_t += timedelta(seconds=config_soi.time_interval)
            update_t = time_start + delta_t  # 待计算位置、速度的对应时刻
            # # 计算
            pos, vel = gp_start.update_position_velocity(update_time=update_t)  # teme

            if teme2j2000:
                pos, vel = teme_to_j2000_skyfield(update_t, pos, vel)  # j2000

            times.append(update_t)
            poss.append(pos)
            vels.append(vel)

    # b.离开始、结束时间最近记录的GP数据index相差为1
    elif delta_index == 1:
        epoch_middle_sec = (epoch_last_sec + epoch_first_sec) / 2
        for i in range(total_num):
            delta_t += timedelta(seconds=config_soi.time_interval)
            update_t = time_start + delta_t  # 待计算位置、速度的对应时刻

            # # 计算
            if time_s+delta_t.total_seconds() <= epoch_middle_sec:
                pos, vel = gp_start.update_position_velocity(update_time=update_t)  # teme
            else:
                pos, vel = gp_end.update_position_velocity(update_time=update_t)  # teme

            if teme2j2000:
                pos, vel = teme_to_j2000_skyfield(update_t, pos, vel)  # j2000

            times.append(update_t)
            poss.append(pos)
            vels.append(vel)

    # c.离开始、结束时间最近记录的GP数据之间有多条数据
    else:
        epochs_sec = []
        for index in range(index_start, index_end + 1):
            epoch_sec = (gps[index]['EPOCH'] - time_reference).total_seconds()
            epochs_sec.append(epoch_sec)
        epochs_middle_sec = []
        for j in range(len(epochs_sec) - 1):
            epoch_middle_sec = (epochs_sec[j] + epochs_sec[j + 1]) / 2
            epochs_middle_sec.append(epoch_middle_sec)

        index_offset = 0
        index = index_start + index_offset
        for i in range(total_num):
            delta_t = timedelta(seconds=config_soi.time_interval * i)
            update_t = time_start + delta_t  # 待计算位置、速度的对应时刻

            # 寻找最优的TLE起点，以epoch最接近作为依据
            if index_offset < len(epochs_middle_sec) and time_s + delta_t.total_seconds() > epochs_middle_sec[index_offset]:
                index_offset += 1
                index = index_start + index_offset
            # # 计算
            pos, vel = gps[index].update_position_velocity(update_time=update_t)  # teme

            if teme2j2000:
                pos, vel = teme_to_j2000_skyfield(update_t, pos, vel)  # j2000

            times.append(update_t)
            poss.append(pos)
            vels.append(vel)
    """ ---------------------------------------------- """

    # 保存输出文件
    save_orbit_txt(times, poss, vels, save_txt)
    save_orbit_jpg(times, poss, vels, save_jpg)


def predict_orbit(norad_id: int, time_start: Union[str, datetime], time_end: Union[str, datetime], teme2j2000=True, save_txt: Union[str, bool] = False, save_jpg: Union[str, bool] = False):
    """
    由SGP4每隔一段时间根据空间目标TLE，更新一次位置、速度信息（UTC+TEME），并输出相关文件。
    当teme2j2000=True时输出UTC+J2000下的结果
    :param norad_id: 空间目标，以GRACE-FO 1, norad_id：43476为例
    :param time_start: 据此选择一条GP数据，该条数据记录时刻最接近 start time to predict
    :param time_end: 据此选择一条GP数据，该条数据记录时刻最接近 end time to predict
    :param teme2j2000: 是否将结果转至J2000
    :param save_txt: 是否保存txt？是，保存为 save_txt
    :param save_jpg: 是否保存jpg？是，保存为 save_jpg
    :return:
    """
    print('-' * 25 + ' ' + 'Predict Orbit' + ' ' + '-' * 25)

    # 读取GP数据
    gp_filename = 'NoradCatID' + '_' + str(norad_id).zfill(5) + '.txt'
    gp_file = os.path.join(config_soi.gp_dataset_dir, gp_filename)
    gps = SOGP(GP_file=gp_file)

    predict_orbit_gps(gps, time_start, time_end, teme2j2000, save_txt, save_jpg)


def predict_orbit_simplify(norad_id: int, time_start: datetime, time_end: datetime, time_interval=60, teme2j2000=True, save_txt: Union[bool, str] = False, save_jpg: Union[bool, str] = False):
    """
    由SGP4每10s处理空间目标TLE，更新一次位置、速度信息（UTC+TEME），并输出相关文件。
    当teme2j2000=True时输出UTC+J2000下的结果
    :param norad_id: 空间目标，以GRACE-FO 1, norad_id：43476为例
    :param time_start: 据此选择一条GP数据，该条数据记录时刻最接近time_start，并每10s处理空间目标TLE，更新一次位置、速度信息, datetime(2022, 12, 31, 10, 00, 30, 263616)
    :param time_end: 据此选择一条GP数据，该条数据记录时刻最接近time_end，并每10s处理空间目标TLE，更新一次位置、速度信息, datetime(2022, 12, 31, 10, 00, 30, 263616)
    :param time_interval: 更新频率，并每time_interval处理空间目标，默认为10s
    :param teme2j2000: 是否将结果转至J2000
    :param save_txt: 是否将结果保存为save_txt
    :param save_jpg: 是否将结果保存为save_jpg
    :return:
    """
    print('-' * 25 + ' ' + 'Predict Orbit' + ' ' + '-' * 25)
    print(f"predicting: from {time_start} to {time_end} {norad_id}'s track")

    # 从数据库搜索相关TLE信息
    # if os.path.exists('../FeatureNCF/spaceTargets'):
    gp_filename = 'NoradCatID' + '_' + str(norad_id).zfill(5) + '.txt'
    gp_file = os.path.join(config_soi.gp_dataset_dir, gp_filename)
    gps = SpaceObjectGP(GP_file=gp_file)

    # 将字符串转为datetime对象
    if isinstance(time_start, str):
        time_start = datetime.strptime(time_start, config_soi.time_in_out_format)
    elif isinstance(time_start, datetime):
        pass
    else:
        raise TypeError
    if isinstance(time_end, str):
        time_end = datetime.strptime(time_end, config_soi.time_in_out_format)
    elif isinstance(time_end, datetime):
        pass
    else:
        raise TypeError

    # 搜索最接近time_start的一条GP数据
    gp_search, index_search_start = gps.search_nearest(search_time=time_start)

    # 把时间转为相对于某一时刻的相对数值时间，单位：秒(s)
    total_time = (time_end - time_start).total_seconds()

    # 细化轨道位置、速度信息
    times, poss, vels = [], [], []
    total_num = int(total_time / time_interval)

    for i in range(total_num):
        delta_t = timedelta(seconds=time_interval*i)
        update_t = time_start + delta_t  # 待计算位置、速度的对应时刻
        # # 计算
        pos_teme, vel_teme = gp_search.update_position_velocity(update_time=update_t)

        if teme2j2000:
            pos_j2000, vel_j2000 = teme_to_j2000_skyfield(update_t, pos_teme, vel_teme)
            pos_j2000, vel_j2000 = pos_j2000.tolist(), vel_j2000.tolist()

            times.append(update_t)
            poss.append(pos_j2000)
            vels.append(vel_j2000)

    # 保存输出文件
    save_orbit_txt(times, poss, vels, save_txt)
    save_orbit_jpg(times, poss, vels, save_jpg)


def pridict_orbit_from_gp_searched_simplify(gp_search: GP, time_start: datetime, time_end: datetime, time_interval=60, teme2j2000=True, save_txt: Union[bool, str] = False,  save_jpg: Union[bool, str] = False):
    """
    由SGP4每10s处理空间目标TLE，更新一次位置、速度信息（UTC+TEME），并输出相关文件。
    当teme2j2000=True时输出UTC+J2000下的结果
    :param gp_search: 空间目标，以GRACE-FO 1, norad_id：43476为例
    :param time_start: 据此选择一条GP数据，该条数据记录时刻最接近time_start
    :param time_end: 据此选择一条GP数据，该条数据记录时刻最接近time_end
    :param time_interval: 更新频率，并每time_interval处理空间目标，默认为10s
    :param teme2j2000: 是否将结果转至J2000
    :param save_txt: 是否将结果保存为save_txt
    :param save_jpg: 是否将结果保存为save_jpg
    :return:
    """
    print('-'*25 + ' ' + 'Predict Orbit' + ' ' + '-'*25)
    norad_id = gp_search['NORAD_CAT_ID']
    print(f"predicting: from {time_start} to {time_end} {norad_id}'s track")

    # 把时间转为相对于某一时刻的相对数值时间，单位：秒(s)
    total_time = (time_end - time_start).total_seconds()

    # 细化轨道位置、速度信息
    times, poss, vels = [], [], []
    total_num = int(total_time / time_interval)

    for i in range(total_num):
        delta_t = timedelta(seconds=time_interval*i)
        update_t = time_start + delta_t  # 待计算位置、速度的对应时刻
        # # 计算
        pos_teme, vel_teme = gp_search.update_position_velocity(update_time=update_t)

        times.append(update_t)
        poss.append(pos_teme)  # teme
        vels.append(vel_teme)  # teme

    if teme2j2000:
        poss, vels = teme_to_j2000_series_skyfield(times, poss, vels)  # j2000

    # 保存输出文件
    save_orbit_txt(times, poss, vels, save_txt)
    save_orbit_jpg(times, poss, vels, save_jpg)


def demo_predict_orbit():
    norad_id = 43476
    ts = datetime(2019, 1, 1)
    te = ts + timedelta(days=1)
    predict_orbit(norad_id=norad_id, time_start=ts, time_end=te, teme2j2000=True, save_txt=True, save_jpg=True)


if __name__ == "__main__":
    '''轨道预报'''
    demo_predict_orbit()





# -*- coding = utf-8 -*-
# @Time:2023/9/12,16:34
# @Author:邓云龙
# @File:GP.py
# @Software:PyCharm

import os
import re
import copy
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
from sgp4.api import Satrec, jday
from sgp4.conveniences import sat_epoch_datetime


from enum import Enum
from datetime import datetime, timedelta


class Keys(Enum):
    HEADER = ['CCSDS_OMM_VERS', 'COMMENT', 'CREATION_DATE', 'ORIGINATOR']
    META = ["OBJECT_NAME", "OBJECT_ID", "EPOCH", "MEAN_MOTION", "ECCENTRICITY", "INCLINATION", "RA_OF_ASC_NODE",
            "ARG_OF_PERICENTER", "MEAN_ANOMALY", "EPHEMERIS_TYPE", "CLASSIFICATION_TYPE", "NORAD_CAT_ID",
            "ELEMENT_SET_NO", "REV_AT_EPOCH", "BSTAR", "MEAN_MOTION_DOT", "MEAN_MOTION_DDOT", "SEMIMAJOR_AXIS",
            "PERIOD", "APOAPSIS", "PERIAPSIS", "OBJECT_TYPE", "RCS_SIZE",
            "GP_ID", "TLE_LINE0", "TLE_LINE1", "TLE_LINE2"]
    STATE = ['X', 'Y', 'Z', 'X_DOT', 'Y_DOT', 'Z_DOT']
    EXTRA = ["COUNTRY_CODE", "LAUNCH_DATE", "SITE", "DECAY_DATE", "FILE"]


class GeneralPerturbation:
    def __init__(self):
        self.data = {
            key: {} for key in [Keys.META]  # 初始化字典结构
        }

    def __getitem__(self, key):  # 建立类似字典访问结构
        return self.to_dict()[key]

    def set(self, key, value):
        try:
            if key in Keys.META.value:
                if key == 'EPOCH':
                    value = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%f")
                self.data[Keys.META][key] = value
            else:
                raise ValueError(f'Invalid key: {key}')
        except Exception as e:
            print(f"Error setting {key}: {e}")

    # 使用@property装饰器简化getter和setter方法
    @property
    def metadata(self):
        return self.data[Keys.META]

    # 其他属性...
    def keys(self):
        return Keys.META.value

    def from_series(self, series: pd.Series):
        for key in series.keys():
            self.set(key, series.loc[key])

    def to_dict(self):
        return {**self.data[Keys.META]}

    def to_dataframe(self):
        return pd.DataFrame([self.to_dict()])

    # 其他方法...
    def _calculate_position_velocity(self, datetime_value):
        satellite = Satrec.twoline2rv(self['TLE_LINE1'], self['TLE_LINE2'])
        second = datetime_value.second + datetime_value.microsecond * 1e-6
        jd, jr = jday(*datetime_value.timetuple()[:5], second)
        return satellite.sgp4(jd, jr)

    def calculate_epoch_position_velocity(self):
        """
        计算tle记载时刻epoch下的位置、速度，把相应值补充到特征X、Y、Z，X_DOT、Y_DPT、Z_DOT中，并返回
        :return: position, velocity
        """
        error, position, velocity = self._calculate_position_velocity(self['EPOCH'])
        return position, velocity

    def update_position_velocity(self, update_time: datetime = None, delta_time: timedelta = None):
        """
        解算某一时刻的目标位置、速度，并返回相应值
        :param update_time: 待更新时刻
        :param delta_time
        :return: position, velocity
        """
        if delta_time and not update_time:
            update_time = self['EPOCH'] + delta_time
        elif not delta_time and not update_time:
            raise ValueError("Must provide either update_time or delta_time")
        error, position, velocity = self._calculate_position_velocity(update_time)
        return position, velocity


GP = GeneralPerturbation


def use_tle_compute_pos_vel(line_1, line_2, update_time: datetime = None, delta_time: timedelta = None):
        satellite = Satrec.twoline2rv(line_1, line_2)
        if delta_time is not None and update_time is None:
            epoch = sat_epoch_datetime(satellite)
            update_time = epoch + delta_time
        elif update_time is not None and delta_time is None:
            pass
        else:
            raise ValueError("只应输入1个时间参数！！！")
        year, month, day, hour, minute, second = update_time.year, update_time.month, update_time.day, update_time.hour, update_time.minute, update_time.second
        microsecond = update_time.microsecond
        second = second + (microsecond/1000000)
        jd, jr = jday(year, month, day, hour, minute, second)
        error, position, velocity = satellite.sgp4(jd, jr)  # 单位：km, km/s, TEME
        return position, velocity


def test():
    """
    '1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753'
    '2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667'
        3 days in the future (day = 182.784 950 62) from the TLE epoch.

            rTEME = −9060.473 735 69 4658.709 525 02 813.686 731 53 km
            vTEME = −2.232 832 783 −4.110 453 490 −3.157 345 433 km/s

            rJ2000 = 3961.744 260 3 6010.215 610 9 4619.362 575 8 km
            vJ2000 = −5.314 643 386 3.964 357 585 1.752 939 153 km/s
    :return:
    """
    from datetime import datetime, timedelta
    l1 = '1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753'
    l2 = '2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667'
    p, v = use_tle_compute_pos_vel(l1, l2, delta_time=timedelta(days=3))
    p, v = np.array(p), np.array(v)
    print(f'teme pos: {p}\nteme vel: {v}')

    pTEME = (-9060.47373569, 4658.70952502, 813.68673153)
    vTEME = (-2.232832783, -4.110453490, -3.157345433)
    pTEME, vTEME = np.array(pTEME), np.array(vTEME)
    print(f'teme pos error: {p-pTEME} km\nteme vel error: {v-vTEME} km/s')


if __name__ == "__main__":
    test()







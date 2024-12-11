import os
import time
from skyfield.sgp4lib import TEME_to_ITRF
from skyfield.positionlib import ITRF_to_GCRS2
from skyfield.constants import AU_KM, DAY_S
from skyfield.api import load

from datetime import timedelta, datetime
from typing import Union, List
from astropy.coordinates import TEME, CartesianDifferential, CartesianRepresentation, ITRS, GCRS
from astropy import coordinates as coord, units as u
from astropy.time import Time
import numpy as np


def eci_to_ecef_astropy(time, r_eci: list):
    time_convert = Time(time)  # 定义转换时间

    # 地心惯性系坐标转换为地心地固系坐标
    r_eci = CartesianRepresentation(r_eci * u.km)
    p_gcrs = GCRS(r_eci, obstime=time_convert)
    r_itrs = p_gcrs.transform_to(ITRS(obstime=time_convert))

    x_ecef, y_ecef, z_ecef = r_itrs.cartesian.xyz.value

    return x_ecef, y_ecef, z_ecef


def ecef_to_eci_astropy(time, r_ecef: list):
    time_convert = Time(time)  # 定义转换时间

    # 地心惯性系坐标转换为地心地固系坐标
    r_ecef = CartesianRepresentation(r_ecef * u.km)
    p_itrs = ITRS(r_ecef, obstime=time_convert)
    r_gcrs = p_itrs.transform_to(GCRS(obstime=time_convert))

    x_eci, y_eci, z_eci = r_gcrs.cartesian.xyz.value

    return x_eci, y_eci, z_eci


def teme_to_j2000_astropy(time_convert_utc, r_teme, v_teme):  # pos,vel是vectors, result of SGP4 in TEME frame
    time_convert = Time(time_convert_utc)
    time0 = time.time()
    p_teme = CartesianRepresentation(r_teme * u.km)
    v_teme = CartesianDifferential(v_teme * u.km / u.s)
    time1 = time.time()
    print('笛卡尔坐标系：', time1 - time0, 's')
    teme = TEME(p_teme.with_differentials(v_teme), obstime=time_convert)
    time2 = time.time()
    print('teme坐标系：', time2 - time1, 's')
    itrs = teme.transform_to(ITRS(obstime=time_convert))
    time3 = time.time()
    print('teme2itrs：', time3 - time2, 's')
    gcrs = itrs.transform_to(GCRS(obstime=time_convert))
    time4 = time.time()
    print('itrs2j2000一次：', time4 - time2, 's')
    p_gcrs, v_gcrs = gcrs.cartesian.xyz.value, gcrs.velocity.d_xyz.value

    return p_gcrs, v_gcrs


def teme_to_j2000_skyfield(time_convert_utc: datetime, r_teme, v_teme):
    ts = load.timescale()
    year = time_convert_utc.year
    month = time_convert_utc.month
    day = time_convert_utc.day
    hour = time_convert_utc.hour
    minute = time_convert_utc.minute
    second = time_convert_utc.second
    micro_sec = time_convert_utc.microsecond
    second = second + micro_sec*1e-6
    t = ts.utc(year, month, day, hour, minute, second)

    r_teme = np.array(r_teme)
    v_teme = np.array(v_teme)
    r_teme /= AU_KM
    v_teme /= AU_KM
    v_teme *= DAY_S

    r_itrf, v_itrf = TEME_to_ITRF(t.whole, r_teme, v_teme, 0.0, 0.0, t.ut1_fraction)

    r_gcrs, v_gcrs = ITRF_to_GCRS2(t, r_itrf, v_itrf)
    r_j2000, v_2000 = r_gcrs * AU_KM, v_gcrs / DAY_S * AU_KM

    return r_j2000, v_2000


def teme_to_j2000_series_skyfield(time_convert_utcs: List[datetime], poss_teme: List[List[float]], vels_teme: List[List[float]]):
    poss_j2000, vels_j2000 = [], []
    for i in range(len(time_convert_utcs)):
        r_j2000, v_j2000 = teme_to_j2000_skyfield(time_convert_utcs[i], poss_teme[i], vels_teme[i])

        poss_j2000.append(r_j2000)
        vels_j2000.append(v_j2000)

    return poss_j2000, vels_j2000



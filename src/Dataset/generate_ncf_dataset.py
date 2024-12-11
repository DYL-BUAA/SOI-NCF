import os
import math
import sys
import time

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from multiprocessing import Pool
from functools import partial
from astropy.time import Time
import matplotlib.pyplot as plt


try:
    from ..config import ConfigSOI
    from ..FeatureNCF.SpaceObjectGP import SOGP
    from ..FeatureNCF.main_ncf_from_tle import get_ncf_from_norad, get_ncf_from_gps

except ImportError:
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'FeatureNCF'))
    from SpaceObjectGP import SOGP
    from main_ncf_from_tle import get_ncf_from_norad, get_ncf_from_gps

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from config import ConfigSOI

config_soi = ConfigSOI()


def check_label_consistency(label_counts, info):
    """
    当标签内容只含不确定和某一确定类别时，返回True；其余，返回False
    :param label_counts:
    :param info:
    :return:
    """
    unknown_string = ['nan', 'null', 'UNKNOWN', 'TBA', 'TBD']
    if len(label_counts) == 1:
        pass  # 标签始终不变，可信
    elif len(label_counts) == 2:
        if any(label in unknown_string for label in label_counts.index):
            pass  # 两个标签中有一个是未知，可信
        else:
            print(f' {info}: 两个标签都有相当确定性，不可信!')
            return False
    else:
        print(f' {info}: 三个及以上个标签都有相当确定性，不可信!')
        return False
    return True


def so_ncf_dataset(time_gen_start: datetime, time_gen_end: datetime, norad_id: int):
    print(f'正在处理SO: {norad_id} {time_gen_start}--{time_gen_end} ......')
    # 读取GP数据
    gp_file = os.path.join(config_soi.gp_dataset_dir, f'NoradCatID_{norad_id:05d}.txt')
    gps = SOGP(GP_file=gp_file)
    gps_dataframe = gps.to_dataframe()

    # 检查标签一致性
    obj_type = gps_dataframe['OBJECT_TYPE']
    rcs_size = gps_dataframe['RCS_SIZE']
    obj_type_counts = obj_type.value_counts()
    rcs_size_counts = rcs_size.value_counts()
    if not check_label_consistency(obj_type_counts, info='Object Type'):
        return None  # 或者您可以选择记录日志等操作
    if not check_label_consistency(rcs_size_counts, info='RCS Size'):
        return None

    # label
    category = obj_type_counts.index[0] + '_' + rcs_size_counts.index[0]
    label = config_soi.label[category]

    # 生成数据集的保存路径
    output_dir = config_soi.output_dir + '/' + f'{norad_id}_{label}'
    os.makedirs(output_dir, exist_ok=True)

    # 总时间
    total_time = (time_gen_end - time_gen_start).total_seconds()
    total_time = int(total_time / 86400)

    # 初始化采样时间点
    time_start_ncf = time_gen_start
    period = gps[0]['PERIOD']  # 单位：分钟
    time_orbit_pred = config_soi.num_period * period / (24 * 60)  # 天
    delta_time_ncf = int(time_orbit_pred) + 1
    for days in range(0, total_time, delta_time_ncf):
        # ncf计算相关时间
        time_ncf_start = time_start_ncf + timedelta(days=days)
        time_ncf_end = time_ncf_start + timedelta(days=time_orbit_pred)

        try:
            if time_ncf_start < config_soi.epoch_eop_start_utc:  # 不应早于EOP文件的第一条时间
                print('采样时刻早于EOP文件的第一条时间!')
                raise ValueError
            # 执行实际的处理函数
            get_ncf_from_gps(gps=gps, time_start=time_ncf_start, time_end=time_ncf_end, output_dir=output_dir)

        except FileNotFoundError:
            print(f'没找到 相关文件！')
        except ValueError:
            print(f'没找到 {norad_id} 在 {time_ncf_start}--{time_ncf_end} 的数据！')
        except TypeError:
            print(f'数据类型存在错误！')


if __name__ == '__main__':
    # 数据集中的所有 norad_id
    norad_ids_dir = config_soi.gp_dataset_dir  # \NoradCatID_89496.txt
    norad_id_filenames = os.listdir(norad_ids_dir)
    file_num = len(norad_id_filenames)
    norad_ids = []
    for i in range(file_num):
        norad_ids.append(int(norad_id_filenames[i][-9:-4]))
    norad_ids.sort()

    # 使用多进程池
    with Pool(processes=4) as pool:  # 你可以根据你的CPU核心数调整进程数
        results = pool.map(partial(so_ncf_dataset, config_soi.epoch_start_utc, config_soi.epoch_end_utc), norad_ids)

    # 处理结果
    for result in results:
        if result is not None:
            # 这里处理每个NORAD ID的返回结果
            print('-' * 50, f'{result} 计算正常', '-' * 50)








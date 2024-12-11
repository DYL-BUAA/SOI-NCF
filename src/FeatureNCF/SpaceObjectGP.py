import os
import re
import copy
import numpy as np
from typing import Optional, Union, List, Tuple
import matplotlib.pyplot as plt
import pandas as pd
from sgp4.api import Satrec, jday
from datetime import datetime, timedelta

from GP import GP


class SpaceObjectGP:
    def __init__(self, GPs: Optional[List[GP]]=None, GP_file: Optional[str]=None):
        """
        Initialize with a list of GP objects or a file path to a GP data file.
        :param GPs: A list of GP objects containing the General Perturbation data.
        :param GP_file: A path to a text or CSV file containing GP data sorted by epoch.
        """
        self._GPs: List[GP] = GPs if GPs else []
        self._dataframe: Optional[pd.DataFrame] = None
        if GP_file:
            self._dataframe = self._load_GPs_from_file(GP_file)

    def _load_GPs_from_file(self, GP_file: str) -> Optional[pd.DataFrame]:
        if not os.path.isfile(GP_file):
            raise FileNotFoundError(f"The file {GP_file} does not exist.")
        if os.path.splitext(GP_file)[-1] not in ['.txt', '.csv']:
            raise ValueError("GP_file must be a .txt or .csv file.")

        df = pd.read_csv(GP_file)
        df.columns = df.columns.str.strip()
        gp = GP()
        cols_to_keep = gp.keys()
        df = df.drop(columns=[col for col in df.columns if col not in cols_to_keep])
        # df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        df = df.apply(lambda x: x.map(lambda y: y.strip() if isinstance(y, str) else y))
        return df

    def __getitem__(self, index: int):
        gp = GP()
        _dataframe = self._dataframe.iloc[index, :]
        if isinstance(index, slice):
            _GPs = []
            for i in range(_dataframe.shape[0]):
                gp.from_series(_dataframe.iloc[i])
                _GPs.append(gp)
            return _GPs
        else:
            gp.from_series(_dataframe)
            return gp

    def __len__(self) -> int:
        return len(self._GPs) if self._GPs else len(self._dataframe)

    def to_list(self) -> List[GP]:
        gp = GP()
        for i in range(self._dataframe.shape[0]):
            gp.from_series(self._dataframe.iloc[i])
            self._GPs.append(gp)
        return self._GPs

    def to_dataframe(self):
        return self._dataframe

    def add(self, gp):
        if isinstance(gp, GP):
            self._GPs.append(gp)
        elif isinstance(gp, list):
            [self.add(g) for g in gp]
        else:
            raise TypeError('Expecting a single GP or a list of GPs')

    def search_nearest(self, search_time: Union[str, datetime, None] = None, max_delta_time: int = 86400) -> Optional[Tuple[GP, int]]:   #
        """
        Searches for the GP record closest to the search time within the allowed time delta.

        :param search_time: The search time in 'YYYY-MM-DDTHH:MM:SS' format or as a datetime object.
        :param max_delta_time: The maximum allowed time difference in seconds.
        :return: The closest GP object within the time delta, or None if no match is found.
        """
        if search_time is None:
            if not self._dataframe.empty:
                return self[-1], -1
            else:
                raise ValueError("not given search time and GPs is None.")

        # Convert search_time to a datetime object if it's a string.
        if isinstance(search_time, str):
            try:
                search_time = datetime.strptime(search_time, "%Y-%m-%dT%H:%M:%S")
            except ValueError:
                raise ValueError("search_time must be in 'YYYY-MM-DDTHH:MM:SS' format.")

        # Ensure search_time is within the range of available data.
        if not self._dataframe.empty and (search_time < self[0]['EPOCH'] - timedelta(seconds=max_delta_time) or search_time > self[-1]['EPOCH'] + timedelta(seconds=max_delta_time)):
            raise ValueError('Search time is out of the range of available data.')

        # Perform the search using a linear search algorithm.
        latest_gp = None
        latest_diff = float('inf')  # Start with an infinitely large difference.
        index_gp = -1
        for index_gp in range(len(self._dataframe)):
            epoch_time = self[index_gp]['EPOCH']
            if isinstance(epoch_time, str):
                epoch_time = datetime.strptime(epoch_time, "%Y-%m-%dT%H:%M:%S.%f")
            current_diff = abs((epoch_time - search_time).total_seconds())
            if current_diff <= max_delta_time and current_diff < latest_diff:
                latest_gp = self[index_gp]
                latest_diff = current_diff
            elif current_diff >= latest_diff:
                break  # Since the DataFrame is sorted by EPOCH, no need to continue.
            index_gp += 1
        if latest_gp is None:
            raise ValueError('No data that satisfies the condition exists within the allowed search time range.')
        else:
            return self[index_gp], index_gp

    def to_csv(self, save_file: str) -> None:
        if self._dataframe is not None:
            if os.path.exists(save_file):
                existing_df = pd.read_csv(save_file)
                if existing_df.iloc[-1]['CREATION_DATE'] != self._dataframe.iloc[-1]['CREATION_DATE']:
                    self._dataframe.to_csv(save_file, mode='a', header=False)
            else:
                self._dataframe.to_csv(save_file)


SOGP = SpaceObjectGP


def search_demo():
    # 输入待搜索目标
    norad_cat_id_search = input("请输入待查找的norad_cat_id:")  # 示例：28227
    gp_file_search = 'E:\项目\基于非保守力特征的空间目标识别\原始数据库\spaceTargets' + '\\' + 'NoradCatID_' + str(norad_cat_id_search) + '.txt'
    time_search = input("请输入待查找的时间(%Y-%m-%dT%K:%M:%S):")
    delta_time = int(input("允许的最大时间偏差（S）:"))

    # 搜索
    obj = SpaceObjectGP(GP_file=gp_file_search)  # SpaceObject[i]:第i条GP
    gp_search, index = obj.search_nearest(search_time=time_search, max_delta_time=delta_time)
    print('查找到的对应时间GP数据：', gp_search)
    if isinstance(gp_search, GP):
        pos, vel = gp_search.calculate_epoch_position_velocity()
        print('当前查找时间对应的position：{}\n         velocity：{}'.format(pos, vel))


if __name__ == "__main__":
    search_demo()

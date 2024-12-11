import os
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from astropy.time import Time, TimeGPS
from typing import Union, List, Tuple, Dict, Set

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import ConfigSOI


colors = ['#1f77b4', '#ff7f0e', 'g']


def read_orbit_file(file: str):
    try:
        times, poss, vels = [], [], []
        with open(file, 'r') as f:
            title = f.readline().split(',')
            # print(title)
            lines = f.read().splitlines()
            for line in lines:
                values = line.replace(' ', '').split(',')
                values_pv = []
                for i in range(1, len(values), 1):
                    values_pv.append(float(values[i]))

                times.append(values[0])
                poss.append(values_pv[0:3])
                vels.append(values_pv[3:6])
        return times, poss, vels

    except FileNotFoundError:
        print(f'文件{file}不存在！')
    except PermissionError:
        print(f'无权限访问文件{file}！')
    except Exception as e:
        print(f'发生未知错误：{e}')


def output_orbit_file(times: List[str], poss: List[list], vels: List[list], savefile: str):
    with open(savefile, 'w') as f:
        # 30 ','
        # 写入title
        f.write(' ' * 19 + 'time (UTC),')
        f.write(' ' * 23 + 'x (km),')
        f.write(' ' * 23 + 'y (km),')
        f.write(' ' * 23 + 'z (km),')
        f.write(' ' * 17 + 'x_dot (km/s),')
        f.write(' ' * 17 + 'y_dot (km/s),')
        f.write(' ' * 18 + 'z_dot (km/s)\n')

        for i in range(len(times)):
            time = times[i]
            f.write(f"{time:>29},")

            f.write(f"{poss[i][0]:>29.15f},")
            f.write(f"{poss[i][1]:>29.15f},")
            f.write(f"{poss[i][2]:>29.15f},")

            f.write(f"{vels[i][0]:>29.15f},")
            f.write(f"{vels[i][1]:>29.15f},")
            f.write(f"{vels[i][2]:>30.15f}\n")
    print(f'finish output_refine_file: {os.path.split(savefile)[-1]}')


def plot_predict_orbit(times_value: Union[List[int], List[float]], poss: List[list], vels: List[list], savefile=False):
    fig, ax = plt.subplots(2, 1, sharex='col')
    for i in range(3):
        ax[0].plot(times_value, np.array(poss)[:, i].tolist(), colors[i])
        ax[1].plot(times_value, np.array(vels)[:, i].tolist(), colors[i])
    label_p, label_v = ['x', 'y', 'z'], ['vx', 'vy', 'vz']
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Position (km)')
    ax[0].legend(label_p)

    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('Velocity (km/s)')
    ax[1].legend(label_v)
    # 保存
    if isinstance(savefile, str):
        plt.savefig(savefile)
        plt.close()
    elif savefile:  # True / False
        plt.show()
    print(f'finish plot_predict_orbit: 保存-{savefile}')


def read_numerical_differentiation_file(file):
    try:
        times, pos_2order_diffs = [], []
        with open(file, 'r') as file:
            titles_ = file.readline().split(',')
            # print(f'titles:{titles_}')
            line = file.readline()
            while line:
                a = []
                sub_lines = line.split(',')
                for index, _ in enumerate(sub_lines):
                    if index == 0:
                        times.append(_.strip())
                    else:
                        a.append(float(_.strip()))
                pos_2order_diffs.append(a)
                line = file.readline()
            return times, pos_2order_diffs
    except FileNotFoundError:
        print(f'文件{file}不存在！')
    except PermissionError:
        print(f'无权限访问文件{file}！')
    except Exception as e:
        print(f'发生未知错误：{e}')


def output_numerical_differentiation_file(times: List[str], diff_accs: List[list], savefile: str):
    with open(savefile, 'w') as f:
        # 写入title
        f.write(' ' * 19 + 'time (UTC),')
        f.write(' ' * 9 + 'x_dot_diff (nm/s**2),')
        f.write(' ' * 9 + 'y_dot_diff (nm/s**2),')
        f.write(' ' * 10 + 'z_dot_diff (nm/s**2)\n')

        for i in range(len(times)):
            time = times[i]
            f.write(f"{time:>29},")
            f.write(f"{diff_accs[i][0] * 1e12:>29.15e},")
            f.write(f"{diff_accs[i][1] * 1e12:>29.15e},")
            f.write(f"{diff_accs[i][2] * 1e12:>30.15e}\n")
    print(f'finish output_numerical_differentiation_file: {os.path.split(savefile)[-1]}')


def plot_numerical_differentiation_acc(times_value: Union[List[int], List[float]], diff_accs: List[list], savefile=False):
    diff_accs = np.array(diff_accs)

    labels = ['ncf_acc_x', 'ncf_acc_y', 'ncf_acc_z']
    plt.subplots(nrows=3, ncols=1, sharex='col')
    plt.suptitle('position 2-order differentiation')
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.plot(times_value, diff_accs[:, i] * 1e12, colors[i], label=labels[i])
        # plt.ylim(-3000, 3000)
        plt.xlabel('Time (s)')
        plt.ylabel('Acceleration (nm/s$^2$)')
        plt.legend()

    if isinstance(savefile, str):
        plt.savefig(savefile)
        plt.close()
    elif savefile:  # True / False
        plt.show()
    print(f'finish plot_numerical_differentiation_file: {savefile}')


def read_gni(file):
    import pandas as pd
    df = pd.read_csv(file, skiprows=148, header=None, sep=' ')
    time = df.iloc[:, 0]
    times = np.array(time).tolist()
    pos = df.iloc[:, 3:6]
    positions = np.array(pos).tolist()
    vel = df.iloc[:, 9:12]
    velocities = np.array(vel).tolist()

    return times, positions, velocities


def read_gforcemodel_file(file):
    try:
        times, positions, velocities = [], [], []
        gfm_gravity, gfm_n_body, gfm_general_relativity = [], [], []

        with open(file, 'r') as file:
            titles_ = file.readline()
            line = file.readline()
            while line:
                sub_lines_ = line.split(' ')
                sub_lines = []
                for i in range(len(sub_lines_)):
                    if sub_lines_[i] != '':
                        sub_lines.append(sub_lines_[i])
                # print(sub_lines)
                times.append(sub_lines[0][:-3])

                p = sub_lines[4:7]
                positions.append([float(p[0]), float(p[1]), float(p[2])])

                v = sub_lines[7:10]
                velocities.append([float(v[0]), float(v[1]), float(v[2])])

                a = sub_lines[10:13]
                gfm_gravity.append([float(a[0]), float(a[1]), float(a[2])])
                a = sub_lines[13:16]
                gfm_n_body.append([float(a[0]), float(a[1]), float(a[2])])
                a = sub_lines[16:19]
                gfm_general_relativity.append([float(a[0]), float(a[1]), float(a[2])])

                line = file.readline()
        return times, positions, velocities, gfm_gravity, gfm_n_body, gfm_general_relativity
    except FileNotFoundError:
        print('文件不存在！')
    except PermissionError:
        print('无权限访问文件！')
    except Exception as e:
        print(f'发生未知错误：{e}')


def read_conservative_force_file(file):
    try:
        times, cf_accs = [], []

        with open(file, 'r') as file:
            titles_ = file.readline().split(',')
            # print(f'titles:{titles_}')
            line = file.readline()
            while line:
                a = []
                sub_lines = line.split(',')
                for index, _ in enumerate(sub_lines):
                    if index == 0:
                        times.append(_.strip())
                    else:
                        a.append(float(_.strip()))
                cf_accs.append(a)
                line = file.readline()
        return times, cf_accs
    except FileNotFoundError:
        print('文件不存在！')
    except PermissionError:
        print('无权限访问文件！')
    except Exception as e:
        print(f'发生未知错误：{e}')


def output_conservative_force_file(times: List[str], gfm_gravity: List[list], gfm_n_body: List[list], gfm_general_relativity: List[list], savefile: str):
    cf_accs = np.array(gfm_gravity) + np.array(gfm_n_body) + np.array(gfm_general_relativity)
    cf_accs = cf_accs.tolist()

    with open(savefile, 'w') as f:
        # 写入title
        f.write(' ' * 19 + 'time (UTC),')
        f.write(' ' * 11 + 'cf_acc_x (nm/s**2),')
        f.write(' ' * 11 + 'cf_acc_y (nm/s**2),')
        f.write(' ' * 12 + 'cf_acc_z (nm/s**2)\n')

        for i in range(len(cf_accs)):
            time = times[i]
            f.write(f"{time:>29},")

            f.write(f"{cf_accs[i][0]:>29.15e},")
            f.write(f"{cf_accs[i][1]:>29.15e},")
            f.write(f"{cf_accs[i][2]:>30.15e}\n")
    print(f'finish output_CF_file: {os.path.split(savefile)[-1]}')


def plot_conservative_force_acc(times, poss, vels, cf_accs):
    times, poss, vels, cf_accs = [], [], [], []
    pass


def read_non_conservative_force_file(file):
    try:
        times, ncf_accs = [], []
        with open(file, 'r') as file:
            titles_ = file.readline().split(',')
            line = file.readline()
            while line:
                a = []
                sub_lines = line.split(',')
                for index, _ in enumerate(sub_lines):
                    if index == 0:
                        times.append(_.strip())
                    else:
                        a.append(float(_.strip()))
                ncf_accs.append(a)
                line = file.readline()
        return times, ncf_accs
    except FileNotFoundError:
        print(f'文件{file}不存在！')
    except PermissionError:
        print(f'无权限访问文件{file}！')
    except Exception as e:
        print(f'发生未知错误：{e}')


def output_non_conservative_force_file(times: List[str], ncf_accs: List[list], savefile: str):
    with open(savefile, 'w') as f:
        # f.write('J2000' + '\n')
        # 30 ','
        # 写入title
        f.write(' ' * 19 + 'time (UTC),')
        f.write(' ' * 10 + 'ncf_acc_x (nm/s**2),')
        f.write(' ' * 10 + 'ncf_acc_y (nm/s**2),')
        f.write(' ' * 11 + 'ncf_acc_z (nm/s**2)\n')

        for i in range(len(ncf_accs)):
            time = times[i]
            f.write(f"{time:>29},")
            f.write(f"{ncf_accs[i][0]:>29.15e},")
            f.write(f"{ncf_accs[i][1]:>29.15e},")
            f.write(f"{ncf_accs[i][2]:>30.15e}\n")
    print(f'finish output_NCF_file: {os.path.split(savefile)[-1]}')


def plot_non_conservative_force_acc(times_value: list, ncf_accs: List[list], savefile: Union[bool, str] = False):
    # 绘制非保守力加速度
    plt.subplots(nrows=3, ncols=1, sharex='col')
    ncf_accs = np.array(ncf_accs)
    labels = ['ncf_acc_x', 'ncf_acc_y', 'ncf_acc_z']
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.plot(times_value, ncf_accs[:, i], color=colors[i], label=labels[i])
        plt.xlabel('Time (s)')
        plt.legend()
    plt.ylabel('Non-conservative Force Acceleration (nm/s$^2$)', labelpad=10, y=1.75, rotation=90)
    if isinstance(savefile, str):
        plt.savefig(savefile)
        plt.close()
    elif savefile:  # True / False
        plt.show()
    print(f'finish plot_NCF_file: {savefile}')


def read_gni_file(file: str):
    with open(file, 'r') as f:
        line = f.readline()
        while line:
            if '# End of YAML header' in line:
                break
            else:  # 更新 line
                line = f.readline()
        line = f.readline()  # end 之后 1 行
        times_gpst, poss, vels = [], [], []
        while line:
            container = line.split(' ')
            times_gpst.append(int(container[0]))  # gps_time, 单位：s, Continuous seconds past 01-Jan-2000 11:59:47 UTC
            x_pos, y_pos, z_pos = float(container[3]), float(container[4]), float(container[5])  # x, y_train, z (ECI), 单位：m
            poss.append([x_pos, y_pos, z_pos])
            x_vel, y_vel, z_vel = float(container[9]), float(container[10]), float(container[11])  # vx, vy, vz (ECI), 单位：m/s
            vels.append([x_vel, y_vel, z_vel])

            # 更新 line
            line = f.readline()

        return times_gpst, poss, vels


def output_gni_to_tleorbit(gni_file, orbit_file):
    times_gps, poss, vels = read_gni_file(gni_file)

    # 常用时间系统：tt(行星时)，ut（世界时），utc（协调时），tai（原子时）
    # 形式：iso、datetime、gps、jd、mjd等
    time_ref_gni = datetime(2000, 1, 1, 11, 59, 47, 0)
    # time_ref_gni = datetime(2000, 1, 1, 12, 0, 0, 0)
    time_ref_gni_gps = Time(time_ref_gni, format='datetime', scale='utc').tai.gps

    config_soi = ConfigSOI()
    times = []
    for i in range(len(times_gps)):
        t_gps = times_gps[i] + time_ref_gni_gps
        t = Time(t_gps, format='gps', scale='tai').utc.datetime
        if isinstance(t, datetime):
            t = t.strftime(config_soi.time_in_out_format)
        times.append(t)

    with open(orbit_file, 'w') as f:
        # 30 ','
        # 写入title
        f.write(' ' * 19 + 'time (UTC),')
        f.write(' ' * 23 + 'x (km),')
        f.write(' ' * 23 + 'y_train (km),')
        f.write(' ' * 23 + 'z (km),')
        f.write(' ' * 17 + 'x_dot (km/s),')
        f.write(' ' * 17 + 'y_dot (km/s),')
        f.write(' ' * 18 + 'z_dot (km/s)\n')

        for i in range(len(times)):
            time = times[i]
            f.write(f"{time:>29},")

            f.write(f"{poss[i][0]*1e-3:>29.15f},")
            f.write(f"{poss[i][1]*1e-3:>29.15f},")
            f.write(f"{poss[i][2]*1e-3:>29.15f},")

            f.write(f"{vels[i][0]*1e-3:>29.15f},")
            f.write(f"{vels[i][1]*1e-3:>29.15f},")
            f.write(f"{vels[i][2]*1e-3:>30.15f}\n")
    print(f'finish output_gni_to_tleorbit: {orbit_file}')


def read_sca_file(file: str):
    """
    读取sca1b文件
    :param file:
    :return: 时间和四元数
    """
    with open(file, 'r') as f:
        line = f.readline()
        while line:
            if '# End of YAML header' in line:
                break
            else:  # 更新 line
                line = f.readline()
        line = f.readline()  # end 之后 1 行
        times_gpst, quaternions = [], []
        while line:
            container = line.split(' ')
            times_gpst.append(int(container[0]))  # gps_time, 单位：s, Continuous seconds past 01-Jan-2000 11:59:47 UTC
            quat_angle = float(container[3])    # ECI转BCS四元数 w
            quat_i_coeff = float(container[4])  # ECI转BCS四元数 i
            quat_j_coeff = float(container[5])  # ECI转BCS四元数 j
            quat_k_coeff = float(container[6])  # ECI转BCS四元数 k

            quaternions.append([quat_angle, quat_i_coeff, quat_j_coeff, quat_k_coeff])

            # 更新 line
            line = f.readline()

        return times_gpst, quaternions


def read_act_file(file: str):
    with open(file, 'r') as f:
        line = f.readline()
        while line:
            if '# End of YAML header' in line:
                break
            else:  # 更新 line
                line = f.readline()
        line = f.readline()  # end 之后 1 行
        times_gpst, lin_accs = [], []
        while line:
            container = line.split(' ')
            times_gpst.append(int(container[0]))  # gps_time, 单位：s, Continuous seconds past 01-Jan-2000 11:59:47 UTC
            lin_acc_x = float(container[2])  # 加速度计 x 方向分量
            lin_acc_y = float(container[3])  # 加速度计 y 方向分量
            lin_acc_z = float(container[4])  # 加速度计 z 方向分量

            lin_accs.append([lin_acc_x, lin_acc_y, lin_acc_z])

            # 更新 line
            line = f.readline()

        return times_gpst, lin_accs




import os
import sys


from input_output import read_gforcemodel_file, read_conservative_force_file, output_conservative_force_file
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import ConfigSOI


config_soi = ConfigSOI()


def run_force_model(bin, constellation, geometry, cfg, sat, sca, gni, out_acc):
    """
    # call the exe to generate force model files
    :param bin: 保守力模型可执行文件
    :param constellation: 保守力模型配置文件--constellation
    :param geometry: 保守力模型配置文件--geometry
    :param cfg: 保守力模型配置文件--力模型cfg
    :param sat: 保守力模型配置--卫星类型
    :param sca: 保守力模型输入--姿态四元素文件
    :param gni: 保守力模型输入--星历文件(eci坐标)
    :param out_acc: 保守力模型输入输出路径
    :return: None
    """
    command = f'{bin} {constellation} {geometry} {cfg} {sat} {sca} {gni} {out_acc}'
    bin_dir = os.path.split(bin)[0]
    r_v = os.system(f'cd {bin_dir} && {command}')
    print(f"GFM.exe 运行情况: {'success' if r_v==0 else 'fail'}")

    # print(f'finish calculate gfm_acc: {gfm_acc_file}')


def calculate_cf_acc(orbit_file: str, cf_acc_file: str):
    """
    计算精密保守力加速度
    :param orbit_file: 由TLE计算的position、velocity文件
    :param cf_acc_file: 保守力文件保存路径
    :return:
    """
    print('-'*25+' '+'Conservative Force'+' '+'-'*25)

    # 目录
    cf_model_dir = config_soi.cf_model_dir

    # 参数
    sat_id_gfc = 'L04'
    config_dir = os.path.join(cf_model_dir, "config")

    bin_exe = cf_model_dir + "/" + "processing" + '/' + "GForceTLE"
    # bin_exe = cf_model_dir + "/" + "processing" + '/' + "GForce_gni"

    constellation_file = config_dir + '/' + "constellation.txt"
    spacecraft_geometry = config_dir + '/' + "spacecraft.geometry"
    force_model_cfg = config_dir + '/' + "forcemodels.cfg"

    ceres_file_config = config_dir + '/' + "EarthFlux" + '/'
    # copy f107_ap and eop_file to config directory
    f107_ap = config_dir + '/' + "solar_indices.txt"  # 太阳活动
    eop_file = config_dir + '/' + "EOP" + '/' + "finals2000A.data"  # 地球定向参数

    # 卫星数据
    sca_file = os.path.join(cf_model_dir, "data/XX04-01A/SCA1B/TH04-01_SCA1B_2022-04-15_A_00_20220416102308.asc")
    sca_file = os.path.join(cf_model_dir, "data/gracefo/gracefo_1B_2019-01-01_RL04.ascii.noLRI/SCA1B_2019-01-01_C_04.txt")

    # 建立gfm_acc_file保存目录
    gfm_acc_file = orbit_file.replace(config_soi.orbit_suffix, config_soi.gfm_suffix).replace('orbit', 'gfm')
    os.makedirs(os.path.split(gfm_acc_file)[0], exist_ok=True)

    run_force_model(bin_exe, constellation_file, spacecraft_geometry, force_model_cfg, sat_id_gfc, sca_file, orbit_file, gfm_acc_file)
    print(f'finish calculate gfm_acc: {os.path.split(gfm_acc_file)[-1]}')

    # 格式转换
    t, p, v, g, n_body, gr = read_gforcemodel_file(gfm_acc_file)
    output_conservative_force_file(times=t, gfm_gravity=g, gfm_n_body=n_body, gfm_general_relativity=gr, savefile=cf_acc_file)
    print(f'finish convert from gfm_acc to cf_acc: {os.path.split(cf_acc_file)[-1]}')






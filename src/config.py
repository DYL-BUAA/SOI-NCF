import os
import yaml
from datetime import datetime, timedelta


class ConfigSOI:
    def __init__(self):
        ''' 配置文件 '''
        if os.name.startswith('nt'):  # windows平台
            config_filename = 'config_windows.yaml'
        elif os.name.startswith('posix'):  # linux平台
            config_filename = 'config_linux.yaml'
        else:
            config_filename = 'config_java.yaml'
        current_work_dir = os.getcwd()
        src_dir = os.path.split(current_work_dir)[0]
        SOI_dir = os.path.split(src_dir)[0]
        config_dir = os.path.join(SOI_dir, 'config')
        config_file = os.path.join(config_dir, config_filename)
        with open(config_file, encoding='utf-8') as yaml_file:
            configure = yaml.safe_load(yaml_file)

        # 目录
        self.gp_dataset_dir = configure['SOI']['directories']['dataset directory']
        self.process_data_dir = configure['SOI']['directories']['process data directory']
        self.output_dir = configure['SOI']['directories']['output directory']
        self.cf_model_dir = configure['SOI']['directories']['conservative force model directory']

        # 时间格式
        self.time_in_out_format = configure['SOI']['formats']['time utc in input and output']
        self.time_filename_format = configure['SOI']['formats']['time utc in filename']
        # 命名结尾格式
        self.orbit_suffix = configure['SOI']['formats']['orbit_suffix']
        self.diff_suffix = configure['SOI']['formats']['differentiation_suffix']
        self.gfm_suffix = configure['SOI']['formats']['gfm_suffix']
        self.cf_suffix = configure['SOI']['formats']['conservative_force_suffix']
        self.ncf_suffix = configure['SOI']['formats']['non_conservative_force_suffix']

        # 参考时间
        self.time_ref = configure['SOI']['parameters']['time_reference']
        self.epoch_start = configure['SOI']['parameters']['epoch_start']
        self.epoch_end = configure['SOI']['parameters']['epoch_end']
        self.time_ref_utc = datetime.strptime(self.time_ref, "%Y-%m-%dT%H:%M:%S.%f")
        self.epoch_start_utc = datetime.strptime(self.epoch_start, "%Y-%m-%dT%H:%M:%S.%f")
        self.epoch_end_utc = datetime.strptime(self.epoch_end, "%Y-%m-%dT%H:%M:%S.%f")
        self.epoch_eop_start = configure['SOI']['parameters']['epoch_eop_start']
        self.epoch_eop_start_utc = datetime.strptime(self.epoch_eop_start, "%Y-%m-%dT%H:%M:%S.%f")
        # 数值微分
        self.num_diff_win_size = configure['SOI']['parameters']['num_diff_win_size']
        # 轨道推算时间间隔
        self.time_interval = configure['SOI']['parameters']['orbit_interval']
        # 轨道推算总时间（多少个周期）
        self.num_period = configure['SOI']['parameters']['predict_period']
        # 采样间隔（day）
        self.sample_interval = configure['SOI']['parameters']['sample_interval']

        # 采样对象
        self.ids = configure['SOI']['objects']['ids']

        # GP特征
        self.gp_features = configure['SOI']['features']['gp features']

        # SO 对应GP的label
        self.label = {}
        for i, obj_type in enumerate(configure['SOI']['labels']['object_type']):
            for j, rcs_size in enumerate(configure['SOI']['labels']['rcs_size']):
                self.label[f'{obj_type}_{rcs_size}'] = 3 * i + j


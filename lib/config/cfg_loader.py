"""
-*- coding: utf-8 -*-
@Author : Leezed
@Time : 2024/3/20 20:58
"""
from lib.utils.YamlUtil import YamlUtil


class CfgLoader(dict):

    def __getattr__(self, item):
        if item in self:
            value = self[item]
            if isinstance(value, dict):
                return CfgLoader(value)
            else:
                return value
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{item}'")


def get_cfg(path, cfg_name):
    return YamlUtil(path, cfg_name).content


def env_setting(cfg_name):
    """
    获取配置文件 ，默认是local.yaml
    :param cfg_name:  如果不想获取local.yaml，可以传入其他的配置文件名
    :return: 返回配置文件
    """
    cfg = CfgLoader(get_cfg('configs', cfg_name if cfg_name is not None else 'local.yaml'))
    return cfg

"""
-*- coding: utf-8 -*-
@Author : Leezed
@Time : 2024/3/18 13:33
"""

import os

import yaml


class YamlUtil():
    def __init__(self, yaml_path, yaml_name):
        self.yaml_path = yaml_path
        self.yaml_name = yaml_name
        self.yaml_file = os.path.join(self.yaml_path, self.yaml_name)

        # 判断当前路径下文件是否存在，如果存在则读取，如果不存在则创建
        if not os.path.exists(self.yaml_file):
            with open(self.yaml_file, 'w') as f:
                f.write('')

        self.content = self.read_yaml()  # 读取到的yaml文件内容

    def read_yaml(self):
        """
        读取整个yaml文件
        :return: 读取到的文件内容
        """
        with open(self.yaml_file, 'r') as f:
            content = yaml.safe_load(f)
        return content

    def write_yaml(self):
        """
        将content写入yaml文件
        """
        with open(self.yaml_file, 'w') as f:
            yaml.safe_dump(self.content, f)

    # 判断yaml中是否存在某个key
    def is_exist_key(self, key_path):
        """
        判断yaml中是否存在某个key路径
        :param key_path: key_path,数组形式 ['a','b','c'] 或者字符串形式 'a.b.c'
        :return: True/False
        """
        # 判断key_path的类型，如果是字符串则转换为数组
        really_key_path = key_path.split('.') if isinstance(key_path, str) else key_path

        assert isinstance(really_key_path, list), 'key_path必须是list或者str类型'

        content = self.content
        for key in really_key_path:
            if key in content:
                content = content[key]
            else:
                return False
        return True

"""
-*- coding: utf-8 -*-
@Author : Leezed
@Time : 2024/3/18 16:37
"""
from lib.config.genrate_cfg import default_cfg_generate
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Create default cfg file on ITP or PAI')
    parser.add_argument("--cfg_name", type=str, required=True)
    parser.add_argument("--workspace_dir", type=str)
    parser.add_argument("--data_dir", type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    cfg_name = args.cfg_name if args.cfg_name.endswith('.yaml') else args.cfg_name + '.yaml'
    workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../")) if args.workspace_dir is None \
        else os.path.relpath(args.workspace_dir)
    datasets_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../datasets")) if args.data_dir is None \
        else os.path.relpath(args.data_dir)

    default_cfg_generate(cfg_name, workspace_dir, datasets_dir)

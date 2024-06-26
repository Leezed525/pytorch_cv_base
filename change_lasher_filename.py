"""
-*- coding: utf-8 -*-
@Author : Leezed
@Time : 2024/4/14 0:25
"""

import os
import re

LasHeR_path = "/home/lzd/dataset/LasHeR/TrainingSet/trainingset"
bowblkboy1_quezhen_path = "/media/star/data/Leezed/dataset/LasHeR/TrainingSet/trainingset/bowblkboy1-quezhen/infrared"
nightrightboy1_path = "/media/star/data/Leezed/dataset/LasHeR/TrainingSet/trainingset/nightrightboy1/"
biked_path = "/media/star/data/Leezed/dataset/LasHeR/TrainingSet/trainingset/biked/"
fogboyscoming1_quezhen_inf_heiying_path = "/media/star/data/Leezed/dataset/LasHeR/TrainingSet/trainingset/fogboyscoming1_quezhen_inf_heiying/"
lastblkboy1_quezhen_path = "/media/star/data/Leezed/dataset/LasHeR/TrainingSet/trainingset/lastblkboy1_quezhen/"
manglass2_path = "/media/star/data/Leezed/dataset/LasHeR/TrainingSet/trainingset/manglass2/"


def change_bowblkboy1_quezhen_filename():
    for root, dirs, files in os.walk(bowblkboy1_quezhen_path):
        # 将文件名中改为{:06d}.jpg
        if root.endswith('infrared'):
            print(root)
            for file in files:
                if file.endswith('.jpg'):
                    print("old filename = " + file)
                    flag = 'i'
                    num = re.findall(r'\d', file)
                    num = ''.join(num)
                    new_name = flag + '{:06d}.jpg'.format(int(num) - 1000)
                    print("new name = " + new_name)
                    os.rename(os.path.join(root, file), os.path.join(root, new_name))
        else:
            continue


def change_nightrightboy1_filename():
    for root, dirs, files in os.walk(nightrightboy1_path):
        # 将文件名中改为{:06d}.jpg
        print(root)
        if root.endswith('visible') or root.endswith('infrared'):
            for file in files:
                if file.endswith('.jpg'):
                    print("old filename = " + file)
                    flag = 'v' if root.endswith('visible') else 'i'
                    num = re.findall(r'\d', file)
                    num = ''.join(num)
                    new_name = flag + '{:06d}.jpg'.format(int(num) - 10000)
                    print("new name = " + new_name)
                    os.rename(os.path.join(root, file), os.path.join(root, new_name))
        else:
            continue


def change_biked_filename():
    for root, dirs, files in os.walk(biked_path):
        # 将文件名中改为{:06d}.jpg
        print(root)
        if root.endswith('visible') or root.endswith('infrared'):
            for file in files:
                if file.endswith('.jpg'):
                    print("old filename = " + file)
                    flag = 'v' if root.endswith('visible') else 'i'
                    num = re.findall(r'\d', file)
                    num = ''.join(num)
                    new_name = flag + '{:06d}.jpg'.format(int(num) - 1)
                    print("new name = " + new_name)
                    os.rename(os.path.join(root, file), os.path.join(root, new_name))
        else:
            continue


def change_fogboyscoming1_quezhen_inf_heiying_filename():
    for root, dirs, files in os.walk(fogboyscoming1_quezhen_inf_heiying_path):
        # 将文件名中改为{:06d}.jpg
        print(root)
        if root.endswith('visible') or root.endswith('infrared'):
            for file in files:
                if file.endswith('.jpg'):
                    print("old filename = " + file)
                    flag = 'v' if root.endswith('visible') else 'i'
                    num = re.findall(r'\d', file)
                    num = ''.join(num)
                    new_name = flag + '{:06d}.jpg'.format(int(num) - 1000)
                    print("new name = " + new_name)
                    os.rename(os.path.join(root, file), os.path.join(root, new_name))
        else:
            continue


def change_question_filename(question_name, vis_change, infra_change, sub_num):
    path = "/media/star/data/Leezed/dataset/LasHeR/TrainingSet/trainingset/" + question_name
    for root, dirs, files in os.walk(path):
        # 将文件名中改为{:06d}.jpg
        print(root)
        if (root.endswith('visible') and vis_change) or (root.endswith('infrared') and infra_change):
            for file in files:
                if file.endswith('.jpg'):
                    print("old filename = " + file)
                    flag = 'v' if root.endswith('visible') else 'i'
                    num = re.findall(r'\d', file)
                    num = ''.join(num)
                    new_name = flag + '{:06d}.jpg'.format(int(num) - sub_num)
                    print("new name = " + new_name)
                    os.rename(os.path.join(root, file), os.path.join(root, new_name))
        else:
            continue


def change_all_filename():
    for root, dirs, files in os.walk(LasHeR_path):
        # 将文件名中改为{:06d}.jpg
        file_number = 0
        old_filename_list = []

        if root.endswith('visible') or root.endswith('infrared'):
            print(root)
            for file in files:
                if file.endswith('.jpg'):
                    file_number += 1
                    old_filename_list.append(file)
            flag = 'v' if root.endswith('visible') else 'i'
            old_filename_list.sort()

            for index, old_filename in enumerate(old_filename_list):
                new_name = flag + '{:06d}.jpg'.format(index)
                print("old filename = " + old_filename)
                print("new name = " + new_name)
                os.rename(os.path.join(root, old_filename), os.path.join(root, new_name))
        else:
            continue


if __name__ == '__main__':
    change_all_filename()
    # change_bowblkboy1_quezhen_filename()
    # change_nightrightboy1_filename()
    # change_biked_filename()
    # change_fogboyscoming1_quezhen_inf_heiying_filename()
    # change_question_filename('whitecarturn85', True, True, 1)

"""
-*- coding: utf-8 -*-
@Author : Leezed
@Time : 2024/4/14 0:25
"""

import os
import re

path = "/home/***/dataset/LasHeR/TrainingSet/trainingset"

if __name__ == '__main__':
    # print(int("001"))
    for root, dirs, files in os.walk(path):
        # 将文件名中改为{:06d}.jpg
        if root.endswith('visible') or root.endswith('infrared'):
            print(root)
            for file in files:
                if file.endswith('.jpg'):
                    print("old filename = " + file )
                    flag = 'v' if root.endswith('visible') else 'i'
                    num = re.findall(r'\d', file)
                    num = ''.join(num)
                    new_name = flag + '{:06d}.jpg'.format(int(num))
                    print("new name = " + new_name)
                    os.rename(os.path.join(root, file), os.path.join(root, new_name))
        else:
            continue

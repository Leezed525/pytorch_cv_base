"""
-*- coding: utf-8 -*-
@Author : Leezed
@Time : 2024/3/19 16:01
"""


class Transform():
    """
    用于数据增强的transform

    init中传入的transforms是一个列表，每个元素是一个transform函数

    call中传入的inputs是一个字典，包含了需要增强的数据，以及一些参数
        数据类型包括：
            image: 输入的图像数据
            coords: 输入的坐标数据 [y,x]
            bbox: 输入的边界框数据 [x,y,w,h]
            mask: 输入的mask数据
            att: 输入的注意力数据
        参数包括：
            joint: 如果为True,则对输入的所有数据进行相同的变换，否则对每个数据进行(random)不同的变换 (默认为True)
            new_roll: 如果为False，则不执行新的随机滚动，而是使用上一次滚动中保存的结果 (默认为True)
    """

    def __init__(self, *transforms):
        # 如果transforms是一个列表或者元组，那么就将其展开(防止传入的时候多加了一层括号[])
        if len(transforms) == 1 and isinstance(transforms[0], (list, tuple)):
            transforms = transforms[0]
        self.transforms = transforms

        self.__valid_inputs = ['image', 'coords', 'bbox', 'mask', 'att']
        self.__valid_args = ['joint', 'new_roll']

    def __call__(self, **inputs):
        input_name = [k for k in inputs.keys() if k in self.__valid_inputs]

        # 检查输入的参数是否合法
        for v in inputs.keys():
            if v not in self.__valid_inputs + self.__valid_args:
                raise ValueError(f"Invalid input name {v}, valid inputs are {self.__valid_inputs} and args are {self.__valid_args}")

        joint_mode = inputs.get('joint', True)
        new_roll = inputs.get('new_roll', True)

        if not joint_mode: # 如果joint_mode为False，则对每个输入的数据进行不同的变换，就是把数据进行拆分然后走一遍这个流程
            out = zip(*[self(**inp) for inp in self._split_input(inputs)])
            return tuple(list(o) for o in out)

        out = {k:v for k, v in inputs.items() if k in self.__valid_inputs}

        for t in self.transforms:
            out = t(**out,joint=joint_mode, new_roll=new_roll)

        if len(input_name) == 1:
            return out[input_name[0]]

        return tuple(out[k] for k in input_name)



    def _split_input(self, inputs):
        input_names = [k for k in inputs.keys() if k in self.__valid_inputs]
        # 将数据转换成[{'image': [1, 1, 1, 1], 'bbox': [1, 2, 3, 4]}, {'image': [2, 2, 2, 2], 'bbox': [5, 6, 7, 8]},,,,]这种形式
        split_inputs = [{k: v for k, v, in zip(input_names, input)} for input in zip(*[inputs[input_name] for input_name in input_names])]

        # 将参数赋值给每组输入的数据
        for arg_name, arg_value in filter(lambda it: it[0] != 'joint' and it[0] in self.__valid_args, inputs.items()):
            if isinstance(arg_value, list):  # 如果是列表，则将每个输入的参数都赋值按照列表的顺序赋值
                for input, arg_v in zip(split_inputs, arg_value):
                    input[arg_name] = arg_v
            else:
                for input in split_inputs:
                    input[arg_name] = arg_value

        return split_inputs

#
# if __name__ == '__main__':
#     def test1(**kwargs):
#         print(kwargs)
#         return kwargs
#
#
#     tfm = Transform([test1])
#     images = [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]]
#     bbox = [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]]
#
#     # print(tfm(image=images, bbox=bbox, joint=False, new_roll=False))
#     tfm(image=images, bbox=bbox, joint=False, new_roll=False)
#     for input in zip(*[images, bbox]):
#         print(input)

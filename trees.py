# _*_ coding: utf-8 _*_
# @Time     : 2017/10/12 17:06
# @Author    : Ligb
# @File     : trees.py

from math import log
from numpy import *


def calc_shannon_ent(data_set):
    """
    计算给定数据集的信息熵
    :param data_set:输入的数据集，每行为一个样本，最后一列是标签
    :return:返回信息熵的值
    """
    class_set = {}
    ent = 0
    sample_size = len(data_set)

    # 逐行获取标签，并给每种标签计数
    for line in range(sample_size):
        sample_label = data_set[line][-1]
        if sample_label not in class_set.keys():
            class_set[sample_label] = 1
        else:
            class_set[sample_label] += 1

    # 计算信息熵
    for class_count in class_set.values():
        class_ratio = float(class_count / sample_size)
        ent += class_ratio * log2(class_ratio) * -1
    return ent


def split_data_set(data_set, axis, value):
    """
    将数据集按照属性划分子集
    :param data_set: 待划分的数据集
    :param axis: 划分子集的属性
    :param value: 划分出的子集的属性值
    :return: 划分后的子集
    """
    sub_data_set = []
    sub_data_set_raw = []
    sample_size = len(data_set)
    for sample_line in range(sample_size):
        if data_set[sample_line][axis] == value:
            sub_data_set_raw = data_set[sample_line][:axis]
            sub_data_set_raw.extend(data_set[sample_line][axis + 1:])
            sub_data_set.append(sub_data_set_raw)
    return sub_data_set


def best_feature_to_split():


data = [[1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no'],
        [1, 1, 'maybe']]

print(data)
print(calc_shannon_ent(data))
print(str(split_data_set(data, 0, 0)))

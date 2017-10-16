# _*_ coding: utf-8 _*_
# @Time     : 2017/10/12 17:06
# @Author    : Ligb
# @File     : trees.py

"""
决策树的基本算法实现,基于ID3算法
要求使用的数据是一个由列表元素组成的列表，并且
不能有缺失值，且每个样本最后一列是样本标签；
test_decision_tree（）中是一个简单的例子；
决策树的剪枝、连续值属性以及缺失值处理未完成，后续补充...
"""

from math import log
from numpy import *
import operator
import pickle

import tree_plot


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


def best_attribute_to_split(data_set):
    """
    计算各轴属性的信息增益，选择信息增益最大的轴作为划分属性
    :param data_set: 待划分的数据集
    :return: 信息增益最大的轴
    """
    info_gain = 0.0
    best_attribute = 0

    # 父结点样本数
    sample_size = len(data_set)
    root_ent = calc_shannon_ent(data_set)
    attribute_list = []
    axises = len(data_set[0]) - 1

    # 依次计算每个属性轴
    for axis in range(axises):
        node_ent_sum = 0

        # 对应该轴，逐行添加每个样本对应的属性值，合并重复的属性值
        for sample in range(sample_size):
            attribute_list.append(data_set[sample][axis])
            simplified_attribute_list = set(attribute_list)

        # 按该轴的每种属性值获取子集并计算信息增益
        for attribute in simplified_attribute_list:
            sub_data_set = split_data_set(data_set, axis, attribute)
            node_ent_sum += len(sub_data_set) / float(sample_size) * calc_shannon_ent(sub_data_set)

        # 依次比较各轴的信息增益，返回增益最大的轴
        if root_ent - node_ent_sum > info_gain:
            info_gain = root_ent - node_ent_sum
            best_attribute = axis
    return best_attribute


def leaf_node_class(node_data_set):
    """
    确定叶结点的分类类别，返回叶结点中样本数最多的类别
    :param node_data_set:叶结点数据集
    :return: 返回的类别
    """
    node_labels = {}
    node_size = len(node_data_set)

    # 逐行统计各类别的样本数
    for node_sample in range(node_size):
        node_label = node_data_set[node_sample][-1]
        if node_label not in node_labels.keys():
            node_labels[node_label] = 1
        else:
            node_labels[node_label] += 1

    # 返回的是一个以元组为元素的列表
    sorted_labels_count = sorted(node_labels.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_labels_count[0][0]


def create_tree(data_set, data_labels):
    """
    递归建立决策树，决策结点的value用字典的形式表示，叶结点的value就是普通数据
    :param data_set: 结点内包含的数据集
    :param data_labels: 结点的标签
    :return: 建立的树
    """
    class_list = [sample[-1] for sample in data_set]

    # 当节点内的样本全是同一分类的，结束递归，产生叶结点，将类别标签作为叶结点的标签
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]

    # 当节点内的样本已无可划分属性，结束递归，产生叶结点，将类别标签作为叶结点的标签
    if len(data_set[0]) == 1:
        return leaf_node_class(data_set)
    tree_node = {}
    best_attribute = best_attribute_to_split(data_set)

    # 获取内部结点的标签，对于离散属性，在划分过后该属性将不再出现在其子集中
    tree_node_key = data_labels[best_attribute]
    del data_labels[best_attribute]
    tree_node[tree_node_key] = {}

    # 获取划分属性的特征值
    attribute_classes = set([sample[best_attribute] for sample in data_set])
    for attribute_class in attribute_classes:
        sub_node_data_set = split_data_set(data_set, best_attribute, attribute_class)
        sub_labels = data_labels[:]

        # 将类别标签作为分支的键，值为下一结点的字典
        tree_node[tree_node_key][attribute_class] = create_tree(sub_node_data_set, sub_labels)
    return tree_node


def classify(input_tree, attribute_labels, test_sample):
    """
    递归实现一个样本的分类
    :param input_tree: 决策树的模型
    :param attribute_labels: 样本中的各个属性的标签组成的列表
    :param test_sample: 测试样本
    :return: 分类结果
    """
    node_keys_list = list(input_tree.keys())
    root_node = node_keys_list[0]
    sub_node = input_tree[root_node]

    # 获取根结点标签对应的属性索引值，因为在建树的时候已经保证结点不会重复，所以不必将attribute_labels中用过的标签剔除
    attribute_index = attribute_labels.index(root_node)
    for key in sub_node.keys():
        if test_sample[attribute_index] == key:
            if type(sub_node[key]).__name__ == 'dict':
                class_label = classify(sub_node[key], attribute_labels, test_sample)
            else:
                class_label = sub_node[key]
    return class_label


def test_tree():
    """
    测试trees.py模块
    :return:None
    """
    data = [[1, 1, 'yes'],
            [1, 1, 'yes'],
            [1, 0, 'no'],
            [0, 1, 'no'],
            [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    my_tree = create_tree(data, labels)
    print(my_tree)
    store_tree(my_tree, 'testtree.txt')
    print(grab_tree('testtree.txt'))
    print(data)
    print(leaf_node_class(data))
    print(best_attribute_to_split(data))
    print(calc_shannon_ent(data))
    print(str(split_data_set(data, 0, 0)))
    print(tree_plot.get_leaf_numbers(my_tree))
    print(tree_plot.get_tree_depth(my_tree))

    labels = ['no surfacing', 'flippers']
    print(classify(my_tree, labels, [1, 1]))


def test_glasses():
    """
    根据人的数据，推荐相应的眼镜
    :return: None
    """
    lenses = []
    with open('lenses.txt') as txt:
        for line in txt.readlines():
            lenses.append(line.strip().split('\t'))
    lenses_labels = ['age', 'prescript', 'astigmatic', 'tear_rate']
    lenses_tree = create_tree(lenses, lenses_labels)
    store_tree(lenses_tree, 'lens_tree.txt')
    tree_plot.create_plot(lenses_tree)


def store_tree(tree, file_name):
    """
    储存已构造的决策树,注意打开方式，因为pickle默认为二进制
    :param tree: 待保存的树
    :param file_name: 保存路径
    :return: None
    """
    with open(file_name, 'wb+') as ft:
        pickle.dump(tree, ft, 0)


def grab_tree(file_name):
    """
    读取已构造的决策树
    :param file_name: 文件路径
    :return: 读取的树
    """
    with open(file_name, 'rb+') as ft:
        return pickle.load(ft)


if __name__ == '__main__':
    test_tree()
    test_glasses()


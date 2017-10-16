# _*_ coding: utf-8 _*_
# @Time     : 2017/10/14 22:31
# @Author    : Ligb
# @File     : tree_plot.py

import matplotlib.pyplot as plt


def plot_node(node_text, center_pt, parent_pt, node_type):
    """
    绘制结点，axl仅是一个属性，怎么命名都行。。。
    :param node_text:结点文本
    :param center_pt:结点的位置
    :param parent_pt:注释的位置
    :param node_type:结点类型
    :return:None
    """
    create_plot.ax1.annotate(node_text, xy=parent_pt, xycoords='axes fraction',
                             xytext=center_pt, textcoords='axes fraction', va='center',
                             ha='center', bbox=node_type, arrowprops=dict(arrowstyle='<-'))


def create_plot(tree):
    """
    绘制决策树
    :param tree:输入的树
    :return:None
    """
    fig = plt.figure(1, facecolor='white')
    fig.clf()

    # 将xy轴的轴不予显示
    axis_props = dict(xticks=[], yticks=[])
    create_plot.ax1 = plt.subplot(111, frameon=False, **axis_props)
    plot_tree.totalW = float(get_leaf_numbers(tree))
    plot_tree.totalD = float(get_tree_depth(tree))
    plot_tree.xoff = -0.5 / plot_tree.totalW
    plot_tree.yoff = 1.0
    plot_tree(tree, (0.5, 1.0), '')
    # plot_node('a decision node', (0.5, 0.1), (0.1, 0.5), dict(boxstyle='sawtooth', fc='0.8'))
    # plot_node('a leaf node ', (0.8, 0.1), (0.3, 0.8), dict(boxstyle='round4', fc='0.8'))
    plt.show()


def get_leaf_numbers(my_tree):
    """
    递归统计叶结点的数目
    :param my_tree: 结点的数据集
    :return: 叶结点的数目
    """
    leaf_numbers = 0
    root_node_keys_list = list(my_tree.keys())
    root_node = root_node_keys_list[0]
    sub_node = my_tree[root_node]
    for key in sub_node.keys():
        if type(sub_node[key]).__name__ == 'dict':
            leaf_numbers += get_leaf_numbers(sub_node[key])
        else:
            leaf_numbers += 1
    return leaf_numbers


def get_tree_depth(my_tree):
    """
    递归计算树的深度，计算每一个branch的深度找最深的那个
    :param my_tree: 输入的树（字典结构）
    :return: 树的深度
    """
    tree_depth = 0
    root_node_keys_list = list(my_tree.keys())
    root_node = root_node_keys_list[0]
    sub_node = my_tree[root_node]
    for key in sub_node.keys():
        if type(sub_node[key]).__name__ == 'dict':
            current_branch_depth = get_tree_depth(sub_node[key]) + 1
        else:
            current_branch_depth = 1
        if current_branch_depth > tree_depth:
            tree_depth = current_branch_depth
    return tree_depth


def retrieve_tree(i):
    """
    储存两颗样例树，方便检验功能
    :param i: 树的索引
    :return: 选择的树
    """
    tree_list = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}, 2: 'maybe'}},
                 {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}]
    return tree_list[i]


def plot_mid_text(center_pt, parent_pt,text_string):
    """
    在两个节点的连线上添加文本信息
    :param center_pt: 子节点位置
    :param parent_pt: 父结点位置
    :param text_string: 注释信息
    :return: None
    """
    xMid = (parent_pt[0] - center_pt[0]) / 2.0 + center_pt[0]
    yMid = (parent_pt[1] - center_pt[1]) / 2.0 + center_pt[1]
    create_plot.ax1.text(xMid, yMid, text_string)


def plot_tree(tree, parent_pt, node_text):
    """
    绘制决策树
    :param tree: 待绘制的树的信息
    :param parent_pt: 父结点的坐标信息
    :param node_text: 结点文本
    :return: None
    """
    leaf_numbers = get_leaf_numbers(tree)
    node_key_list = list(tree.keys())
    root_node = node_key_list[0]
    center_pt = (plot_tree.xoff + (1.0 + float(leaf_numbers)) / 2.0 / plot_tree.totalW, plot_tree.yoff)
    plot_mid_text(center_pt, parent_pt, node_text)
    plot_node(root_node, center_pt, parent_pt, dict(boxstyle='sawtooth', fc='0.8'))
    sub_node = tree[root_node]
    plot_tree.yoff = plot_tree.yoff - 1.0 / plot_tree.totalD
    for key in sub_node.keys():
        if type(sub_node[key]).__name__ == 'dict':
            plot_tree(sub_node[key], center_pt, str(key))
        else:
            plot_tree.xoff = plot_tree.xoff + 1.0 / plot_tree.totalW
            plot_node(sub_node[key], (plot_tree.xoff, plot_tree.yoff), center_pt, dict(boxstyle='round4', fc='0.8'))
            plot_mid_text((plot_tree.xoff, plot_tree.yoff), center_pt, str(key))
    plot_tree.yoff = plot_tree.yoff + 1.0 / plot_tree.totalD


def test_plot_tree():
    """
    测试本模块代码
    :return: None
    """
    tree = retrieve_tree(0)
    print(get_tree_depth(tree))
    print(get_leaf_numbers(tree))
    create_plot(tree)

if __name__ == '__main__':
    test_plot_tree()
# -*- coding: utf-8 -*-
# @Time    : 2018/4/12 10:56
# @Author  : 
# @简介    : 逻辑回归分类
# @File    : logistic_regression.py

import numpy as np


def sigmoid(x):
    """
    sigmoid function
    :param x: (mat), feature * w
    :return: (mat) sigmoid value
    """
    return 1.0 / (1 + np.exp(-x))


def lr_train(feature, label, max_iter=150, alpha=0.05):
    """
    梯度下降法训练LR模型
    :param feature: (mat) 特征
    :param label: (mat) 标签
    :param max_iter: 最大迭代次数
    :param alpha: 下降率
    :return: w (mat) weight
    """
    n = np.shape(feature)[1]        # r * n r:样本个数 n:特征个数
    weights = np.mat(np.ones((n, 1)))
    i = 0
    while i <= max_iter:
        i += 1
        h = sigmoid(feature * weights)
        err = label - h
        weights = weights + alpha * feature.T * err
    return weights

#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2018/2/2 13:10
# @Author  : Dervean
# @Email   : derveanme@gmail.com
# @File    : PA.py
import numpy as np


def file2matrix(filename):
    data_set = None
    with open(filename) as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip().split('\t')
            nums = [float(num) for num in line[0].split(' ')]
            nums.append(float(line[-1]))
            if data_set is None:
                data_set = np.array(nums)
            else:
                data_set = np.vstack((data_set, np.array(nums)))
    # 矩阵行列数目
    n_row, n_column = data_set.shape
    one = np.ones(n_row)
    data_set = np.insert(data_set, 0, values=one, axis=1)
    return data_set


def print_message(*messages):
    print("\n**************************************************************************")
    for message in messages:
        print(message,end='\t')
    print("\n**************************************************************************")


class PA:
    def __init__(self, train_filename='hw1_18_train.dat'):
        self.train_matrix = file2matrix(train_filename)
        # 矩阵行列数目
        n_row, n_column = self.train_matrix.shape
        # 装在袋子里的w
        self.w = np.zeros(n_column - 1)

    def training(self, step=1.0, rounds=50):
        """
        :param step:
        :param rounds: 最大的轮次数目
        :return: 训练集的错误率error_rate
        """
        # 矩阵行列数目
        n_row, n_column = self.train_matrix.shape
        # 随机生成一个具有标准正态分布的权重w
        w = np.random.randn(n_column - 1)
        # 初始化错误率为1
        error_rate = 1
        print_message('初始化权重:', str(w))
        for i in range(rounds):
            if (i+1) % 10 == 0:
                print("round ...%d" % (i+1))
            flag = True
            for row in self.train_matrix:
                if np.dot(w, row[:n_column - 1] * row[-1]) < 0:
                    w = w + step * row[:n_column - 1] * row[-1]
                    e_rate = self.get_error_rate(self.train_matrix, w)
                    if e_rate < error_rate:
                        error_rate = e_rate
                        self.w = w
                    flag = False
            # 如果已经找到 w 可以将数据分开，则提前结束循环
            if flag:
                break
        print_message('最终权重:', str(self.w))
        print_message('错误率:', str(error_rate))
        return error_rate

    def testing(self,filename='hw1_18_train.dat'):
        test_matrix = file2matrix(filename)
        error_rate = self.get_error_rate(test_matrix, self.w)
        print_message('在测试集上测试的正确率:', str(1 - error_rate))
        return error_rate

    def get_error_rate(self, matrix, w):
        # 矩阵行列数目
        n_row, n_column = matrix.shape
        n_error = 0
        for row in matrix:
            if np.dot(w, row[:n_column - 1] * row[-1]) < 0:
                n_error += 1
        return n_error / n_row


if __name__ == '__main__':
    pa = PA()
    error_rate = pa.training(step=0.5)
    error_rate = pa.testing()
#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2018/2/1 16:19
# @Author  : Dervean
# @Email   : derveanme@gmail.com
# @File    : PLA.py

"""
perceptron learning algorithm（PLA)
"""
import numpy as np


class PLA:

    def file2matrix(self, filename):
        """
        生成一个矩阵，第一列是全1列矩阵，最后一列是实际yn，中间几列是xn
        :param filename:
        :return: matrix
        """
        with open(filename) as f:
            lines = f.readlines()
            data_set = np.zeros((len(lines), 5))
            index = 0
            for line in lines:
                row = line.strip().split('\t')
                nums = [float(str_n) for str_n in row[0].split(' ')]
                nums.append(float(row[-1]))
                data_set[index] = nums
                index += 1
        one = np.ones(len(lines))
        data_set = np.insert(data_set, 0, values=one, axis=1)
        return data_set

    def __init__(self,filename):
        self.matrix = self.file2matrix(filename)

    def classify(self, data_set, step=1):
        """
        :param data_set: self.matrix
        :param step: w的更新步长权重，标准PLA算法为1，可以更改来减少迭代次数（例如0.5）
        :return:
        """
        # 矩阵行列数目
        n_row, n_column = self.matrix.shape
        # 随机生成一个具有标准正态分布的权重w
        w = np.random.randn(n_column - 1)
        self.print_message('初始化权重:', str(w))
        # 轮次数目
        cnt = 0
        while True:
            # 判断这一轮过后是否能停止
            flag = True
            for data in data_set:
                # 判断sign(wT xn) 和 yn 符号是否相同
                if np.dot(w, data[:n_column - 1]) * data[-1] >= 0:
                    continue
                else:
                    # 轮次自增
                    cnt += 1
                    flag = False
                    # 更新w
                    w = w + step * data[:n_column - 1] * data[-1]
            if flag:
                break
        self.print_message('迭代次数:', str(cnt))
        self.print_message('更新后权重:', str(w))
        return w, cnt

    def print_message(self, *messages):
        print("\n**************************************************************************")
        for message in messages:
            print(message,end='\t')
        print("\n**************************************************************************")


if __name__ == '__main__':
    pla = PLA('hw1_15_train.dat')
    print(pla.matrix)

    w, cnt = pla.classify(pla.matrix)
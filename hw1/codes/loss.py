from __future__ import division
import numpy as np


class EuclideanLoss(object):
    def __init__(self, name):
        self.name = name


    def forward(self, input, target):
        # TODO START
        '''
        :param input: (batch_size=N, 10)
        :param target: (batch_size=N, 10)
        :return loss: (1, 1)
        '''
        N = input.shape[0]
        return np.power(input - target, 2).sum() / (2.0 * N)

        # TODO END

    def backward(self, input, target):
		# TODO START
        '''
        :param input: (batch_size=N, 10)
        :param target: (batch_size=N, 10)
        :return 
        '''
        N = input.shape[0]
        return (input - target) / N
		# TODO END


class SoftmaxCrossEntropyLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        # TODO START
        '''
        :param input: K * N (K: label_num = 10, N: batch_size)
        :param target: K * N 
        '''
        mid = np.exp(input)  # K * N
        h = mid / np.sum(mid, 0)  # sum of the cols  K * N
        E = - np.sum(target * np.log(h), 0)  # sum of the cols  1 * N
        return np.average(E) 
        # TODO END

    def backward(self, input, target):
        # TODO START
        '''Your codes here'''
        pass
        # TODO END


class HingeLoss(object):
    def __init__(self, name, margin=5):
        self.name = name
        self.margin = margin

    def forward(self, input, target):
        # TODO START 
        '''
        :param input: K * N (K: label_num = 10, N: batch_size)
        :param target: 1 * N
        '''

        K, N = input.shape
        Margin = self.margin * np.ones(input.shape)  # (K, N)
        idx_arr = input[target, np.array(range(N))]  # for x^n_n  (1, N)
        idx_matrix = np.ones((K, 1)) * idx_arr  # (K, 1) * (1, N)

        delta_matrix = np.zeros(input.shape)  # (K, N)
        delta_matrix[target, np.array(range(N))] = self.margin  # (K, N)

        input = input - delta_matrix  # 修正
        res = Margin - idx_matrix + input

        return res * (res > 0)  # max(0, res)
        # TODO END

    def backward(self, input, target):
        # TODO START
        '''Your codes here'''
        pass
        # TODO END


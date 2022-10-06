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
        :param input: (batch_size=N, 10)
        :param target: (batch_size=N, 10)
        '''
        mid = np.exp(input)  # (N, K)
        h = mid / np.sum(mid, axis=1, keepdims=True)  # sum of the rows  (N, K)
        E = - np.sum(target * np.log(h), axis=1)  # sum of the rows  (N,)
        return np.average(E) 
        # TODO END

    def backward(self, input, target):
        # TODO START
        '''
        :param input: (batch_size=N, 10)
        :param target: (batch_size=N, 10)
        '''
        N, K = input.shape
        mid = np.exp(input)  # (N, K)
        h = mid / np.sum(mid, axis=1, keepdims=True)  # sum of the rows  (N, K)
        tr_sum = np.matmul(target.sum(axis=1, keepdims=True), np.ones(K).reshape(1, K))  # 对 target 按行求和再扩充 (N, K)
        return - (target - tr_sum * h) / N
        # TODO END


class HingeLoss(object):
    def __init__(self, name, margin=5):
        self.name = name
        self.margin = margin

    def forward(self, input, target):
        # TODO START 
        '''
        :param input: (batch_size=N, K=10)
        :param target: (batch_size=N, K=10)
        '''
        N, K = input.shape
        r_expand = np.matmul(input[target > 0].reshape(N, 1), np.ones(K).reshape(1, K))  # (N, K)
        input[target > 0] -= self.margin  # 修正 input，抵消 k = t_n 情况
        res = self.margin - r_expand + input
        res = res * (res > 0)
        return res.sum() / N
        # TODO END

    def backward(self, input, target):
        # TODO START
        '''
        :param input: (batch_size=N, K=10)
        :param target: (batch_size=N, K=10)
        '''
        N, K = input.shape

        r_expand = np.matmul(input[target > 0].reshape(N, 1), np.ones(K).reshape(1, K))  # (N, K)
        mid = self.margin - r_expand + input

        res = np.zeros(input.shape)
        res[mid > 0] = 1
        res[target > 0] -= res.sum(axis=1)
        return res
        # TODO END


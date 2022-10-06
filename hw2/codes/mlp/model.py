# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter

class BatchNorm1d(nn.Module):
	# TODO START
	def __init__(self, num_features, eps=1e-5, momentum=0.1, is_train=True):
		super(BatchNorm1d, self).__init__()
		self.num_features = num_features
		self.momentum = momentum
		self.is_train = is_train
		self.eps = eps

		# Parameters
		self.weight = Parameter(torch.Tensor(num_features)) # will be optimized
		self.bias = Parameter(torch.Tensor(num_features))   # will be optimized

		# Store the average mean and variance
		self.register_buffer('running_mean', torch.zeros(num_features))
		self.register_buffer('running_var', torch.ones(num_features))
		
		# Initialize your parameter
		init.ones_(self.weight)
		init.zeros_(self.bias)

	def forward(self, input):
		# input: [batch_size, num_feature_map * height * width]
		if self.is_train: # training
			mu = input.mean(axis=1).reshape(-1, 1) # 按行求平均 (bsz, 1)
			var = input.var(axis=1).reshape(-1, 1) # (bsz, 1)
			self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu
			self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
		else: # eval
			mu = self.running_mean
			var = self.running_var
		res = (input - mu) / torch.sqrt(var + self.eps)
		return self.weight * res + self.bias
	# TODO END

class Dropout(nn.Module):
	# TODO START
	def __init__(self, p=0.5, is_train=True):
		super(Dropout, self).__init__()
		self.p = p
		self.is_train = is_train

	def forward(self, input):
		# input: [batch_size, num_feature_map * height * width]
		if self.is_train: # training
			return input / (1. - self.p)
		# eval
		return input
	# TODO END

class Model(nn.Module):
	def __init__(self, drop_rate=0.5):
		super(Model, self).__init__()
		# TODO START
		# Define your layers here
		self.linear1 = nn.Linear(32 * 32 * 3, 1024)
		self.BN = BatchNorm1d(num_features=1024)
		self.relu = nn.ReLU()
		self.dropout = Dropout(p=drop_rate)
		self.linear2 = nn.Linear(1024, 10)
		# TODO END
		self.loss = nn.CrossEntropyLoss()

	def forward(self, x, y=None):
		# TODO START
		# the 10-class prediction output is named as "logits"
		logits = self.linear1(x)
		logits = self.BN(logits)
		logits = self.relu(logits)
		logits = self.dropout(logits)
		logits = self.linear2(logits)
		y = y.to(torch.long)
		# TODO END

		pred = torch.argmax(logits, 1)  # Calculate the prediction result
		if y is None:
			return pred
		loss = self.loss(logits, y)
		correct_pred = (pred.int() == y.int())
		acc = torch.mean(correct_pred.float())  # Calculate the accuracy in this mini-batch

		return loss, acc

# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter
class BatchNorm2d(nn.Module):
	# TODO START
	def __init__(self, num_features, eps=1e-5, momentum=0.1):
		super(BatchNorm2d, self).__init__()
		self.num_features = num_features
		self.momentum = momentum
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
		# input: [batch_size, num_feature_map, height, width]
		if self.training: # training
			mu = input.mean([0, 2, 3])
			var = input.var([0, 2, 3])
			self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu
			self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
		else: # eval
			mu = self.running_mean
			var = self.running_var
		res = (input - mu[:, None, None]) / torch.sqrt(var[:, None, None] + self.eps)
		return self.weight[:, None, None] * res + self.bias[:, None, None]
	# TODO END

class Dropout(nn.Module):
	# TODO START
	def __init__(self, p=0.5):
		super(Dropout, self).__init__()
		self.p = p

	def forward(self, input):
		# input: [batch_size, num_feature_map, height, width]
		if self.training: # training
			'''Dropout2d'''
			mask = torch.ones(input.shape[:2]).cuda()
			mask *= (1 - self.p)
			return input * torch.bernoulli(mask).unsqueeze(-1).unsqueeze(-1) / (1. - self.p)
			'''Dropout1d'''
			# mask = torch.ones(input.shape).cuda()
			# mask *= (1 - self.p)
			# return input * torch.bernoulli(mask) / (1. - self.p)
		# eval
		return input
	# TODO END

class Model(nn.Module):
	def __init__(self, drop_rate=0.5):
		super(Model, self).__init__()
		# TODO START
		# Define your layers here
		hidden_channels = [256, 256]
		kernel_size = [5, 5]
		self.network = nn.Sequential(
			# (batch_size, 3, 32, 32)
			nn.Conv2d(in_channels=3, out_channels=hidden_channels[0], kernel_size=kernel_size[0]),
			# (batch_size, 256, 28, 28)
			BatchNorm2d(hidden_channels[0]),
			nn.ReLU(),
			Dropout(p=drop_rate),
			nn.MaxPool2d(kernel_size=2, stride=2),
			# (batch_size, 256, 14, 14)
			nn.Conv2d(in_channels=hidden_channels[0], out_channels=hidden_channels[1], kernel_size=kernel_size[1]),
			# (batch_size, 256, 10, 10)
			BatchNorm2d(hidden_channels[1]),
			nn.ReLU(),
			Dropout(p=drop_rate),
			nn.MaxPool2d(kernel_size=2, stride=2),
			# (batch_size, 256, 5, 5)
		)
		self.linear = nn.Linear(hidden_channels[1] * 5 * 5, 10)
		# TODO END2
		self.loss = nn.CrossEntropyLoss()

	def forward(self, x, y=None):
		# TODO START
		# the 10-class prediction output is named as "logits"
		logits = self.network(x)
		logits = logits.reshape(logits.shape[0], -1)
		logits = self.linear(logits)
		# TODO END

		pred = torch.argmax(logits, 1)  # Calculate the prediction result
		if y is None:
			return pred
		loss = self.loss(logits, y)
		correct_pred = (pred.int() == y.int())
		acc = torch.mean(correct_pred.float())  # Calculate the accuracy in this mini-batch

		return loss, acc

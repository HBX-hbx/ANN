# -*- coding: utf-8 -*-
import sys
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt

from model import Model
from load_data import load_cifar_4d

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=100,
	help='Batch size for mini-batch training and evaluating. Default: 100')
parser.add_argument('--num_epochs', type=int, default=20,
	help='Number of training epoch. Default: 20')
parser.add_argument('--learning_rate', type=float, default=1e-3,
	help='Learning rate during optimization. Default: 1e-3')
parser.add_argument('--drop_rate', type=float, default=0.5,
	help='Drop rate of the Dropout Layer. Default: 0.5')
parser.add_argument('--is_train', type=bool, default=True,
	help='True to train and False to inference. Default: True')
parser.add_argument('--data_dir', type=str, default='../cifar-10_data',
	help='Data directory. Default: ../cifar-10_data')
parser.add_argument('--train_dir', type=str, default='./train',
	help='Training directory for saving model. Default: ./train')
parser.add_argument('--inference_version', type=int, default=0,
	help='The version for inference. Set 0 to use latest checkpoint. Default: 0')
parser.add_argument('--figure_path', type=str, default=os.path.join(os.getcwd(), 'figures'),
	help='The path to save figures for cnn. Default: ./figures')
parser.add_argument('--log_path', type=str, default=os.path.join(os.getcwd(), 'logs'),
	help='The path to save logs for cnn. Default: ./logs')
args = parser.parse_args()

torch.cuda.manual_seed(42)

setting_path = 'bsz_' + str(args.batch_size) + '_lr_' + str(args.learning_rate) + '_drop_' + str(args.drop_rate)
save_setting_path = '_bsz_' + str(args.batch_size) + '_lr_' + str(args.learning_rate)[2:] + '_drop_' + str(args.drop_rate)[2:]
train_loss_list = []  # loss every display
train_acc_list  = []  # accuracy every display
valid_loss_list = []  # loss every display
valid_acc_list =  []  # accuracy every display
final_test_loss = 1e18
final_test_acc = 0.0
best_val_acc = 0.0


def draw():
    print('saving figures to %s' % args.figure_path)
    if not os.path.exists(args.figure_path):
        os.makedirs(args.figure_path)

    loss_path = os.path.join(args.figure_path, 'Loss' + save_setting_path)
    acc_path = os.path.join(args.figure_path, 'Acc' + save_setting_path)

    epoch_list = list(range(len(train_loss_list)))

    plt.plot(epoch_list, train_loss_list)
    plt.plot(epoch_list, valid_loss_list)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(('Train', 'Valid'), loc='center right')
    plt.title(setting_path)
    print(loss_path)
    plt.savefig(loss_path)

    plt.clf()

    plt.plot(epoch_list, train_acc_list)
    plt.plot(epoch_list, valid_acc_list)
    plt.xlabel('Epochs')
    plt.ylabel('ACC')
    plt.legend(('Train', 'Valid'), loc='center right')
    plt.title(setting_path)
    plt.savefig(acc_path)


def log():
	print('saving logs to %s' % args.log_path)
	if not os.path.exists(args.log_path):
		os.makedirs(args.log_path)
	with open(os.path.join(args.log_path, setting_path), 'w+') as f:
		f.write("  training loss:                 " + str(train_loss_list[-1]) + "\n")
		f.write("  training accuracy:             " + str(train_acc_list[-1]) + "\n")
		f.write("  validation loss:               " + str(valid_loss_list[-1]) + "\n")
		f.write("  validation accuracy:           " + str(valid_acc_list[-1]) + "\n")
		f.write("  best validation accuracy:      " + str(best_val_acc) + "\n")
		f.write("  final test loss:               " + str(test_loss) + "\n")
		f.write("  final test accuracy:           " + str(test_acc) + "\n")


def shuffle(X, y, shuffle_parts):
	chunk_size = int(len(X) / shuffle_parts)
	shuffled_range = list(range(chunk_size))

	X_buffer = np.copy(X[0:chunk_size])
	y_buffer = np.copy(y[0:chunk_size])

	for k in range(shuffle_parts):
		np.random.shuffle(shuffled_range)
		for i in range(chunk_size):
			X_buffer[i] = X[k * chunk_size + shuffled_range[i]]
			y_buffer[i] = y[k * chunk_size + shuffled_range[i]]

		X[k * chunk_size:(k + 1) * chunk_size] = X_buffer
		y[k * chunk_size:(k + 1) * chunk_size] = y_buffer

	return X, y


def train_epoch(model, X, y, optimizer): # Training Process
	model.train()
	loss, acc = 0.0, 0.0
	st, ed, times = 0, args.batch_size, 0
	while st < len(X) and ed <= len(X):
		optimizer.zero_grad()
		X_batch, y_batch = torch.from_numpy(X[st:ed]).to(device), torch.from_numpy(y[st:ed]).to(device)
		loss_, acc_ = model(X_batch, y_batch)

		loss_.backward()
		optimizer.step()

		loss += loss_.cpu().data.numpy()
		acc += acc_.cpu().data.numpy()
		st, ed = ed, ed + args.batch_size
		times += 1
	loss /= times
	acc /= times
	return acc, loss


def valid_epoch(model, X, y): # Valid Process
	model.eval()
	loss, acc = 0.0, 0.0
	st, ed, times = 0, args.batch_size, 0
	while st < len(X) and ed <= len(X):
		X_batch, y_batch = torch.from_numpy(X[st:ed]).to(device), torch.from_numpy(y[st:ed]).to(device)
		loss_, acc_ = model(X_batch, y_batch)

		loss += loss_.cpu().data.numpy()
		acc += acc_.cpu().data.numpy()

		st, ed = ed, ed + args.batch_size
		times += 1
	loss /= times
	acc /= times
	return acc, loss


def inference(model, X): # Test Process
	model.eval()
	pred_ = model(torch.from_numpy(X).to(device))
	return pred_.cpu().data.numpy()


if __name__ == '__main__':
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	if not os.path.exists(args.train_dir):
		os.mkdir(args.train_dir)
	if args.is_train:
		X_train, X_test, y_train, y_test = load_cifar_4d(args.data_dir)
		X_val, y_val = X_train[40000:], y_train[40000:]
		X_train, y_train = X_train[:40000], y_train[:40000]
		cnn_model = Model(drop_rate=args.drop_rate)
		cnn_model.to(device)
		print(cnn_model)
		optimizer = optim.Adam(cnn_model.parameters(), lr=args.learning_rate)

		# model_path = os.path.join(args.train_dir, 'checkpoint_%d.pth.tar' % args.inference_version)
		# if os.path.exists(model_path):
		# 	cnn_model = torch.load(model_path)

		pre_losses = [1e18] * 3
		best_val_acc = 0.0
		for epoch in range(1, args.num_epochs+1):
			start_time = time.time()
			train_acc, train_loss = train_epoch(cnn_model, X_train, y_train, optimizer)
			X_train, y_train = shuffle(X_train, y_train, 1)

			val_acc, val_loss = valid_epoch(cnn_model, X_val, y_val)

			if val_acc >= best_val_acc:
				best_val_acc = val_acc
				best_epoch = epoch
				test_acc, test_loss = valid_epoch(cnn_model, X_test, y_test)
				final_test_loss = test_loss
				final_test_acc = test_acc
				# with open(os.path.join(args.train_dir, 'checkpoint_{}.pth.tar'.format(epoch)), 'wb') as fout:
				# 	torch.save(cnn_model, fout)
				# with open(os.path.join(args.train_dir, 'checkpoint_0.pth.tar'), 'wb') as fout:
				# 	torch.save(cnn_model, fout)

			epoch_time = time.time() - start_time
			print("Epoch " + str(epoch) + " of " + str(args.num_epochs) + " took " + str(epoch_time) + "s")
			print("  learning rate:                 " + str(optimizer.param_groups[0]['lr']))
			print("  training loss:                 " + str(train_loss))
			print("  training accuracy:             " + str(train_acc))
			print("  validation loss:               " + str(val_loss))
			print("  validation accuracy:           " + str(val_acc))
			print("  best epoch:                    " + str(best_epoch))
			print("  best validation accuracy:      " + str(best_val_acc))
			print("  test loss:                     " + str(test_loss))
			print("  test accuracy:                 " + str(test_acc))
			train_loss_list.append(train_loss)
			train_acc_list.append(train_acc)
			valid_loss_list.append(val_loss)
			valid_acc_list.append(val_acc)

			if train_loss > max(pre_losses):
				for param_group in optimizer.param_groups:
					param_group['lr'] = param_group['lr'] * 0.9995
			pre_losses = pre_losses[1:] + [train_loss]

	else:
		print("begin testing")
		cnn_model = Model()
		cnn_model.to(device)
		model_path = os.path.join(args.train_dir, 'checkpoint_%d.pth.tar' % args.inference_version)
		if os.path.exists(model_path):
			cnn_model = torch.load(model_path)

		X_train, X_test, y_train, y_test = load_cifar_4d(args.data_dir)

		count = 0
		for i in range(len(X_test)):
			test_image = X_test[i].reshape((1, 3, 32, 32))
			result = inference(cnn_model, test_image)[0]
			if result == y_test[i]:
				count += 1
		print("test accuracy: {}".format(float(count) / len(X_test)))

	draw()
	log()

from network import Network
from utils import LOG_INFO
from layers import Relu, Sigmoid, Linear, Gelu
from loss import EuclideanLoss, SoftmaxCrossEntropyLoss, HingeLoss
from solve_net import train_net, test_net
from load_data import load_mnist_2d
from matplotlib import pyplot as plt
import numpy as np
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument('--loss', type=int, default=0, help='0: EuclideanLoss\n1: SoftmaxCrossEntropyLoss\n2: HingeLoss')
parser.add_argument('--act', type=int, default=0, help='0: Sigmoid\n1: ReLU\n2: GeLU')
args = parser.parse_args()

train_data, test_data, train_label, test_label = load_mnist_2d('data')

train_loss_list = []  # loss every display
train_acc_list  = []  # accuracy every display
test_loss_list = []  # loss every display
test_acc_list  = []  # accuracy every display


np.random.seed(42)

# Your model defintion here
# You should explore different model architecture

def one_hidden_layer():
    linear1 = Linear('fc1', 784, 128, 0.01)
    linear2 = Linear('fc2', 128, 10, 0.01)
    setting_path = ''
    loss = None
    if args.loss == 0:
        loss = EuclideanLoss(name='loss')
        setting_path += '_EuclideanLoss'
        LOG_INFO('using loss: EuclideanLoss')
    elif args.loss == 1:
        loss = SoftmaxCrossEntropyLoss(name='loss')
        setting_path += '_SoftmaxCrossEntropyLoss'
        LOG_INFO('using loss: SoftmaxCrossEntropyLoss')
    elif args.loss == 2:
        loss = HingeLoss(name='loss')
        setting_path += '_HingeLoss'
        LOG_INFO('using loss: HingeLoss')
    else:
        raise ValueError('Undefined loss: %d' % args.loss)

    activate_func = None
    if args.act == 0:
        activate_func = Sigmoid('activate')
        setting_path += '_Sigmoid'
        LOG_INFO('using act: Sigmoid')
    elif args.act == 1:
        activate_func = Relu('activate')
        setting_path += '_Relu'
        LOG_INFO('using act: Relu')
    elif args.act == 2:
        activate_func = Gelu('activate')
        setting_path += '_Gelu'
        LOG_INFO('using act: Gelu')
    else:
        raise ValueError('Undefined activate function: %d' % args.act)

    model = Network()
    model.add(linear1)
    model.add(activate_func)
    model.add(linear2)

    return model, loss, setting_path

def two_hidden_layer():
    linear1 = Linear('fc1', 784, 256, 0.01)
    linear2 = Linear('fc2', 256, 128, 0.01)
    linear3 = Linear('fc3', 128, 10, 0.01)
    setting_path = ''
    loss = None
    if args.loss == 0:
        loss = EuclideanLoss(name='loss')
        setting_path += '_EuclideanLoss'
        LOG_INFO('using loss: EuclideanLoss')
    elif args.loss == 1:
        loss = SoftmaxCrossEntropyLoss(name='loss')
        setting_path += '_SoftmaxCrossEntropyLoss'
        LOG_INFO('using loss: SoftmaxCrossEntropyLoss')
    elif args.loss == 2:
        loss = HingeLoss(name='loss')
        setting_path += '_HingeLoss'
        LOG_INFO('using loss: HingeLoss')
    else:
        raise ValueError('Undefined loss: %d' % args.loss)

    activate_func1 = None
    activate_func2 = None
    if args.act == 0:
        activate_func1 = Sigmoid('activate')
        activate_func2 = Sigmoid('activate')
        setting_path += '_Sigmoid'
        LOG_INFO('using act: Sigmoid')
    elif args.act == 1:
        activate_func1 = Relu('activate')
        activate_func2 = Relu('activate')
        setting_path += '_Relu'
        LOG_INFO('using act: Relu')
    elif args.act == 2:
        activate_func1 = Gelu('activate')
        activate_func2 = Gelu('activate')
        setting_path += '_Gelu'
        LOG_INFO('using act: Gelu')
    else:
        raise ValueError('Undefined activate function: %d' % args.act)

    model = Network()
    model.add(linear1)
    model.add(activate_func1)
    model.add(linear2)
    model.add(activate_func2)
    model.add(linear3)

    return model, loss, setting_path

def draw():
    LOG_INFO('saving figures to %s' % figure_path)
    loss_path = os.path.join(figure_path, 'Loss' + setting_path)
    acc_path = os.path.join(figure_path, 'Acc' + setting_path)
    epoch_list = list(range(len(train_loss_list)))

    plt.plot(epoch_list, train_loss_list)
    plt.plot(epoch_list, test_loss_list)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(('Train', 'Test'), loc='center right')
    plt.title(setting_path[1:])
    plt.savefig(loss_path)

    plt.clf()

    plt.plot(epoch_list, train_acc_list)
    plt.plot(epoch_list, test_acc_list)
    plt.xlabel('Epochs')
    plt.ylabel('ACC')
    plt.legend(('Train', 'Test'), loc='center right')
    plt.title(setting_path[1:])
    plt.savefig(acc_path)


model, loss, setting_path = one_hidden_layer()
figure_path = os.path.join(os.getcwd(), 'figures1')

# model, loss, setting_path = two_hidden_layer()
# figure_path = os.path.join(os.getcwd(), 'figures2')


# Training configuration
# You should adjust these hyperparameters
# NOTE: one iteration means model forward-backwards one batch of samples.
#       one epoch means model has gone through all the training samples.
#       'disp_freq' denotes number of iterations in one epoch to display information.

config = {
    'learning_rate': 1e-4,
    'weight_decay': 2e-4,
    'momentum': 0.9,
    'batch_size': 100,
    'max_epoch': 100,
    'disp_freq': 100,
    'test_epoch': 1
}


for epoch in range(config['max_epoch']):
    LOG_INFO('Training @ %d epoch...' % (epoch))
    train_net(model, loss, config, train_data, train_label, config['batch_size'], config['disp_freq'], train_loss_list, train_acc_list)

    if epoch % config['test_epoch'] == 0:
        LOG_INFO('Testing @ %d epoch...' % (epoch))
        test_net(model, loss, test_data, test_label, config['batch_size'], test_loss_list, test_acc_list)

draw()
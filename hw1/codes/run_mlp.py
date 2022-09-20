from network import Network
from utils import LOG_INFO
from layers import Relu, Sigmoid, Linear, Gelu
from loss import EuclideanLoss, SoftmaxCrossEntropyLoss, HingeLoss
from solve_net import train_net, test_net
from load_data import load_mnist_2d
import numpy as np


train_data, test_data, train_label, test_label = load_mnist_2d('data')

np.random.seed(42)

# Your model defintion here
# You should explore different model architecture

model = Network()
# model.add(Linear('fc1', 784, 10, 0.01))
# model.add(Sigmoid('g1'))
# model.add(Linear('fc2', 256, 10, 0.01))
# model.add(Sigmoid('sg2'))

model = Network()
model.add(Linear('fc1', 784, 10, 0.01))
# model.add(Sigmoid('re1'))
# model.add(Linear('fc2', 256, 10, 0.01))
# model.add(Sigmoid('re2'))

# loss = EuclideanLoss(name='loss')
# loss = SoftmaxCrossEntropyLoss(name='loss')
loss = HingeLoss(name='loss')

# Training configuration
# You should adjust these hyperparameters
# NOTE: one iteration means model forward-backwards one batch of samples.
#       one epoch means model has gone through all the training samples.
#       'disp_freq' denotes number of iterations in one epoch to display information.

config = {
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'momentum': 0.09,
    'batch_size': 100,
    'max_epoch': 100,
    'disp_freq': 50,
    'test_epoch': 5
}


for epoch in range(config['max_epoch']):
    LOG_INFO('Training @ %d epoch...' % (epoch))
    train_net(model, loss, config, train_data, train_label, config['batch_size'], config['disp_freq'])

    if epoch % config['test_epoch'] == 0:
        LOG_INFO('Testing @ %d epoch...' % (epoch))
        test_net(model, loss, test_data, test_label, config['batch_size'])

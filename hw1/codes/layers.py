import numpy as np


class Layer(object):
    def __init__(self, name, trainable=False):
        self.name = name
        self.trainable = trainable
        self._saved_tensor = None

    def forward(self, input):
        pass

    def backward(self, grad_output):
        pass

    def update(self, config):
        pass

    def _saved_for_backward(self, tensor):
        '''The intermediate results computed during forward stage
        can be saved and reused for backward, for saving computation'''

        self._saved_tensor = tensor

class Relu(Layer):
    def __init__(self, name):
        super(Relu, self).__init__(name)

    def forward(self, input):
        # TODO START
        '''Your codes here'''
        self._saved_for_backward(input)
        return (abs(input) + input) / 2  # max(0, input)
        # TODO END

    def backward(self, grad_output):
        # TODO START
        '''Your codes here'''
        grad_output[self._saved_tensor <= 0] = 0
        return grad_output
        # TODO END

class Sigmoid(Layer):
    def __init__(self, name):
        super(Sigmoid, self).__init__(name)

    def forward(self, input):
        # TODO START
        '''Your codes here'''
        res = 1.0 / (1 + np.exp(-input))
        self._saved_for_backward(res)
        return res
        # TODO END

    def backward(self, grad_output):
        # TODO START
        '''
        :param grad_output: (1, output_dim)
        '''
        return self._saved_tensor * (1.0 - self._saved_tensor) * grad_output
        # TODO END

class Gelu(Layer):
    def __init__(self, name):
        super(Gelu, self).__init__(name)

    def forward(self, input):
        # TODO START
        '''Your codes here'''
        self._saved_for_backward(input)
        mid1 = np.sqrt(2 / np.pi) * (input + 0.044715 * np.power(input, 3))
        mid2 = 1 + np.tanh(mid1)
        return 0.5 * input * mid2
        # TODO END
    
    def backward(self, grad_output):
        # TODO START
        '''Your codes here'''
        delta = 1e-5
        x1 = self._saved_tensor + delta
        x2 = self._saved_tensor
        y1 = 0.5 * x1 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x1 + 0.044715 * np.power(x1, 3))))
        y2 = 0.5 * x2 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x2 + 0.044715 * np.power(x2, 3))))
        return grad_output * (y1 - y2) / delta
        pass
        # TODO END

class Linear(Layer):
    def __init__(self, name, in_num, out_num, init_std):
        super(Linear, self).__init__(name, trainable=True)
        self.in_num = in_num
        self.out_num = out_num
        self.W = np.random.randn(in_num, out_num) * init_std  # in_num * out_num
        self.b = np.zeros(out_num)  # 1 * out_num

        self.grad_W = np.zeros((in_num, out_num))  # in_num * out_num
        self.grad_b = np.zeros(out_num)  # 1 * out_num

        self.diff_W = np.zeros((in_num, out_num))  # in_num * out_num
        self.diff_b = np.zeros(out_num)  # 1 * out_num

    def forward(self, input):
        # TODO START
        '''
        :param input: batch_size * in_num
        '''
        self._saved_for_backward(input)
        # print('W: ', self.W)
        return np.matmul(input, self.W) + self.b  # xW + b: 1 * out_num
        # TODO END

    def backward(self, grad_output):
        # TODO START
        '''
        :param grad_output: (batch_size, out_num) 对输出 output 的梯度
        :return (batch_size, in_dim) 对输入 input 的梯度
        '''
        self.grad_W = np.matmul(self._saved_tensor.T, grad_output)  # (in_dim, bsz) * (bsz, ou_dim)
        self.grad_b = grad_output.mean(axis=0)  # 按行求平均
        return np.matmul(grad_output, self.W.T)  # (bsz, out_dim) * (out_dim, in_dim)
        # TODO END

    def update(self, config):
        mm = config['momentum']
        lr = config['learning_rate']
        wd = config['weight_decay']

        self.diff_W = mm * self.diff_W + (self.grad_W + wd * self.W)
        self.W = self.W - lr * self.diff_W

        self.diff_b = mm * self.diff_b + (self.grad_b + wd * self.b)
        self.b = self.b - lr * self.diff_b

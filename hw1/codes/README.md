## 1. 修改

### 1.1 `run_mlp.py`

+ 绘图
  + 导入了 `matplotlib`
  + 新增了 `draw()` 函数用于绘图并保存到当前文件夹
  + train_loss_list、train_acc_list、test_loss_list、test_acc_list 四个列表用于绘图的 `x` 和 `y` 轴
  + 将上述四个变量传入 `train_net()` 和 `test_net()` 函数，用于计算每个 epoch 的 loss
+ 选择损失函数和激活函数
  + 导入了 `argparse`
  + 根据 `--loss` 和 `--act` 指定所使用的损失函数和激活函数
+ 单双层 `MLP` 模型
  + 新增了 `one_hidden_layer()` 和 `two_hidden_layer()` 两个函数，用于初始化两种网络架构，并根据 `args` 选择损失函数和激活函数
+ 复现结果
  + 导入了 `numpy` 用于设置 `seed`

### 1.2 `solve_net.py`

+ 绘图
  + 为 `train_net()` 和 `test_net()` 函数各新增了两个参数，用于记录每个 `epoch` 的 `accuracy` 和 `loss`，即 `run_mlp.py` 中传过来的四个 `list`，最后用于绘图

## 2. 使用

首先安装好必要的依赖：`numpy`、`matplotlib`，并将数据集解压在 `codes` 的 `data` 目录中

接着：

```shell
python run_mlp.py [--loss LOSS] [--act ACT]
```

其中：

+ `--loss`：指定损失函数，$0$ 表示 `EuclideanLoss`，$1$ 表示 `SoftmaxCrossEntropyLoss`， $2$ 表示 `HingeLoss`
+ `--act`：指定激活函数，$0$ 表示 `Sigmoid`，$1$ 表示 `Relu`，$2$ 表示 `Gelu`

示例：

```shell
python run_mlp.py  # 默认 --loss 0 --act 0，即使用 EuclideanLoss 和 Sigmoid
python run_mlp.py --loss 1 --act 2  # 使用 SoftmaxCrossEntropyLoss 和 Gelu
```

`MLP` 层数选择：

+ 单层：启用 `line 151、152`，注释掉 `line 154、155`
+ 双层：注释掉 `line 151、152`，启用 `line 154、155`
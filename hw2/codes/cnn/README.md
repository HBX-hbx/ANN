## 1. 修改

### 1.1 新增文件：`run.sh`

该脚本用于跑超参实验，即调整 `learning_rate`、`batch_size` 和 `dropout_rate` 这三个参数，共 $4\times 4\times 4=64$ 组实验。

### 1.2 `main.py`

#### 1.2.1 绘图

+ 导入 `matplotlib.pyplot`

  ```python
  from matplotlib import pyplot as plt
  ```

+ `draw()` 函数用于绘图

+ 新增变量（用于记录 `loss` 和 `acc`，或者记录实验 setting）：

  ```python
  setting_path = 'bsz_' + str(args.batch_size) + '_lr_' + str(args.learning_rate) + '_drop_' + str(args.drop_rate)
  save_setting_path = '_bsz_' + str(args.batch_size) + '_lr_' + str(args.learning_rate)[2:] + '_drop_' + str(args.drop_rate)[2:]
  train_loss_list = []  # loss every display
  train_acc_list  = []  # accuracy every display
  valid_loss_list = []  # loss every display
  valid_acc_list =  []  # accuracy every display
  ```

#### 1.2.2 保存实验结果

+ 新增参数：`figure_path` 和 `log_path`，作为输出图片和以及输出信息的保存路径

  输出信息形如：

  ```shell
  training loss:                 0.45609002649784086
  training accuracy:             0.8525249785184861
  validation loss:               1.4961878669261932
  validation accuracy:           0.540699987411499
  best validation accuracy:      0.5606999871134758
  final test loss:               1.3196451157331466
  final test accuracy:           0.5540999901294709
  ```

+ `log()` 函数：用于保存每一组实验的最后一个 `epoch` 输出

## 1.3 其余更改

+ 注释了保存 `checkpoints` 的代码
+ `model.py` 中 `BatchNorm2d` 的类名

## 2. 使用

+ 设置的 `seed` 为 $42$
+ 需要在有 `cuda` 环境下跑
+ 要测试 `bonus` 第一项时，可以在 `model.py` 下 `Dropout` 类的 `forward` 方法中，在 `Dropout1d` 和 `Dropout2d` 之间通过注释来切换。
+ 要测试 `bonus` 第二项时，可以直接在 `cnn` 目录下跑 `run.sh` 脚本：`bash ./run.sh`，跑完之后会默认在当前目录出现 `figures/` 和 `logs/` 文件夹，里面分别存有图像和 `log` 信息。
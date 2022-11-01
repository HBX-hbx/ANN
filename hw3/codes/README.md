## 1. 修改

### 1.1 新增文件：`ds_run.sh`

该脚本用于实验报告中第二节部分的实验，即调整 `decode_strategy`、`temperature` 对不同的模型进行实验。

### 1.2 缺少文件：`.DS_Store`

不小心删除了...

### 1.3 `main.py`

#### 1.3.1 画图

+ 导入了 `SummaryWriter` 包
+ 设置了 `writer`
+ 调用了 `add_scalars` 函数进行画图

#### 1.3.2 路径设置

+ 新增了 `args: id`，取值 $1、2、3$，用于跑 `Bonus` 实验里的第二个小实验，即取不同层的 `12-layers` 的实验编号
+ 对一些参数的默认值进行更改，比如 `--train_dir` 和 `--pretrain_dir`
+ 设置路径：`setting_path`
+ `fast_evaluate` 函数中：

```python
all_loss += loss.cpu().numpy().tolist()
# 改为
all_loss += [loss.cpu().numpy().tolist()]
```

对于一个数用 `tolist()` 函数得到的还是一个数，需要加上 `[]`，不然会报错

#### 1.3.3 load_model 函数

根据 `args.id` 加载 `12-layers` 的某三层，并初始化好三层模型的参数，返回生成的模型

#### 1.3.4 device

调整了 `device` 的初始化。

#### 1.3.5 `train/val` 结果的记录

将 `print` 放在每个 `epoch` 下而不是只有在 `val_ppl < best_val_ppl` 时输出，并注释掉了 `early_stop` 的部分

同时将最后一个 `epoch` 下的 `train/val` 结果记录在一个 `txt` 文件中

#### 1.3.6 `test` 结果记录

修改了 `test` 时的顺序，先 `evaluate` 再去 `decode`，这样便于记录 `test` 结果到 `txt` 文件中，同时修改了记录的路径，改为 `setting_path`。

## 2. 使用

我们按照实验报告里的顺序分别介绍如何复现，过程中会生成不少 `txt` 文件，记录 `train/val/test` 结果以及生成的句子：

### 2.1 第一节

我们把预训练的模型放在 `pretrain/3_layers` 和 `pretrain/12_layers` 下。

#### 2.1.1 训练

```shell
python main.py --name Tfmr_scratch
python main.py --name Tfmr_finetune --pretrain_dir ./pretrain/3_layers
```

#### 2.1.2 测试

```
python main.py --test Tfmr_scratch
python main.py --test Tfmr_finetune
```

### 2.2 第二节

直接运行 `ds_run.sh` 即可。

### 2.3 第三节

在 `output_*.txt` 中可以看到对应实验的生成的句子。

### 2.4 第五节

#### 2.4.1 不同层数 Transformer

+ 训练

```shell
python main.py --name Tfmr_finetune --pretrain_dir ./pretrain/3_layers
python main.py --name Tfmr_finetune——12 --pretrain_dir ./pretrain/12_layers
```

+ 测试

```
python main.py --test Tfmr_finetune --decode_strategy top-p --temperature 0.7 --top_p 0.9
python main.py --test Tfmr_finetune_12 --decode_strategy top-p --temperature 0.7 --top_p 0.9
```

#### 2.4.2 不同层 `Transformer`

+ 训练

```python
python main.py --pretrain_dir ./pretrain/12_layers --name Tfmr_first --id 1
python main.py --pretrain_dir ./pretrain/12_layers --name Tfmr_jump --id 2
python main.py --pretrain_dir ./pretrain/12_layers --name Tfmr_last --id 3
```

+ 测试

```python
python main.py --test Tfmr_first --decode_strategy top-p --temperature 0.7 --top_p 0.9
python main.py --test Tfmr_jump --decode_strategy top-p --temperature 0.7 --top_p 0.9
python main.py --test Tfmr_last --decode_strategy top-p --temperature 0.7 --top_p 0.9
```

#### 2.4.3 不同 head 数目

+ 训练和测试跟上面基本一致，只需修改 `config.json` 的 `head` 数目即可
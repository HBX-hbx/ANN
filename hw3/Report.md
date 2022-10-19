<center><font size=6>ANN Lab3 Report</font></center>

<center><font size=4>何秉翔 计04 2020010944</font></center>

## 0. 前言

Lab3 实验主要使用 `PyTorch` 框架实现一个基于 Transformer 的语言模型，我们需要实现 `Multi-head Self Attention` 的机制，需要全面理解 `Attention` 的原理以及 `Transformer` 的架构。

## 1. 模型训练（`Tfmr-scratch` && `Tfmr-finetune`）

我们未修改任何超参，使用默认超参训练 $20$ 个 `epoch`。

### 1.1 `Tfmr-scratch`

下面分别汇报该模型在 `train` 和 `val` 数据集上的 `loss` 图像，以及在 `val` 上表现最好的模型在 `test` 集上的结果。

#### 1.1.1 train/val loss



#### 1.1.2 test 结果

|    Metric     | `Tfmr-scratch` |
| :-----------: | :------------: |
|  Perplexity   |                |
| Forward BLEU  |                |
| Backward BLEU |                |
| Harmonic BLEU |                |

### 1.2 `Tfmr-finetune`

下面分别汇报该模型在 `train` 和 `val` 数据集上的 `loss` 图像，以及在 `val` 上表现最好的模型在 `test` 集上的结果。

#### 1.1.1 train/val loss



#### 1.1.2 test 结果

|    Metric     | `Tfmr-finetune` |
| :-----------: | :-------------: |
|  Perplexity   |                 |
| Forward BLEU  |                 |
| Backward BLEU |                 |
| Harmonic BLEU |                 |

### 1.3 实验结果对比与分析

## 2. decode_strategy 探究

### 2.1 实验设置

在本部分，我们直接对任务 $1$ 里的两个模型进行 `inference`。在 `inference` 阶段，
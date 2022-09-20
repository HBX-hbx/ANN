<center><font size=6>ANN Lab1 Report</font></center>

<center><font size=4>何秉翔 计04 2020010944</font></center>

## 1. 单隐藏层 MLP

### 1.1 实验环境

在该部分中，我们构建一个具有一层隐藏层的 MLP，并对三种激活函数和三种损失函数进行组合，共九种组合。其余超参按如下给定：

```json
config = {
    
}
```

对于隐藏层的维度，

对于 `Hinge Loss`，我们选取 `margin = 5`

### 1.2 实验结果

#### 1.2.1 Train

最后一步 Train 之后的结果为：（ACC / Loss）

| Accuracy / Loss | EuclideanLoss | SoftmaxCELoss | HingeLoss |
| :-------------: | :-----------: | :-----------: | :-------: |
|   **Sigmoid**   |               |               |           |
|    **ReLU**     |               |               |           |
|    **GeLU**     |               |               |           |



#### 1.2.2 Test

最后一步 Test 之后的结果为：（ACC / Loss）

| Accuracy / Loss | EuclideanLoss | SoftmaxCELoss | HingeLoss |
| :-------------: | :-----------: | :-----------: | :-------: |
|   **Sigmoid**   |               |               |           |
|    **ReLU**     |               |               |           |
|    **GeLU**     |               |               |           |

## 2. 双隐藏层 MLP


---
title: Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting
authors: Bing Yu, Haoteng Yin, Zhanxing Zhu
year: 2018
---

# Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting 时空图卷积网络

## 问题

### 1. 什么是图神经网络？

- GNN是提取特征的方法，假如我们已知节点的标签和图的结构信息，通过GNN聚合、更新、循环，多层聚合后的节点表示相当于获得了多阶邻居的标签，可以用于分类、关联预测等。

### 2. 时空图神经网络的时空指的是什么？

- spatial即空间，指的是图的拓扑结构；temporal指图的时间序列。

### 3. 具有完整卷积结构的模型？图卷积GCN？

- 形如RNN从序列数据、CNN从网格结构，GCN是从图结构的数据中提取特征的方法，同样也是一个神经网络层。
- 图像的卷积很明显，那么graph上的卷积怎么做呢？离散卷积的本质是加权求和，GCN理论要找到针对图这种结构数据的卷积参数，傅立叶变换+Laplacian算子为基础数学推导出了GCN公式。

### 4. 图的多尺度指的是什么？

- 对信号的不同粒度的采样

### 5. 使用STGCN相比与已有的算法可以解决什么问题？

- 大多数方法在短期交通（5-30min）预测表现良好，但在长期预测（>30min）上效果较差。
- 现有的方法可以分为两类，一是动态建模，用物理知识和数学工具模拟解决交通问题，缺点是耗费大量时间和计算力；二是数据驱动方式，又分为经典统计ARIMA和机器学习。
- 深度学习方法如DBN、SAE被应用于交通任务，但是从稠密网络中提取时空特征领域还有困难。有些方法用LSTM+CNN来处理这类问题但是却有难以训练、计算力需求大的缺点。

### 6. LSTM的具体原理？

- RNN循环神经网络处理序列数据，LSTM相比与普通RNN解决了训练过程中梯度消失和梯度爆炸的问题。
- LSTM多了一个变换慢的传输状态，RNN原本的传输状态变化很快。
- LSTM内部分为三个阶段，忘记、选择记忆、输出，由三个门控状态来负责。
- 因为LSTM训练难度大，可以选择GRU取代他。

### 7. 梯度消失和梯度爆炸的概念？

- 损失函数计算的误差通过梯度反向传播，对NN中的权值进行更新，梯度值特别大或接近0，两者在本质原理上是一样的。
- 造成梯度消失的原因？深度网络+不合适的激活函数，靠近输入层隐藏层权值更新缓慢，靠近输出层隐藏层梯度相对正常，相当于只有最后几层网络在学习。
- 造成梯度爆炸的原因？深度网络+权值初始化值太大

### 8.这篇文章做了什么？

- 提出深度学习框架STGCN，用于处理交通领域的时间序列预测问题。
- 用近似傅立叶变换的两种策略来降低时间复杂度。其一为切比雪夫多项式近似，其二是1阶近似。
- 基于RNN的方法费时、门控复杂、对动态变化很缓慢，本文用卷积结构在时间轴上来捕捉时间特征。

### 9. GCN的数学原理？傅立叶变化和拉普拉斯算子的基础知识。

- 拉普拉斯矩阵是顶点的度矩阵和邻接矩阵之差。因为是对称矩阵，可以用于特征分解（谱分解）

### 10. 如何理解“结构化数据”？

- 不方便用数据库二维逻辑表来表现的数据即称为非结构化数据

### 11. 如何理解捕捉时间特征的CNN？什么是GLU？

- GLU能够处理并行数据的CNN网络架构，利用CNN+门控机制实现了RNN的功能。相比与RNN的优点是保留了时序位置并加快了计算速度。

### 12. bottleneck strategy和residual connection的意思?

- bottleneck、skip connection、residual block三者都和ResNet有关。bottleneck用1x1的卷积来进行升维降维，skip connection顾名思义“跳连接”，用于构造深层网络。

### 13. 上采样upstreaming？

- 缩小图像是下采样，放大图像是上采样

### 14. 什么是Batch Normalization?

- BN是一种数据归一化方法，常用于激活层之前。作用是加快收敛速度，避免梯度消失或爆炸，可以达到更好的精度。

## 相关阅读

### 1. https://zhuanlan.zhihu.com/p/286445515
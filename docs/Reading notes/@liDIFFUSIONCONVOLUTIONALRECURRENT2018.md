---
title: DIFFUSION CONVOLUTIONAL RECURRENT NEURAL NETWORK: DATA-DRIVEN TRAFFIC FORECASTING
authors: Yaguang Li, Rose Yu, Cyrus Shahabi, Yan Liu
year: 2018
---

# DCRNN

## 问题

### 1. 交通预测面临哪些困难/挑战？

- 路网的复杂空间依赖；路况条件非线性时间动态变换；长时间预测的困难性

### 2. 如何理解diffusion?

- 某传感器的交通信息与其邻居相关，交通流沿路网扩散出去。

### 3. DCRNN用了什么架构的深度学习模型来解决问题？

- 空间：双向随机游走，捕捉交通上下游信息
- 时间：定时采样的encoder-decoder architecture 
- 交通的空间结构是非欧几里得的，基于有向图
- diffusion convolution + GRU + Seq2Seq + scheduled sampling

### 4. encoder-decoder和auto-encoder的区别？

- encoder-decoder通常用于处理输入输出序列变长的情况，decoder通常是RNN。auto-encoder是一种encoder-decoder模式

### 5. 交通预测的历史

- ARIMA等知识驱动的方法，依赖于平稳性假设
- 一些深度学习模型忽略了空间结构，构建交通于2D空间或无向图都是错误
- DCRNN将交通流的变动建模为一个动态的过程，用diffusion convolution来捕获

### 6. 捕获交通流扩散为什么采用卷积方法？

- GCN用谱图分解来实现，局限于无向图。从频谱域到顶点域，扩散卷积神经把卷积定义为从节点扩散过程。

### 7. GRU?

- RNN的一种，两个门控，与LSTM有相当的表现，计算上更容易。

### 8. Seq2Seq?

*XMind: ZEN - Trial Version*
---
Title: 29-卷积神经网络AlexNet
Date: 2021-12-07
---

## keywords
#Dropout #ReLU #池化 
## 笔记

AlexNet的改进：

1. $ReLU(x)=max(0,x)$ 网络训练速度以更快的收敛。
2. 降采样 -> 池化。Max Pooling 非线性
3. Dropout 训练时以概率p丢弃神经元，测试时用完整的网络结构参数(w,b)乘以(1-p)。
4. 增加训练样本
5. GPU加速



## 总结



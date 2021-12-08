---
Title: 28-卷积神经网络Lenet
Date: 2021-12-07
---

## keywords
#卷积神经网络
## 笔记

早期：人为地设计卷积核。

卷积神经网络：自动学习卷积核

图像 = $height * width * channel$
视频 = $height * width * channel * time$

![](assets/Pasted%20image%2020211207122344.png)

图像卷积 类似 全连接层权值共享

步长 Stride  ：卷积核的移动

卷积得到的张量被称为feature map特征图

![](assets/Pasted%20image%2020211207123208.png)

特征图大小：$(k,l) | k \le (M-m)/u+1, l \le (N-n)/v$

padding 补零zero-padding

多卷积核 -> 特征图的channel数

是否要带偏置？自己选择。

subsampling 下采样

## 总结
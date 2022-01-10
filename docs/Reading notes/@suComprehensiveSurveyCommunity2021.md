---
title: A Comprehensive Survey on Community Detection with Deep Learning
authors: Xing Su, Shan Xue, Fanzhen Liu, Jia Wu, Jian Yang, Chuan Zhou, Wenbin Hu, Cecile Paris, Surya Nepal, Di Jin, Quan Z. Sheng, Philip S. Yu
year: 2021
---

# 摘要

社区内成员的特征和连接 有别于 其他网络中的社区

传统方法：谱聚类 统计推断

深度学习方法——在高维网络数据表现出优势

综述survey 提出了新的taxonomy


# 介绍

2002 graph partition

利用网络的**拓扑结构**和**语意信息**，对**动态/静态**、**小型/巨型** 网络进行社区检测。

在社区检测中 分析网络动态和社区影响

物以类聚，人以群分 六度空间

社区检测的实际意义：广告推荐、引文网络、生物蛋白网络、大脑网络。

传统方法不适用的原因：数据中含有丰富的非线性信息，计算的代价昂贵。

深度学习方法：非线性特征、低维网络嵌入、多种信息。

现在的文章关注：特殊技巧、网络类型、社区类型、应用场景

## 定义

有向图：邻接矩阵不对称（描述有无）
权重图：描述边权
符号图：边权为1或-1
属性图：节点带属性

社区：disjoint 分离 overlapping重叠

输入：邻接矩阵、属性矩阵、测量矩阵
输出：社区（重叠和分离）

## 社区检测的发展

**Graph Partition** 又称为图聚类，给定社区数量K，将网络分割。

两种代表性算法：
- Kernighan-Lin 
- Spectral bisection

**Statistical Inference** 代表性算法SBM及其变种

**Hierarchical Clustering** 分层聚类 

- divisive分裂：GN算法（通过连续的去除边导致新社区出现？）
- agglomerative凝结：FastQ算法
- hybrid：CDASS合并上述两种策略

**Dynamical Methods** 利用随机游走来动态监测社区

Dynamical Methods的代表性算法有InfoMaps 和 LPA

**Spectral Clustering** 图谱反映了社区结构

**Density-based Algorithms** 基于密度的算法 

代表性算法 DBSCAN SCAN LCCD

**Optimizations** Modularity (Q)是最经典的优化目标函数

**为什么社区检测需要深度学习？**

直接从连接信息中可能会获取次佳的社区结果。

深度学习方法从高维的复杂结构关系数据中学习**低维向量**，然后用上sota的技术。

深度学习框架可以**嵌入非结构特征**。

可以将边、节点、邻居、多图的**信息组合**在一起进行社区检测，有更好的效果。

现实世界中大数据、大规模、高稀疏、复杂结构、动态网络的使用场景通过深度学习方法来发掘。

## 分类法

卷积网络、图注意力网络、对抗生成网络、自动编码器、深度非负矩阵分解、深度稀疏过滤。

![](assets/Pasted%20image%2020211218232939.png)


## 基于卷积的社区检测

两种形式：CNN GCN

CNN为网格形式的拓扑数据设计（如图片），减少了计算代价，池化操作保证了特征表示的鲁棒性

GCN 为图结构的数据所提出的，一阶近似性

$H^{l+1}=\sigma(\tilde D^{-\frac{1}{2}} \tilde A \tilde D^{-\frac{1}{2}} H^{(l)}W^{(l)})$

其中$H^{(0)}=X$即节点属性矩阵

直觉上 就是对邻居加权平均

### CNN

对输入数据有严格限制，需要预处理为图像格式和标注的数据？

节点和边预处理为图数据 （具体如何做？）

节点分为不同种类，边分为类间和类内

先预处理为图像数据，再多个卷积层，然后全连接输出表示。然后分开边和点，节点分类为不同社区，边分类为类间和类内两种类型，去掉类间的边，用某个指标（测量）来优化。

**不完整的拓扑结构** 导致传统算法的准确率降低

（CNN）TINs被踢出 2层CNN+1层全连接，可以从基本的输入中恢复较为完整的社区结构，高阶邻居表示提升了精确度

为了解决大规模社交网络中的高稀疏性，两种方法：SparseConv和SparseConv2D

ComNet-R 边到图像的模型，将边分为类间和类内，去除类见的边作为初始化，优化过程用局部模块化指标（合并社区）

### GCN

两种社区检测方法：
- 监督/半监督社区分类
- 无监督社区聚类

LGNN（监督学习） 改进了SBM

GCN最初被设计时没有关注学习节点嵌入，MRFasGCN（半监督）

SGCN（无监督）聚合网络拓扑和节点属性



## 遗留问题

GCN的数学原理
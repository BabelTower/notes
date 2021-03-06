---
title: Network representation learning: an overview
authors: Cheng Yang, Zhiyuan Liu, Cunchao Tu, Maosong Sun
year: 2017
---

# 图嵌入中文综述

## 问题

### 1. 如何理解可伸缩性？

- 在图嵌入领域，可伸缩性指的是某种方法能处理的网络规模的大小，通常受到时间复杂度的限制，当算法的时间复杂度为顶点数的平方时，我们常说这个算法的可伸缩性不好。

### 2. 一阶近似、二阶近似？

- 一阶近似可以直观的理解为节点之间的边权，边权表示节点间的相似度，值越高两个顶点越相似。
- 二阶近似描述了节点领域结构的相似性，衡量两个节点邻域的相似度，通过比较节点和其他所有节点一阶近似构成的向量。

### 3. 什么是余弦距离？

- 余弦距离=1-余弦相似度
- 余弦相似度=两个向量间夹角的余弦值
- 取值范围在[0,2]，具有非负性
- 余弦距离显示方向上的相对差异，欧式距离显示数值上的绝对差异

### 4. 网络表示学习的作用

- 在原始数据（网络）与应用任务（节点分类、链接预测、社区检测）之间构建桥梁，（无监督或半监督的）学习每个节点的向量表示，用于后续的网络应用任务（已有的机器学习算法）。

### 5. 网络的拉普拉斯矩阵？

- 图的拉普拉斯矩阵是图上的（离散的）拉普拉斯算子，拉普拉斯算子可以计算一个点到它所有自由度微小扰动的增益。L=D-W
- 拉普拉斯矩阵与拉普拉斯算子的关系 - superbrother的文章 - 知乎
https://zhuanlan.zhihu.com/p/85287578

### 6. 什么是流形？

- 参考花书

### 7. 什么是谱聚类？

### 基于网络结构

- 8. 基于矩阵特征向量计算

	- 先定义一个节点表示的线性或二次损失函数，再将最优化问题转化成某个关系矩阵的特征向量计算
	- 缺点：复杂度，特征向量的计算时间是非线性的，非常消耗时间和空间（要将关系矩阵整体存于内存之中）

- 9. 基于简单神经网络的算法

	- DeepWalk使用随机游走序列，有两点优势：1、只依赖于局部信息，适用于分布式和在线系统2、降低0-1二值邻接矩阵的方差和不确定性？
	- LINE适用于有向带权图，对一阶相似度和二阶相似度进行了概率建模，并最小化了概率分布和经验分布的KL距离

- 10. 基于矩阵分解的方法

	- GrapRep的关系矩阵等价于节点通过k步随机游走抵达的概率，将不同k步对应的表示拼接起来，缺点是计算效率低
	- 寻找一种间接近似高阶的关系矩阵而不增加计算复杂度的方法NEU

- 11. 基于深层神经网络的方法

	- SDNE，非线性建模，两个组成部分：1、拉普拉斯矩阵——一阶相似度2、深层自编码器——二阶相似度

- 12. 基于社区发现的算法

	- 节点表示的每一维对应节点从属于某一社区的强度
	- BIGCLAM学习非负向量表示，两节点的向量表示内积越大，形成边的概率也越大

- 13. 保存特殊性质的网络

	- HOPE保存网络的非对称性信息
	- CNRL在节点表示中嵌入网络隐藏的社区信息

### 基于外部信息

- 14. 文本信息

	- TADW将节点的文本特征引入
	- CANE利用文本信息对节点间的关系进行解释，根据不同邻居学习上下文相关的表示。节点表示由文本表示向量和结构表示向量

- 15. 半监督

	- 把已标注的节点类别或标签利用起来，增加区分性，问题在于如何将部分标记的节点标签利用起来？
	- MMDW学习有区分性的网络表示，同时学习矩阵分解形式的网络表示模型和最大间隔分类器。
	- DDRW与MMDW类似，同时训练了DeepWalk模型和最大间隔分类器
	- node2vec改变随机游走序列的生成方式，引入了dfs和bfs
	- GCN设计了作用于网络结构上的卷积神经网络，基于边的标签
	- Planetoid联合预测邻居节点和类别标签

- 16. 边上标签

	- 节点之间也存在丰富的交互（语义）信息，已有的网络表示学习模型更侧重于节点本身的信息，忽略了节点之间的具体关系
	- TransNet平移机制来解决社会关系抽取问题

### 17. 了解什么是网格搜索参数？

- 适用于3-4个参数，排列组合，选择在验证集上误差最小的一组参数
- 较多参数的情况适用随机搜索

### 18. 什么是分布式表示方案？

- 也可称为“分散式”

### 19. GRL和GNN的区别

- GRL要学习的网络节点的低维向量表示，以供后续的机器学习任务；GNN是端到端的系统，可以说囊括了GRL。


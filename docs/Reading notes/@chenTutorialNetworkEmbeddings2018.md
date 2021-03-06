---
title: A Tutorial on Network Embeddings
authors: Haochen Chen, Bryan Perozzi, Rami Al-Rfou, Steven Skiena
year: 2018
---

## 前置知识/术语

### Word2Vec之Skip-Gram

> Word2Vec其实就是通过学习文本来用词向量的方式表征词的语义信息，即通过一个嵌入空间使得语义上相似的单词在该空间内距离很近。Embedding其实就是一个映射，将单词从原先所属的空间映射到新的多维空间中，也就是把原先词所在空间嵌入到一个新的空间中去。

两种模型：
- Skip-Gram : 给定input word来预测上下文
- CBOW : 给定上下文，来预测input word

[理解 Word2Vec 之 Skip-Gram 模型](https://zhuanlan.zhihu.com/p/27234078)

### Modularity 模块化

> 要理解modularity的定义，首先需要理解community。目前比较广泛的认识，community是指网络中的一组节点，它们之间的link/connection要比和外界的link/connection更多。modularity正是基于这个定义，通过比较community内部和外部connection来衡量切分的优劣。

$$ Q = \sum_i(e_{ii} - a_i^2) = Tre - \parallel e^2 \parallel $$

当modularity这个度量被认可后，后续很多算法的思路就是如何找到一个partitioning的方法，使得modularity最大。将community detection转化成了最优化的问题。

**缺陷：** 在large network中，基于modularity的方法找不到那些small community，即便这些small community的结构都很明显。

[Community Detection – Modularity的概念](https://greatpowerlaw.wordpress.com/2013/02/24/community-detection-modularity/)

### 不同性质的图/网络

- heterogeneous network 节点和边类型不止一种。
- signed graph 边带权重。
- attributed graphs 节点、边带有属性，如图片、文本。


## 总结与感受

Network Embeddings的本质是一个将网络节点映射为低维向量的函数，一次压缩的过程。邻接矩阵的大小是节点个数的平方，如果直接作为数据无疑过大，在保证网络信息（拓扑关系）不丢失的情况，通过降维/压缩/嵌入转化成时间复杂度上可以接受的数据。

在图的性质方面，提出了无向图、有向图、有符号图、异构图和属性图等分类方法，然后从第2节到第5节大篇幅在讲述不同性质的图有什么模型/method适用。

读完后的感受的话，理解了文中介绍的无向图DeepWalk原理，而其他网络类型罗列的方法太多，暂时没有具体去看懂每个的原理。

## 范式/DeepWalk
**作用：** 在 network embeddings 和 word embeddings 架起了桥梁，语言建模和网络建模内在存在相似性。

**方法：** 将点视作单词，生成 short random walks 作为句子。然后使用 neural language model（如 Skip-gram ）来获取 network embedding 。

**受欢迎的原因** 
- 在线算法
- 可并行化
- 引入了范式
- 可扩展性好（图的复杂性，范式中第2步和第3步所能采用的策略）

Skip-gram预测临近单词，即上下文单词出现的条件概率，分为两阶段：
1. 识别出上下文的单词
2. 最大化条件概率

DeepWalk同样分为两阶段，可以观察到与Skip-gram的相似性：
1. 识别上下文节点，生成随机游走
2. 学习嵌入，最大化预测上下文的可能性
---
如上所说，DeepWalk的贡献就是给NE/NRL引入了范式，以下是**范式的步骤：**
1. 选择一个与图相关的矩阵（ random walk transition matrix / normalized Laplacian matrix / the  powers of the adjacency matrix）
2. 图采样生成节点序列（可选的步骤）
3. 从矩阵/生成序列中学习node embeddings (DeepWalk采用了Skip-gram)

**上下文节点的来源**和**嵌入学习算法**，导致了无监督表示学习的不同算法。

!> **超参**非常重要，控制图中每个节点的上下文节点的分布。

## 结论
- 在**图嵌入的应用**领域大有可为，大多数研究关注一般方法
- right context : GraphAttention
- Improved Losses / Optimization Models : 对于特定任务，与专门设计的端到端模型相比，现存的方法并不理想。**为特定任务设计损失函数和优化模型**

## 问题
?> DeepWalk的算法实现中为什么要建立一棵二叉树？在DeepWalk的算法实现后续中并没有再次用到这个变量。

https://zhuanlan.zhihu.com/p/104731569

?> 2.3Signed Graph Embeddings的数学公式(14)有错？

## 参考阅读
[1] [网络表示学习综述：一文理解Network Embedding](https://zhuanlan.zhihu.com/p/42022918) 

[2] [A Tutorial on Network Embeddings](https://www.cnblogs.com/chaoran/p/9720667.html) 

[3] [DeepWalk：算法原理，实现和应用](https://zhuanlan.zhihu.com/p/56380812) 
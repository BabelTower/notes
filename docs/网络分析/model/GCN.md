GCN-3在《SEMI-SUPERVISED CLASSIFICATION WITH GRAPH CONVOLUTIONAL NETWORKS》一文中所提出。

节点分类是一个图的半监督学习，采用的损失函数有：
$$
\mathcal{L} = \mathcal{L}_0 + \lambda\mathcal{L}_reg, \quad with\quad 
\mathcal{L}_reg=\sum_{i,j}{A_{ij}\Vert{f(X_i)-f(X_j)}\Vert^2}
=f(X)^T\Delta f(X)
$$
这里假设了连接的节点有相似的标签，但实际情况中节点的边并不一定意味着相似性。

GCN以$\mathcal{L}_0$为目标函数，训练$f(X,A)$学习节点表示。





## 相关阅读

| 序号 | 标题                                                         | 笔记                                                         |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1    | [何时能懂你的心——图卷积神经网络（GCN）](https://zhuanlan.zhihu.com/p/71200936) | CV（网格）、NLP（序列）都是属于欧式空间的数据。CNN的核心是**平移不变性+参数共享**，RNN的核心是**门控+长短时记忆**。但他们都不适用于图数据。<br />GCN是**对称归一化**的一阶邻域做**卷积（参数共享）**操作。在节点分类的例子中，通过两层GCN，激活函数分别采用ReLU和Softmax，损失函数为cross entropy，这种模型是半监督分类（只有少量node有label）。<br />从**空域的角度**看GCN公式的推导，也是从$f(H^{(l)},A)=\sigma(AH^{(l)}W^{(l)})$对所有邻域聚合的简单神经网络层，到$f(H^{(l)},A)=\sigma({\hat D^{-\frac{1}{2}}}{\hat A}{\hat D^{-\frac{1}{2}}}H^{(l)}W^{(l)})$的对称且归一化邻接矩阵，其中$\hat A=A+I$表示特征聚合的过程中加上自身，$\hat D$是$\hat A$的度矩阵。<br />GCN在实验中的表现：不经过训练、随机初始化的参数，GCN提取的特征就十分优秀（20220123：我觉得是加入了先验知识的原因）。加入少量标注信息后（半监督），GCN的效果会更加出色。<br />其他的一些注意点：对于没有特征矩阵X的网络，用单位矩阵I来替换；GCN网络的层数不宜过多，2-3层效果适宜（20220123：我觉得层数上去会导致感受野过大，节点表示的特征就不是局部的了）。<br />这篇文章十分适合新手入门，避开了繁琐的数学公式，从空域上来解读GCN，旨在对GCN树立起直观的概念。并且后面对GCN实验的点评也让人体会到了其powerful的地方。 |
| 2    | [如何理解 Graph Convolutional Network（GCN）？](https://www.zhihu.com/question/54504471/answer/332657604) |                                                              |
| 3    | [2020年，我终于决定入门GCN](https://zhuanlan.zhihu.com/p/112277874) |                                                              |
| 4    | [如何理解 Graph Convolutional Network（GCN）？](https://www.zhihu.com/question/54504471/answer/611222866) |                                                              |
| 5    | [GRAPH CONVOLUTIONAL NETWORKS](http://tkipf.github.io/graph-convolutional-networks/) |                                                              |
| 6    | [【GNN】万字长文带你入门 GCN - 阿泽的文章 - 知乎]( https://zhuanlan.zhihu.com/p/120311352) | 图上卷积的思路：借助**谱图理论（Spectral Graph Theory）**来实现在拓扑图上的卷积操作，大致步骤为将空域中的拓扑图结构通过**傅立叶变换**映射到频域中并进行卷积，然后利用逆变换返回空域，从而完成了图卷积操作。<br />希尔伯特空间（Hilbert Space）是有限维欧几里得空间的一个推广，是一个完备的内积空间，其定义了一个带有内积的完备向量空间。在希尔伯空间中，一个抽象元素通常被称为向量，它可以是一个复数或者函数。傅立叶分析的一个重要目的是将一个给定的函数表示成一族给定的基底函数的和，而希尔伯特空间为傅立叶分析提供了一种有效的表述方式。 |
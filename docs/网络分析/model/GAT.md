## GAT的DGL实现



## 相关阅读

| 序号 | 标题                                                         | 笔记                                                         |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1    | [深入理解图注意力机制](https://zhuanlan.zhihu.com/p/57180498) | 本质上，GAT 只是将GCN的**标准化常数（对称归一化后的邻接矩阵）**替换为使用**注意力权重的邻居节点特征**聚合函数。<br />$$z_i^{(l)}=W^{(l)}h_i^{(l)},  \\ e_{ij}^{(l)} = LeakyReLU(\vec{a}^{(l)^T}(z_i^{(l)}||z_j^{(l)})), \\ a_{ij}^{(l)} = \frac{exp(e_{ij}^{(l)})}{\sum_{k\in \mathcal{N}(i)}exp(e_{ik}^{(l)})}, \\ h_i^{(l+1)}=\sigma(\sum_{j\in \mathcal{N}(i)}a_{ij}^{(l)}z_j^{(l)})$$<br />GAT中的注意力机制使邻域的权重取决于节点特征，独立于拓扑结构。前三行计公式旨在计算**softmax归一化后的注意力系数**（GAT中采用的是拼接成对节点的embedding，即加性注意力），最后一行是基于注意力做邻域的aggregate。 |
| 2    | [向往的GAT（图注意力模型）](https://zhuanlan.zhihu.com/p/81350196) | 图学习利用到拓扑关系和节点属性两方面的信息，GCN局限于tranductive，且难以处理有向图。GCN和GAT都是将邻域的特征aggregate到节点上，不同的是GCN利用拉普拉斯矩阵，GAT利用attention系数。<br />GAT分为两类：Global和Mask。前者对所有节点attention，但是未利用上connectivity，且计算成本高昂；后者只对邻域attention，DGL和论文中采用的都是这种。<br />多头注意力机制(multi-head attention) 运用了类似ensemble的方法。其中拼接的公式如下：$h_i'(K)=\Vert_{k=1}^K\sigma(\sum_{j\in\mathcal{N}_i}a_{ij}^kW^kh_j)$。 |


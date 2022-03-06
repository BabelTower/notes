## 理论

### 文献

- GCN: SEMI-SUPERVISED CLASSIFICATION WITH GRAPH CONVOLUTIONAL NETWORKS
- GraphSAGE: Inductive Representation Learning on Large Graphs
- GAT: GRAPH ATTENTION NETWORKS

### GraphSAGE

目标：学习大型网络中节点的低维嵌入，之前的多数方法是transductive的，在训练时需要整张图（所有节点）。

解决方案：提出了一个通用的inductive框架，不是为每个节点分别学习嵌入，而是学习一个从局部邻域**采样和聚合**特征的**函数**。

难点：需要将新的节点**“对齐”**到训练好的节点嵌入中，inductive框架需要学习到**局部和全局**的结构属性。

贡献：将GCN推广到inductive且unsupervised，并泛化简单的卷积操作成**可训练的聚合函数**。

特点：

1. 利用了节点特征
2. 训练的是一组聚合函数（对应不同hop/depth）
3. 设计了无监督损失函数

benchmark实验：**evolving** information graphs(citation and Reddit post)、**completely unseen** graphs(multi-graph dataset of protein-protein interactions)

### GAT

贡献：解决了“基于谱域的GNN”的许多关键挑战，可以处理inductive和transductive问题。

方法：采用了self-attention策略


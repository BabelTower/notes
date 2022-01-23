## 有监督or无监督or半监督

机器学习可分为有监督学习、无监督学习、半监督学习。

- 有监督学习(supervised learning)，训练集为$\mathcal D=\{X_{train},y_{train}\}$，测试集为$X_{test}$，$y_{test}$是预测的目标。
- 无监督学习(unsupervised learning)，所有数据都是未标记的。
- 半监督学习(semi-supervised learning)，训练集为$\mathcal D=\{X_{train},y_{train},X_{unknown}\}$，测试集为$X_{test}$，$y_{test}$是预测的目标。其中$X_{test}$和$X_{unknown}$都是未标记的。

## inductive learning和transductive learning
归纳式学习(inductive learning)：从已有数据中归纳出模式来，应用于**新的数据和任务**。我们常用的机器学习模式，就是这样的：根据已有数据，学习分类器，然后应用于新的数据或任务。

直推式学习(transductive learning)：由当前学习的知识直接推广到给定的数据上。其实相当于是**给了一些测试数据**的情况下，结合已有的训练数据，看能不能推广到测试数据上。

inductive和transductive的根本区别在于：是否提前知道了测试数据，或者说，测试的数据是否出现在了训练过程中？如果不是，那么我们称它是inductive的；如果不是，我们称它是transductive。形象来说，就是**开卷考和闭卷考**的区别。

如半监督学习中，如果去预测$X_{unknown}$就是transductive，去预测$X_{test}$就是inductive。

通常transductive比inductive的效果要好，因为inductive需要从训练generalize到测试。

总结自知乎问题[如何理解 inductive learning 与 transductive learning?](https://www.zhihu.com/question/68275921)下的回答。

---

这里我们关注GNN的学习模式，GCN是transductive，而GraphSAGE是inductive的。

所谓直推式学习（GCN等），就是训练期间可以看得到没有标注的节点（训练需要整个图里面所有的节点参与），那么

1. 需要将整个图作为输入
2. 模型是基于某个具体的图的，如Cora，换个图，模型就需要从头开始训练

所谓归纳式学习，在训练期间，看不到没有标注的节点常见的有GraphSAGE等

1. 训练只需要图的局部，不需要整个图都参与训练。对于大型图，可以通过子采样的方法进行训练，不需要一次性输入整个图
2. 由于这个特性，归纳式学习是可以迁移的。即，在这个图上训练好的模型，可以迁移到另外一个图中使用。

转载自CSDN文章[图神经网络的直推式(Transductive)学习与归纳(Inductive)学习](https://blog.csdn.net/tagagi/article/details/121470121)。

---


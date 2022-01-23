



## 相关阅读

关于inductive和tranductive的区别阅读：[ML pattern](网络分析/术语/ML%20pattern.md)。

| 序号 | 标题                                                         | 笔记                                                         |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1    | [GraphSAGE：我寻思GCN也没我牛逼](https://zhuanlan.zhihu.com/p/74242097) | GCN是transductive的，对于新节点（加入后会改变旧节点的表示）要重新训练整张网络；而GraphSAGE是inductive的，节点表示不是固定的，这种学习表示方法可以用在动态网络$\mathcal G_{(t)}=(V_t,E_t)$上。<br />节点的embedding由其邻域节点的特征聚合而来，GraphSAGE就是这么一种框架，其节点的embedding随邻域的变化（包括节点特征和边连接的增减）而变化。GraphSAGE循环k轮聚合邻域特征，得到最终的节点embedding。<br />学习聚合函数的参数需要损失函数，根损失函数据任务目标设计，有监督用cross entropy，无监督应让临近节点有相似表示：$J_G(z_u)=-log(\sigma(z_u^Tz_v))-Q\cdot\Bbb E_{V_n\sim P_{n}(v)}\log{\sigma(-z_u^Tz_{v_n})}$。<br />聚合函数包括Mean、GCN、LSTM（20220123：LSTM学习包括了序列的先后顺序关系，而图学习中这种先后关系并不存在，即使用了随机打乱的技巧，这种聚合函数仍然不符合直觉/不合理，从直觉上我们应该公平的对待所有邻域，但在实验中LSTM在Reddit、PPI数据集上展现优秀的性能）、Pooling四种。<br />GraphGCN的邻域采样是一阶的，通过不断stack层数来聚合更高阶邻域的特征（和GCN类似，层数K选择为2，邻域采样次数为20）。 |


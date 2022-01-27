参考[DGL官方用户指南](https://docs.dgl.ai/guide_cn/index.html)

新建一张图，`dgl.graph`接受邻接表（有向边）的输入

```python
import dgl

u = torch.tensor([0, 0, 0, 1])
v = torch.tensor([2, 1, 3, 3])

g = dgl.graph((u, v)) 
# 如果具有最大ID的节点没有边，在创建图的时候，用户需要明确地指明节点的数量。
# g = dgl.graph((u, v), num_nodes=8)
print(g)
```

获取节点ID

```python
g.nodes()
```

获取边对应端点和边ID

```python
# 获取边的对应端点
g.edges()
# 获取边的对应端点和边ID
g.edges(form='all')
```

转化成无向图

```python
bg = dgl.to_bidirected(g)
bg.edges()
```

节点和边ID的数据类型

```python
edges = torch.tensor([2, 5, 3]), torch.tensor([3, 5, 0])  # 边：2->3, 5->5, 3->0
g64 = dgl.graph(edges)  # DGL默认使用int64
print(g64.idtype)
g32 = dgl.graph(edges, idtype=torch.int32)  # 使用int32构建图
g32.idtype
```

ID数据类型转换

```python
g64_2 = g32.long()  # 转换成int64
g64_2.idtype
g32_2 = g64.int()  # 转换成int32
g32_2.idtype
```

节点和边的特征通过`ndata`和`edata`接口访问

```python
g.ndata['x'] = torch.ones(g.num_nodes(), 3)   
g.edata['x'] = torch.ones(g.num_edges(), dtype=torch.int32)   
g
```

带权图构建示例

```python
# 边 0->1, 0->2, 0->3, 1->3
edges = torch.tensor([0, 0, 0, 1]), torch.tensor([1, 2, 3, 3])
weights = torch.tensor([0.1, 0.6, 0.9, 0.7])  # 每条边的权重
g = dgl.graph(edges)
g.edata['w'] = weights  # 将其命名为 'w'
g
```

`dgl`的`DGLgraph`可以从`networkX`和`scipy`中创建，也可从磁盘加载和保存。


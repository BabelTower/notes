## 0. 介绍

本文复现的是下面三篇文献中的模型，旨在通过理论结合实践，加深对图神经网络经典模型的理解，也给和我一样刚刚入门这方面的同学一些参考，作者本人水平也有限，如有错处，希望各位能在评论区指出。

- GCN: SEMI-SUPERVISED CLASSIFICATION WITH GRAPH CONVOLUTIONAL NETWORKS
- GraphSAGE: Inductive Representation Learning on Large Graphs
- GAT: GRAPH ATTENTION NETWORKS

本次实验是从易到难的，一步步扩展对DGL框架的了解，分别包括了以下三步：

1. 调用DGL**预定义的Module**，采用transductive的方法，即全图训练，主要工作包括三步：Cora数据集的导入、构建出GAT、GCN和GraphSAGE模型、训练和测试模型。
2. 通过DGL提供的**用户自定义**函数（UDF），在消息传递（Message Passing）范式下，手工实现GCN Layer、GAT Layer、Multi Head GAT Layer模块，并以此为基础构建模型和训练测试。
3. GraphSAGE是为了**大型网络的inductive学习**而诞生的，在实验内容的第一点中显然没有完全满足，在这里加入了分Batch训练、邻居采样、离线推断三种技术的实现。

实验使用的是DGL内部的Cora数据集，目标是完成节点分类（Node Classification）的任务。

## 1. 快速入门DGL——调包

DGL库的导入建议放在torch的下面，这样会自动确定backend为PyTorch。

```python
# ======== torch ========
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF

# ======== dgl ========
import dgl
import dgl.data
from dgl.nn import GraphConv, SAGEConv, GATConv

# ======== tool ========
import time
import math
import numpy as np
```

### 1.1 数据——Cora引文网络

关于Cora数据集的详细介绍可以查看下面这个链接。

[Papers with Code - Cora Dataset](https://paperswithcode.com/dataset/cora)

需要了解的关键是，这是一个带有节点属性的无向图，边只有存在与不存在（0/1-valued）两种状态。

```python
# ======== 数据 ========
# 用DGL自带的数据，直接导入DGLGraph类
dataset = dgl.data.CoraGraphDataset() 
g = dataset[0] 

print('type of dataset:', type(dataset)) 
print('type of g:', type(g)) 

------------------------以下为输出------------------------

  NumNodes: 2708
  NumEdges: 10556
  NumFeats: 1433
  NumClasses: 7
  NumTrainingSamples: 140
  NumValidationSamples: 500
  NumTestSamples: 1000
Done loading data from cached files.
type of dataset: <class 'dgl.data.citation_graph.CoraGraphDataset'>
type of g: <class 'dgl.heterograph.DGLHeteroGraph'>
```

`ndata`、`edata`是`graph`中节点和边的数据接口，Cora数据中只有节点属性，没有边属性，节点属性包括`feat, label, val_mask, test_mask, train_mask `5个类别。

```python
print(g)

------------------------以下为输出------------------------

Graph(num_nodes=2708, num_edges=10556,
      ndata_schemes={'feat': Scheme(shape=(1433,), dtype=torch.float32), 'label': Scheme(shape=(), dtype=torch.int64), 'val_mask': Scheme(shape=(), dtype=torch.bool), 'test_mask': Scheme(shape=(), dtype=torch.bool), 'train_mask': Scheme(shape=(), dtype=torch.bool)}
      edata_schemes={})
```

### 1.2 模型——使用DGL中预定义的Module

```python
# ======== 采用DGL预定义的Module ========
# 分别实现GCN、GAT、GraphSAGE模型，参考论文的实验设置

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, out_feats):
        # 调用父类初始化函数
        super(GCN, self).__init__() 
        
        # 定义子模块
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, out_feats)
        
    def forward(self, g, in_feat):
        # 调用子模块
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

class GAT(nn.Module):
    def __init__(self, in_feats, out_feats):
        # 调用父类初始化函数
        super(GAT, self).__init__() 
        
        # 定义子模块
        # The first layer consists of K = 8 attention heads computing F ′ = 8 features each (for a total of 64 features)
        self.conv1 = GATConv(in_feats, 8, num_heads=8)
        # The second layer is used for classification: a single attention head that computes C features
        self.conv2 = GATConv(8*8, out_feats, num_heads=1)
        
    def forward(self, g, in_feat):
        # 调用子模块
        h = self.conv1(g, in_feat)
        h = torch.flatten(h, 1)
        h = F.elu(h) # followed by an exponential linear unit (ELU)
        h = self.conv2(g, h)
        h = torch.mean(h, 1)
        # h = F.log_softmax(h, 1)
        return h
    
    
class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats, out_feats, aggregator_type):
        # 调用父类初始化函数
        super(GraphSAGE, self).__init__() 
        
        # 定义子模块
        self.conv1 = SAGEConv(in_feats, h_feats, aggregator_type)
        self.conv2 = SAGEConv(h_feats, out_feats, aggregator_type)
        
    def forward(self, g, in_feat):
        # 调用子模块
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h
```

### 1.3 训练和测试模型

```python
def train(g, model, epochs, lr, weight_decay=0):
    # optimizer, loss_fn = F.cross_entropy
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay=weight_decay)
    
    # graph data    
    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    
    # best metric score
    best_val_acc = 0
    best_test_acc = 0
    
    dur = []
    
    # train loop
    for e in range(epochs):
        
        if e >= 3:
            t0 = time.time()
        
        # Forward
        logits = model(g, features)
        pred = logits.argmax(axis=1)
        
        # loss 这里input参数用的是logits，而不是pred，具体查询cross_entropy的参数规定
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Compute accuracy
        train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
        val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        test_acc = (pred[test_mask] == labels[test_mask]).float().mean()
        
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            
        if e >= 3: 
            dur.append(time.time() - t0)
            
        if (e + 1) % 10 == 0:
            print('Epoch {:2d} | Loss {:.3f} | Train Acc {:.3f} | '.format(e, loss, train_acc), 'Val Acc {:.3f} (best: {:.3f}) | '.format(val_acc, best_val_acc), 'Test Acc {:.3f} (best: {:.3f}) | Time (s) {:.4f}'.format(test_acc, best_test_acc, np.mean(dur)))
            
    print('Total Time (s): {:.4f}, Best Val Acc: {:.4f}, Best Test Acc: {:.4f}'.format(dur[-1], best_val_acc, best_test_acc))
```

GCN

```python
# Ceate the GCN model
model = GCN(g.ndata['feat'].shape[1], 64, dataset.num_classes)
train(g, model, epochs=100, lr=5e-3, weight_decay=5e-4)

------------------------以下为输出------------------------

Epoch  9 | Loss 1.850 | Train Acc 0.936 |  Val Acc 0.628 (best: 0.668) |  Test Acc 0.629 (best: 0.649) | Time (s) 0.0260
Epoch 19 | Loss 1.663 | Train Acc 0.950 |  Val Acc 0.688 (best: 0.688) |  Test Acc 0.674 (best: 0.674) | Time (s) 0.0260
Epoch 29 | Loss 1.412 | Train Acc 0.957 |  Val Acc 0.746 (best: 0.746) |  Test Acc 0.756 (best: 0.756) | Time (s) 0.0265
Epoch 39 | Loss 1.135 | Train Acc 0.971 |  Val Acc 0.764 (best: 0.764) |  Test Acc 0.779 (best: 0.779) | Time (s) 0.0264
Epoch 49 | Loss 0.885 | Train Acc 0.971 |  Val Acc 0.782 (best: 0.784) |  Test Acc 0.800 (best: 0.796) | Time (s) 0.0269
Epoch 59 | Loss 0.695 | Train Acc 0.971 |  Val Acc 0.784 (best: 0.786) |  Test Acc 0.803 (best: 0.805) | Time (s) 0.0271
Epoch 69 | Loss 0.565 | Train Acc 0.986 |  Val Acc 0.780 (best: 0.786) |  Test Acc 0.811 (best: 0.805) | Time (s) 0.0270
Epoch 79 | Loss 0.478 | Train Acc 0.993 |  Val Acc 0.772 (best: 0.786) |  Test Acc 0.811 (best: 0.805) | Time (s) 0.0269
Epoch 89 | Loss 0.417 | Train Acc 0.993 |  Val Acc 0.770 (best: 0.786) |  Test Acc 0.811 (best: 0.805) | Time (s) 0.0270
Epoch 99 | Loss 0.372 | Train Acc 0.993 |  Val Acc 0.776 (best: 0.786) |  Test Acc 0.808 (best: 0.805) | Time (s) 0.0269
Total Time (s): 0.0266, Best Val Acc: 0.7860, Best Test Acc: 0.8050
```

GAT

```python
# Ceate the model
model = GAT(g.ndata['feat'].shape[1], dataset.num_classes)
train(g, model, epochs=100, lr=5e-3, weight_decay=5e-4)

------------------------以下为输出------------------------

Epoch  9 | Loss 1.745 | Train Acc 0.979 |  Val Acc 0.778 (best: 0.780) |  Test Acc 0.780 (best: 0.779) | Time (s) 0.0363
Epoch 19 | Loss 1.470 | Train Acc 0.964 |  Val Acc 0.778 (best: 0.780) |  Test Acc 0.775 (best: 0.779) | Time (s) 0.0390
Epoch 29 | Loss 1.161 | Train Acc 0.964 |  Val Acc 0.782 (best: 0.782) |  Test Acc 0.788 (best: 0.783) | Time (s) 0.0405
Epoch 39 | Loss 0.868 | Train Acc 0.971 |  Val Acc 0.778 (best: 0.784) |  Test Acc 0.791 (best: 0.788) | Time (s) 0.0402
Epoch 49 | Loss 0.639 | Train Acc 0.986 |  Val Acc 0.774 (best: 0.784) |  Test Acc 0.791 (best: 0.788) | Time (s) 0.0393
Epoch 59 | Loss 0.486 | Train Acc 0.993 |  Val Acc 0.774 (best: 0.784) |  Test Acc 0.798 (best: 0.788) | Time (s) 0.0379
Epoch 69 | Loss 0.390 | Train Acc 0.993 |  Val Acc 0.790 (best: 0.790) |  Test Acc 0.800 (best: 0.800) | Time (s) 0.0370
Epoch 79 | Loss 0.323 | Train Acc 0.993 |  Val Acc 0.780 (best: 0.790) |  Test Acc 0.792 (best: 0.800) | Time (s) 0.0364
Epoch 89 | Loss 0.273 | Train Acc 1.000 |  Val Acc 0.778 (best: 0.790) |  Test Acc 0.788 (best: 0.800) | Time (s) 0.0358
Epoch 99 | Loss 0.232 | Train Acc 1.000 |  Val Acc 0.766 (best: 0.790) |  Test Acc 0.784 (best: 0.800) | Time (s) 0.0357
Total Time (s): 0.0348, Best Val Acc: 0.7900, Best Test Acc: 0.8000
```

GraphSAGE

```python
# Ceate the model
model = GraphSAGE(g.ndata['feat'].shape[1], 64, dataset.num_classes, aggregator_type='gcn')
train(g, model, epochs=100, lr=5e-3, weight_decay=5e-4)

------------------------以下为输出------------------------

Epoch  9 | Loss 1.798 | Train Acc 0.950 |  Val Acc 0.738 (best: 0.738) |  Test Acc 0.736 (best: 0.736) | Time (s) 0.0290
Epoch 19 | Loss 1.551 | Train Acc 0.957 |  Val Acc 0.778 (best: 0.780) |  Test Acc 0.778 (best: 0.778) | Time (s) 0.0259
Epoch 29 | Loss 1.260 | Train Acc 0.957 |  Val Acc 0.784 (best: 0.786) |  Test Acc 0.794 (best: 0.786) | Time (s) 0.0247
Epoch 39 | Loss 0.975 | Train Acc 0.971 |  Val Acc 0.794 (best: 0.794) |  Test Acc 0.802 (best: 0.802) | Time (s) 0.0248
Epoch 49 | Loss 0.742 | Train Acc 0.979 |  Val Acc 0.798 (best: 0.798) |  Test Acc 0.807 (best: 0.808) | Time (s) 0.0246
Epoch 59 | Loss 0.577 | Train Acc 0.986 |  Val Acc 0.798 (best: 0.798) |  Test Acc 0.814 (best: 0.808) | Time (s) 0.0246
Epoch 69 | Loss 0.469 | Train Acc 0.986 |  Val Acc 0.798 (best: 0.802) |  Test Acc 0.822 (best: 0.817) | Time (s) 0.0244
Epoch 79 | Loss 0.399 | Train Acc 0.986 |  Val Acc 0.800 (best: 0.802) |  Test Acc 0.819 (best: 0.817) | Time (s) 0.0244
Epoch 89 | Loss 0.350 | Train Acc 0.993 |  Val Acc 0.794 (best: 0.802) |  Test Acc 0.817 (best: 0.817) | Time (s) 0.0241
Epoch 99 | Loss 0.314 | Train Acc 0.993 |  Val Acc 0.794 (best: 0.802) |  Test Acc 0.821 (best: 0.817) | Time (s) 0.0240
Total Time (s): 0.0249, Best Val Acc: 0.8020, Best Test Acc: 0.8170
```

## 2 深入DGL——消息传递范式、用户自定义函数

### 2.1 GCN Layer

```python
# ======= 实现GCN =======
# Refer: https://zhuanlan.zhihu.com/p/139359188
# 定义消息函数(message)、聚合函数(reduce)和更新函数(apply)的用户自定义函数(UDF)

def gcn_message(edges):
    """
    消息函数(message)：从源节点向目标节点传递信息，存储在目标节点的信箱(mailbox)中。

    Args:
        edges (EdgeBatch): 表示一批边，包括src、dst、data三个成员属性。

    Return:
        _ (Dictionary): 传递出的消息，键值表示字段名。
    """
    
    msg = edges.src['h'] * edges.src['norm'] # 按位置相乘
    return {'m': msg}
    
def gcn_reduce(nodes):
    """
    聚合函数(reduce)：处理节点信箱(mailbox)中收到的消息。

    Args:
        nodes (NodeBatch): 表示一批节点，包括data、mailbox两个成员属性。
        
    Return:
        _ (Dictionary): 处理后的消息，键值表示字段名。
    """
    
    sum = torch.sum(nodes.mailbox['m'], dim=1) # dim=0是不同的节点，dim=1是每个节点收到的消息
    h = sum * nodes.data['norm']
    return {'h': h}
    
    
# 定义 GCN Layer    
class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, dropout, bias=True):
        super(GCNLayer, self).__init__()
        
        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.bias = None
        self.reset_parameter()
        
        if dropout is not None:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None
        
    def reset_parameter(self):
        stdv = 1. / math.sqrt(self.weight.size(1))   
        self.weight.data.uniform_(-stdv, stdv) 
        if self.bias is not None:
            stdv = 1. / math.sqrt(self.bias.size(0))
            self.bias.data.uniform_(-stdv, stdv)
        
    def forward(self, g, in_feat):
        with g.local_scope():
            if self.dropout is not None:
                h = self.dropout(in_feat)
            else:
                h = in_feat
            g.ndata['h'] = torch.mm(h, self.weight)
            g.update_all(gcn_message, gcn_reduce)
            h = g.ndata['h']
            if self.bias is not None:
                h = h + self.bias
            return h
        
        
class GCN_by_hand(nn.Module):
    def __init__(self, in_feats, h_feats, out_feats, dropout=None, bias=True):
        super(GCN_by_hand, self).__init__()
        
        # 定义子模块
        self.conv1 = GCNLayer(in_feats, h_feats, dropout=dropout, bias=bias)
        self.conv2 = GCNLayer(h_feats, out_feats, dropout=dropout, bias=bias)
        
    def forward(self, g, in_feat):
        # 调用子模块
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h
```

在开始训练之前需要提前计算节点归一化的度数。

```python
with g.local_scope():
    # 归一化入度
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    g.ndata['norm'] = norm.unsqueeze(1)

    # Ceate the model
    model = GCN_by_hand(g.ndata['feat'].shape[1], 64, dataset.num_classes, bias=True)
    train(g, model, epochs=100, lr=5e-3, weight_decay=5e-4)
    
------------------------以下为输出------------------------   

Epoch  9 | Loss 1.854 | Train Acc 0.421 |  Val Acc 0.254 (best: 0.254) |  Test Acc 0.239 (best: 0.239) | Time (s) 0.0774
Epoch 19 | Loss 1.714 | Train Acc 0.743 |  Val Acc 0.498 (best: 0.498) |  Test Acc 0.514 (best: 0.514) | Time (s) 0.0754
Epoch 29 | Loss 1.519 | Train Acc 0.900 |  Val Acc 0.682 (best: 0.682) |  Test Acc 0.699 (best: 0.699) | Time (s) 0.0732
Epoch 39 | Loss 1.281 | Train Acc 0.957 |  Val Acc 0.744 (best: 0.748) |  Test Acc 0.771 (best: 0.767) | Time (s) 0.0755
Epoch 49 | Loss 1.033 | Train Acc 0.971 |  Val Acc 0.770 (best: 0.770) |  Test Acc 0.799 (best: 0.799) | Time (s) 0.0753
Epoch 59 | Loss 0.814 | Train Acc 0.979 |  Val Acc 0.778 (best: 0.778) |  Test Acc 0.810 (best: 0.808) | Time (s) 0.0748
Epoch 69 | Loss 0.646 | Train Acc 0.986 |  Val Acc 0.776 (best: 0.778) |  Test Acc 0.812 (best: 0.808) | Time (s) 0.0785
Epoch 79 | Loss 0.530 | Train Acc 0.993 |  Val Acc 0.772 (best: 0.782) |  Test Acc 0.815 (best: 0.813) | Time (s) 0.0807
Epoch 89 | Loss 0.450 | Train Acc 0.993 |  Val Acc 0.774 (best: 0.782) |  Test Acc 0.816 (best: 0.813) | Time (s) 0.0787
Epoch 99 | Loss 0.394 | Train Acc 0.993 |  Val Acc 0.774 (best: 0.782) |  Test Acc 0.814 (best: 0.813) | Time (s) 0.0775
Total Time (s): 0.0669, Best Val Acc: 0.7820, Best Test Acc: 0.8130
```

### 2.2 GAT Layer、Multi Head GAT Layer

```python
# ====== 实现GAT ======
# Refer: https://docs.dgl.ai/tutorials/models/1_gnn/9_gat.html

class GATLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GATLayer, self).__init__()
        
        # 定义子模块
        self.fc = nn.Linear(in_feats, out_feats, bias=False)
        self.atten_fc = nn.Linear(2 * out_feats, 1, bias=False)
        self.reset_parameters()
    
    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.atten_fc.weight, gain=gain)
        
    def gat_edge_apply(self, edges):
        z2 = torch.concat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.atten_fc(z2)
        return {'e': F.leaky_relu(a)}
    
    def gat_message(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}
    
    def gat_reduce(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}
        
    def forward(self, g, in_feat):
        with g.local_scope():
            z = self.fc(in_feat)
            g.ndata['z'] = z
            g.apply_edges(self.gat_edge_apply)
            g.update_all(self.gat_message, self.gat_reduce)
            return g.ndata['h']
            
            
class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_feats, out_feats, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(in_feats, out_feats))
        self.merge = merge
        
    def forward(self, g, in_feat):
        head_outs = [head(g, in_feat) for head in self.heads]
        
        if self.merge == 'cat':
            return torch.cat(head_outs, dim=1)
        else:
            return torch.mean(torch.stack(head_outs), dim=0)
        
        
class GAT_by_hand(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GAT_by_hand, self).__init__()
        
        # 定义子模块
        self.conv1 = MultiHeadGATLayer(in_feats, 8, num_heads=8)
        self.conv2 = MultiHeadGATLayer(8*8, out_feats, num_heads=1, merge='avg')
        
    def forward(self, g, in_feat):
        # 调用子模块
        h = self.conv1(g, in_feat)
        h = F.elu(h) 
        h = self.conv2(g, h)
        # h = torch.mean(h, 1)
        # h = F.log_softmax(h, 1)
        return h
```



```python
# Ceate the GAT model
model = GAT_by_hand(g.ndata['feat'].shape[1], dataset.num_classes)
train(g, model, epochs=100, lr=5e-3, weight_decay=5e-4)

------------------------以下为输出------------------------

Epoch  9 | Loss 1.738 | Train Acc 0.957 |  Val Acc 0.750 (best: 0.750) |  Test Acc 0.746 (best: 0.746) | Time (s) 0.3104
Epoch 19 | Loss 1.465 | Train Acc 0.964 |  Val Acc 0.766 (best: 0.766) |  Test Acc 0.778 (best: 0.772) | Time (s) 0.3040
Epoch 29 | Loss 1.167 | Train Acc 0.964 |  Val Acc 0.776 (best: 0.776) |  Test Acc 0.784 (best: 0.783) | Time (s) 0.2976
Epoch 39 | Loss 0.887 | Train Acc 0.971 |  Val Acc 0.782 (best: 0.782) |  Test Acc 0.790 (best: 0.789) | Time (s) 0.2998
Epoch 49 | Loss 0.660 | Train Acc 0.979 |  Val Acc 0.784 (best: 0.784) |  Test Acc 0.795 (best: 0.792) | Time (s) 0.3066
Epoch 59 | Loss 0.498 | Train Acc 0.993 |  Val Acc 0.782 (best: 0.786) |  Test Acc 0.787 (best: 0.795) | Time (s) 0.3028
Epoch 69 | Loss 0.391 | Train Acc 0.993 |  Val Acc 0.772 (best: 0.786) |  Test Acc 0.781 (best: 0.795) | Time (s) 0.2998
Epoch 79 | Loss 0.326 | Train Acc 0.993 |  Val Acc 0.768 (best: 0.786) |  Test Acc 0.770 (best: 0.795) | Time (s) 0.2980
Epoch 89 | Loss 0.283 | Train Acc 1.000 |  Val Acc 0.764 (best: 0.786) |  Test Acc 0.768 (best: 0.795) | Time (s) 0.2967
Epoch 99 | Loss 0.253 | Train Acc 1.000 |  Val Acc 0.762 (best: 0.786) |  Test Acc 0.761 (best: 0.795) | Time (s) 0.2953
Total Time (s): 0.2827, Best Val Acc: 0.7860, Best Test Acc: 0.7950
```

## 3 进阶DGL——小批次训练、邻居采样、离线推断

模型构建

```python
# ====== GraphSAGE实现 ======
# Refer: 
#   1. https://github.com/dmlc/dgl/blob/master/examples/pytorch/graphsage/node_classification.py 
#   2. https://docs.dgl.ai/guide_cn/minibatch-node.html#guide-cn-minibatch-node-classification-sampler

class GraphSAGE_batch(nn.Module):
    def __init__(self, in_feats, h_feats, out_feats):
        super(GraphSAGE_batch, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, 'gcn')
        self.conv2 = SAGEConv(h_feats, out_feats, 'gcn')
        self.dropout = nn.Dropout(p=0.5)
        self.h_feats = h_feats
        self.out_feats = out_feats
        self.n_layers = 2

    def forward(self, blocks, in_feat):
        h = in_feat
        h = self.conv1(blocks[0], h)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.conv2(blocks[1], h)
        return h
    
    def inference(self, g, device, batch_size):
        # 用该模块进行离线推断，在GPU显存有限的情况下通过小批次处理和邻居采样实现全图前向传播的方法
        # Refer: https://docs.dgl.ai/guide_cn/minibatch-inference.html
        g.ndata['h'] = g.ndata['feat']
        
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        dataloader = dgl.dataloading.NodeDataLoader(
            g, torch.arange(g.num_nodes()).to(g.device), sampler, device=device,
            batch_size=batch_size, shuffle=False, drop_last=False
        )
        
        # 推断算法将包含一个外循环以迭代执行各层，和一个内循环以迭代处理各个节点小批次。
        for l, layer in enumerate([self.conv1, self.conv2]):
            y = torch.zeros(g.num_nodes(), self.h_feats if l != self.n_layers - 1 else self.out_feats)
            
            for input_nodes, output_nodes, blocks in dataloader:
                # 计算输出
                block = blocks[0]
                x = block.srcdata['h']
                h = layer(block, x)
                if l != self.n_layers - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                y[output_nodes] = h
            g.ndata['h'] = y
            
        return y
```

训练循环

```python 
def train_batch(g, model, epochs, lr, batch_size, weight_decay=0, device='cpu'):
    # optimizer, loss_fn = F.cross_entropy
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay=weight_decay)
    
    # graph data    
    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    
    train_nids = g.nodes()[train_mask]
    val_nids = g.nodes()[val_mask]
    test_nids = g.nodes()[test_mask]
    
    # 定义邻居采样器和dataloader
    sampler = dgl.dataloading.MultiLayerNeighborSampler([25, 10])
    train_dataloader = dgl.dataloading.NodeDataLoader(
        g, train_nids, sampler, device=device, batch_size=batch_size, 
        shuffle=False, drop_last=False
    )
    val_dataloader = dgl.dataloading.NodeDataLoader(
        g, val_nids, sampler, device=device, batch_size=batch_size, 
        shuffle=False, drop_last=False
    )
    
    # best metric score
    best_val_acc = 0
    best_test_acc = 0
    
    dur = []
    
    # train loop
    for e in range(epochs):
        model.train()
        
        if e >= 3:
            t0 = time.time()
        
        for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
            x = blocks[0].srcdata['feat']
            y = blocks[-1].dstdata['label']
            
            # Forward
            y_hat = model(blocks, x)
            
            # Loss
            loss = F.cross_entropy(y_hat, y)
        
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Output
            if it % 10 == 0:
                train_acc = MF.accuracy(y_hat, y) # Compute accuracy
                print('Loss {:.4f} | Train Acc {:.4f}'.format(loss, train_acc))
        
        if e >= 3: 
            dur.append(time.time() - t0)
        
        model.eval()
        ys = []
        y_hats = []
        for it, (input_nodes, output_nodes, blocks) in enumerate(val_dataloader):
            with torch.no_grad():
                x = blocks[0].srcdata['feat']
                ys.append(blocks[-1].dstdata['label'])
                y_hats.append(model(blocks, x))
        
        val_acc = MF.accuracy(torch.cat(y_hats), torch.cat(ys))
        
        print('Epoch {:3d} | Val acc: {:.4f} '.format(e, val_acc.item()), '| Times (s): {:.4f}'.format(np.mean(dur)))
        
    # Test accuracy and offline inference of all nodes
    model.eval()
    with torch.no_grad():
        pred = model.inference(g, device=device, batch_size=batch_size)
        acc = MF.accuracy(pred[test_mask], labels[test_mask])
        print('Test acc {:.4f}'.format(acc.item()))
```



```python
devive = 'cpu'

model = GraphSAGE_batch(g.ndata['feat'].shape[1], 256, dataset.num_classes).to(devive)
train_batch(g, model, 100, lr=5e-3, weight_decay=5e-4, batch_size=1024)

------------------------以下为输出------------------------

Loss 1.9460 | Train Acc 0.1429
Epoch   0 | Val acc: 0.4160  | Times (s): nan
Loss 1.9235 | Train Acc 0.6714
Epoch   1 | Val acc: 0.6300  | Times (s): nan
Loss 1.9022 | Train Acc 0.8571
Epoch   2 | Val acc: 0.7480  | Times (s): nan
Loss 1.8823 | Train Acc 0.8786
Epoch   3 | Val acc: 0.7540  | Times (s): 0.0496
Loss 1.8546 | Train Acc 0.9429
Epoch   4 | Val acc: 0.7640  | Times (s): 0.0478
Loss 1.8304 | Train Acc 0.9571
Epoch   5 | Val acc: 0.7780  | Times (s): 0.0474
Loss 1.7946 | Train Acc 0.9714
Epoch   6 | Val acc: 0.7780  | Times (s): 0.0461
Loss 1.7661 | Train Acc 0.9714
Epoch   7 | Val acc: 0.7860  | Times (s): 0.0451
Loss 1.7247 | Train Acc 0.9429
Epoch   8 | Val acc: 0.7840  | Times (s): 0.0447
Loss 1.6930 | Train Acc 0.9357
Epoch   9 | Val acc: 0.7840  | Times (s): 0.0447
Loss 1.6564 | Train Acc 0.9643
Epoch  10 | Val acc: 0.7900  | Times (s): 0.0446
Loss 1.6121 | Train Acc 0.9714
Epoch  11 | Val acc: 0.7880  | Times (s): 0.0447
Loss 1.5651 | Train Acc 0.9714
Loss 0.2051 | Train Acc 1.0000
Epoch  98 | Val acc: 0.7960  | Times (s): 0.0439
Loss 0.1932 | Train Acc 1.0000
Epoch  99 | Val acc: 0.7940  | Times (s): 0.0439
Test acc 0.8140
```




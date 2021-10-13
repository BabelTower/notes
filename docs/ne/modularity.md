# Modularity 模块化

modularity用于衡量网络切分好坏的指标。

$$ Q = \sum_i(e_{ii} - a_i^2) = Tre - \parallel e^2 \parallel $$

当modularity这个度量被认可后，后续很多算法的思路就是如何找到一个partitioning的方法，使得modularity最大。将community detection转化成了最优化的问题。

**缺陷：** 在large network中，基于modularity的方法找不到那些small community，即便这些small community的结构都很明显。

## refer
[Community Detection – Modularity的概念](https://greatpowerlaw.wordpress.com/2013/02/24/community-detection-modularity/)
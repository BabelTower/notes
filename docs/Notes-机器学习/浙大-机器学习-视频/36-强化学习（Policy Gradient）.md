---
Title: 36-强化学习（Policy Gradient）
Date: 2021-12-09
---

## keywords
#强化学习 
## 笔记

原始的想法：过程简单，$r_t$的调整很难。

改进：$r_t$减去一个估值函数$V(s)$，用神经网络求$V(s)$。（AlphaGo当前棋局的胜率）

Actor-Critic算法：

![](assets/Pasted%20image%2020211209133339.png)


## 总结
---
Title: 35-强化学习Q-learnig
Date: 2021-12-08
---

## keywords
#强化学习 
## 笔记

### 监督学习和强化学习的区别
1. 监督学习数据和标签一一对应，强化学习只有奖励函数。
2. 训练数据通过Action给出
3. Action会影响Reward
4. 训练的目的：“状态->行为”

![](assets/Pasted%20image%2020211208223333.png)

### 定义

$R_t$：t时刻的奖励函数值

$S_t$：t时刻的状态

$A_t$：t时刻的行为

马尔可夫假设：
1. $t+1$时刻的状态只和$t$时刻有关
2. 下一个时刻的状态只与当前的状态和行为有关
3. 下一个时刻的奖励只与当前的状态和行为有关

MDP马尔可夫过程：
![](assets/Pasted%20image%2020211208230136.png)

目标函数：奖励函数的加权和
![](assets/Pasted%20image%2020211208235417.png)

学习的函数（状态到行为的映射）：$\pi(S_t,a_t)=p(a_t|S_t)$

估值函数衡量（状态），Q函数衡量（状态+行为）的组合。

所以，估值函数 = 所有行为的 **Q函数 \*  行为的概率** 之和。

$$
\begin{aligned}

V^\pi(S) & =  E_n\left[\sum_{t=0}^{+\infty}\gamma^tr_t|S_0=s,\pi\right] \\
& =  E_n\left[r_0 +\gamma\sum_{t=0}^{+\infty}\gamma^tr_{t+1}|S_0=s,\pi\right] \\
& = \sum_{a\in A} \pi(s,a) \sum_{s^{\prime}\in S}P_{ss^{\prime}}^a(R_s^a+\gamma V^\pi(s^\prime))

\end{aligned}

$$

其中$Q^\pi(s,a) =\sum_{s^{\prime}\in S}P_{ss^{\prime}}^a(R_s^a+\gamma V^\pi(s^\prime))$，$\pi(s,a)=p(a|s)$。

策略和估值函数都是未知的，训练的方法是先初始化一个策略，算出估值函数，在进行调整策略。（最佳策略的迭代算法 -> 收敛的）

![](assets/Pasted%20image%2020211209004522.png)
![](assets/Pasted%20image%2020211209004456.png)

缺点：状态数、行为数很多时不可行！

深度学习解决这一缺点，DQN所做的事：用深度神经网络替换Q函数。

![](assets/Pasted%20image%2020211209011307.png)

加入随机性， $\epsilon-greedy$ 算法，避免陷入一种模式。

![](assets/Pasted%20image%2020211209011656.png)


## 总结
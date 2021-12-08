---
Title: Lex08_AX=b：可解性及解的结构
Date: 2021-12-08
---

## keywords

## 笔记

对系数矩阵作消元（行向量的线性组合）对等式AX=b右侧向量做相同的行变换，等式两侧保持一致。

构造增广矩阵Augmented matrix = [A b] 

可解性：b在A的行空间中，即b能由A的行向量线性组合而成。等价的（另一种描述），如果A的各行通过线性组合能得到“0行”，右侧b向量同样的组合也要得到0。

### 求解AX=b的所有解
1. 先找一个特解，将自由变元设为0，求解所有主元，得到$x_{particular}$。
2. 求解零空间$x_{nullspace}$
3. 所有解$x_{complete}=x_{nullspace}+x_{particular}$

所有解构成的空间不是向量空间（不通过原点）

$m \times n \text{ matrix A of rank r}$  满足 $r \le m,n$


当$rank=n$时，0个解或一个解，$x=x_p$，$x_n=0$

当$rank=m$时，对于任意b都有解。

对于方阵$(r=m=n)$时，$R=I$。

## 总结

矩阵的rank决定了解的个数：

![](assets/Pasted%20image%2020211208205638.png)
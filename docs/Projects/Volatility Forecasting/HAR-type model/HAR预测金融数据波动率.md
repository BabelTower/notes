## keywords
#时间序列 #线性回归 #风控 #项目

## 项目描述

使用**HAR-type volatility models**来预测金融数据的波动率，并测试评估其预测性能。

HAR-type volatility models指的是 [^1] 中汇总的一系列线性模型。

采用金融数据（即预测对象）可以粗略地分为两类，一种是个股，另一种“综合性”标的物，如ETF基金、板块指数等。

-   ETF基金：交易所交易型基金（Exchange Traded Funds），在场内像股票一样交易的基金。
-   指数：多只股票的加权平均，是一个股票组合。
-   板块：一些具有相同特征的股票的集合。

## 逻辑架构

1.  从原始数据集中提取出我们关心的特征，主要是参考[^1] 中的5种 **Volatility components** 。
2.  定义不同的HAR-type volatility model，生成相应的训练/测试数据集，定义不同的回归学习算法。
3.  使用 [^1] 中的预测评估方法 **rolling window prediction method** ，计算16种HAR-type volatility model及3种回归学习算法的**R2决定系数**，来评估预测表现。

大致可以看作，第1步是数据集处理和特征工程，第2步是算法模型，第3步是目标函数和评估指标。文档中不包括数据获取部分。

## 数据集处理

## 特征工程

## 算法模型

## 算法模型

### 16种HAR-type volatility models（代码已实现8种）

> ==问题== 关于波动率 $RV_t^d$计算是否要开方？这一点还有待确认。 [^5] 中给出有别与 [^1]的模型。
>
> 现在版本的代码中，默认为开方的形式，加上
> $$
> RV_t^d = \sqrt{RV_t^d}
> $$
> 

1. HAR-RV model

$$
RV^d_{t+1}=c+\alpha_1RV_t^d+\alpha_2RV_t^w+\alpha_3RV_t^m+\epsilon_{t+1}
$$

2. HAR-RV-J model

$$
RV^d_{t+1}=c+\alpha_1RV_t^d+\alpha_2RV_t^w+\alpha_3RV_t^m+\beta_1J^d_t+\epsilon_{t+1}
$$

3. HAR-CJ model

$$
RV^d_{t+1}=c+\alpha_1RV_t^d+\alpha_2RV_t^w+\alpha_3RV_t^m
+\beta_1J^d_t+\beta_2J^w_t+\beta_3J^m_t+\epsilon_{t+1}
$$

4. HAR-RSV model

$$
RV^d_{t+1}=c+\alpha_1RSV_t^{d+}+\alpha_2RSV_t^{w+}+\alpha_3RSV_t^{m+}
+\beta_1RSV_t^{d-}+\beta_2RSV_t^{w-}+\beta_3RSV_t^{m-}+\epsilon_{t+1}
$$

5. HAR-RSV-J model

$$
RV^d_{t+1}=c+\alpha_1RSV_t^{d+}+\alpha_2RSV_t^{w+}+\alpha_3RSV_t^{m+}\\
+\beta_1RSV_t^{d-}+\beta_2RSV_t^{w-}+\beta_3RSV_t^{m-}
+\phi_1J^d_t+\epsilon_{t+1}
$$

6. HAR-RV-SJ model

$$
RV^d_{t+1}=c+\alpha_1RV_t^d+\alpha_2RV_t^w+\alpha_3RV_t^m+\beta_1SJ^d_t+\epsilon_{t+1}
$$

7. HAR-RV-SSJ(1) model

$$
RV^d_{t+1}=c+\alpha_1RV_t^d+\alpha_2RV_t^w+\alpha_3RV_t^m
+\beta_1SSJ^{d+}_t+\phi_1SSJ^{d-}_t+\epsilon_{t+1}
$$

8. HAR-RV-SSJ(2) model

$$
RV^d_{t+1}=c+\alpha_1RV_t^d+\alpha_2RV_t^w+\alpha_3RV_t^m
+\beta_1SSJ^{d+}_t+\beta_2SSJ^{w+}_t+\beta_3SSJ^{m+}_t\\
+\phi_1SSJ^{d-}_t+\phi_2SSJ^{w-}_t+\phi_3SSJ^{m-}_t+\epsilon_{t+1}
$$

### 库函数scikit-learn中的3种回归算法

线性模型的学习算法使用的是[scikit-learn.linear_model](https://scikit-learn.org/stable/modules/linear_model.html)中的三种，分别是线性回归、Lasso回归、岭回归。

1. 线性回归目标函数

$$
J(β)=∑(y−Xβ)^2
$$
2. Lasso回归目标函数，惩罚项为L1范数。其中 $E S S ( β )$ 表示误差平方和，$\lambda l_1(\beta)$表示惩罚项

$$
J(β)=∑(y−Xβ)^2+λ\norm{β}_1\\
=∑(y−Xβ)^2+∑λ\abs{β}\\
=ESS(\beta)+\lambda l_1(\beta)
$$
3. 岭回归目标函数，惩罚项为L2范数

$$
J(β)=∑(y−Xβ)^2+λ\norm{β}_2^2\\
=∑(y−Xβ)^2+∑λβ^2
$$

## 预测评估



[^1]: Forecasting the volatility of crude oil futures using HAR-type models with structural breaks
[^2]: 2012 \_HAR modeling for RV forecasting Chap15 Handbook
[^3]: A reduced form framework for modeling volatility of speculative prices based on realized variation measures
[^4]: Forecasting Return Volatility of the CSI 300 Index Using the Stochastic Volatility Model with Continuous Volatility and Jumps
[^5]: The volatility of realized volatility
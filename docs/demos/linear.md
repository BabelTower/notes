# 线性回归

```python
import torch
import matplotlib.pyplot as plt
torch.manual_seed(10)

# 学习率
lr = 0.05

# 创建训练数据
X = torch.rand(20, 1) * 10 # rand 平均分布的随机数
y = 2*X + 5 + torch.randn(20, 1) # randn 正态分布的随机数

# 构建线性回归参数
w = torch.rand((1), requires_grad=True)
b = torch.rand((1), requires_grad=True)

# 迭代训练 1000 次
epochs = 1000
for iteration in range(epochs):

    # forward
    y_pred = torch.add(torch.mul(w, X), b)

    # loss
    loss = (1/2. * (y - y_pred) ** 2).mean()

    # backward
    loss.backward() # 算出梯度

    # update parameters
    b.data.sub_(lr * b.grad)
    w.data.sub_(lr * w.grad)

    # zero grad
    w.grad.zero_()
    b.grad.zero_()

    if(iteration % 20 == 0):
        print('第{}次训练，MSE={}。'.format(iteration+1, loss))
        # TODO: pyplot绘图
```
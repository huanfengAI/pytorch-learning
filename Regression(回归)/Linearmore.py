import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
#先随便初始化参数，用于生成训练集样本
w_target = np.array([0.6, 3.8, 1.4]) # 定义参数
b_target = np.array([0.8]) # 定义参数
x_sample = np.arange(-3, 3.1, 0.1)
y_sample = b_target[0] + w_target[0] * x_sample + w_target[1] * x_sample ** 2 + w_target[2] * x_sample ** 3
x_train =np.stack([x_sample** i for i in range(1,4)],axis=1)#axis=1表示按行来组合x_train**i
x_train=torch.from_numpy(x_train).float()#一定要转成float，因为不转就是double类型
y_train = torch.from_numpy(y_sample).float().unsqueeze(1)#在1维增加一维，也就是将（64），变成（64*1）

#定义参数
w=torch.randn(3,1,requires_grad=True)
b=torch.randn(1,requires_grad=True)
#定义模型,一元的时候w是标量，x是标量，而此时x是
def Linear(x):
	return torch.mm(x,w) + b#torch.mm（a，b）的意思是矩阵a×矩阵b，维度必须一致,本例中x为3,1而样本为64*3，所以我们需要orch.mm(x,w)，而不能orch.mm(w,x)我们只需要记住一点，无论样本是列排还是行排都不是问题。

#定义损失函数
def get_loss(y1,y2):
	return torch.mean((y1-y2)**2)

#开启梯度下降，梯度下降100次
for e in range(100):
    y_pred = Linear(x_train)
    loss = get_loss(y_pred, y_train)
    if e!=0:
        w.grad.data.zero_()#只有进行backward之后我们才可以将梯度进行归0操作
        b.grad.data.zero_()
    loss.backward()
    # 更新参数
    w.data = w.data - 0.001 * w.grad.data
    b.data = b.data - 0.001 * b.grad.data
    if (e + 1) % 20 == 0:
        print('epoch {}, Loss: {:.5f}'.format(e+1, loss.item()))
   
y_pred = Linear(x_train)
plt.plot(x_train.data.numpy()[:, 0], y_pred.data.numpy(), label='fitting curve', color='r')
plt.plot(x_train.data.numpy()[:, 0], y_train.data.numpy(), label='real curve', color='b')
plt.legend()
plt.show()


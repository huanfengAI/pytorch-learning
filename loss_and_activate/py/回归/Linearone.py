import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
#本例是一元线性回归模型
x_train =np.array([[3.3],[4.4],[5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)
y_train =np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)
x_train=torch.from_numpy(x_train)#转成tensor向量形式
y_train=torch.from_numpy(y_train)
#定义一元线性回归模型的参数w和b,因为是一元的所以w也是一个标量
w=Variable(torch.randn(1),requires_grad=True)
b=Variable(torch.randn(1),requires_grad=True)
#定义线性回归模型
def Linear(x):
	y=w*x+b
	return y
#有了模型之后，我们需要定义损失函数,这里我们使用均方误差来定义损失，y1是样本真实标签而y2是样本的预测标签
def get_loss(y1,y2):
	return torch.mean((y1-y2)**2)
#接下来我们就可以完成梯度下降了，具体过程是这样的，我们将样本传入到模型中，得到预测值，然后计算损失loss，然后进行梯度下降更新参数
#第一步，得到预测值
#x_train是一批数据，我们可以将一批数据同时喂给我们的模型，因为python的广播机制，它会对这些样本同时进行处理
y=Linear(x_train)
#第二步，计算损失loss
loss=get_loss(y_train,y)
#反向传播,计算参数的梯度
loss.backward()
wgrad=w.grad
bgrad=b.grad
print(wgrad)
print(bgrad)
#梯度下降更新参数,此时我们使用学习率是1e-2
#w.data表示w的值
#w.grad.data表示w的梯度的值
w.data=w.data-1e-2*w.grad.data
b.data=w.data-1e-2*w.grad.data
#此时是仅仅进行了一次梯度下降，我们可以使用for循环来循环进行多次梯度下降
for i in range(10):
	y=Linear(x_train)
	loss=get_loss(y_train,y)
	#归0梯度
	w.grad.zero_()#zero_后面有一个_,这个很方便，它可以将归零之后的梯度从新赋值给w
	b.grad.zero_()
	loss.backward()
	w.data=w.data-1e-2*w.grad.data
	b.data=w.data-1e-2*w.grad.data
#运行完十次梯度下降算法之后，我们的模型已经已经调的差不多了，此时我们可以将样本在重新预测一次，看最新的预测结果怎么样？	
#要想画图我们需要使用numpy类型才可以画出来，要想转成numpy需要先转成tensor再转成numpy
y = Linear(x_train)
plt.plot(x_train.data.numpy(), y_train.data.numpy(), 'bo', label='real')
plt.plot(x_train.data.numpy(), y.data.numpy(), 'ro', label='estimated')
plt.legend()
plt.show()

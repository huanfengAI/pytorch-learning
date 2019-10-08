import torch
from torch import nn
import numpy as np
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
w=torch.randn(1,requires_grad=True)
b=torch.randn(1,requires_grad=True)
#定义线性回归模型
def Linear(x):
	y=w*x+b
	return y
#有了模型之后，我们需要定义损失函数,这里我们使用均方误差来定义损失，y1是样本真实标签而y2是样本的预测标签
def get_loss(y1,y2):
	return torch.mean((y1-y2)**2)


#此时是仅仅进行了一次梯度下降，我们可以使用for循环来循环进行多次梯度下降
for i in range(10):
	y=Linear(x_train)
	loss=get_loss(y_train,y)
	#归0梯度
	print("当前损失",loss.item())
	if i != 0:
	    w.grad.data.zero_()
	    b.grad.data.zero_()
	   
	loss.backward()
	
	w.data=w.data-1e-2*w.grad.data
	b.data=b.data-1e-2*b.grad.data

	
#要想画图我们需要使用numpy类型才可以画出来，要想转成numpy需要先转成tensor再转成numpy
y = Linear(x_train)
plt.plot(x_train.data.numpy(), y_train.data.numpy(), 'bo', label='real')
plt.plot(x_train.data.numpy(), y.data.numpy(), 'ro', label='estimated')
plt.legend()
plt.show()

import sys
sys.path.append('..')
import numpy as np
from torchvision.datasets import mnist
from torch.utils.data import DataLoader
from torch import nn
import torch
from torchvision import transforms
from datetime import datetime
#训练的时候使用了批归一化，而测试的时候不使用肯定会导致结果出现偏差，但是测试的时候如果只有一个数据集，那么均值不就是这个值，方差就是 0 啊，这显然是随机的，所以测试的时候不能用测试的数据集去算均值和方差，而是用训练的时候算出的移动平均均值和方差去代替
#为什么要设置net.train()和net.eval()
#训练时是正对每个min-batch的，但是在测试中往往是针对单张图片，即不存在min-batch的概念。由于网络训练完毕后参数都是固定的，因此每个批次的均值和方差都是不变的，因此直接结算所有batch的均值和方差。所有Batch Normalization的训练和测试时的操作不同
#在训练中，每个隐层的神经元先乘概率P，然后在进行激活，在测试中，所有的神经元先进行激活，然后每个隐层神经元的输出乘P。

train_set =mnist.MNIST('./data',train=True)
test_set  =mnist.MNIST('./data',train=False)

def batch_norm_1d(x, gamma, beta, is_training, moving_mean, moving_var, moving_momentum=0.1):
    eps = 1e-5
    x_mean = torch.mean(x, dim=0, keepdim=True) # 保留维度进行 broadcast
    x_var = torch.mean((x - x_mean) ** 2, dim=0, keepdim=True)
    if is_training:
        x_hat = (x - x_mean) / torch.sqrt(x_var + eps)
        moving_mean[:] = moving_momentum * moving_mean + (1. - moving_momentum) * x_mean
        moving_var[:] = moving_momentum * moving_var + (1. - moving_momentum) * x_var
    else:
        x_hat = (x - moving_mean) / torch.sqrt(moving_var + eps)
       
    return gamma.view_as(x_mean) * x_hat + beta.view_as(x_mean)




def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)#求每行的最大就是最有可能的类别
    num_correct = (pred_label == label).sum().float()
    return num_correct / total
#def data_tf(x):
#	x=np.array(x,dtype='float32')
#	x=(x - 0.5) /0.5
#	x= x.reshape((-1,))
#	x=torch.from_numpy(x)
#	return x
data_tf=transforms.Compose(
[transforms.ToTensor(),
 transforms.Normalize([0.5],[0.5])
]
)
train_set =mnist.MNIST('./data',train=True,transform=data_tf,download=True)
test_set  =mnist.MNIST('./data',train=False,transform=data_tf,download=True)
train_data=DataLoader(train_set,batch_size=64,shuffle=True)
test_data =DataLoader(test_set,batch_size=128,shuffle=True)
class multi_network(nn.Module):
	def __init__(self):
		super(multi_network,self).__init__()
		self.layer1=nn.Linear(784,100)
		self.relu=nn.ReLU(True)
		self.layer2=nn.Linear(100,10)
		self.beta=nn.Parameter(torch.randn(100))
		self.gamma=nn.Parameter(torch.randn(100))
		self.moving_mean=torch.zeros(100)
		self.moving_var=torch.zeros(100)
	def forward(self,x,is_train=True):
                print(is_train)
                x= self.layer1(x)
                x = batch_norm_1d(x, self.gamma, self.beta, is_train, self.moving_mean, self.moving_var)
                x=self.relu(x)
                x=self.layer2(x)
                return x
net=multi_network()

criterion =nn.CrossEntropyLoss()#定义损失函数
optimizer =torch.optim.SGD(net.parameters(),1e-1)
#训练
prev_time=datetime.now()
for epoch in range(30):
	train_loss=0
	train_acc =0
	net =net.train()
	for im ,label in train_data:#im,label为一批数据，也就是64个样本
		#前向传播并计算损失
		#print(im.size())#im=im.view(im.size(0),-1)torch.Size([64, 1, 28, 28])
		im=im.view(im.size(0),-1)
		#print(im.size())torch.Size([64, 784])
		output =net(im)
		loss =criterion(output ,label)
		#反向传播
		optimizer.zero_grad()
		
		loss.backward()
		optimizer.step()
		
		#print(loss.data)
		train_loss +=loss.data.float()
		train_acc +=get_acc(output,label)
		#print(train_acc/len(train_data))
		#print(train_acc/64)
	#测试
	print("------------------------------")
	cur_time =datetime.now()
	h,remainder =divmod((cur_time-prev_time).seconds,3600)
	m,s=divmod(remainder,60)
	time_str ="Time %02d:%02d:%02d"%(h,m,s)
	valid_loss=0
	valid_acc=0
	net =net.eval()
	for im,label in test_data:
		im=im.view(im.size(0),-1)
		
		output =net(im)
		
		loss= criterion(output,label)
		valid_loss +=loss.data.float()
		valid_acc +=get_acc(output,label)
	epoch_str=(
			"Epoch %d. Train Loss %f,Train Acc:%f,Valid Loss: %f,Valid Acc: %f ,"
			%(epoch,train_loss/len(train_data),
			  train_acc /len(train_data),
			  valid_loss/len(test_data),
			  valid_acc /len(test_data)))
	prev_time=cur_time
	print(epoch_str+time_str)#训练一批测试一批,time_str为每次epoch运行的时间00:00:07表示7秒
	print(net.moving_mean[:10])#训练一批测试一批,time_str为每次epoch运行的时间00:00:07表示7秒



import torch
from torch import nn
from torch import optim
class Net(nn.Module):
	def __init__(self):
		super(Net,self).__init__()
		self.features=nn.Sequential(
			nn.Conv2d(3,6,5),
			nn.ReLU(True),
			nn.MaxPool2d(2,2),
			nn.Conv2d(6,16,5),
			nn.ReLU(),
			nn.MaxPool2d(2,2)
		)
		self.classifier=nn.Sequential(
			nn.Linear(16*5*5,120),
			nn.ReLU(),
			nn.Linear(120,84),
			nn.ReLU(),
			nn.Linear(84,10)
		)
	def forward(self,x):
			x=self.features(x)
			x=x.view(x.shape[0],-1)
			x=self.classifier(x)
			return x

net=Net()
#定义一个优化器，优化方法为SGD
#优化的参数是net.parameters()也就是所有的参数
#对这些所有的参数的优化的学习率为1
optimizer=optim.SGD(params=net.parameters(),lr=1)
#print(optimizer)
#SGD (
#Parameter Group 0
#    dampening: 0
#    lr: 1
#    momentum: 0
#    nesterov: False
#    weight_decay: 0
#)
optimizer.zero_grad()#梯度清零，还可以使用net.zero_grad()完成梯度清零操作
input=torch.randn(1,3,32,32)#创造一个样本batch-size为1，通道为3，大小为32×32
out=net(input)
#print(input.size())torch.Size([1, 3, 32, 32])

###############为不同的子网络设置不同的学习率
#使用SGD的优化方式
#对features的优化学习率为1e-5
#对classifier的优化学习率为1e-2
#这样就实现了对不同子网络的优化
optimizer=optim.SGD([
					{'params':net.features.parameters()},
					{'params':net.classifier.parameters(),'lr':1e-4}
					],lr=1e-5)
#print(optimizer)
#SGD (
#Parameter Group 0
#    dampening: 0
#    lr: 1e-05
#    momentum: 0
#    nesterov: False
#    weight_decay: 0

#Parameter Group 1
#    dampening: 0
#    lr: 0.01
#    momentum: 0
#    nesterov: False
#    weight_decay: 0
#)
#每有一个params参数就有一个Group组
optimizer.zero_grad()#梯度清零

special_layers=nn.ModuleList([net.classifier[0],net.classifier[2]])
#print(special_layers)
#ModuleList(
#  (0): Linear(in_features=400, out_features=120, bias=True)
#  (1): Linear(in_features=120, out_features=84, bias=True)
#)

special_layers_params=list(map(id,special_layers.parameters()))
#id是一个函数
#map也是一个函数
#map的作用是将special_layers.parameters()进行id函数操作，id的作用是获取对象的内存地址
#其中special_layers.parameters()有四个对象，第一个全连接层的w，b，第二个全连接层的w，b
#print(special_layers_params)[139886702441024, 139886702441096, 139886702441312, 139886702441384]

#filter() 函数用于过滤序列，过滤掉不符合条件的元素，返回由符合条件元素组成的新列表。
#lambda是匿名函数
#p是全体神经网络参数net.parameters(),对p进行id(p) not in special_layers_params操作

base_params=filter(lambda p :id(p) not in special_layers_params,
					net.parameters())
#最终的base_params就是除了全连接层之外的所有层的学习参数

optimizer =torch.optim.SGD([
				{'params':base_params},
				{'params':special_layers.parameters(),'lr':0.01}
			],lr=0.001)
#print(optimizer)

####################
##调整学习率，新建一个optimizer，这种做法可能导致动量的优化器丢失动量等状态信息
old_lr =0.1
optimizer1=optim.SGD([
						{'params':net.features.parameters()},
						{'params':net.classifier.parameters(),'lr':old_lr*0.1}			
						],lr=1e-5)

方法二：手动调整学习率
for param_group in optimizer.param_groups:
	param_group['lr']*=0.1



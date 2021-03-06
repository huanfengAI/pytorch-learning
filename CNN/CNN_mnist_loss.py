import numpy as np
from torchvision.datasets import mnist
from torch.utils.data import DataLoader
from torch import nn
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from datetime import datetime


train_set =mnist.MNIST('./adata',train=True)
test_set  =mnist.MNIST('./adata',train=False)
def set_learning_rate(optimizer,lr):
	for param_group in optimizer.param_groups:
		param_group['lr']=lr
def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)#求每行的最大就是最有可能的类别
    num_correct = (pred_label == label).sum().float()
    return num_correct / total

data_tf=transforms.Compose(
[transforms.ToTensor(),
 transforms.Normalize([0.5],[0.5])
]
)
train_set =mnist.MNIST('./adata',train=True,transform=data_tf,download=True)
test_set  =mnist.MNIST('./adata',train=False,transform=data_tf,download=True)
train_data=DataLoader(train_set,batch_size=64,shuffle=True)
test_data =DataLoader(test_set,batch_size=128,shuffle=True)

class multi_network(nn.Module):
	def __init__(self):
		super(multi_network,self).__init__()
		self.layer1=nn.Sequential(
			nn.Conv2d(1,6,3,padding=1),
			nn.ReLU(True),
			nn.MaxPool2d(2,2),
			nn.Conv2d(6,16,5),
			nn.ReLU(True),
			nn.MaxPool2d(2,2)
			
		
		)
		self.classfy=nn.Linear(400,10)
		
	def forward(self,x):
		x= self.layer1(x)
		x=x.view(x.shape[0],-1)
		x=self.classfy(x)
		return x
net=multi_network()

criterion =nn.CrossEntropyLoss()#定义损失函数
optimizer =torch.optim.SGD(net.parameters(),1e-1)
#训练
prev_time=datetime.now()
train_losses=[]
valid_losses=[]
for epoch in range(10):
	if epoch==20:
		set_learning_rate(optimizer,0.01)
	train_loss=0
	train_acc =0
	
	net =net.train()
	for im ,label in train_data:#im,label为一批数据，也就是64个样本
		output =net(im)
		
		loss =criterion(output ,label)
		#反向传播
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		
		train_loss +=loss.data.float()
		train_acc +=get_acc(output,label)
	#测试
	cur_time =datetime.now()
	h,remainder =divmod((cur_time-prev_time).seconds,3600)
	m,s=divmod(remainder,60)
	time_str ="Time %02d:%02d:%02d"%(h,m,s)
	valid_loss=0
	valid_acc=0
	net =net.eval()
	for im,label in test_data:
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
	train_losses.append(train_loss/len(train_data))
	valid_losses.append(valid_loss/len(test_data))
	print(epoch_str+time_str)#训练一批测试一批,time_str为每次epoch运行的时间00:00:07表示7秒
	
plt.plot(train_losses, label='train')
plt.plot(valid_losses, label='valid')
plt.xlabel('epoch')
plt.legend(loc='best')
plt.show()

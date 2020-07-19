import torch
from torch import nn
import numpy as np
from torchvision.datasets import mnist
from torch.utils.data import DataLoader
from torchvision import transforms
from datetime import datetime
from torchvision.datasets import CIFAR10


def get_acc(output, label):
    total = output.shape[0]#本次有多少个样本,训练的时候是64个
    _, pred_label = output.max(1)#求每行的最大就是最有可能的类别.pred_label为长为64的列表
    num_correct = (pred_label == label).sum().float()
    return num_correct / total

data_tf=transforms.Compose(
[transforms.ToTensor(),
 transforms.Normalize([0.5],[0.5])
]
)
train_set = CIFAR10('./data', train=True, transform=data_tf,download=True)
test_set = CIFAR10('./data', train=False, transform=data_tf,download=True)
test_data =DataLoader(test_set, batch_size=64, shuffle=True)
train_data =DataLoader(train_set, batch_size=128, shuffle=True)


class VGG(nn.Module):
	def __init__(self):
		super(VGG,self).__init__()
		self.features=nn.Sequential(
			nn.Conv2d(3,64,kernel_size=3,padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(64,64,kernel_size=3,padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2,stride=2),
			nn.Conv2d(64,128,kernel_size=3,padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(128,128,kernel_size=3,padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2,stride=2),
			nn.Conv2d(128,256,kernel_size=3,padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(256,256,kernel_size=3,padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(256,256,kernel_size=4,padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2,stride=2),
			nn.Conv2d(256,512,kernel_size=3,padding=1) ,
			nn.ReLU (True) , 
			nn.Conv2d (512,512, kernel_size=3, padding=1) , 
			nn.ReLU (True) , 
			nn.Conv2d(512, 512 , kernel_size=3,padding=1) , 
			nn.ReLU(True) , 
			nn.MaxPool2d(kernel_size=2,stride=2) , 
			nn.Conv2d(512,512,kernel_size=3,padding=1) , 
			nn.ReLU(True), 
			nn.Conv2d(512, 512, kernel_size=3,padding=1),
			nn.ReLU(True) , 
			nn.Conv2d(512, 512, kernel_size=3, padding=1) , 
			nn.ReLU(True) , 
			nn.MaxPool2d(kernel_size=2,stride=2 ) 
		)
		self.classifier = nn.Sequential (
			nn.Linear(512, 4096), 
			nn.ReLU (True) , 
			nn.Dropout() , 
			nn.Linear(4096,4096) , 
			nn.ReLU(True), 
			nn.Dropout(), 
			nn.Linear(4096,10))
		
	def forward(self,x):
			x=self.features(x)
			x=x.view(x.shape[0],-1)
			x=self.classifier(x)
			return x
net=VGG()


criterion =nn.CrossEntropyLoss()#定义损失函数
optimizer =torch.optim.SGD(net.parameters(),lr=1e-1,weight_decay=1e-4)#使用Lr正则化，lambda为1e-4
#optimizer =torch.optim.SGD(net.parameters(),lr=1e-1)
#训练
prev_time=datetime.now()
for epoch in range(30):
	train_loss=0
	train_acc =0
	net =net.train()
	for im ,label in train_data:#im,label为一批数据，也就是64个样本
		#前向传播并计算损失
		#print(im.size())#im=im.view(im.size(0),-1)torch.Size([64, 1, 28, 28])
		#im=im.view(im.size(0),-1)
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
		#print(train_acc)
		#print(train_acc/len(train_data))
		#print(train_acc/64)
	#测试
	cur_time =datetime.now()
	h,remainder =divmod((cur_time-prev_time).seconds,3600)
	m,s=divmod(remainder,60)
	time_str ="Time %02d:%02d:%02d"%(h,m,s)
	valid_loss=0
	valid_acc=0
	net =net.eval()
	for im,label in test_data:
		#im=im.view(im.size(0),-1)
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

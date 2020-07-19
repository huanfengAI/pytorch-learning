import numpy as np
from torchvision.datasets import mnist
from torch.utils.data import DataLoader
from torch import nn
import torch
from torchvision import transforms
from datetime import datetime

def get_acc(output, label):#准确率
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
test_data=DataLoader(test_set,batch_size=64,shuffle=True)

class CNN(nn.Module):
       def __init__(self):
              super(CNN,self).__init__()
              self.layer1=nn.Sequential(
                     nn.Conv2d(1,16,kernel_size=3),
                     nn.BatchNorm2d(16),
                     nn.ReLU(inplace=True),           
              )
              self.layer2=nn.Sequential(
                     nn.Conv2d(16,32,kernel_size=3),
                     nn.BatchNorm2d(32),
                     nn.ReLU(inplace=True),
                     nn.MaxPool2d(kernel_size=2,stride=2)
              )
              self.layer3=nn.Sequential(
                     nn.Conv2d(32,64,kernel_size=3),
                     nn.BatchNorm2d(64),
                     nn.ReLU(inplace=True)
              )
              self.layer4=nn.Sequential(
                     nn.Conv2d(64,128,kernel_size=3),
                     nn.BatchNorm2d(128),
                     nn.ReLU(inplace=True),
                     nn.MaxPool2d(kernel_size=2,stride=2)
              )
              self.fc=nn.Sequential(
                     nn.Linear(128*4*4,1024),
                     nn.ReLU(inplace=True),
                     nn.Linear(1024,128),
                     nn.ReLU(inplace=True),
                     nn.Linear(128,10)
              )
       def forward(self,x):
              x=self.layer1(x)
              x=self.layer2(x)
              x=self.layer3(x)
              x=self.layer4(x)
              in_put=x.view(x.size(0),-1)
              out_put=self.fc(in_put)
              return out_put
net=CNN()

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
	print(epoch_str+time_str)#训练一批测试一批,time_str为每次epoch运行的时间00:00:07表示7秒



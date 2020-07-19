
import numpy as np
from torchvision.datasets import mnist
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torch import nn
import torch
from torchvision import transforms
from datetime import datetime

def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)#求每行的最大就是最有可能的类别
    num_correct = (pred_label == label).sum().float()
    return num_correct / total

data_tf=transforms.Compose(
[
transforms.Resize(96),# 将图片放大到 96 x 96这样本网络才可以处理
transforms.ToTensor(),
 transforms.Normalize([0.5],[0.5])
]
)
train_set = CIFAR10('./data', train=True, transform=data_tf,download=True)
test_set = CIFAR10('./data', train=False, transform=data_tf,download=True)
test_data =DataLoader(test_set, batch_size=32, shuffle=True)
train_data =DataLoader(train_set, batch_size=64, shuffle=True)


def conv_relu(in_channel, out_channel, kernel, stride=1, padding=0):
    layer = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel, stride, padding),
        nn.BatchNorm2d(out_channel, eps=1e-3),
        nn.ReLU(True)
    )
    return layer
     
class inception(nn.Module):
	def __init__(self,in_channel,out1_1,out2_1,out2_3,out3_1,out3_5,out4_1):
		super(inception,self).__init__()
		self.branch1x1=conv_relu(in_channel,out1_1,1)
		
		self.branch3x3=nn.Sequential(
			conv_relu(in_channel,out2_1,1),
			conv_relu(out2_1,out2_3,3,padding=1)
		)
		self.branch5x5=nn.Sequential(
			conv_relu(in_channel,out3_1,1),
			conv_relu(out3_1,out3_5,5,padding=2)
		)
		self.branch_pool=nn.Sequential(
			nn.MaxPool2d(3,stride=1,padding=1),
			conv_relu(in_channel,out4_1,1)
		)
		
	def forward(self,x):
			f1 = self.branch1x1(x)
			f2 = self.branch3x3(x)
			f3 = self.branch5x5(x)
			f4 = self.branch_pool(x)
			output = torch.cat((f1, f2, f3, f4), dim=1)
			return output
   
class GoogLeNet2(nn.Module):
    def __init__(self, in_channel, num_classes):
        super(GoogLeNet2, self).__init__()
       
        
        self.block1 = nn.Sequential(
            conv_relu(in_channel, out_channel=64, kernel=7, stride=2, padding=3),
            nn.MaxPool2d(3, 2)
        )
        
        self.block2 = nn.Sequential(
            conv_relu(64, 64, kernel=1),
            conv_relu(64, 192, kernel=3, padding=1),
            nn.MaxPool2d(3, 2)
        )
        
        self.block3 = nn.Sequential(
            inception(192, 64, 96, 128, 16, 32, 32),
            inception(256, 128, 128, 192, 32, 96, 64),
            nn.MaxPool2d(3, 2)
        )
        
        self.block4 = nn.Sequential(
            inception(480, 192, 96, 208, 16, 48, 64),
            inception(512, 160, 112, 224, 24, 64, 64),
            inception(512, 128, 128, 256, 24, 64, 64),
            inception(512, 112, 144, 288, 32, 64, 64),
            inception(528, 256, 160, 320, 32, 128, 128),
            nn.MaxPool2d(3, 2)
        )
        
        self.block5 = nn.Sequential(
            inception(832, 256, 160, 320, 32, 128, 128),
            inception(832, 384, 182, 384, 48, 128, 128),
            nn.AvgPool2d(2)
        )
        
        self.classifier = nn.Linear(1024, num_classes)
        
    def forward(self, x):
        x = self.block1(x)
        
        x = self.block2(x)
        
        x = self.block3(x)
     
        x = self.block4(x)
      
        x = self.block5(x)
      
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x
        
net=GoogLeNet2(3,10)
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

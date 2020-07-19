import numpy as np
from torchvision.datasets import mnist
from torch.utils.data import DataLoader
from torch import nn
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from datetime import datetime
from torchvision.datasets import CIFAR10
import torch.nn.functional as F


def set_learning_rate(optimizer,lr):
	for param_group in optimizer.param_groups:
		param_group['lr']=lr
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
train_set = CIFAR10('./data', train=True, transform=data_tf,download=True)
test_set = CIFAR10('./data', train=False, transform=data_tf,download=True)
test_data =DataLoader(test_set, batch_size=128, shuffle=True)
train_data =DataLoader(train_set, batch_size=64, shuffle=True)




def conv3x3(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, 3, stride=stride, padding=1, bias=False)

class Resnet(nn.Module):
	def __init__(self,in_channel,out_channel,same_shape=True):
		super(Resnet,self).__init__()
		self.same_shape=same_shape
		stride=1 if self.same_shape else 2
		
		self.conv1=conv3x3(in_channel,out_channel,stride=stride)
		self.bn1=nn.BatchNorm2d(out_channel)

		self.conv2=conv3x3(out_channel,out_channel)
		self.bn2=nn.BatchNorm2d(out_channel)
		if not self.same_shape:
			self.conv3 =nn.Conv2d(in_channel,out_channel,1,stride=stride)
	def forward(self,x):
		out=self.conv1(x)
		out=F.relu(self.bn1(out),True)
		out=self.conv2(out)
		out=F.relu(self.bn2(out),True)

		if not self.same_shape:
			x=self.conv3(x)
		
		return F.relu(x+out,True)
class Resnet1(nn.Module):
    def __init__(self, in_channel, num_classes, verbose=False):
        super(Resnet1, self).__init__()
        self.verbose = verbose
        
        self.block1 = nn.Conv2d(in_channel, 64, 7, 2)
        
        self.block2 = nn.Sequential(
            nn.MaxPool2d(3, 2),
            Resnet(64, 64),
            Resnet(64, 64)
        )
        
        self.block3 = nn.Sequential(
            Resnet(64, 128, False),
            Resnet(128, 128)
        )
        
        self.block4 = nn.Sequential(
            Resnet(128, 256, False),
            Resnet(256, 256)
        )
        
        self.block5 = nn.Sequential(
            Resnet(256, 512, False),
            Resnet(512, 512),
            nn.AvgPool2d(3)
        )
        
        self.classifier = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.block1(x)
        if self.verbose:
            print('block 1 output: {}'.format(x.shape))
        x = self.block2(x)
        if self.verbose:
            print('block 2 output: {}'.format(x.shape))
        x = self.block3(x)
        if self.verbose:
            print('block 3 output: {}'.format(x.shape))
        x = self.block4(x)
        if self.verbose:
            print('block 4 output: {}'.format(x.shape))
        x = self.block5(x)
        if self.verbose:
            print('block 5 output: {}'.format(x.shape))
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x
net=Resnet1(3,10)

criterion =nn.CrossEntropyLoss()#定义损失函数
optimizer =torch.optim.SGD(net.parameters(),1e-1)
#训练
prev_time=datetime.now()
train_losses=[]
valid_losses=[]
for epoch in range(30):
	if epoch==20:
		set_learning_rate(optimizer,0.01)
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
	train_losses.append(train_loss/len(train_data))
	train_losses.append(valid_loss/len(test_data))
	print(epoch_str+time_str)#训练一批测试一批,time_str为每次epoch运行的时间00:00:07表示7秒
	print(train_losses)
	print(train_losses)
plt.plot(train_losses, label='train')
plt.plot(valid_losses, label='valid')
plt.xlabel('epoch')
plt.legend(loc='best')
plt.show()

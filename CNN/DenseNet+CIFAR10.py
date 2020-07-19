import sys
sys.path.append('..')

from torchvision.datasets import mnist
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms
from datetime import datetime
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torchvision.datasets import CIFAR10


# 除了 dense block，DenseNet 中还有一个模块叫过渡层（transition block），因为 DenseNet 会不断地对维度进行拼接， 所以当层数很高的时候，输出的通道数就会越来越大，参数和计算量也会越来越大，为了避免这个问题，需要引入过渡层将输出通道降低下来，同时也将输入的长宽减半，这个过渡层可以使用 1 x 1 的卷积

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



def transition(in_channel, out_channel):
    trans_layer = nn.Sequential(
        nn.BatchNorm2d(in_channel),
        nn.ReLU(True),
        nn.Conv2d(in_channel, out_channel, 1),
        nn.AvgPool2d(2, 2)
    )
    return trans_layer






def conv_block(in_channel, out_channel):
    layer = nn.Sequential(
        nn.BatchNorm2d(in_channel),
        nn.ReLU(True),
        nn.Conv2d(in_channel, out_channel, 3, padding=1, bias=False)
    )
    return layer

class dense_block(nn.Module):
    def __init__(self, in_channel, growth_rate, num_layers):
        super(dense_block, self).__init__()
        block = []
        channel = in_channel
        for i in range(num_layers):
            block.append(conv_block(channel, growth_rate))
            channel += growth_rate
            
        self.net = nn.Sequential(*block)
        
    def forward(self, x):
        for layer in self.net:
            out = layer(x)
            x = torch.cat((out, x), dim=1)
        return x
        
        
        
class densenet(nn.Module):
    def __init__(self, in_channel, num_classes, growth_rate=32, block_layers=[6, 12, 24, 16]):
        super(densenet, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channel, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2, padding=1)
        )
        
        channels = 64
        block = []
        for i, layers in enumerate(block_layers):
            block.append(dense_block(channels, growth_rate, layers))
            channels += layers * growth_rate
            if i != len(block_layers) - 1:
                block.append(transition(channels, channels // 2)) # 通过 transition 层将大小减半，通道数减半
                channels = channels // 2
        
        self.block2 = nn.Sequential(*block)
        self.block2.add_module('bn', nn.BatchNorm2d(channels))
        self.block2.add_module('relu', nn.ReLU(True))
        self.block2.add_module('avg_pool', nn.AvgPool2d(3))
        
        self.classifier = nn.Linear(channels, num_classes)
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x
net=densenet(3,10)

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
		optimizer.zero_grad()#梯度归0
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
	valid_losses.append(valid_loss/len(test_data))
	print(epoch_str+time_str)#训练一批测试一批,time_str为每次epoch运行的时间00:00:07表示7秒
	
plt.plot(train_losses, label='train')
plt.plot(valid_losses, label='valid')
plt.xlabel('epoch')
plt.legend(loc='best')
plt.show()

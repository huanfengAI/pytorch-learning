import torch
from torch import nn
from torchvision.datasets import mnist
from torch.utils.data import DataLoader

from torchvision import transforms
from datetime import datetime
train_set =mnist.MNIST('./data',train=True)
test_set  =mnist.MNIST('./data',train=False)
def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)#求每行的最大就是最有可能的类别
    print(pred_label)
    print(label)
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
class LeNet(nn.Module):
	def __init__(self):
		super(LeNet,self).__init__()
		layer1=nn.Sequential()
		layer1.add_module('conv1',nn.Conv2d(1,6,5,padding=1))
		layer1.add_module('pool1',nn.MaxPool2d(2,2))
		self.layer1=layer1
		
		layer2=nn.Sequential()
		layer2.add_module('conv2',nn.Conv2d(6,16,5))
		layer2.add_module('poo2',nn.MaxPool2d(2,2))
		self.layer2=layer2
		
		layer3=nn.Sequential()
		layer3.add_module('fc1',nn.Linear(256,120))
		layer3.add_module('fc2',nn.Linear(120,84))
		layer3.add_module('fc3',nn.Linear(84,10))
		self.layer3=layer3
	def forward(self,x):
		x=self.layer1(x)
		x=self.layer2(x)
		x=x.view(x.shape[0],-1)
		x=self.layer3(x)
		return x
net=LeNet()

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
	

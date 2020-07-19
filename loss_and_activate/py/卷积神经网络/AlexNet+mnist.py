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
    num_correct = (pred_label == label).sum().float()
    return num_correct / total
#def data_tf(x):
#	x=np.array(x,dtype='float32')
#	x=(x - 0.5) /0.5
#	x= x.reshape((-1,))
#	x=torch.from_numpy(x)
#	return x
data_tf=transforms.Compose(
[transforms.Resize(224),#拉伸到224
 transforms.ToTensor(),
 transforms.Normalize([0.5],[0.5])
 
]
)
train_set =mnist.MNIST('./data',train=True,transform=data_tf,download=True)
test_set  =mnist.MNIST('./data',train=False,transform=data_tf,download=True)
train_data=DataLoader(train_set,batch_size=64,shuffle=True)
test_data =DataLoader(test_set,batch_size=128,shuffle=True)


class AlexNet(nn.Module):
	def __init__(self):
		super(AlexNet,self).__init__()
		self.features=nn.Sequential(
			nn.Conv2d(1,64,kernel_size=11,stride=4,padding=2),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3,stride=2),
			
			nn.Conv2d(64,192,kernel_size=5,padding=2),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3,stride=2),
			
			nn.Conv2d(192,384,kernel_size=3,padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(384,256,kernel_size=3,padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(256,256,kernel_size=3,padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3,stride=2)
			)	
		self.classfier=nn.Sequential(
			nn.Dropout(),
			nn.Linear(9216,4096),
			nn.ReLU(inplace=True),
			nn.Dropout(),
			nn.Linear(4096,4096),
			nn.ReLU(inplace=True),
			nn.Linear(4096,10)#做10分类的话就是10
			)
	def forward(self,x):
		x=self.features(x)
		x=x.view(x.shape[0],-1)
		x=self.classfier(x)
		return x
		
net=AlexNet()
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

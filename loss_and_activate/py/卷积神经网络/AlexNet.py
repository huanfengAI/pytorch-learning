import torch
from torch import nn

class AlexNet(nn.Module):
	def __init__(self):
		super(AlexNet,self).__init__()
		self.features=nn.Sequential(
			nn.Conv2d(3,64,kernel_size=11,stride=4,padding=2),
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
			nn.Linear(256*6*6,4096),
			nn.ReLU(inplace=True),
			nn.Dropout(),
			nn.Linear(4096,4096),
			nn.ReLU(inplace=True),
			nn.Linear(4096,10)#做10分类的话就是10
			)
net=AlexNet()
print(*list(net.parameters()))	

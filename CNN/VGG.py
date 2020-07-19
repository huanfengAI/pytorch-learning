import torch
from torch import nn
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
			nn.Linear (512 * 7 * 7 , 4096) , 
			nn.ReLU (True) , 
			nn.Dropout() , 
			nn.Linear (4096,4096) , 
			nn.ReLU (True) , 
			nn.Dropout ( ) , 
			nn.Linear(4096 ,10)
		) 
		
		  
	def forward(self,x):
			x=self.features(x)
			x=x.view(x.size(0),-1)
			x=self.classifier(x)
net=VGG()
print(net)

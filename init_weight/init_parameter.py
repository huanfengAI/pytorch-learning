import numpy as np
import torch
from torch import nn
from torch.nn import init
net1=nn.Sequential(#设置一个sequential
	nn.Linear(50,100),
	nn.ReLU(True),
	nn.Linear(100,200),
	nn.ReLU(True),
	nn.Linear(200,10)
)
for i in net1.parameters():
	print(i)
#net1[0]为获取第一层Linear(50,100)
#net1[0].weight.data= torch.from_numpy(np.random.uniform(3, 5, size=(50, 100)))
#w1=net1[0].weight#weight为获取权重参数
#b1=net1[0].bias#bias为获取偏置单元
#print(w1)
#print(b1)
init.xavier_uniform_(net1[0].weight)#初始化
for i in net1:#遍历这个sequential，只要里面是linear层，我们就可以为期进行初始化参数了
	if isinstance(i,nn.Linear):
		i.weight.data=torch.from_numpy(np.random.normal(0,0.5,size=i.weight.shape))




class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.l1 = nn.Sequential(
            nn.Linear(30, 40),
            nn.ReLU()
        )
        self.l1[0].weight.data = torch.randn(40, 30) # 直接对某一层初始化
    def forward(self, x):
        x = self.l1(x)
        return x



class sim_net(nn.Module):
    def __init__(self):
        super(sim_net, self).__init__()
        self.l1 = nn.Sequential(
            nn.Linear(30, 40),
            nn.ReLU()
        )
        
        self.l2 = nn.Sequential(
            nn.Linear(40, 50),
            nn.ReLU()
        )
        
        self.l3 = nn.Sequential(
            nn.Linear(50, 10),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.l1(x)
        x =self.l2(x)
        x = self.l3(x)
        return x

net2 = sim_net()
#for i in net2.children():
 #   print(i)
#for i in net2.modules():
#    print(i)



for layer in net2.modules():
    if isinstance(layer, nn.Linear):
        param_shape = layer.weight.shape
        layer.weight.data = torch.from_numpy(np.random.normal(0, 0.5, size=param_shape)) 

model=nn.Sequential(*list(net2.children())[:2])
#print(model)




class multi_network(nn.Module):
	def __init__(self):
		super(multi_network,self).__init__()
		self.layer1=nn.Sequential(
			nn.Conv2d(1,6,3,padding=1),
			
			nn.BatchNorm2d(6),
			nn.ReLU(True),
			nn.MaxPool2d(2,2),
			nn.Conv2d(6,16,5),
			nn.BatchNorm2d(16),
			nn.ReLU(True),
			nn.MaxPool2d(2,2)
			
		
		)
		self.classfy=nn.Linear(400,10)
		
	def forward(self,x):
		x= self.layer1(x)
		print(x.shape)#64,16,5,5
		x=x.view(x.shape[0],-1)
		x=self.classfy(x)
		return x
model=multi_network()
#获取一个网络的所有卷积层
count=0
conv_model=nn.Sequential()#创建一个空的序列
for layer in model.named_modules():#
	#print(layer[0])
	#print(layer[1])	
	if isinstance(layer[1],nn.Conv2d):
		conv_model.add_module(str(count),layer[1])
		count=count+1
for layer in conv_model:
	print(layer)


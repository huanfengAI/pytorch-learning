#当我们只训练神经网络的全连接，而前面的所有层都固定的时候，假如epoch=30，那么所有的数据集就要进行30个epoch。
#本类用于生成向量
import os
import shutil
import torch
from torch import nn
from torchvision import datasets,transforms,models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from datetime import datetime
from torch.autograd import Variable
import h5py
#conda install h5py
#Tqdm 是一个快速，可扩展的Python进度条，可以在 Python 长循环中添加一个进度提示信息，用户只需要封装任意的迭代器 tqdm(iterator)。

#其实因为前面是固定的，所以每一次epoch的前向传播都是一样的，我们可以计算一次前向传播，然后将数据保存下来
#这个feature_net的作用就是我们传递给它我们想要的模型的名称，它就会帮助我们创建这个模型，然后会将这个模型的非全连接层部分赋值给featurs属性
class feature_net(nn.Module):
	def __init__(self,model):
		super(feature_net,self).__init__()
		
		if model =='vgg':
			vgg=models.vgg19(pretrained=True)
			self.feature=nn.Sequential(*list(vgg.children())[:-1])
			self.feature.add_module('global average',nn.AvgPool2d(9))
			#最后加上一个平均池化层，将结果转化为特征向量
		
		elif model =='inceptionv3':
			inception =models.inception_v3(pretrained=True)
			self.feature=nn.Sequential(*list(inception.children())[:-1])
			self.feature._modules.pop('13')
			self.feature.add_module('global average',nn.AvgPool2d(35))
		elif model =='resnet152':
			resnet =models.resnet152(pretrained=True)
			self.feature =nn.Sequential(*list(resnet.children())[:-1])
			#resnet最后是有一个平均池化层，所以我们就没有不要给它添加了

#self.feature._modules.pop表示删除掉神经网络第13个模块
#(13): InceptionAux(
#      (conv0): BasicConv2d(
#        (conv): Conv2d(768, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#        (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#      )
#      (conv1): BasicConv2d(
#        (conv): Conv2d(128, 768, kernel_size=(5, 5), stride=(1, 1), bias=False)
#        (bn): BatchNorm2d(768, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#      )
#      (fc): Linear(in_features=768, out_features=1000, bias=True)
#    )
	def forward(self,x):
		x=self.feature(x)
		print("11111111")
		print(x.shape)
		x=x.view(x.size(0),-1)
		return x
		

class classfier(nn.Module):
	def __init__(self,dim,n_classes):
		super(classfier,self).__init__()
		self.fc=nn.Sequential(
			nn.Linear(dim,1000),
			nn.ReLU(True),
			nn.Dropout(0.5),
			nn.Linear(1000,n_classes)
		)
	def forward(x):
		x=self.fc(x)
		return x

transform =transforms.Compose([
	transforms.Scale(328),
	transforms.CenterCrop(299),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_set =ImageFolder('dataset/train/',transform=transform)#加载数据，并且进行数据处理
test_set =ImageFolder('dataset/val/',transform=transform)#加载数据，并且进行数据处理


test_data =DataLoader(test_set, batch_size=2, shuffle=False,num_workers=2)
train_data =DataLoader(train_set, batch_size=2, shuffle=False,num_workers=2)
use_gpu = torch.cuda.is_available()
def CreateFeature(model):
	net=feature_net(model)
	print(net)
	if use_gpu:
		net.cuda()
	
	feature_map=torch.FloatTensor()
	label_map=torch.LongTensor()
	len=0
	for data in test_data:#######这里需要指定我们遍历的数据集
		img,label=data
		
		print(img.shape)
		if use_gpu:
        		img = Variable(img, volatile=True).cuda()
               
		else:
        		img = Variable(img, volatile=True)
		
		out=net(img)
		print(img.shape)
		print(len)
		len=len+1
		feature_map=torch.cat((feature_map,out.cpu().data),0)
		label_map=torch.cat((label_map,label),0)
	feature_map=feature_map.detach().numpy()#直接转成numpy不行
	label_name =label_map.detach().numpy()	
	file_name = 'test_data_feature_{}.hd5f'.format(model)#_feature_vgg.hd5f
	################################这里是要指定我们的文件名称
	with h5py.File(file_name,'w') as h:
		h.create_dataset('data',data=feature_map)
		h.create_dataset('lable',data=label_map)
		
CreateFeature('inceptionv3')##############################################这里是指定我们要使用哪个神经网络



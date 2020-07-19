import torch
from torch import nn
from torch.utils import data
from torch.utils.data import DataLoader
import os
from PIL import Image
import numpy as np
from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import random
#使用Dataset构建数据集需要准备三个方法，第一个方法__init__,第二个方法__getitem__,第三个方法__len__
#在__init__方法中把数据（图片数据就是图片的名称）放到一个属性上
class DogCat(data.Dataset):
	def __init__(self,root):
		imgs =os.listdir(root)
		#['cat.12484.jpg', 'dog.12496.jpg', 'dog.12499.jpg', 'cat.12486.jpg', 'dog.12497.jpg', 'cat.12487.jpg', 'dog.12498.jpg', 'cat.12485.jpg']
		self.imgs=[os.path.join(root,img) for img in imgs]
		#print(self.imgs)
	    #['./data/dogcat/cat.12484.jpg', './data/dogcat/dog.12496.jpg', './data/dogcat/dog.12499.jpg', './data/dogcat/cat.12486.jpg', './data/dogcat/dog.12497.jpg', './data/dogcat/cat.12487.jpg', './data/dogcat/dog.12498.jpg', './data/dogcat/cat.12485.jpg']
	def __getitem__(self,index):
	    #getitem根据索引返回样本和样本标签
		img_path =self.imgs[index]
		label =1 if 'dog' in img_path.split('/')[-1] else 0
		pil_img =Image.open(img_path)
		array =np.asarray(pil_img)
		data =torch.from_numpy(array)
		return data,label
	def __len__(self):
		#__len__用于返回样本的长度
		return len(self.imgs)
		
dataset=DogCat('./data/dogcat/')
img,label =dataset[0]#这个是获取dataset中的第一个样本，它等价于dataset.__getitem__(0)

for img,label in dataset:
	print(img.size(),img.float().mean(),label)
#torch.Size([500, 497, 3]) tensor(106.4915) 0
#torch.Size([375, 499, 3]) tensor(116.8138) 1
#torch.Size([400, 300, 3]) tensor(128.1550) 1
#torch.Size([236, 289, 3]) tensor(130.3004) 0
#torch.Size([375, 499, 3]) tensor(150.5079) 1
#torch.Size([374, 499, 3]) tensor(115.5177) 0
#torch.Size([377, 499, 3]) tensor(151.7174) 1
#torch.Size([499, 379, 3]) tensor(171.8085) 0
#这样就获取到了所有的数据了，但是这个图片数据大小不一样，而且返回样本的数值较大，未归一化至[-1, 1]
#可以使用工具包torchvision来完成这个任务

transform =transforms.Compose([
	transforms.Resize(224),
	transforms.CenterCrop(224),
	transforms.ToTensor(),#将图片image转成tensor，他会自动归一化到[0,1]
	transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5])#标准化至[-1,1],规定均值和标准差

])

class DogCat(data.Dataset):
	def __init__(self,root,transforms=None):
		imgs=os.listdir(root)
		self.imgs=[os.path.join(root,img) for img in imgs]
		self.transforms=transforms
	
	def __getitem__(self,index):
		img_path=self.imgs[index]
		label= 0 if 'dog' in img_path.split('/')[-1] else 1
		data =Image.open(img_path)
		if self.transforms:
			data=self.transforms(data)
		return data,label
	def __len__(self):
		return len(self.imgs)

dataset=DogCat('./data/dogcat/',transforms=transform)
for img,label in dataset:
	print(img.size() ,label)
#torch.Size([3, 224, 224]) 1
#torch.Size([3, 224, 224]) 0
#torch.Size([3, 224, 224]) 0
#torch.Size([3, 224, 224]) 1
#torch.Size([3, 224, 224]) 0
#torch.Size([3, 224, 224]) 1
#torch.Size([3, 224, 224]) 0
#torch.Size([3, 224, 224]) 1

#以上的数据处理中一个文件中有多个类别的数据，要是一个文件中的样本的类别都是一样的，文件夹名就是类名，那我们就不用构建dataset了，我们只需要使用ImageFolder来完成这个操作
#ImageFolder(root, transform=None, target_transform=None, loader=default_loader)
#root在指定的路径下寻找图片
#transform对PIL Image进行的转换操作，transform的输入是使用loader读取图片的返回对象
#target_transform`：对label的转换
#`loader`：给定路径后如何读取图片，默认读取为RGB格式的PIL Image对象
dataset=ImageFolder('data/dogcat_2/')#dataset就是所有的数据啦
#print(dataset.class_to_idx){'cat': 0, 'dog': 1}标签并不是文件夹的名称，而是从0到1开始排的
#print(dataset.imgs)#返回图片的路径（样本），和所对应的标签
#[('data/dogcat_2/cat/cat.12484.jpg', 0), ('data/dogcat_2/cat/cat.12485.jpg', 0), ('data/dogcat_2/cat/cat.12486.jpg', 0), ('data/dogcat_2/cat/cat.12487.jpg', 0), ('data/dogcat_2/dog/dog.12496.jpg', 1), ('data/dogcat_2/dog/dog.12497.jpg', 1), ('data/dogcat_2/dog/dog.12498.jpg', 1), ('data/dogcat_2/dog/dog.12499.jpg', 1)]
#print(dataset[3])获取数据集中的第四个样本
#(<PIL.Image.Image image mode=RGB size=499x374 at 0x7FC5F0ADEE80>, 0)
#我们可以看到这个第四个样本还是图片格式并不是tensor，所以我们可以使用transform进行数据处理
transform =transforms.Compose([
	transforms.RandomResizedCrop(224),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.4,0.4,0.4],std=[0.2,0.2,0.2])
])

dataset =ImageFolder('data/dogcat_2/',transform=transform)#加载数据，并且进行数据处理
#dataset是所有的数据
#dataset[0]是所有数据中的第一个样本，格式是这样的（样本，标签）

#print(dataset[0])
#print(dataset[0][1])获取第一个样本的标签
#print(dataset[0][0].size())#torch.Size([3, 224, 224])
#现在的dataset[0][0]是tensor，我们可以将其转换为图片格式
to_img =transforms.ToPILImage()
a=to_img(dataset[0][0]*0.2+0.4)#转换为图片，0.2是标准差，0.4是均值
#plt.imshow(a)
#plt.show()


#############至此我们已经学会了两种图片的处理方式，分别为一个文件中有多个类别的，还有一个是一个文件中就是一个类别的
##现在使用dataset一次只能取一个数据，要想一次取多个数据，我们可以使用Dataset，可以一次取一个batchsize
#DataLoader的函数定义如下： 
# `DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, num_workers=0, collate_fn=default_collate, pin_memory=False, drop_last=False)`
#dataset：加载的数据集对象
#sampler： 样本抽样，后续会详细介绍
# - num_workers：使用多进程加载的进程数，0代表不使用多进程
# - collate_fn： 如何将多个样本数据拼接成一个batch，一般使用默认的拼接方式即可
# - pin_memory：是否将数据保存在pin memory区，pin memory中的数据转到GPU会快一些
# - drop_last：dataset中的数据个数可能不是batch_size的整数倍，drop_last为True会将多出来不足一个batch的数据丢弃

dataloader =DataLoader(dataset,batch_size=3,shuffle=True,num_workers=0,drop_last=False)
dataiter=iter(dataloader)#dataiter为迭代器对象
imgs ,labels=next(dataiter)#取一个batchsize
#print(imgs.size())#torch.Size([3, 3, 224, 224])

#########################有问题的样本##############################
#在数据处理的过程中，如果某个样本无法读取，如果没有办法读取的话，那么我们执行dataset的getitem方法就会产生异常，那么程序就会停止，要想解决这个问题，我们可以新建一个数据集类，然后捕捉异常，然后哦返回None，然后他会自动调用collate_fn来处理，所以我们只需要重写collate——fn方法就ok啦

class NewDogCat(DogCat): #继承前面实现的DogCat数据集
	def __getitem__(self,index):
		try:
			return super(NewDogCat,self).__getitem__(index)
		except:
			return None,None

from torch.utils.data.dataloader import default_collate # 导入默认的拼接方式

def my_collate_fn(batch):#batch表示一个batch中的所有的数据
	batch =list(filter(lambda x:x[0] is not None ,batch))#过滤掉有问题的样本，这里有问题的样本我们上面已经处理返回None
	if len(batch)==0:return torch.tensor()
	return default_collate(batch)
	

dataset=NewDogCat('data/dogcat_wrong/',transforms=transform)
#print(dataset[8])#获取第8个数据
#print(dataset[9])#获取第9个数据，因为总共有8个数据集，第9个数据并没有存在，此时返回None
#通过collate_fn参数，指定自定义的my_collate_fn
dataloader =DataLoader(dataset,batch_size=2,collate_fn=my_collate_fn,num_workers=1,shuffle=True)
#for batch_datas,batch_labels in dataloader:
#	print(batch_datas.size(),batch_labels.size())
#torch.Size([2, 3, 224, 224]) torch.Size([2])
#torch.Size([2, 3, 224, 224]) torch.Size([2])
#torch.Size([2, 3, 224, 224]) torch.Size([2])
#torch.Size([1, 3, 224, 224]) torch.Size([1])
#因为最后一张图片有问题，所以最后一个batch_size为1，如果我们要是想要丢弃这个batch我们可以使用drop_last=True来丢弃最后一个不足batch_size的batch
#以上是一种方式，就是过滤掉有问题的样本，还有一种方法就是只要样本存在问题，就随机取一张图片代替


class NewDogCat(DogCat):
     def __getitem__(self, index):
         try:
             return super(NewDogCat, self).__getitem__(index)
         except:
             new_index = random.randint(0, len(self)-1)
             return self[new_index]	
dataset=NewDogCat('data/dogcat_wrong/',transforms=transform)	
dataloader =DataLoader(dataset,batch_size=2,collate_fn=my_collate_fn,num_workers=1,shuffle=True)
#for batch_datas,batch_labels in dataloader:
#	print(batch_datas.size(),batch_labels.size())	
#data/dagcat_wrong下一共有9张图片，其中8张正常图片，1张不正常的图片，当不正常的图片出现问题的时候，就会随机找一张替代它
#torch.Size([2, 3, 224, 224]) torch.Size([2])
#torch.Size([2, 3, 224, 224]) torch.Size([2])
#torch.Size([2, 3, 224, 224]) torch.Size([2])
#torch.Size([2, 3, 224, 224]) torch.Size([2])
#torch.Size([1, 3, 224, 224]) torch.Size([1])
	
# 1. 高负载的操作放在`__getitem__`中，如加载图片等。
#dataset中应尽量只包含只读对象，避免修改任何可变对象，这样多线程操作就不会出现问题。
#class BadDataset(Dataset):
#     def __init__(self):
#         self.datas = range(100)
#         self.num = 0 # 取数据的次数
#     def __getitem__(self, index):
#         self.num += 1
#         return self.datas[index]
# ```
# 当我们使用dataloader的时候，有一个参数是shuffle=True的时候，它就相当于使用随机采样器RandomSampler实现打乱数据。默认的是采用`SequentialSampler`，它会按顺序一个一个进行采样。

#还有一个是weightedRandomSampler，它可以根据每个样本的权重选取数据，在样本比例不均衡的问题中，我们可以使用它来完成重采样。
#要想使用它需要提供两个参数，第一个参数是每个样本的权重weights，共选取的样本的总数num_samples,还有一个可选参数replacement（默认为true，是否可以重复选取某一个样本）即允许在一个epoch中重复采样某一个数据。如果设为False，则当某一类的样本被全部选取完，但其样本数目仍未达到num_samples时，sampler将不会再从该类中选择数据，此时可能导致`weights`参数失效。
dataset =DogCat('data/dogcat/',transforms=transform)
#下面我们设置一个权重列表，狗的图片被取出的概率是猫的概率的两倍，两类图片被取出的概率与weights的绝对大小无关，只和比值有关
weights=[2 if label == 1 else 1 for data , label in dataset]
#print(weights)[2, 1, 1, 2, 1, 2, 1, 2]权重参数和样本的顺序是一一对应的
from torch.utils.data.sampler import  WeightedRandomSampler

#下面我们先来构建一个Sampler对象
sampler=WeightedRandomSampler(weights,num_samples=9,replacement=True)
#然后我们使用类加载器的时候，可以使用Sampler
dataloader=DataLoader(dataset,batch_size=3,sampler=sampler)
#for datas,labels in dataloader:
#	print(labels.tolist())
#[1, 1, 1]
#[1, 1, 1]
#[0, 0, 0]
#一共有8个样本，但是返回了9个肯定有重复返回的，这就是replacement的作用

#也就是说如果dataloader指定了sampler，那么shuffle将不在生效，并且sampler.num_samples会覆盖dataset的实际大小，即一个epoch返回的图片总数取决于`sampler.num_samples`，如果实际dataset不够用的话，我们可以设置replacement=True


#计算机视觉工具包torchvision
#torchvision主要包含三个方面
#第一个方面是models：提供深度学习中各种经典网络的网络结构以及预训练好的模型，包括`AlexNet`、VGG系列、ResNet系列、Inception系列等。
# - datasets： 提供常用的数据集加载，设计上都是继承`torhc.utils.data.Dataset`，主要包括`MNIST`、`CIFAR10/100`、`ImageNet`、`COCO`等。
# - transforms：提供常用的数据预处理操作，主要包括对Tensor以及PIL Image对象的操作。
#我们下面来看一下如何才能加载已经训练好的模型
from torchvision import models
#resnet34=models.squeezenet1_1(pretrained=True,num_classes=1000)
#resnet34.fc=nn.Linear(512,10)#然后修改其全连接层，使其变为10分类的问题
#下载的未知为/home/dongxinge/.torch/models/squeezenet1_1-f364aa15.pth
from torchvision import datasets
#dataset=datasets.MNIST('data/',download=True,train=False,transform=transforms)
#data/为数据集下载的位置
#train=False为下载的数据为测试集
#transform为下载的数据进行转换操作

from torchvision import transforms
to_pil =transforms.ToPILImage()#构建一个transform转换操作
a=to_pil(torch.randn(3,64,64))#对torch.randn(3,64,64)进行转换操作。此时它还是图片，并不是张量

#torchvison的两个常用的函数，
#make_grid,它能够将多张图片拼接成一个网格中
#save_img它能将tensor保存成图片

dataloader =DataLoader(dataset,shuffle=True,batch_size=16)
from torchvision.utils import make_grid,save_image
dataiter =iter(dataloader)
#print(len(next(dataiter)[0]))8,因为一共有8张图片，所以虽然batch_size为16，但是只能取出8张来
img =make_grid(next(dataiter)[0],4)#拼接成4*4的网格图片，并且会将图片转成3通道的
#print(img)<PIL.Image.Image image mode=RGB size=906x454 at 0x7F102350C240>
a=to_img(img)#将tensor转成图片
#print(a)#<PIL.Image.Image image mode=RGB size=906x454 at 0x7F2C637872B0>
#save_image(img, 'a.png')#保存图片，名称是a.png
#Image.open('a.png')
